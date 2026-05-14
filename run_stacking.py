"""Stacking Ensemble: meta-learner поверх предсказаний базовых моделей.

Вместо простого усреднения (averaging ensemble) обучает Ridge-регрессию
на предсказаниях базовых моделей, используя validation set для подбора
оптимальных весов каждой модели.

Стратегии:
  1. Averaging (baseline) — среднее предсказаний
  2. Inverse-MAE weighting — веса обратно пропорциональны MAE на val
  3. Ridge stacking — Ridge(alpha) обучается на val predictions
  4. Ridge stacking + uncertainty — добавляет sigma каждой модели как features

Usage:
    python run_stacking.py --output_dir /path/to/experiments
"""
import argparse
import json
import re
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler as SkScaler

from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(data_dir="data", seed=42):
    ds = load_and_unify_datasets(data_dir)
    split = grouped_stratified_split(ds, seed=seed)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()
    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
    data = {}
    for key in ["train", "val", "test"]:
        idx = split[key]
        data[key] = {
            "x": feat_scaler.transform(x_all[idx]),
            "y": tgt_scaler.transform(y_all[idx].reshape(-1, 1)).ravel(),
            "ages": ages_all[idx],
        }
    data["input_dim"] = data["train"]["x"].shape[1]
    data["feat_scaler"] = feat_scaler
    data["tgt_scaler"] = tgt_scaler
    data["n_train"] = len(data["train"]["x"])
    return data


def parse_hidden_dims(filename):
    """Extract hidden_dims from checkpoint filename."""
    m = re.search(r'_h([\dx]+)_', filename)
    if m:
        return [int(x) for x in m.group(1).split('x')]
    return [256, 128, 64]


def load_model(ckpt_path, input_dim):
    """Load a generator from checkpoint, auto-detecting architecture."""
    hidden = parse_hidden_dims(ckpt_path.name)
    gen = ConcreteGenerator(GeneratorConfig(
        input_dim=input_dim, hidden_dims=hidden, dropout=0.1, seed=42,
    ))
    gen.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True))
    gen.eval()
    return gen, hidden


def get_predictions(model, x, tgt_scaler, mc_samples=30):
    """Get (mu, sigma) predictions in original scale."""
    model = model.cpu()
    mu_s, sigma_s = model.predict(x, mc_samples=mc_samples)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    sigma = sigma_s.ravel() * tgt_scaler.scale[0]
    return mu, sigma


def compute_metrics(y_true, y_pred, y_std=None, ages=None):
    """Compute MAE, RMSE, R2, PICP."""
    ev = evaluate_model(y_true=y_true, y_pred=y_pred, y_std=y_std, age_days=ages)
    r = {"mae": ev.regression.mae, "rmse": ev.regression.rmse,
         "r2": ev.regression.r2, "mape": ev.regression.mape}
    if ev.calibration:
        r["picp"] = ev.calibration.picp
        r["mpiw"] = ev.calibration.mpiw
    return r


# ====================================================================
# STACKING STRATEGIES
# ====================================================================

def stacking_average(val_preds, val_y, test_preds, test_sigmas):
    """Simple averaging baseline."""
    test_mu = np.mean(test_preds, axis=0)
    test_sigma = np.sqrt(
        np.mean(test_sigmas**2, axis=0) + np.var(test_preds, axis=0))
    return test_mu, test_sigma


def stacking_inverse_mae(val_preds, val_y, test_preds, test_sigmas):
    """Weight each model inversely proportional to its validation MAE."""
    n_models = val_preds.shape[0]
    maes = np.array([np.mean(np.abs(val_preds[i] - val_y)) for i in range(n_models)])
    weights = 1.0 / (maes + 1e-6)
    weights /= weights.sum()

    log(f"    Inverse-MAE weights: {[f'{w:.3f}' for w in weights]}")

    test_mu = np.average(test_preds, axis=0, weights=weights)
    test_sigma = np.sqrt(
        np.average(test_sigmas**2, axis=0, weights=weights)
        + np.average((test_preds - test_mu)**2, axis=0, weights=weights))
    return test_mu, test_sigma


def stacking_ridge(val_preds, val_y, test_preds, test_sigmas):
    """Ridge regression meta-learner on base model predictions."""
    X_val = val_preds.T    # [n_val, n_models]
    X_test = test_preds.T  # [n_test, n_models]

    scaler = SkScaler()
    X_val_s = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    meta = RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                   fit_intercept=True)
    meta.fit(X_val_s, val_y)

    log(f"    Ridge alpha={meta.alpha_:.2f}, coefs={[f'{c:.3f}' for c in meta.coef_]}")

    test_mu = meta.predict(X_test_s)

    # Uncertainty: propagate through linear combination
    # σ² = Σ w_i² σ_i² + model disagreement
    w = meta.coef_ / (scaler.scale_ + 1e-8)
    weighted_var = np.sum((w**2)[:, None] * test_sigmas**2, axis=0)
    disagreement = np.var(test_preds, axis=0)
    test_sigma = np.sqrt(weighted_var + disagreement)

    return test_mu, test_sigma


def stacking_ridge_with_sigma(val_preds, val_sigmas, val_y,
                              test_preds, test_sigmas):
    """Ridge on [predictions, uncertainties] — 2×n_models features."""
    X_val = np.vstack([val_preds, val_sigmas]).T    # [n_val, 2*n_models]
    X_test = np.vstack([test_preds, test_sigmas]).T

    scaler = SkScaler()
    X_val_s = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    meta = RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                   fit_intercept=True)
    meta.fit(X_val_s, val_y)

    log(f"    Ridge+sigma alpha={meta.alpha_:.2f}")

    test_mu = meta.predict(X_test_s)
    test_sigma = np.sqrt(np.var(test_preds, axis=0)
                         + np.mean(test_sigmas**2, axis=0))

    return test_mu, test_sigma


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    device = get_device()
    log(f"Device: {device}")
    log("Loading data...")

    data = load_data(args.data_dir)
    tgt_scaler = data["tgt_scaler"]
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    tracker = ExperimentTracker(args.output_dir)

    log(f"Data: train={data['n_train']}, input_dim={data['input_dim']}")

    # ── 1. Load all models ──────────────────────────────────────────
    log("=" * 60)
    log("LOADING BASE MODELS")
    log("=" * 60)

    models = []
    model_names = []

    # GAN models
    gan_ckpts = sorted(checkpoint_dir.glob("gan_*.pt"))
    for ckpt in gan_ckpts:
        try:
            gen, hidden = load_model(ckpt, data["input_dim"])
            models.append(gen)
            model_names.append(ckpt.stem)
            log(f"  ✓ {ckpt.name} ({hidden})")
        except Exception as e:
            log(f"  ✗ {ckpt.name}: {e}")

    # Multi-seed supervised models
    seed_ckpts = sorted(checkpoint_dir.glob("seed_*.pt"))
    for ckpt in seed_ckpts:
        try:
            gen, hidden = load_model(ckpt, data["input_dim"])
            models.append(gen)
            model_names.append(ckpt.stem)
            log(f"  ✓ {ckpt.name} ({hidden})")
        except Exception as e:
            log(f"  ✗ {ckpt.name}: {e}")

    n_models = len(models)
    log(f"\nTotal base models: {n_models}")

    if n_models < 2:
        log("ERROR: Need at least 2 models for stacking!")
        return

    # ── 2. Get predictions on val and test ───────────────────────────
    log("\n" + "=" * 60)
    log("COLLECTING PREDICTIONS")
    log("=" * 60)

    y_val_orig = tgt_scaler.inverse_transform(
        data["val"]["y"].reshape(-1, 1)).ravel()
    y_test_orig = tgt_scaler.inverse_transform(
        data["test"]["y"].reshape(-1, 1)).ravel()
    ages_test = data["test"]["ages"]

    val_preds = np.zeros((n_models, len(y_val_orig)))
    val_sigmas = np.zeros((n_models, len(y_val_orig)))
    test_preds = np.zeros((n_models, len(y_test_orig)))
    test_sigmas = np.zeros((n_models, len(y_test_orig)))

    for i, (model, name) in enumerate(zip(models, model_names)):
        mu_v, sigma_v = get_predictions(model, data["val"]["x"], tgt_scaler)
        mu_t, sigma_t = get_predictions(model, data["test"]["x"], tgt_scaler)

        val_preds[i] = mu_v
        val_sigmas[i] = sigma_v
        test_preds[i] = mu_t
        test_sigmas[i] = sigma_t

        val_mae = np.mean(np.abs(mu_v - y_val_orig))
        test_mae = np.mean(np.abs(mu_t - y_test_orig))
        log(f"  [{i+1}/{n_models}] {name}: val_MAE={val_mae:.2f}, test_MAE={test_mae:.2f}")

    # ── 3. Run stacking strategies ───────────────────────────────────
    log("\n" + "=" * 60)
    log("STACKING STRATEGIES")
    log("=" * 60)

    results = {}

    strategies = [
        ("averaging", lambda: stacking_average(
            val_preds, y_val_orig, test_preds, test_sigmas)),
        ("inverse_mae", lambda: stacking_inverse_mae(
            val_preds, y_val_orig, test_preds, test_sigmas)),
        ("ridge", lambda: stacking_ridge(
            val_preds, y_val_orig, test_preds, test_sigmas)),
        ("ridge_sigma", lambda: stacking_ridge_with_sigma(
            val_preds, val_sigmas, y_val_orig, test_preds, test_sigmas)),
    ]

    # Also try GAN-only stacking (exclude multi-seed supervised)
    n_gan = len(gan_ckpts)
    if n_gan >= 2 and n_gan < n_models:
        strategies.extend([
            ("ridge_gan_only", lambda: stacking_ridge(
                val_preds[:n_gan], y_val_orig, test_preds[:n_gan], test_sigmas[:n_gan])),
            ("inv_mae_gan_only", lambda: stacking_inverse_mae(
                val_preds[:n_gan], y_val_orig, test_preds[:n_gan], test_sigmas[:n_gan])),
        ])

    for name, strategy_fn in strategies:
        log(f"\n  === {name} ===")
        t0 = time.time()
        try:
            with tracker.run(f"stack_{name}", tags=["stacking"]) as run:
                mu, sigma = strategy_fn()
                m = compute_metrics(y_test_orig, mu, sigma, ages_test)
                run.log_metrics(m)
                dt = time.time() - t0

                results[name] = m
                log(f"    MAE={m['mae']:.2f}, R2={m['r2']:.4f}, "
                    f"PICP={m.get('picp', 'N/A')}, RMSE={m['rmse']:.2f} ({dt:.1f}s)")
        except Exception as e:
            log(f"    ERROR: {e}")

    # ── 4. Summary ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("STACKING RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Strategy':<25} {'MAE':>8} {'R2':>8} {'RMSE':>8} {'PICP':>8}")
    log("-" * 60)

    best_mae = float("inf")
    best_name = ""
    for name, m in sorted(results.items(), key=lambda x: x[1]["mae"]):
        picp_str = f"{m['picp']:.4f}" if "picp" in m else "N/A"
        marker = ""
        if m["mae"] < best_mae:
            best_mae = m["mae"]
            best_name = name
            marker = " ★"
        log(f"{name:<25} {m['mae']:8.2f} {m['r2']:8.4f} "
            f"{m['rmse']:8.2f} {picp_str:>8}{marker}")

    log(f"\n🏆 Best: {best_name} → MAE={best_mae:.2f}")

    # Save results
    with open(checkpoint_dir / "stacking_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log("\nDone!")


if __name__ == "__main__":
    main()
