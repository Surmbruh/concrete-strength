"""Final optimization: evaluate and train for t=28 days only.

Key insight from organizers: 28-day strength is THE main criterion.
Our current MAE=5.03 includes all ages (t=1,3,7,28,...).
Filtering to t=28 should give much better metrics.

Usage:
    python run_final_t28.py --output_dir /path/to/experiments
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.metrics import evaluate_model
from materialgen.tracker import get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_enhanced_features(x):
    cement, water = x[:, 0:1], x[:, 1:2]
    sand, coarse_agg = x[:, 2:3], x[:, 3:4]
    w_c, log_age = x[:, 7:8], x[:, 9:10]
    extras = np.hstack([
        cement * log_age, w_c * log_age, cement * water,
        log_age ** 2, sand + coarse_agg,
    ])
    return np.hstack([x, extras])


def load_all_data(data_dir="data", seed=42):
    """Load data and separate t=28 subset."""
    ds = load_and_unify_datasets(data_dir)
    split = stratified_split(ds, seed=seed)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()

    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

    data = {"feat_scaler": feat_scaler, "tgt_scaler": tgt_scaler}
    for key in ["train", "val", "test"]:
        idx = split[key]
        ages = ages_all[idx]
        data[key] = {
            "x": feat_scaler.transform(x_all[idx]),
            "y": tgt_scaler.transform(y_all[idx].reshape(-1, 1)).ravel(),
            "y_orig": y_all[idx],
            "ages": ages,
        }
        # t=28 subset
        m28 = (ages == 28)
        data[f"{key}_28"] = {
            "x": feat_scaler.transform(x_all[idx][m28]),
            "y": tgt_scaler.transform(y_all[idx][m28].reshape(-1, 1)).ravel(),
            "y_orig": y_all[idx][m28],
            "ages": ages[m28],
        }
    data["input_dim"] = data["train"]["x"].shape[1]
    log(f"Train: {len(data['train']['x'])} (t28: {len(data['train_28']['x'])})")
    log(f"Val:   {len(data['val']['x'])} (t28: {len(data['val_28']['x'])})")
    log(f"Test:  {len(data['test']['x'])} (t28: {len(data['test_28']['x'])})")
    return data


def eval_on(gen, x, y_orig, ages, tgt_scaler, mc=20):
    mu_s, sig_s = gen.predict(x, mc_samples=mc)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    sigma = sig_s.ravel() * tgt_scaler.scale[0]
    ev = evaluate_model(y_true=y_orig, y_pred=mu, y_std=sigma, age_days=ages)
    return {"mae": ev.regression.mae, "r2": ev.regression.r2,
            "rmse": ev.regression.rmse,
            "picp": ev.calibration.picp if ev.calibration else None}


# ====================================================================
# PART 1: Evaluate existing models on t=28 only
# ====================================================================

def evaluate_existing_t28(data, checkpoint_dir):
    log("=" * 60)
    log("EXISTING MODELS — t=28 EVALUATION")
    log("=" * 60)

    tgt = data["tgt_scaler"]
    x28, y28, a28 = data["test_28"]["x"], data["test_28"]["y_orig"], data["test_28"]["ages"]
    results = []

    # GAN models
    for ckpt in sorted(checkpoint_dir.glob("gan_*.pt")):
        for hid in [[256,128,64],[256,128,64,32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hid, dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                m = eval_on(gen, x28, y28, a28, tgt)
                results.append({"name": ckpt.stem, "type": "GAN", **m})
                log(f"  {ckpt.stem}: MAE={m['mae']:.2f}, R2={m['r2']:.4f}")
                break
            except Exception:
                continue

    results.sort(key=lambda r: r["mae"])
    log(f"\nBest individual t=28: {results[0]['name']} → MAE={results[0]['mae']:.2f}")
    return results


# ====================================================================
# PART 2: Train t=28-only models
# ====================================================================

def train_t28_models(data, checkpoint_dir, n_seeds=5):
    log("\n" + "=" * 60)
    log("TRAINING t=28-ONLY MODELS")
    log("=" * 60)

    results = []
    for seed in range(42, 42 + n_seeds):
        tag = f"t28_s{seed}"
        ckpt = checkpoint_dir / f"{tag}.pt"
        t0 = time.time()

        gen = ConcreteGenerator(GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=[256, 128, 64],
            epochs=400, batch_size=64, learning_rate=1e-3,
            weight_decay=1e-4, dropout=0.1, seed=seed))

        if ckpt.exists():
            gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            log(f"  {tag}: loaded")
        else:
            train_generator_supervised(
                gen, data["train_28"]["x"], data["train_28"]["y"],
                data["val_28"]["x"], data["val_28"]["y"], config=gen.config)
            gen = gen.cpu()
            torch.save(gen.state_dict(), ckpt)

        m = eval_on(gen, data["test_28"]["x"], data["test_28"]["y_orig"],
                    data["test_28"]["ages"], data["tgt_scaler"])
        results.append({"tag": tag, **m})
        log(f"  {tag}: MAE={m['mae']:.2f}, R2={m['r2']:.4f} ({time.time()-t0:.0f}s)")

    return results


# ====================================================================
# PART 3: Stacking on t=28 test
# ====================================================================

def stacking_t28(data, checkpoint_dir):
    log("\n" + "=" * 60)
    log("STACKING — t=28 EVALUATION")
    log("=" * 60)

    tgt = data["tgt_scaler"]

    # Combine train+val (all ages) for meta-train
    x_tv = np.concatenate([data["train"]["x"], data["val"]["x"]])
    y_tv = np.concatenate([data["train"]["y_orig"], data["val"]["y_orig"]])
    ages_tv = np.concatenate([data["train"]["ages"], data["val"]["ages"]])
    mask28_tv = (ages_tv == 28)

    # t=28 test
    x28 = data["test_28"]["x"]
    y28 = data["test_28"]["y_orig"]
    a28 = data["test_28"]["ages"]

    # Enhanced features
    enh_scaler = StandardScaler.fit(add_enhanced_features(data["train"]["x"]))

    # Collect predictions
    names, preds_tv, preds_t28, sigmas_t28 = [], [], [], []

    # GAN models
    for ckpt in sorted(checkpoint_dir.glob("gan_*.pt")):
        for hid in [[256,128,64],[256,128,64,32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hid, dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                mu_tv, _ = gen.predict(x_tv, mc_samples=20)
                mu_t, sig_t = gen.predict(x28, mc_samples=20)
                preds_tv.append(tgt.inverse_transform(mu_tv).ravel())
                preds_t28.append(tgt.inverse_transform(mu_t).ravel())
                sigmas_t28.append(sig_t.ravel() * tgt.scale[0])
                names.append(f"GAN:{ckpt.stem}")
                break
            except Exception:
                continue

    # Enhanced models
    for ckpt in sorted(checkpoint_dir.glob("enh_*.pt")):
        for hid in [[256,128,64],[512,256,128]]:
            try:
                dim_enh = data["input_dim"] + 5
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=dim_enh, hidden_dims=hid, dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                x_tv_e = enh_scaler.transform(add_enhanced_features(x_tv))
                x28_e = enh_scaler.transform(add_enhanced_features(x28))
                mu_tv, _ = gen.predict(x_tv_e, mc_samples=20)
                mu_t, sig_t = gen.predict(x28_e, mc_samples=20)
                preds_tv.append(tgt.inverse_transform(mu_tv).ravel())
                preds_t28.append(tgt.inverse_transform(mu_t).ravel())
                sigmas_t28.append(sig_t.ravel() * tgt.scale[0])
                names.append(f"ENH:{ckpt.stem}")
                break
            except Exception:
                continue

    # t=28-only models
    for ckpt in sorted(checkpoint_dir.glob("t28_*.pt")):
        try:
            gen = ConcreteGenerator(GeneratorConfig(
                input_dim=data["input_dim"], hidden_dims=[256,128,64], dropout=0.1, seed=42))
            gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            mu_tv, _ = gen.predict(x_tv, mc_samples=20)
            mu_t, sig_t = gen.predict(x28, mc_samples=20)
            preds_tv.append(tgt.inverse_transform(mu_tv).ravel())
            preds_t28.append(tgt.inverse_transform(mu_t).ravel())
            sigmas_t28.append(sig_t.ravel() * tgt.scale[0])
            names.append(f"T28:{ckpt.stem}")
            break
        except Exception:
            continue

    log(f"Total base models: {len(names)}")

    X_tv = np.stack(preds_tv, axis=1)
    X_t28 = np.stack(preds_t28, axis=1)
    S_t28 = np.stack(sigmas_t28, axis=1)

    # Simple average
    simple_mu = X_t28.mean(axis=1)
    simple_sig = np.sqrt(X_t28.std(axis=1)**2 + S_t28.mean(axis=1)**2)
    ev = evaluate_model(y_true=y28, y_pred=simple_mu, y_std=simple_sig, age_days=a28)
    log(f"\n  Simple avg t=28: MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")

    # K-fold stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, factory in [
        ("ridge_a10", lambda: Ridge(alpha=10)),
        ("ridge_a100", lambda: Ridge(alpha=100)),
        ("gbr_n30", lambda: GradientBoostingRegressor(
            n_estimators=30, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)),
        ("gbr_n50", lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)),
    ]:
        oof = np.zeros(len(y_tv))
        test_folds = []
        fold_maes = []

        for tr_idx, va_idx in kf.split(X_tv):
            m = factory()
            m.fit(X_tv[tr_idx], y_tv[tr_idx])
            oof[va_idx] = m.predict(X_tv[va_idx])
            fold_maes.append(np.mean(np.abs(y_tv[va_idx] - oof[va_idx])))
            test_folds.append(m.predict(X_t28))

        oof_mae = np.mean(np.abs(y_tv - oof))
        # OOF on t=28 subset of train+val
        oof_28_mae = np.mean(np.abs(y_tv[mask28_tv] - oof[mask28_tv]))

        test_mu = np.stack(test_folds).mean(axis=0)
        w = np.ones(X_t28.shape[1]) / X_t28.shape[1]
        test_sig = np.sqrt(X_t28.std(axis=1)**2 + (S_t28 * w).sum(axis=1)**2)

        ev = evaluate_model(y_true=y28, y_pred=test_mu, y_std=test_sig, age_days=a28)
        results[name] = {
            "oof_mae_all": float(oof_mae),
            "oof_mae_t28": float(oof_28_mae),
            "test_mae_t28": ev.regression.mae,
            "test_r2_t28": ev.regression.r2,
            "test_rmse_t28": ev.regression.rmse,
            "picp": ev.calibration.picp if ev.calibration else None,
        }
        log(f"  {name:<15}: OOF(all)={oof_mae:.2f}, OOF(t28)={oof_28_mae:.2f}, "
            f"Test(t28)={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")

    # Weighted average
    from scipy.optimize import minimize
    def obj(w):
        wn = np.abs(w) / np.abs(w).sum()
        return np.mean(np.abs(y_tv - (X_tv * wn).sum(axis=1)))

    best_w = None
    best_obj = float("inf")
    for s in range(30):
        w0 = np.random.default_rng(s).dirichlet(np.ones(X_tv.shape[1]))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
        if res.fun < best_obj:
            best_obj = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()

    wt_mu = (X_t28 * best_w).sum(axis=1)
    wt_sig = np.sqrt(X_t28.std(axis=1)**2 + (S_t28 * best_w).sum(axis=1)**2)
    ev = evaluate_model(y_true=y28, y_pred=wt_mu, y_std=wt_sig, age_days=a28)
    results["weighted"] = {
        "test_mae_t28": ev.regression.mae, "test_r2_t28": ev.regression.r2,
        "picp": ev.calibration.picp if ev.calibration else None,
    }
    log(f"  weighted:        Test(t28)={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")

    results["simple_avg"] = {
        "test_mae_t28": float(np.mean(np.abs(y28 - simple_mu))),
        "test_r2_t28": float(1 - np.sum((y28-simple_mu)**2)/np.sum((y28-y28.mean())**2)),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    log(f"Device: {get_device()}")
    ckpt = Path(args.output_dir) / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    data = load_all_data(args.data_dir)

    # 1. Evaluate existing on t=28
    ind_results = evaluate_existing_t28(data, ckpt)

    # 2. Train t=28-only models
    t28_results = train_t28_models(data, ckpt, n_seeds=5)

    # 3. Stacking on t=28
    stack_results = stacking_t28(data, ckpt)

    # Final summary
    log("\n" + "=" * 60)
    log("FINAL t=28 RESULTS")
    log("=" * 60)
    log(f"{'Method':<25} {'MAE(t28)':>10} {'R2(t28)':>10}")
    log("-" * 45)
    for n, r in sorted(stack_results.items(), key=lambda x: x[1]["test_mae_t28"]):
        log(f"{n:<25} {r['test_mae_t28']:10.3f} {r['test_r2_t28']:10.4f}")

    best = min(stack_results, key=lambda k: stack_results[k]["test_mae_t28"])
    log(f"\nBEST t=28: {best} → MAE={stack_results[best]['test_mae_t28']:.3f}")

    all_res = {"individual_t28": ind_results, "t28_models": t28_results,
               "stacking_t28": stack_results}
    with open(ckpt / "final_t28_results.json", "w") as f:
        json.dump(all_res, f, indent=2, default=str)
    log("Done!")


if __name__ == "__main__":
    main()
