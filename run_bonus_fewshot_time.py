"""Бонус 2: Few-Shot Evaluation + Прочность во времени.

Часть A: Few-shot — как модель работает при ограниченных данных (n=50,100,200,500).
Часть B: Time curves — модель предсказывает прочность для разных возрастов по одному составу.

Usage:
    python run_bonus_fewshot_time.py --output_dir /path/to/experiments
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ====================================================================
# PART A: FEW-SHOT EVALUATION
# ====================================================================

def run_fewshot(data, ds, feat_scaler, tgt_scaler, tracker, checkpoint_dir):
    """Обучить модель на подмножествах разного размера.
    
    Показывает как качество деградирует при уменьшении данных —
    и насколько наша архитектура устойчива к малым выборкам.
    """
    log("=" * 60)
    log("FEW-SHOT EVALUATION")
    log("=" * 60)
    
    sample_sizes = [50, 100, 200, 500, 1000, 2000]
    n_seeds = 3  # Повторяем 3 раза для оценки дисперсии
    results = []
    
    x_train_full = data["train"]["x"]
    y_train_full = data["train"]["y"]
    x_val = data["val"]["x"]
    y_val = data["val"]["y"]
    
    for n_samples in sample_sizes:
        if n_samples > len(x_train_full):
            log(f"  n={n_samples}: skip (only {len(x_train_full)} available)")
            continue
        
        maes, r2s = [], []
        for seed_offset in range(n_seeds):
            seed = 42 + seed_offset
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x_train_full), n_samples, replace=False)
            
            tag = f"fewshot_n{n_samples}_s{seed}"
            t0 = time.time()
            
            gen = ConcreteGenerator(GeneratorConfig(
                input_dim=data["input_dim"],
                hidden_dims=[256, 128, 64] if n_samples >= 200 else [128, 64],
                epochs=300, batch_size=min(32, n_samples // 4),
                learning_rate=1e-3, weight_decay=1e-3,  # stronger reg for small data
                dropout=0.2 if n_samples < 200 else 0.1,
                seed=seed,
            ))
            train_generator_supervised(
                gen, x_train_full[idx], y_train_full[idx],
                x_val, y_val, config=gen.config,
            )
            gen = gen.cpu()
            
            # Evaluate on full test set
            mu_s, sigma_s = gen.predict(data["test"]["x"], mc_samples=20)
            mu = tgt_scaler.inverse_transform(mu_s).ravel()
            sigma = sigma_s.ravel() * tgt_scaler.scale[0]
            y_orig = tgt_scaler.inverse_transform(
                data["test"]["y"].reshape(-1, 1)).ravel()
            
            ev = evaluate_model(y_true=y_orig, y_pred=mu, y_std=sigma,
                                age_days=data["test"]["ages"])
            dt = time.time() - t0
            
            maes.append(ev.regression.mae)
            r2s.append(ev.regression.r2)
            log(f"  n={n_samples}, seed={seed}: MAE={ev.regression.mae:.2f}, "
                f"R2={ev.regression.r2:.4f} ({dt:.0f}s)")
        
        result = {
            "n_samples": n_samples,
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }
        results.append(result)
        log(f"  → n={n_samples}: MAE={result['mae_mean']:.2f}±{result['mae_std']:.2f}, "
            f"R2={result['r2_mean']:.4f}±{result['r2_std']:.4f}")
    
    # Summary
    log("\nFEW-SHOT SUMMARY:")
    log(f"{'n_train':<10} {'MAE':>12} {'R2':>12}")
    log("-" * 34)
    for r in results:
        log(f"{r['n_samples']:<10} {r['mae_mean']:7.2f}±{r['mae_std']:.2f} "
            f"{r['r2_mean']:7.4f}±{r['r2_std']:.4f}")
    
    with open(checkpoint_dir / "fewshot_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


# ====================================================================
# PART B: TIME-DEPENDENT PREDICTION
# ====================================================================

def run_time_curves(data, ds, feat_scaler, tgt_scaler, tracker, checkpoint_dir):
    """Оценить как модель предсказывает прочность для разных возрастов.
    
    Берём составы, для которых есть измерения при нескольких возрастах,
    и проверяем:
    1. Монотонность: f(t=28) > f(t=7) > f(t=3) > f(t=1)?
    2. Точность: MAE по возрастным группам
    3. Кривая: для 5 случайных составов строим f(t) vs реальность
    """
    log("=" * 60)
    log("TIME-DEPENDENT STRENGTH PREDICTION")
    log("=" * 60)
    
    # Load or create best model
    gen = ConcreteGenerator(GeneratorConfig(
        input_dim=data["input_dim"], hidden_dims=[256, 128, 64],
        dropout=0.1, seed=42))
    
    ckpt_paths = list(checkpoint_dir.glob("*best*.pt")) + list(
        checkpoint_dir.glob("sup_lr0.001_h256x128x64_d0.1*.pt"))
    if ckpt_paths:
        gen.load_state_dict(
            torch.load(ckpt_paths[0], map_location="cpu", weights_only=True))
        log(f"Loaded model from {ckpt_paths[0].name}")
    else:
        log("Training model...")
        train_generator_supervised(
            gen, data["train"]["x"], data["train"]["y"],
            data["val"]["x"], data["val"]["y"],
            config=GeneratorConfig(
                input_dim=data["input_dim"], hidden_dims=[256, 128, 64],
                epochs=300, batch_size=64, learning_rate=1e-3, dropout=0.1, seed=42))
        gen = gen.cpu()
    
    # ── Test per age group ──────────────────────────────────────────
    ages = data["test"]["ages"]
    x_test = data["test"]["x"]
    y_test = data["test"]["y"]
    
    age_groups = {}
    for age in sorted(np.unique(ages)):
        mask = ages == age
        if mask.sum() >= 5:
            age_groups[int(age)] = mask
    
    log(f"\nAge groups in test set: {list(age_groups.keys())}")
    
    per_age_results = []
    for age, mask in age_groups.items():
        mu_s, sigma_s = gen.predict(x_test[mask], mc_samples=20)
        mu = tgt_scaler.inverse_transform(mu_s).ravel()
        y_orig = tgt_scaler.inverse_transform(y_test[mask].reshape(-1, 1)).ravel()
        
        mae = float(np.mean(np.abs(y_orig - mu)))
        r2_num = np.sum((y_orig - mu) ** 2)
        r2_den = np.sum((y_orig - y_orig.mean()) ** 2)
        r2 = 1 - r2_num / r2_den if r2_den > 0 else 0.0
        
        per_age_results.append({
            "age_days": age, "n": int(mask.sum()),
            "mae": float(mae), "r2": float(r2),
            "mean_strength": float(y_orig.mean()),
        })
        log(f"  t={age:>3d}d (n={mask.sum():>4d}): MAE={mae:.2f}, R2={r2:.4f}, "
            f"avg_strength={y_orig.mean():.1f}")
    
    # ── Monotonicity test ───────────────────────────────────────────
    # Find compositions with multiple ages in data
    log("\n--- Monotonicity Analysis ---")
    
    # Use full dataset to find compositions with time series
    comp_cols = ds.composition_columns
    features = ds.features
    
    # Group by composition (round to reduce noise)
    compositions = features[comp_cols].round(1)
    compositions["_group"] = compositions.apply(lambda r: tuple(r), axis=1)
    compositions["age_days"] = ds.age_days.values
    compositions["strength"] = ds.target.values
    
    groups = compositions.groupby("_group").filter(lambda g: len(g) >= 3)
    if len(groups) > 0:
        unique_comps = groups["_group"].unique()
        log(f"Found {len(unique_comps)} compositions with 3+ age measurements")
        
        # Test monotonicity on model predictions
        n_monotone = 0
        n_total = 0
        
        for comp in unique_comps[:100]:  # limit to 100
            comp_rows = groups[groups["_group"] == comp].sort_values("age_days")
            if len(comp_rows) < 3:
                continue
            
            # Build input features for this composition at different ages
            comp_features_raw = features.loc[comp_rows.index]
            # Re-compute derived features for different ages
            x_comp = feat_scaler.transform(
                ds.all_features[comp_rows.index])
            
            mu_s, _ = gen.predict(x_comp, mc_samples=10)
            mu = tgt_scaler.inverse_transform(mu_s).ravel()
            
            # Check monotonicity: each subsequent age should have higher strength
            is_monotone = all(mu[i] <= mu[i+1] for i in range(len(mu)-1))
            n_monotone += int(is_monotone)
            n_total += 1
        
        monotonicity_rate = n_monotone / max(n_total, 1)
        log(f"Monotonicity rate: {n_monotone}/{n_total} = {monotonicity_rate:.1%}")
    else:
        log("No compositions with 3+ time points found.")
        monotonicity_rate = None
    
    # ── Time curve for sample compositions ──────────────────────────
    log("\n--- Sample Time Curves ---")
    time_points = np.array([1, 3, 7, 14, 28, 56, 90])
    
    # Pick 5 random test compositions
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(x_test), min(5, len(x_test)), replace=False)
    
    curves = []
    for i, idx in enumerate(sample_idx):
        x_base = x_test[idx].copy()
        true_age = ages[idx]
        true_strength = tgt_scaler.inverse_transform(
            y_test[idx:idx+1].reshape(-1, 1)).ravel()[0]
        
        # Find log_age column index (last column in derived)
        log_age_idx = data["input_dim"] - 1  # log_age is last feature
        
        # Build batch of all time variants at once
        x_batch = np.tile(x_base, (len(time_points), 1))  # [n_times, input_dim]
        for j, t in enumerate(time_points):
            raw_log_age = np.log(t)
            x_batch[j, log_age_idx] = (raw_log_age - feat_scaler.mean[log_age_idx]) / feat_scaler.scale[log_age_idx]
        
        # Predict all at once (batch_size=7, safe for BatchNorm)
        mu_s, sigma_s = gen.predict(x_batch, mc_samples=20)
        mu_orig = tgt_scaler.inverse_transform(mu_s).ravel()
        std_orig = sigma_s.ravel() * tgt_scaler.scale[0]
        
        predictions = []
        for j, t in enumerate(time_points):
            predictions.append({"t": int(t), "pred": float(mu_orig[j]), "std": float(std_orig[j])})
        
        curves.append({
            "sample_idx": int(idx),
            "true_age": int(true_age),
            "true_strength": float(true_strength),
            "predictions": predictions,
        })
        
        preds_str = ", ".join(f"t={p['t']}→{p['pred']:.1f}" for p in predictions)
        log(f"  Sample {i+1} (true: t={true_age}d→{true_strength:.1f}MPa): {preds_str}")
    
    # ── Save results ────────────────────────────────────────────────
    results = {
        "per_age": per_age_results,
        "monotonicity_rate": monotonicity_rate,
        "time_curves": curves,
    }
    
    with open(checkpoint_dir / "time_prediction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--part", default="both", choices=["fewshot", "time", "both"])
    args = parser.parse_args()
    
    device = get_device()
    log(f"Device: {device}")
    
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(args.output_dir)
    
    # Load data
    log("Loading data...")
    ds = load_and_unify_datasets(args.data_dir)
    split = grouped_stratified_split(ds, seed=42)
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
    
    if args.part in ("fewshot", "both"):
        run_fewshot(data, ds, feat_scaler, tgt_scaler, tracker, checkpoint_dir)
    
    if args.part in ("time", "both"):
        run_time_curves(data, ds, feat_scaler, tgt_scaler, tracker, checkpoint_dir)
    
    log("\nDone!")


if __name__ == "__main__":
    main()
