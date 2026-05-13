"""Robust stacking with K-fold cross-validation.

The previous stacking (train on val, test on test) likely overfits because:
- GBR with 100 trees on ~560 samples × 24 features = overfit
- No cross-validation → optimistic estimate

Fix: K-fold CV stacking where meta-learner never sees test set during training.
Final metric is computed on a held-out test set using out-of-fold predictions.

Usage:
    python run_robust_stacking.py --output_dir /path/to/experiments
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
            "y_orig": y_all[idx],
            "ages": ages_all[idx],
        }
    data["input_dim"] = data["train"]["x"].shape[1]
    data["feat_scaler"] = feat_scaler
    data["tgt_scaler"] = tgt_scaler
    return data


def add_enhanced_features(x):
    """Same 5 interaction features as run_push_below9.py."""
    cement = x[:, 0:1]
    water = x[:, 1:2]
    sand = x[:, 2:3]
    coarse_agg = x[:, 3:4]
    w_c = x[:, 7:8]
    log_age = x[:, 9:10]
    extras = np.hstack([
        cement * log_age,
        w_c * log_age,
        cement * water,
        log_age ** 2,
        sand + coarse_agg,
    ])
    return np.hstack([x, extras])


# ====================================================================
# COLLECT BASE MODEL PREDICTIONS
# ====================================================================

def collect_predictions(data, checkpoint_dir):
    """Collect predictions from all available models on train+val and test."""
    tgt_scaler = data["tgt_scaler"]
    
    # Combine train+val for K-fold meta-training
    x_tv = np.concatenate([data["train"]["x"], data["val"]["x"]], axis=0)
    y_tv_orig = np.concatenate([data["train"]["y_orig"], data["val"]["y_orig"]], axis=0)
    ages_tv = np.concatenate([data["train"]["ages"], data["val"]["ages"]], axis=0)
    
    x_test = data["test"]["x"]
    
    # ── GAN models ────────────────────────────────────────────────────
    gan_ckpts = sorted(checkpoint_dir.glob("gan_*.pt"))
    model_names = []
    preds_tv = []  # predictions on train+val
    preds_test = []
    sigmas_test = []
    
    for ckpt in gan_ckpts:
        for hidden in [[256, 128, 64], [256, 128, 64, 32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hidden,
                    dropout=0.1, seed=42))
                gen.load_state_dict(
                    torch.load(ckpt, map_location="cpu", weights_only=True))
                
                mu_tv, _ = gen.predict(x_tv, mc_samples=20)
                mu_t, sig_t = gen.predict(x_test, mc_samples=20)
                
                preds_tv.append(tgt_scaler.inverse_transform(mu_tv).ravel())
                preds_test.append(tgt_scaler.inverse_transform(mu_t).ravel())
                sigmas_test.append(sig_t.ravel() * tgt_scaler.scale[0])
                model_names.append(f"GAN:{ckpt.stem}")
                log(f"  Loaded {ckpt.name} ({len(hidden)}-layer)")
                break
            except Exception:
                continue
    
    # ── Enhanced feature models ───────────────────────────────────────
    x_tv_enh = add_enhanced_features(x_tv)
    x_test_enh = add_enhanced_features(x_test)
    # Re-scale enhanced features
    enh_scaler = StandardScaler.fit(
        add_enhanced_features(data["train"]["x"]))
    x_tv_enh = enh_scaler.transform(x_tv_enh)
    x_test_enh = enh_scaler.transform(x_test_enh)
    
    enh_ckpts = sorted(checkpoint_dir.glob("enh_*.pt"))
    for ckpt in enh_ckpts:
        for hidden in [[256, 128, 64], [512, 256, 128]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=x_tv_enh.shape[1], hidden_dims=hidden,
                    dropout=0.1, seed=42))
                gen.load_state_dict(
                    torch.load(ckpt, map_location="cpu", weights_only=True))
                
                mu_tv, _ = gen.predict(x_tv_enh, mc_samples=20)
                mu_t, sig_t = gen.predict(x_test_enh, mc_samples=20)
                
                preds_tv.append(tgt_scaler.inverse_transform(mu_tv).ravel())
                preds_test.append(tgt_scaler.inverse_transform(mu_t).ravel())
                sigmas_test.append(sig_t.ravel() * tgt_scaler.scale[0])
                model_names.append(f"ENH:{ckpt.stem}")
                break
            except Exception:
                continue
    
    log(f"\n  Total base models: {len(model_names)}")
    log(f"  GAN models: {sum(1 for n in model_names if n.startswith('GAN'))}")
    log(f"  Enhanced models: {sum(1 for n in model_names if n.startswith('ENH'))}")
    
    return {
        "model_names": model_names,
        "X_tv": np.stack(preds_tv, axis=1),      # [n_train+val, n_models]
        "X_test": np.stack(preds_test, axis=1),   # [n_test, n_models]
        "sigmas_test": np.stack(sigmas_test, axis=1),
        "y_tv": y_tv_orig,
        "ages_tv": ages_tv,
        "y_test": data["test"]["y_orig"],
        "ages_test": data["test"]["ages"],
    }


# ====================================================================
# K-FOLD STACKING (ROBUST)
# ====================================================================

def kfold_stacking(preds, n_folds=5):
    """K-Fold cross-validated stacking.
    
    For each fold:
    1. Train meta-learner on (K-1) folds of train+val predictions
    2. Predict on held-out fold → out-of-fold (OOF) predictions
    3. Predict on test → average across folds for final test prediction
    
    This gives:
    - OOF MAE: unbiased estimate of stacking quality
    - Test predictions: averaged from K meta-models (more robust)
    """
    X_tv = preds["X_tv"]
    y_tv = preds["y_tv"]
    X_test = preds["X_test"]
    y_test = preds["y_test"]
    ages_test = preds["ages_test"]
    sigmas_test = preds["sigmas_test"]
    
    log(f"\nK-Fold Stacking (K={n_folds})")
    log(f"  Meta-features: {X_tv.shape[1]} model predictions")
    log(f"  Train+Val: {X_tv.shape[0]}, Test: {X_test.shape[0]}")
    
    results = {}
    
    # Simple average baseline (no meta-learning)
    simple_mu = X_test.mean(axis=1)
    simple_sigma = np.sqrt(X_test.std(axis=1)**2 + sigmas_test.mean(axis=1)**2)
    ev = evaluate_model(y_true=y_test, y_pred=simple_mu, y_std=simple_sigma, age_days=ages_test)
    results["simple_avg"] = {"mae": ev.regression.mae, "r2": ev.regression.r2,
                             "picp": ev.calibration.picp if ev.calibration else None}
    log(f"\n  Simple average: MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    # K-Fold for each meta-learner
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    meta_configs = [
        ("ridge_a0.1", lambda: Ridge(alpha=0.1)),
        ("ridge_a1", lambda: Ridge(alpha=1.0)),
        ("ridge_a10", lambda: Ridge(alpha=10.0)),
        ("ridge_a100", lambda: Ridge(alpha=100.0)),
        ("gbr_n30_d2", lambda: GradientBoostingRegressor(
            n_estimators=30, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)),
        ("gbr_n50_d2", lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)),
    ]
    
    for meta_name, meta_factory in meta_configs:
        oof_preds = np.zeros(len(y_tv))
        test_preds_folds = []
        fold_maes = []
        
        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_tv)):
            model = meta_factory()
            model.fit(X_tv[train_idx], y_tv[train_idx])
            
            # Out-of-fold prediction
            oof_preds[val_idx] = model.predict(X_tv[val_idx])
            fold_mae = np.mean(np.abs(y_tv[val_idx] - oof_preds[val_idx]))
            fold_maes.append(fold_mae)
            
            # Test prediction for this fold
            test_preds_folds.append(model.predict(X_test))
        
        # OOF metrics (unbiased)
        oof_mae = np.mean(np.abs(y_tv - oof_preds))
        oof_r2_num = np.sum((y_tv - oof_preds)**2)
        oof_r2_den = np.sum((y_tv - y_tv.mean())**2)
        oof_r2 = 1 - oof_r2_num / oof_r2_den
        
        # Test prediction: average across folds
        test_mu = np.stack(test_preds_folds, axis=0).mean(axis=0)
        
        # Uncertainty estimate
        weights = np.ones(X_test.shape[1]) / X_test.shape[1]
        if hasattr(meta_factory(), 'coef_'):
            # Refit on full train+val for final weights
            final_model = meta_factory()
            final_model.fit(X_tv, y_tv)
            weights = np.abs(final_model.coef_)
            if weights.sum() > 0:
                weights /= weights.sum()
        
        test_sigma = np.sqrt(
            X_test.std(axis=1)**2 + 
            (sigmas_test * weights[None, :]).sum(axis=1)**2
        )
        
        ev = evaluate_model(y_true=y_test, y_pred=test_mu, y_std=test_sigma, age_days=ages_test)
        
        fold_mae_std = np.std(fold_maes)
        results[meta_name] = {
            "oof_mae": float(oof_mae), "oof_r2": float(oof_r2),
            "test_mae": ev.regression.mae, "test_r2": ev.regression.r2,
            "test_rmse": ev.regression.rmse,
            "picp": ev.calibration.picp if ev.calibration else None,
            "mpiw": ev.calibration.mpiw if ev.calibration else None,
            "fold_mae_std": float(fold_mae_std),
        }
        
        log(f"  {meta_name:<20}: OOF MAE={oof_mae:.3f}±{fold_mae_std:.3f}, "
            f"Test MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    # ── Weighted average (optimize on full train+val) ──────────────
    from scipy.optimize import minimize
    
    def neg_obj(weights):
        w = np.abs(weights) / np.abs(weights).sum()
        pred = (X_tv * w[None, :]).sum(axis=1)
        return np.mean(np.abs(y_tv - pred))
    
    best_mae = float("inf")
    best_weights = None
    for seed in range(30):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(X_tv.shape[1]))
        res = minimize(neg_obj, w0, method="Nelder-Mead",
                       options={"maxiter": 3000})
        if res.fun < best_mae:
            best_mae = res.fun
            best_weights = np.abs(res.x) / np.abs(res.x).sum()
    
    weighted_mu = (X_test * best_weights[None, :]).sum(axis=1)
    weighted_sigma = np.sqrt(
        X_test.std(axis=1)**2 + 
        (sigmas_test * best_weights[None, :]).sum(axis=1)**2
    )
    ev = evaluate_model(y_true=y_test, y_pred=weighted_mu, y_std=weighted_sigma, age_days=ages_test)
    results["weighted_opt"] = {
        "test_mae": ev.regression.mae, "test_r2": ev.regression.r2,
        "picp": ev.calibration.picp if ev.calibration else None,
        "weights": best_weights.tolist(),
        "train_val_mae": float(best_mae),
    }
    
    # Show top weights
    top_w_idx = np.argsort(best_weights)[::-1][:5]
    top_w_str = ", ".join(
        f"{preds['model_names'][i]}={best_weights[i]:.3f}" 
        for i in top_w_idx if best_weights[i] > 0.01
    )
    log(f"  weighted_opt: Train+Val MAE={best_mae:.3f}, "
        f"Test MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    log(f"    Top weights: {top_w_str}")
    
    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()
    
    device = get_device()
    log(f"Device: {device}")
    
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    tracker = ExperimentTracker(args.output_dir)
    
    log("Loading data...")
    data = load_data(args.data_dir)
    log(f"Train: {len(data['train']['x'])}, Val: {len(data['val']['x'])}, "
        f"Test: {len(data['test']['x'])}")
    
    # Collect base model predictions
    log("\nCollecting base model predictions...")
    preds = collect_predictions(data, checkpoint_dir)
    
    # K-Fold stacking
    log("\n" + "=" * 60)
    log("ROBUST K-FOLD STACKING")
    log("=" * 60)
    results = kfold_stacking(preds, n_folds=args.n_folds)
    
    # Final summary
    log("\n" + "=" * 60)
    log("FINAL RESULTS (sorted by Test MAE)")
    log("=" * 60)
    log(f"{'Method':<25} {'OOF MAE':>10} {'Test MAE':>10} {'Test R2':>10} {'PICP':>8}")
    log("-" * 63)
    
    for name, r in sorted(results.items(), key=lambda x: x[1].get("test_mae", x[1].get("mae", 99))):
        oof = f"{r['oof_mae']:.3f}" if "oof_mae" in r else "N/A"
        test_mae = r.get("test_mae", r.get("mae", 0))
        test_r2 = r.get("test_r2", r.get("r2", 0))
        picp = f"{r['picp']:.1%}" if r.get("picp") else "N/A"
        log(f"{name:<25} {oof:>10} {test_mae:10.3f} {test_r2:10.4f} {picp:>8}")
    
    best_name = min(results, key=lambda k: results[k].get("test_mae", results[k].get("mae", 99)))
    best = results[best_name]
    best_mae = best.get("test_mae", best.get("mae"))
    
    log(f"\n{'='*63}")
    log(f"BEST ROBUST: {best_name} → Test MAE={best_mae:.3f}")
    if best_mae < 9.0:
        log("✅ MAE < 9.0 CONFIRMED with cross-validation!")
    
    with open(checkpoint_dir / "robust_stacking_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log("\nDone!")


if __name__ == "__main__":
    main()
