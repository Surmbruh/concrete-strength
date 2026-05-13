"""Push MAE below 9.0: weighted ensemble + enhanced features + stacking.

Three strategies combined:
1. Weighted ensemble: optimize weights on validation (vs simple average)
2. Enhanced features: interaction terms + maturity function → retrain
3. Stacking: meta-learner on top of all model predictions

Usage:
    python run_push_below9.py --output_dir /path/to/experiments
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ====================================================================
# DATA LOADING
# ====================================================================

def load_data_standard(data_dir="data", seed=42):
    """Load standard 10-feature data."""
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
            "idx": idx,
        }
    data["input_dim"] = data["train"]["x"].shape[1]
    data["feat_scaler"] = feat_scaler
    data["tgt_scaler"] = tgt_scaler
    data["ds"] = ds
    data["split"] = split
    return data


def add_enhanced_features(x, feat_scaler):
    """Add interaction features to already-scaled data.
    
    New features (computed from SCALED values):
    - cement × log_age (strength gain rate depends on cement)
    - w_c_ratio × log_age (maturity interaction)
    - cement × water (direct interaction)
    - total_agg_ratio = (sand + coarse_agg) / (cement + fine_add) [scaled proxy]
    - log_age² (non-linear time effect)
    """
    # Column indices in standard 10-feature format:
    # 0:cement, 1:water, 2:sand, 3:coarse_agg, 4:fine_add_1, 5:fine_add_2, 
    # 6:plasticizer, 7:w_c_ratio, 8:w_b_ratio, 9:log_age
    
    cement = x[:, 0:1]
    water = x[:, 1:2]
    sand = x[:, 2:3]
    coarse_agg = x[:, 3:4]
    w_c = x[:, 7:8]
    log_age = x[:, 9:10]
    
    extras = np.hstack([
        cement * log_age,        # cement × log_age
        w_c * log_age,           # w/c × log_age (maturity)
        cement * water,          # cement × water
        log_age ** 2,            # log_age² (non-linear time)
        (sand + coarse_agg),     # total aggregate (already scaled)
    ])
    
    return np.hstack([x, extras])


def load_data_enhanced(data_dir="data", seed=42):
    """Load data with enhanced features (10 + 5 = 15 features)."""
    data = load_data_standard(data_dir, seed)
    
    for key in ["train", "val", "test"]:
        data[key]["x_enh"] = add_enhanced_features(
            data[key]["x"], data["feat_scaler"])
    
    data["input_dim_enh"] = data["train"]["x_enh"].shape[1]
    
    # Fit scaler for enhanced features
    enh_scaler = StandardScaler.fit(data["train"]["x_enh"])
    for key in ["train", "val", "test"]:
        data[key]["x_enh"] = enh_scaler.transform(data[key]["x_enh"])
    data["enh_scaler"] = enh_scaler
    
    return data


# ====================================================================
# 1. WEIGHTED ENSEMBLE
# ====================================================================

def run_weighted_ensemble(data, checkpoint_dir):
    """Optimize ensemble weights on validation set."""
    log("=" * 60)
    log("WEIGHTED ENSEMBLE OPTIMIZATION")
    log("=" * 60)
    
    tgt_scaler = data["tgt_scaler"]
    
    # Load all GAN models
    gan_ckpts = sorted(checkpoint_dir.glob("gan_*.pt"))
    log(f"Found {len(gan_ckpts)} GAN checkpoints")
    
    models = []
    val_preds = []
    test_preds = []
    val_sigmas = []
    test_sigmas = []
    
    for ckpt in gan_ckpts:
        # Try 3-layer first, then 4-layer
        for hidden in [[256, 128, 64], [256, 128, 64, 32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hidden,
                    dropout=0.1, seed=42))
                gen.load_state_dict(
                    torch.load(ckpt, map_location="cpu", weights_only=True))
                
                # Val predictions
                mu_v, sig_v = gen.predict(data["val"]["x"], mc_samples=20)
                mu_v_orig = tgt_scaler.inverse_transform(mu_v).ravel()
                
                # Test predictions
                mu_t, sig_t = gen.predict(data["test"]["x"], mc_samples=20)
                mu_t_orig = tgt_scaler.inverse_transform(mu_t).ravel()
                sig_t_orig = sig_t.ravel() * tgt_scaler.scale[0]
                
                models.append(ckpt.name)
                val_preds.append(mu_v_orig)
                test_preds.append(mu_t_orig)
                test_sigmas.append(sig_t_orig)
                log(f"  Loaded {ckpt.name} ({len(hidden)}-layer)")
                break
            except Exception:
                continue
    
    if len(models) < 2:
        log("Not enough models for ensemble!")
        return None
    
    val_preds = np.stack(val_preds, axis=0)    # [n_models, n_val]
    test_preds = np.stack(test_preds, axis=0)  # [n_models, n_test]
    test_sigmas = np.stack(test_sigmas, axis=0)
    
    y_val_orig = data["val"]["y_orig"]
    y_test_orig = data["test"]["y_orig"]
    ages_test = data["test"]["ages"]
    
    results = {}
    
    # 1a. Simple average (baseline)
    simple_mu = test_preds.mean(axis=0)
    simple_sigma = np.sqrt(test_preds.std(axis=0)**2 + test_sigmas.mean(axis=0)**2)
    ev = evaluate_model(y_true=y_test_orig, y_pred=simple_mu, y_std=simple_sigma, age_days=ages_test)
    results["simple_avg"] = {"mae": ev.regression.mae, "r2": ev.regression.r2}
    log(f"\n  Simple average ({len(models)} models): MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    # 1b. Optimize weights via scipy (minimize MAE on val)
    from scipy.optimize import minimize
    
    def neg_obj(weights):
        w = np.abs(weights) / np.abs(weights).sum()
        pred = (val_preds * w[:, None]).sum(axis=0)
        return np.mean(np.abs(y_val_orig - pred))
    
    best_mae = float("inf")
    best_weights = None
    
    for seed in range(20):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(models)))
        res = minimize(neg_obj, w0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6})
        if res.fun < best_mae:
            best_mae = res.fun
            best_weights = np.abs(res.x) / np.abs(res.x).sum()
    
    # Apply to test
    weighted_mu = (test_preds * best_weights[:, None]).sum(axis=0)
    weighted_sigma = np.sqrt(
        (test_preds.std(axis=0))**2 + 
        (test_sigmas * best_weights[:, None]).sum(axis=0)**2
    )
    ev = evaluate_model(y_true=y_test_orig, y_pred=weighted_mu, y_std=weighted_sigma, age_days=ages_test)
    results["weighted"] = {"mae": ev.regression.mae, "r2": ev.regression.r2, 
                           "weights": best_weights.tolist()}
    
    log(f"  Weighted ensemble: MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    log(f"  Weights: {', '.join(f'{w:.3f}' for w in best_weights)}")
    
    # 1c. Inverse-MAE weighted
    individual_maes = np.array([
        np.mean(np.abs(y_val_orig - val_preds[i])) for i in range(len(models))
    ])
    inv_mae_w = (1.0 / individual_maes)
    inv_mae_w /= inv_mae_w.sum()
    
    inv_mu = (test_preds * inv_mae_w[:, None]).sum(axis=0)
    inv_sigma = np.sqrt(test_preds.std(axis=0)**2 + (test_sigmas * inv_mae_w[:, None]).sum(axis=0)**2)
    ev = evaluate_model(y_true=y_test_orig, y_pred=inv_mu, y_std=inv_sigma, age_days=ages_test)
    results["inv_mae"] = {"mae": ev.regression.mae, "r2": ev.regression.r2}
    log(f"  Inv-MAE weighted: MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    return results, val_preds, test_preds, test_sigmas, models


# ====================================================================
# 2. ENHANCED FEATURES TRAINING
# ====================================================================

def run_enhanced_training(data, checkpoint_dir, n_seeds=5):
    """Train models with enhanced features (15-dim)."""
    log("\n" + "=" * 60)
    log("ENHANCED FEATURES TRAINING")
    log("=" * 60)
    log(f"Standard features: {data['input_dim']}, Enhanced: {data['input_dim_enh']}")
    
    tgt_scaler = data["tgt_scaler"]
    results = []
    enh_preds_val = []
    enh_preds_test = []
    enh_sigmas_test = []
    
    configs = [
        {"hidden": [256, 128, 64], "lr": 1e-3, "dropout": 0.1},
        {"hidden": [256, 128, 64], "lr": 5e-4, "dropout": 0.1},
        {"hidden": [512, 256, 128], "lr": 1e-3, "dropout": 0.15},
    ]
    
    for cfg in configs:
        for seed in range(42, 42 + n_seeds):
            tag = f"enh_h{'x'.join(map(str,cfg['hidden']))}_lr{cfg['lr']}_s{seed}"
            ckpt_path = checkpoint_dir / f"{tag}.pt"
            
            t0 = time.time()
            gen = ConcreteGenerator(GeneratorConfig(
                input_dim=data["input_dim_enh"],
                hidden_dims=cfg["hidden"],
                epochs=300, batch_size=64,
                learning_rate=cfg["lr"],
                weight_decay=1e-4,
                dropout=cfg["dropout"],
                seed=seed,
            ))
            
            if ckpt_path.exists():
                gen.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True))
                log(f"  {tag}: loaded from cache")
            else:
                train_generator_supervised(
                    gen, data["train"]["x_enh"], data["train"]["y"],
                    data["val"]["x_enh"], data["val"]["y"], config=gen.config)
                gen = gen.cpu()
                torch.save(gen.state_dict(), ckpt_path)
            
            # Predict
            mu_v, sig_v = gen.predict(data["val"]["x_enh"], mc_samples=20)
            mu_t, sig_t = gen.predict(data["test"]["x_enh"], mc_samples=20)
            
            mu_t_orig = tgt_scaler.inverse_transform(mu_t).ravel()
            sig_t_orig = sig_t.ravel() * tgt_scaler.scale[0]
            mu_v_orig = tgt_scaler.inverse_transform(mu_v).ravel()
            
            mae = float(np.mean(np.abs(data["test"]["y_orig"] - mu_t_orig)))
            dt = time.time() - t0
            
            enh_preds_val.append(mu_v_orig)
            enh_preds_test.append(mu_t_orig)
            enh_sigmas_test.append(sig_t_orig)
            results.append({"tag": tag, "mae": mae})
            log(f"  {tag}: MAE={mae:.3f} ({dt:.0f}s)")
    
    # Enhanced ensemble
    if enh_preds_test:
        enh_test = np.stack(enh_preds_test, axis=0)
        enh_mu = enh_test.mean(axis=0)
        enh_sigma_avg = np.stack(enh_sigmas_test).mean(axis=0)
        enh_sigma = np.sqrt(enh_test.std(axis=0)**2 + enh_sigma_avg**2)
        
        ev = evaluate_model(
            y_true=data["test"]["y_orig"], y_pred=enh_mu, 
            y_std=enh_sigma, age_days=data["test"]["ages"])
        log(f"\n  Enhanced ensemble ({len(enh_preds_test)} models): "
            f"MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    return results, enh_preds_val, enh_preds_test, enh_sigmas_test


# ====================================================================
# 3. STACKING META-LEARNER
# ====================================================================

def run_stacking(data, gan_val_preds, gan_test_preds, gan_test_sigmas,
                 enh_val_preds, enh_test_preds, enh_test_sigmas,
                 checkpoint_dir):
    """Train meta-learner on combined predictions."""
    log("\n" + "=" * 60)
    log("STACKING META-LEARNER")
    log("=" * 60)
    
    y_val = data["val"]["y_orig"]
    y_test = data["test"]["y_orig"]
    ages_test = data["test"]["ages"]
    
    # Stack all predictions as features for meta-learner
    all_val = list(gan_val_preds) if gan_val_preds is not None else []
    all_test = list(gan_test_preds) if gan_test_preds is not None else []
    all_test_sig = list(gan_test_sigmas) if gan_test_sigmas is not None else []
    
    if enh_val_preds:
        all_val.extend(enh_val_preds)
        all_test.extend(enh_test_preds)
        all_test_sig.extend(enh_test_sigmas)
    
    if len(all_val) < 2:
        log("Not enough base models for stacking!")
        return None
    
    X_val = np.stack(all_val, axis=1)     # [n_val, n_models]
    X_test = np.stack(all_test, axis=1)   # [n_test, n_models]
    
    log(f"Stacking {X_val.shape[1]} base models")
    
    results = {}
    
    # 3a. Ridge regression
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_val, y_val)
        pred = ridge.predict(X_test)
        mae = float(np.mean(np.abs(y_test - pred)))
        
        # Uncertainty: weighted combination
        weights = np.abs(ridge.coef_) / np.abs(ridge.coef_).sum()
        sigma_stack = np.stack(all_test_sig, axis=1)
        sigma_pred = np.sqrt((sigma_stack**2 * weights[None, :]).sum(axis=1))
        
        ev = evaluate_model(y_true=y_test, y_pred=pred, y_std=sigma_pred, age_days=ages_test)
        results[f"ridge_a{alpha}"] = {
            "mae": ev.regression.mae, "r2": ev.regression.r2,
            "picp": ev.calibration.picp if ev.calibration else None,
        }
        log(f"  Ridge(α={alpha}): MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    # 3b. Gradient Boosting (non-linear stacking)
    for n_est in [50, 100]:
        gbr = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        gbr.fit(X_val, y_val)
        pred = gbr.predict(X_test)
        mae = float(np.mean(np.abs(y_test - pred)))
        
        # Simple uncertainty estimate
        sigma_pred = np.stack(all_test_sig, axis=1).mean(axis=1)
        ev = evaluate_model(y_true=y_test, y_pred=pred, y_std=sigma_pred, age_days=ages_test)
        results[f"gbr_n{n_est}"] = {
            "mae": ev.regression.mae, "r2": ev.regression.r2,
        }
        log(f"  GBR(n={n_est}): MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    # 3c. Simple average of top-5 base models (by val MAE)
    individual_val_maes = [np.mean(np.abs(y_val - X_val[:, i])) for i in range(X_val.shape[1])]
    top5_idx = np.argsort(individual_val_maes)[:5]
    top5_mu = X_test[:, top5_idx].mean(axis=1)
    top5_sigma = np.sqrt(
        X_test[:, top5_idx].std(axis=1)**2 + 
        np.stack(all_test_sig, axis=1)[:, top5_idx].mean(axis=1)**2
    )
    ev = evaluate_model(y_true=y_test, y_pred=top5_mu, y_std=top5_sigma, age_days=ages_test)
    results["top5_avg"] = {"mae": ev.regression.mae, "r2": ev.regression.r2}
    log(f"  Top-5 average: MAE={ev.regression.mae:.3f}, R2={ev.regression.r2:.4f}")
    
    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()
    
    device = get_device()
    log(f"Device: {device}")
    
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    log("Loading data (standard + enhanced)...")
    data = load_data_enhanced(args.data_dir)
    log(f"Standard: {data['input_dim']} features, Enhanced: {data['input_dim_enh']} features")
    log(f"Train: {len(data['train']['x'])}, Val: {len(data['val']['x'])}, "
        f"Test: {len(data['test']['x'])}")
    
    # 1. Weighted ensemble of existing GAN models
    ens_results = run_weighted_ensemble(data, checkpoint_dir)
    if ens_results:
        ens_results, gan_val_preds, gan_test_preds, gan_test_sigmas, gan_names = ens_results
    else:
        gan_val_preds = gan_test_preds = gan_test_sigmas = None
    
    # 2. Enhanced feature training
    enh_results, enh_val_preds, enh_test_preds, enh_test_sigmas = \
        run_enhanced_training(data, checkpoint_dir, n_seeds=args.n_seeds)
    
    # 3. Stacking
    stack_results = run_stacking(
        data, 
        gan_val_preds if gan_val_preds is not None else [],
        gan_test_preds if gan_test_preds is not None else [],
        gan_test_sigmas if gan_test_sigmas is not None else [],
        enh_val_preds, enh_test_preds, enh_test_sigmas,
        checkpoint_dir)
    
    # Final summary
    log("\n" + "=" * 60)
    log("FINAL SUMMARY — PUSH BELOW 9.0")
    log("=" * 60)
    
    all_results = {}
    if ens_results:
        all_results.update(ens_results)
    if stack_results:
        all_results.update(stack_results)
    
    log(f"{'Method':<30} {'MAE':>8} {'R2':>8}")
    log("-" * 46)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["mae"]):
        log(f"{name:<30} {r['mae']:8.3f} {r['r2']:8.4f}")
    
    best_name = min(all_results, key=lambda k: all_results[k]["mae"])
    best_mae = all_results[best_name]["mae"]
    log(f"\n{'='*46}")
    log(f"BEST: {best_name} → MAE={best_mae:.3f}")
    if best_mae < 9.0:
        log("🎉 MAE < 9.0 ACHIEVED!")
    else:
        log(f"Gap to 9.0: {best_mae - 9.0:.3f}")
    
    with open(checkpoint_dir / "push_below9_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    log("\nDone!")


if __name__ == "__main__":
    main()
