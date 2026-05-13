"""Feature Importance Analysis.

Three methods:
1. Gradient-based (saliency): |dOutput/dInput| averaged over test set
2. Permutation importance: shuffle each feature, measure MAE delta
3. Feature ablation: zero each feature, measure MAE delta

Uses the best model (GAN no physics from long_gan experiment).
"""
import json
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig

ARTIFACTS = Path("artifacts/feature_importance")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
SEED = 42

FEATURE_NAMES = [
    "cement", "water", "fine_agg", "coarse_agg",
    "fine_add_1", "fine_add_2", "plasticizer",
    "wc_ratio", "wb_ratio", "log_age"
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gradient_importance(gen, x_test_t):
    """Compute mean absolute gradient of output w.r.t. each input feature."""
    gen.eval()
    x = x_test_t.clone().requires_grad_(True)

    mu, _ = gen(x)
    mu.sum().backward()

    grads = x.grad.abs().mean(dim=0).detach().numpy()
    return grads


def permutation_importance(gen, x_test, y_test, tgt_scaler, n_repeats=10):
    """Permutation feature importance: MAE increase when each feature is shuffled."""
    gen.eval()
    rng = np.random.default_rng(SEED)

    # Baseline MAE
    with torch.no_grad():
        mu_base, _ = gen(torch.as_tensor(x_test, dtype=torch.float32))
    mu_base_np = tgt_scaler.inverse_transform(mu_base.numpy()).ravel()
    y_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae_base = np.mean(np.abs(y_orig - mu_base_np))

    importances = np.zeros(x_test.shape[1])

    for j in range(x_test.shape[1]):
        mae_shuffled = 0.0
        for _ in range(n_repeats):
            x_perm = x_test.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            with torch.no_grad():
                mu_perm, _ = gen(torch.as_tensor(x_perm, dtype=torch.float32))
            mu_perm_np = tgt_scaler.inverse_transform(mu_perm.numpy()).ravel()
            mae_shuffled += np.mean(np.abs(y_orig - mu_perm_np))
        mae_shuffled /= n_repeats
        importances[j] = mae_shuffled - mae_base

    return importances, mae_base


def ablation_importance(gen, x_test, y_test, tgt_scaler):
    """Zero-out each feature and measure MAE delta."""
    gen.eval()

    with torch.no_grad():
        mu_base, _ = gen(torch.as_tensor(x_test, dtype=torch.float32))
    mu_base_np = tgt_scaler.inverse_transform(mu_base.numpy()).ravel()
    y_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae_base = np.mean(np.abs(y_orig - mu_base_np))

    importances = np.zeros(x_test.shape[1])

    for j in range(x_test.shape[1]):
        x_abl = x_test.copy()
        x_abl[:, j] = 0.0  # zero in scaled space = set to mean
        with torch.no_grad():
            mu_abl, _ = gen(torch.as_tensor(x_abl, dtype=torch.float32))
        mu_abl_np = tgt_scaler.inverse_transform(mu_abl.numpy()).ravel()
        importances[j] = np.mean(np.abs(y_orig - mu_abl_np)) - mae_base

    return importances, mae_base


def main():
    t0 = time.time()

    # Load data
    log("Loading data...")
    ds = load_and_unify_datasets("data")
    split = grouped_stratified_split(ds, seed=SEED)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()

    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    x_test = x_all[split["test"]]
    y_test = y_all[split["test"]]

    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

    x_test_s = feat_scaler.transform(x_test)
    y_test_s = tgt_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Load best model
    log("Loading best GAN model...")
    gen_config = GeneratorConfig(
        input_dim=x_test_s.shape[1], hidden_dims=[128, 64, 32],
        dropout=0.2, seed=SEED,
    )
    gen = ConcreteGenerator(gen_config)
    model_path = Path("artifacts/long_gan/gan_nophys/generator.pt")
    if not model_path.exists():
        log("Model not found, using supervised baseline")
        model_path = Path("artifacts/long_gan/supervised_baseline.pt")
    gen.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    gen.eval()

    x_test_t = torch.as_tensor(x_test_s, dtype=torch.float32)

    # 1. Gradient importance
    log("\n1. Gradient-based importance...")
    grad_imp = gradient_importance(gen, x_test_t)
    log("Feature importance (gradient saliency):")
    order = np.argsort(grad_imp)[::-1]
    for i in order:
        log(f"  {FEATURE_NAMES[i]:15s}: {grad_imp[i]:.4f}")

    # 2. Permutation importance
    log("\n2. Permutation importance (10 repeats)...")
    perm_imp, perm_base = permutation_importance(gen, x_test_s, y_test_s, tgt_scaler)
    log(f"Baseline MAE: {perm_base:.2f}")
    order = np.argsort(perm_imp)[::-1]
    for i in order:
        sign = "+" if perm_imp[i] > 0 else ""
        log(f"  {FEATURE_NAMES[i]:15s}: {sign}{perm_imp[i]:.4f} MPa")

    # 3. Ablation importance
    log("\n3. Feature ablation importance...")
    abl_imp, abl_base = ablation_importance(gen, x_test_s, y_test_s, tgt_scaler)
    log(f"Baseline MAE: {abl_base:.2f}")
    order = np.argsort(abl_imp)[::-1]
    for i in order:
        sign = "+" if abl_imp[i] > 0 else ""
        log(f"  {FEATURE_NAMES[i]:15s}: {sign}{abl_imp[i]:.4f} MPa")

    # Save results
    results = {
        "feature_names": FEATURE_NAMES,
        "gradient_importance": grad_imp.tolist(),
        "permutation_importance": perm_imp.tolist(),
        "permutation_baseline_mae": perm_base,
        "ablation_importance": abl_imp.tolist(),
        "ablation_baseline_mae": abl_base,
    }
    with open(ARTIFACTS / "feature_importance.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    log("\n" + "=" * 70)
    log("FEATURE IMPORTANCE SUMMARY (normalized to max=1.0)")
    log("=" * 70)
    g_norm = grad_imp / (grad_imp.max() + 1e-8)
    p_norm = np.maximum(perm_imp, 0) / (max(perm_imp.max(), 1e-8))
    a_norm = np.maximum(abl_imp, 0) / (max(abl_imp.max(), 1e-8))
    avg = (g_norm + p_norm + a_norm) / 3

    header = f"{'Feature':15s} {'Gradient':>10} {'Permute':>10} {'Ablate':>10} {'Average':>10}"
    log(header)
    log("-" * len(header))
    order = np.argsort(avg)[::-1]
    for i in order:
        log(f"{FEATURE_NAMES[i]:15s} {g_norm[i]:10.3f} {p_norm[i]:10.3f} "
            f"{a_norm[i]:10.3f} {avg[i]:10.3f}")

    dt = time.time() - t0
    log(f"\nTotal: {dt:.1f}s")


if __name__ == "__main__":
    main()
