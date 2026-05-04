"""Бонус 3: Multi-Property Prediction (прочность + удобоукладываемость).

Задача: предсказывать несколько свойств бетона одновременно.
Удобоукладываемость (осадка конуса / slump) — свойство свежей смеси,
не зависит от времени (в отличие от прочности).

Подход: Multi-head generator с общим backbone и отдельными головами
для каждого свойства.

Usage:
    python run_bonus_multiprop.py --output_dir /path/to/experiments
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from materialgen.data_preparation import (
    load_and_unify_datasets, COMPOSITION_COLUMNS, DERIVED_COLUMNS
)
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig
from materialgen.tracker import ExperimentTracker, get_device
from materialgen.data import read_dataset_frame


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ====================================================================
# MULTI-HEAD GENERATOR
# ====================================================================

class MultiHeadGenerator(nn.Module):
    """Generator with shared backbone + per-property heads.
    
    Architecture:
        Input → SharedBackbone → [strength_head, slump_head]
        
    Strength head: (μ_str, σ_str) — зависит от возраста
    Slump head: (μ_slump, σ_slump) — НЕ зависит от возраста
    """
    
    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.extend([
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
        ])
        self.backbone = nn.Sequential(*layers)
        
        # Strength head (uses full features including log_age)
        self.strength_mu = nn.Linear(hidden_dims[-1], 1)
        self.strength_sigma = nn.Linear(hidden_dims[-1], 1)
        
        # Slump head (workability — no time dependency)
        self.slump_mu = nn.Linear(hidden_dims[-1], 1)
        self.slump_sigma = nn.Linear(hidden_dims[-1], 1)
        
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        """Returns dict with predictions for each property."""
        h = self.backbone(x)
        
        str_mu = self.softplus(self.strength_mu(h))
        str_sigma = self.softplus(self.strength_sigma(h)) + 1e-4
        
        slump_mu = self.softplus(self.slump_mu(h))
        slump_sigma = self.softplus(self.slump_sigma(h)) + 1e-4
        
        return {
            "strength": (str_mu, str_sigma),
            "slump": (slump_mu, slump_sigma),
        }
    
    def predict(self, x, property_name="strength", mc_samples=20):
        """Predict with MC-dropout uncertainty."""
        device = next(self.parameters()).device
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        
        if mc_samples <= 1:
            self.eval()
            with torch.no_grad():
                out = self.forward(x_t)
                mu, sigma = out[property_name]
            return mu.cpu().numpy(), sigma.cpu().numpy()
        
        self.train()
        mus, sigmas = [], []
        with torch.no_grad():
            for _ in range(mc_samples):
                out = self.forward(x_t)
                mu, sigma = out[property_name]
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
        
        mus = np.stack(mus)
        sigmas = np.stack(sigmas)
        mean_mu = mus.mean(axis=0)
        total_std = np.sqrt(mus.std(axis=0)**2 + sigmas.mean(axis=0)**2)
        
        return mean_mu, total_std


def train_multihead(model, x_train, y_str_train, y_slump_train,
                    x_val, y_str_val, y_slump_val,
                    *, lr=1e-3, epochs=300, batch_size=32,
                    strength_weight=1.0, slump_weight=1.0, patience=30):
    """Train multi-head model with combined loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_str_t = torch.as_tensor(y_str_train.reshape(-1, 1), dtype=torch.float32, device=device)
    
    has_slump = y_slump_train is not None
    if has_slump:
        # Mask for samples that have slump data (non-NaN)
        slump_mask_train = ~np.isnan(y_slump_train)
        y_slump_t = torch.as_tensor(
            np.nan_to_num(y_slump_train.reshape(-1, 1)),
            dtype=torch.float32, device=device)
        slump_mask_t = torch.as_tensor(slump_mask_train, device=device)
    
    x_v = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_str_v = torch.as_tensor(y_str_val.reshape(-1, 1), dtype=torch.float32, device=device)
    
    best_val = float("inf")
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_t.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, x_t.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            
            out = model(x_t[idx])
            
            # Strength NLL
            str_mu, str_sigma = out["strength"]
            str_nll = 0.5 * (torch.log(str_sigma**2) + 
                             (y_str_t[idx] - str_mu)**2 / str_sigma**2)
            loss = strength_weight * str_nll.mean()
            
            # Slump NLL (only for samples with slump data)
            if has_slump:
                batch_mask = slump_mask_t[idx]
                if batch_mask.any():
                    slump_mu, slump_sigma = out["slump"]
                    slump_nll = 0.5 * (torch.log(slump_sigma**2) + 
                                       (y_slump_t[idx] - slump_mu)**2 / slump_sigma**2)
                    # Only average over samples with slump data
                    loss += slump_weight * (slump_nll * batch_mask.float().unsqueeze(1)).sum() / max(batch_mask.sum(), 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation (strength only for early stopping)
        model.eval()
        with torch.no_grad():
            out_v = model(x_v)
            str_mu_v, str_sigma_v = out_v["strength"]
            val_nll = 0.5 * (torch.log(str_sigma_v**2) + 
                             (y_str_v - str_mu_v)**2 / str_sigma_v**2)
            val_loss = val_nll.mean().item()
        
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return {"epochs_run": epoch + 1, "best_val_loss": best_val}


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
    
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(args.output_dir)
    
    # ── 1. Load data with slump (workability) ────────────────────────
    log("Loading data with slump information...")
    
    # Load Normal_Concrete_DB which has slump data (column index 9)
    from materialgen.data_preparation import add_derived_features
    
    csv_path = Path(args.data_dir) / "Normal_Concrete_DB.csv"
    frame = read_dataset_frame(csv_path)
    cols = list(frame.columns)
    
    log(f"Normal_Concrete_DB columns ({len(cols)}): {cols[:5]}...")
    
    # Unify and extract slump
    ds = load_and_unify_datasets(args.data_dir)
    
    # Re-read slump from original CSV
    if len(cols) >= 11:
        slump_raw = frame.iloc[:, 9]
        slump_values = slump_raw.astype(str).str.replace(",", ".").apply(
            lambda x: float(x) if x.replace(".", "").replace("-", "").isdigit() else np.nan
        )
        log(f"Slump data: {slump_values.notna().sum()} / {len(slump_values)} values")
        log(f"Slump range: {slump_values.min():.1f} - {slump_values.max():.1f} mm")
    else:
        log("WARNING: Slump column not found in Normal_Concrete_DB")
        slump_values = None
    
    # ── 2. Prepare features ──────────────────────────────────────────
    from materialgen.data_preparation import stratified_split
    split = stratified_split(ds, seed=42)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    
    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
    
    # Map slump data to unified dataset indices
    # Only Normal_Concrete_DB rows have slump
    source = ds.source.to_numpy()
    n_total = len(ds.features)
    y_slump_all = np.full(n_total, np.nan)
    
    if slump_values is not None:
        normal_mask = source == "normal_concrete_db"
        normal_indices = np.where(normal_mask)[0]
        # slump_values aligns with original CSV rows
        for i, idx in enumerate(normal_indices):
            if i < len(slump_values) and not np.isnan(slump_values.iloc[i]):
                y_slump_all[idx] = slump_values.iloc[i]
    
    n_with_slump = np.sum(~np.isnan(y_slump_all))
    log(f"Samples with slump in unified dataset: {n_with_slump}/{n_total}")
    
    # Scale slump
    valid_slump = y_slump_all[~np.isnan(y_slump_all)]
    if len(valid_slump) > 10:
        slump_scaler = StandardScaler.fit(valid_slump.reshape(-1, 1))
        y_slump_scaled = np.full_like(y_slump_all, np.nan)
        mask = ~np.isnan(y_slump_all)
        y_slump_scaled[mask] = slump_scaler.transform(
            y_slump_all[mask].reshape(-1, 1)).ravel()
    else:
        log("Not enough slump data. Running strength-only multi-head as demo.")
        slump_scaler = None
        y_slump_scaled = np.full(n_total, np.nan)
    
    # Prepare splits
    input_dim = x_all.shape[1]
    data = {}
    for key in ["train", "val", "test"]:
        idx = split[key]
        data[key] = {
            "x": feat_scaler.transform(x_all[idx]),
            "y_str": tgt_scaler.transform(y_all[idx].reshape(-1, 1)).ravel(),
            "y_slump": y_slump_scaled[idx],
        }
    
    # ── 3. Train single-head baseline ────────────────────────────────
    log("\n=== Single-head (strength only) ===")
    with tracker.run("multiprop_single", tags=["multiprop"]) as run:
        gen_single = ConcreteGenerator(GeneratorConfig(
            input_dim=input_dim, hidden_dims=[256, 128, 64],
            epochs=300, batch_size=64, learning_rate=1e-3,
            dropout=0.1, seed=42))
        
        from materialgen.generator import train_generator_supervised
        train_generator_supervised(
            gen_single, data["train"]["x"], data["train"]["y_str"],
            data["val"]["x"], data["val"]["y_str"], config=gen_single.config)
        
        gen_single = gen_single.cpu()
        mu_s, sigma_s = gen_single.predict(data["test"]["x"], mc_samples=20)
        mu = tgt_scaler.inverse_transform(mu_s).ravel()
        y_orig = tgt_scaler.inverse_transform(data["test"]["y_str"].reshape(-1, 1)).ravel()
        
        str_mae = float(np.mean(np.abs(y_orig - mu)))
        str_r2 = 1 - np.sum((y_orig - mu)**2) / np.sum((y_orig - y_orig.mean())**2)
        
        run.log_metrics({"str_mae": str_mae, "str_r2": str_r2})
        log(f"  Strength: MAE={str_mae:.2f}, R2={str_r2:.4f}")
    
    # ── 4. Train multi-head ──────────────────────────────────────────
    log("\n=== Multi-head (strength + slump) ===")
    with tracker.run("multiprop_multi", tags=["multiprop"]) as run:
        model = MultiHeadGenerator(input_dim, hidden_dims=[256, 128, 64], dropout=0.1)
        
        hist = train_multihead(
            model, data["train"]["x"], data["train"]["y_str"], data["train"]["y_slump"],
            data["val"]["x"], data["val"]["y_str"], data["val"]["y_slump"],
            lr=1e-3, epochs=300, batch_size=64,
            strength_weight=1.0, slump_weight=0.5)
        
        model = model.cpu()
        
        # Strength evaluation
        mu_s, sigma_s = model.predict(data["test"]["x"], property_name="strength", mc_samples=20)
        mu = tgt_scaler.inverse_transform(mu_s).ravel()
        y_orig = tgt_scaler.inverse_transform(data["test"]["y_str"].reshape(-1, 1)).ravel()
        
        multi_str_mae = float(np.mean(np.abs(y_orig - mu)))
        multi_str_r2 = 1 - np.sum((y_orig - mu)**2) / np.sum((y_orig - y_orig.mean())**2)
        
        # Slump evaluation (only on samples with slump data)
        slump_test = data["test"]["y_slump"]
        slump_mask = ~np.isnan(slump_test)
        
        if slump_mask.sum() > 0 and slump_scaler is not None:
            mu_sl, _ = model.predict(
                data["test"]["x"][slump_mask], property_name="slump", mc_samples=20)
            mu_sl_orig = slump_scaler.inverse_transform(mu_sl).ravel()
            y_sl_orig = slump_scaler.inverse_transform(
                slump_test[slump_mask].reshape(-1, 1)).ravel()
            
            slump_mae = float(np.mean(np.abs(y_sl_orig - mu_sl_orig)))
            slump_r2_num = np.sum((y_sl_orig - mu_sl_orig)**2)
            slump_r2_den = np.sum((y_sl_orig - y_sl_orig.mean())**2)
            slump_r2 = 1 - slump_r2_num / slump_r2_den if slump_r2_den > 0 else 0
            
            log(f"  Strength: MAE={multi_str_mae:.2f}, R2={multi_str_r2:.4f}")
            log(f"  Slump:    MAE={slump_mae:.2f} mm, R2={slump_r2:.4f} (n={slump_mask.sum()})")
            
            run.log_metrics({
                "str_mae": multi_str_mae, "str_r2": multi_str_r2,
                "slump_mae": slump_mae, "slump_r2": slump_r2,
                "slump_n_test": int(slump_mask.sum()),
                "epochs_run": hist["epochs_run"],
            })
        else:
            log(f"  Strength: MAE={multi_str_mae:.2f}, R2={multi_str_r2:.4f}")
            log(f"  Slump: no test data with slump measurements")
            run.log_metrics({
                "str_mae": multi_str_mae, "str_r2": multi_str_r2,
                "epochs_run": hist["epochs_run"],
            })
    
    # ── 5. Summary ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("MULTI-PROPERTY SUMMARY")
    log("=" * 60)
    log(f"{'Model':<25} {'Str MAE':>10} {'Str R2':>10}")
    log("-" * 45)
    log(f"{'Single-head':<25} {str_mae:10.2f} {str_r2:10.4f}")
    log(f"{'Multi-head':<25} {multi_str_mae:10.2f} {multi_str_r2:10.4f}")
    
    diff = multi_str_mae - str_mae
    log(f"\nMulti-head strength delta: {diff:+.2f} MPa "
        f"({'better' if diff < 0 else 'worse'})")
    
    results = {
        "single_head": {"str_mae": str_mae, "str_r2": str_r2},
        "multi_head": {"str_mae": multi_str_mae, "str_r2": multi_str_r2},
    }
    
    with open(checkpoint_dir / "multiprop_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
