"""Бонус 1: Transfer Learning с EWC + Replay Buffer.

Задача: перенести знания модели, обученной на полных данных (3745 samples),
на узкую лабораторную выборку (ограниченный диапазон составов).

Проблема naive fine-tuning: catastrophic forgetting (R2: 0.45 → 0.008).
Решение: EWC (Elastic Weight Consolidation) + Replay Buffer.

EWC: добавляет штраф за отклонение от pre-trained весов, взвешенный по
Fisher Information Matrix (важность каждого параметра).
Replay Buffer: подмешивает 20% данных из pre-training в каждый batch.

Usage:
    python run_bonus_transfer.py --output_dir /path/to/experiments
"""
import argparse
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ====================================================================
# EWC (Elastic Weight Consolidation)
# ====================================================================

class EWC:
    """Elastic Weight Consolidation.
    
    Вычисляет Fisher Information Matrix для pre-trained модели,
    затем добавляет penalty за отклонение от этих весов:
    
    L_ewc = Σ_i F_i * (θ_i - θ*_i)²
    
    где F_i — диагональ Fisher matrix, θ* — pre-trained веса.
    """
    
    def __init__(self, model: nn.Module, dataloader_x, dataloader_y, 
                 device, n_samples=200):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader_x, dataloader_y, n_samples)
    
    def _compute_fisher(self, x_data, y_data, n_samples):
        """Вычислить диагональ Fisher Information Matrix.
        
        Uses eval mode (BatchNorm with running stats) to avoid batch_size=1 crash.
        Gradients from NLL still flow through parameters for Fisher estimation.
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        
        self.model.eval()  # Use running stats for BatchNorm
        n = min(n_samples, len(x_data))
        indices = np.random.choice(len(x_data), n, replace=False)
        
        # Process in mini-batches to be efficient
        batch_size = 8
        n_processed = 0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            x = torch.as_tensor(x_data[batch_idx], dtype=torch.float32, device=self.device)
            y = torch.as_tensor(y_data[batch_idx].reshape(-1, 1), dtype=torch.float32, device=self.device)
            
            self.model.zero_grad()
            mu, sigma = self.model(x)
            nll = 0.5 * (torch.log(sigma ** 2) + (y - mu) ** 2 / (sigma ** 2))
            nll.mean().backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2 * len(batch_idx)
            n_processed += len(batch_idx)
        
        # Normalize
        for name in fisher:
            fisher[name] /= max(n_processed, 1)
        
        return fisher
    
    def penalty(self, model):
        """Вычислить EWC penalty."""
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.params[name]) ** 2).sum()
        return loss


# ====================================================================
# Transfer Training with EWC + Replay
# ====================================================================

def train_with_ewc(
    model, x_train, y_train, x_val, y_val,
    ewc: EWC, 
    replay_x=None, replay_y=None,
    *,
    lr=5e-4, epochs=200, batch_size=32,
    ewc_lambda=1000.0, replay_ratio=0.2,
    patience=30,
):
    """Fine-tune с EWC penalty и replay buffer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    x_v = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_v = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)
    
    if replay_x is not None:
        x_r = torch.as_tensor(replay_x, dtype=torch.float32, device=device)
        y_r = torch.as_tensor(replay_y.reshape(-1, 1), dtype=torch.float32, device=device)
    
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
            xb, yb = x_t[idx], y_t[idx]
            
            # Replay buffer: подмешиваем данные из pre-training
            if replay_x is not None:
                n_replay = max(1, int(len(idx) * replay_ratio))
                r_idx = torch.randint(0, x_r.shape[0], (n_replay,))
                xb = torch.cat([xb, x_r[r_idx]], dim=0)
                yb = torch.cat([yb, y_r[r_idx]], dim=0)
            
            mu, sigma = model(xb)
            nll = 0.5 * (torch.log(sigma ** 2) + (yb - mu) ** 2 / (sigma ** 2))
            task_loss = nll.mean()
            
            # EWC penalty
            ewc_loss = ewc.penalty(model)
            loss = task_loss + ewc_lambda * ewc_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += task_loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            mu_v, sigma_v = model(x_v)
            val_nll = 0.5 * (torch.log(sigma_v ** 2) + (y_v - mu_v) ** 2 / (sigma_v ** 2))
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


def create_narrow_subset(ds, feat_scaler, tgt_scaler, 
                         fraction=0.15, seed=42):
    """Создать узкую лабораторную выборку.
    
    Эмулирует ситуацию: у лаборатории ограниченный диапазон составов.
    Берём подмножество данных из одного источника с ограниченным 
    диапазоном по цементу и воде.
    """
    rng = np.random.default_rng(seed)
    
    # Берём данные из одного источника (эмуляция узкой лаборатории)
    source_mask = ds.source == "normal_concrete_db"
    source_idx = np.where(source_mask.to_numpy())[0]
    
    # Ограничиваем по диапазону цемента (300-450 кг/м³)
    cement = ds.features.iloc[source_idx]["cement"].to_numpy()
    narrow_mask = (cement >= 250) & (cement <= 400)
    narrow_idx = source_idx[narrow_mask]
    
    # Берём fraction от оставшихся
    n = max(50, int(len(narrow_idx) * fraction))
    selected = rng.choice(narrow_idx, size=min(n, len(narrow_idx)), replace=False)
    
    x = feat_scaler.transform(ds.all_features[selected])
    y = tgt_scaler.transform(ds.target.to_numpy()[selected].reshape(-1, 1)).ravel()
    ages = ds.age_days.to_numpy()[selected]
    
    # Split 70/15/15
    n_train = int(len(selected) * 0.7)
    n_val = int(len(selected) * 0.15)
    
    perm = rng.permutation(len(selected))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    
    return {
        "train": {"x": x[train_idx], "y": y[train_idx], "ages": ages[train_idx]},
        "val": {"x": x[val_idx], "y": y[val_idx], "ages": ages[val_idx]},
        "test": {"x": x[test_idx], "y": y[test_idx], "ages": ages[test_idx]},
        "n_total": len(selected),
        "n_train": n_train,
    }


def eval_on_split(gen, data_split, tgt_scaler, mc_samples=30):
    """Evaluate on a data split."""
    gen_cpu = gen.cpu()
    mu_s, sigma_s = gen_cpu.predict(data_split["test"]["x"], mc_samples=mc_samples)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    sigma = sigma_s.ravel() * tgt_scaler.scale[0]
    y_orig = tgt_scaler.inverse_transform(
        data_split["test"]["y"].reshape(-1, 1)).ravel()
    
    ev = evaluate_model(y_true=y_orig, y_pred=mu, y_std=sigma, 
                        age_days=data_split["test"]["ages"])
    return {
        "mae": ev.regression.mae,
        "rmse": ev.regression.rmse,
        "r2": ev.regression.r2,
        "picp": ev.calibration.picp if ev.calibration else None,
        "n_test": len(data_split["test"]["y"]),
    }


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
    
    # ── 1. Load full data ────────────────────────────────────────────
    log("Loading data...")
    ds = load_and_unify_datasets(args.data_dir)
    split = grouped_stratified_split(ds, seed=42)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    
    x_train_full = x_all[split["train"]]
    y_train_full = y_all[split["train"]]
    feat_scaler = StandardScaler.fit(x_train_full)
    tgt_scaler = StandardScaler.fit(y_train_full.reshape(-1, 1))
    
    x_train_s = feat_scaler.transform(x_train_full)
    y_train_s = tgt_scaler.transform(y_train_full.reshape(-1, 1)).ravel()
    
    input_dim = x_train_s.shape[1]
    
    # ── 2. Load or train base model ──────────────────────────────────
    base_cfg = GeneratorConfig(
        input_dim=input_dim, hidden_dims=[256, 128, 64],
        epochs=300, batch_size=64, learning_rate=1e-3,
        weight_decay=1e-4, dropout=0.1, seed=42,
    )
    
    base_model = ConcreteGenerator(base_cfg)
    # Try loading checkpoint from grid search
    ckpt_paths = list(checkpoint_dir.glob("sup_lr0.001_h256x128x64_d0.1*.pt"))
    if ckpt_paths:
        base_model.load_state_dict(
            torch.load(ckpt_paths[0], map_location="cpu", weights_only=True))
        log(f"Loaded base model from {ckpt_paths[0].name}")
    else:
        log("Training base model from scratch...")
        x_val_s = feat_scaler.transform(x_all[split["val"]])
        y_val_s = tgt_scaler.transform(y_all[split["val"]].reshape(-1, 1)).ravel()
        train_generator_supervised(
            base_model, x_train_s, y_train_s, x_val_s, y_val_s, config=base_cfg)
    
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    
    # ── 3. Create narrow subset ──────────────────────────────────────
    log("Creating narrow lab subset...")
    narrow = create_narrow_subset(ds, feat_scaler, tgt_scaler, fraction=0.15)
    log(f"Narrow subset: {narrow['n_total']} total, {narrow['n_train']} train")
    
    # ── 4. Experiments ───────────────────────────────────────────────
    results = []
    
    # 4a. Baseline: base model on narrow test (no adaptation)
    log("\n=== Baseline: no adaptation ===")
    with tracker.run("transfer_no_adapt", tags=["transfer"]) as run:
        m = eval_on_split(base_model, narrow, tgt_scaler)
        run.log_metrics(m)
        results.append({"method": "no_adapt", **m})
        log(f"  MAE={m['mae']:.2f}, R2={m['r2']:.4f}")
    
    # 4b. Naive fine-tune (catastrophic forgetting expected)
    log("\n=== Naive Fine-Tune ===")
    with tracker.run("transfer_naive_ft", tags=["transfer"]) as run:
        model = ConcreteGenerator(GeneratorConfig(
            input_dim=input_dim, hidden_dims=[256, 128, 64], dropout=0.1, seed=42))
        model.load_state_dict({k: v.clone() for k, v in base_state.items()})
        
        cfg = GeneratorConfig(input_dim=input_dim, hidden_dims=[256, 128, 64],
                              epochs=200, batch_size=16, learning_rate=5e-4,
                              dropout=0.1, seed=42)
        train_generator_supervised(
            model, narrow["train"]["x"], narrow["train"]["y"],
            narrow["val"]["x"], narrow["val"]["y"], config=cfg)
        
        model = model.cpu()
        m = eval_on_split(model, narrow, tgt_scaler)
        run.log_metrics(m)
        results.append({"method": "naive_ft", **m})
        log(f"  MAE={m['mae']:.2f}, R2={m['r2']:.4f}")
    
    # 4c. EWC only
    log("\n=== EWC Fine-Tune ===")
    for ewc_lam in [100, 1000, 5000]:
        tag = f"transfer_ewc_{ewc_lam}"
        with tracker.run(tag, config={"ewc_lambda": ewc_lam},
                         tags=["transfer", "ewc"]) as run:
            model = ConcreteGenerator(GeneratorConfig(
                input_dim=input_dim, hidden_dims=[256, 128, 64], dropout=0.1, seed=42))
            model.load_state_dict({k: v.clone() for k, v in base_state.items()})
            
            model_dev = model.to(device)
            ewc = EWC(model_dev, x_train_s, y_train_s, device, n_samples=300)
            
            train_with_ewc(
                model_dev, narrow["train"]["x"], narrow["train"]["y"],
                narrow["val"]["x"], narrow["val"]["y"],
                ewc=ewc, ewc_lambda=ewc_lam, lr=5e-4, epochs=200, batch_size=16)
            
            model = model_dev.cpu()
            m = eval_on_split(model, narrow, tgt_scaler)
            run.log_metrics(m)
            results.append({"method": f"ewc_{ewc_lam}", **m})
            log(f"  λ={ewc_lam}: MAE={m['mae']:.2f}, R2={m['r2']:.4f}")
    
    # 4d. EWC + Replay Buffer
    log("\n=== EWC + Replay Buffer ===")
    for replay_pct in [0.1, 0.2, 0.3]:
        tag = f"transfer_ewc1000_replay{replay_pct}"
        with tracker.run(tag, config={"ewc_lambda": 1000, "replay": replay_pct},
                         tags=["transfer", "ewc", "replay"]) as run:
            model = ConcreteGenerator(GeneratorConfig(
                input_dim=input_dim, hidden_dims=[256, 128, 64], dropout=0.1, seed=42))
            model.load_state_dict({k: v.clone() for k, v in base_state.items()})
            
            model_dev = model.to(device)
            ewc = EWC(model_dev, x_train_s, y_train_s, device, n_samples=300)
            
            # Replay buffer: random subset of full training data
            rng = np.random.default_rng(42)
            n_replay = min(500, len(x_train_s))
            replay_idx = rng.choice(len(x_train_s), n_replay, replace=False)
            
            train_with_ewc(
                model_dev, narrow["train"]["x"], narrow["train"]["y"],
                narrow["val"]["x"], narrow["val"]["y"],
                ewc=ewc, replay_x=x_train_s[replay_idx], replay_y=y_train_s[replay_idx],
                ewc_lambda=1000, replay_ratio=replay_pct,
                lr=5e-4, epochs=200, batch_size=16)
            
            model = model_dev.cpu()
            m = eval_on_split(model, narrow, tgt_scaler)
            run.log_metrics(m)
            results.append({"method": f"ewc1000_replay{replay_pct}", **m})
            log(f"  replay={replay_pct}: MAE={m['mae']:.2f}, R2={m['r2']:.4f}")
    
    # ── 5. Summary ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("TRANSFER LEARNING RESULTS")
    log("=" * 60)
    log(f"{'Method':<30} {'MAE':>8} {'R2':>8} {'PICP':>8} {'n_test':>8}")
    log("-" * 62)
    for r in results:
        picp = f"{r['picp']:.1%}" if r['picp'] else "N/A"
        log(f"{r['method']:<30} {r['mae']:8.2f} {r['r2']:8.4f} {picp:>8} {r['n_test']:>8}")
    
    with open(checkpoint_dir / "transfer_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
