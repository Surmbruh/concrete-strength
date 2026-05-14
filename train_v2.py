#!/usr/bin/env python3
"""
=================================================================
 TRAINING PIPELINE v2  —  Concrete Strength Prediction
=================================================================

Improvements over v1:
  1. Feature engineering: +3 interaction features (13 total)
     - cement × log(age), w/c × log(age), binder/aggregate ratio
  2. GAN epochs: 500 → 1000 (previous best_epoch=495 → training didn't converge)
  3. Cosine annealing LR scheduler for GAN training
  4. AdamW instead of Adam (better weight decay regularization)
  5. Early stopping patience: 50 → 80

Usage in Colab:
  Copy each section into a separate cell.
  
Estimated time (Tesla T4):
  - Supervised grid:  ~15 min
  - GAN tune:         ~45 min  (1000 epochs × 9 configs)
  - Ensemble:         ~5 min
  - Bonus tasks:      ~15 min
  Total:              ~80 min
=================================================================
"""


# ==============================================================
# CELL 1: Setup — run ONCE
# ==============================================================

# %%
# --- Mount Drive & Clone Repo ---
from google.colab import drive
drive.mount('/content/drive')

!cd /content && rm -rf concrete-strength
!cd /content && git clone https://github.com/AICraft-Labs/concrete-strength.git

%cd /content/concrete-strength
!pip install -q neat-python pyro-ppl

# --- Verify new features ---
import importlib, materialgen.data_preparation as dp
importlib.reload(dp)
print(f"Features ({len(dp.ALL_FEATURE_COLUMNS)}):")
for i, f in enumerate(dp.ALL_FEATURE_COLUMNS):
    print(f"  [{i}] {f}")
assert len(dp.ALL_FEATURE_COLUMNS) == 13, f"Expected 13 features, got {len(dp.ALL_FEATURE_COLUMNS)}"
print("\n✅ 13 features confirmed (7 composition + 6 derived)")


# ==============================================================
# CELL 2: Clean old checkpoints — run ONCE before training
# ==============================================================

# %%
import os

EXP_DIR = "/content/drive/MyDrive/concrete_project/experiments"
CKPT_DIR = f"{EXP_DIR}/checkpoints"
LOG_FILE = f"{EXP_DIR}/experiment_log.jsonl"

# Full cleanup
!rm -rf {CKPT_DIR}/*
!rm -f  {LOG_FILE}
os.makedirs(CKPT_DIR, exist_ok=True)

# Verify
remaining = os.listdir(CKPT_DIR)
print(f"Checkpoints after cleanup: {len(remaining)} files")
print(f"experiment_log.jsonl exists: {os.path.exists(LOG_FILE)}")
assert len(remaining) == 0, "Cleanup failed!"
print("✅ Clean slate — ready to train")


# ==============================================================
# CELL 3: Supervised Grid Search (~15 min)
# ==============================================================

# %%
%cd /content/concrete-strength
!python run_experiment.py --mode supervised_grid \
    --output_dir /content/drive/MyDrive/concrete_project/experiments


# ==============================================================
# CELL 4: GAN Tune — 1000 epochs with cosine annealing (~45 min)
# ==============================================================

# %%
!python run_experiment.py --mode gan_tune \
    --output_dir /content/drive/MyDrive/concrete_project/experiments


# ==============================================================
# CELL 5: Multi-seed Ensemble (~5 min)
# ==============================================================

# %%
!python run_ensemble.py \
    --output_dir /content/drive/MyDrive/concrete_project/experiments

# ==============================================================
# CELL 6: Stacking Ensemble (~2 min, no retraining needed)
# ==============================================================

# %%
!pip install -q scikit-learn
!python run_stacking.py \
    --output_dir /content/drive/MyDrive/concrete_project/experiments


# ==============================================================
# CELL 7: Bonus Tasks (~15 min)
# ==============================================================

# %%
%env EXP_DIR=/content/drive/MyDrive/concrete_project/experiments

!python run_bonus_transfer.py   --output_dir $EXP_DIR
!python run_bonus_fewshot_time.py --output_dir $EXP_DIR
!python run_bonus_multiprop.py   --output_dir $EXP_DIR


# ==============================================================
# CELL 7: Quick Sanity Check
# ==============================================================

# %%
import json
from pathlib import Path

ckpt = Path("/content/drive/MyDrive/concrete_project/experiments/checkpoints")

# Check feature dim
import numpy as np
import torch
from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split

ds = load_and_unify_datasets("data")
print(f"Features: {ds.all_features.shape[1]} (expected 13)")
assert ds.all_features.shape[1] == 13

# Check GAN results
gan_file = ckpt / "gan_tune.json"
if gan_file.exists():
    gan_results = json.load(open(gan_file))
    print(f"\nGAN configs trained: {len(gan_results)}")
    print(f"Best GAN: MAE={gan_results[0]['mae']:.2f}, R2={gan_results[0]['r2']:.4f}")
    print(f"  Config: {gan_results[0]['tag']}")
    print(f"  GAN epochs used: {gan_results[0].get('gan_epochs', '?')}")
else:
    print("⚠️ gan_tune.json not found — run Cell 4 first")

# Check ensemble
ens_file = ckpt / "ensemble_results.json"
if ens_file.exists():
    ens = json.load(open(ens_file))
    print(f"\nEnsemble results: {len(ens)} configs")
    for name, metrics in ens.items():
        if "mae" in metrics:
            print(f"  {name}: MAE={metrics['mae']:.2f}")
else:
    print("⚠️ ensemble_results.json not found — run Cell 5 first")

print("\n✅ Pipeline complete!")
"""

# ==============================================================
# CELL 8 (Optional): Compare with v1 results
# ==============================================================

# %%
print("=== v1 vs v2 Comparison ===")
print()
print("v1 results (10 features, 500 GAN epochs):")
print("  Supervised:    MAE=9.16, R²=0.495")
print("  Best GAN:      MAE=8.52, R²=0.533")
print("  Ensemble top3: MAE=8.58, R²=0.529")
print()
print("v2 results (13 features, 1000 GAN epochs + cosine annealing):")
# These will be filled by Cell 7 output ↑
"""
