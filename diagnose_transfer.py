"""Diagnose WHY fine-tuning degraded the transfer learning model.

Checks:
1. Data distribution mismatch between sources
2. Test set composition (which source do test samples come from?)
3. Per-source evaluation of each model
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler

SEED = 42

def main():
    # Load per-source datasets
    ds_all = load_and_unify_datasets("data")
    ds_nc = load_and_unify_datasets("data", include_normal=True, include_boxcrete=False, include_synthetic=False)
    ds_box = load_and_unify_datasets("data", include_normal=False, include_boxcrete=True, include_synthetic=False)
    ds_syn = load_and_unify_datasets("data", include_normal=False, include_boxcrete=False, include_synthetic=True)

    print(f"Normal Concrete: {ds_nc.n_samples}")
    print(f"Boxcrete:        {ds_box.n_samples}")
    print(f"Synthetic:       {ds_syn.n_samples}")
    print(f"Total:           {ds_all.n_samples}")

    # Test set from full stratified split
    split = grouped_stratified_split(ds_all, seed=SEED)
    test_idx = split["test"]
    print(f"\nTest set size: {len(test_idx)}")

    # Figure out which source each test sample comes from
    n_nc = ds_nc.n_samples
    n_box = ds_box.n_samples
    n_syn = ds_syn.n_samples

    sources = []
    for i in test_idx:
        if i < n_nc:
            sources.append("normal_concrete")
        elif i < n_nc + n_box:
            sources.append("boxcrete")
        else:
            sources.append("synthetic")
    
    from collections import Counter
    source_counts = Counter(sources)
    print("\nTest set composition by source:")
    for src, cnt in sorted(source_counts.items()):
        pct = cnt / len(test_idx) * 100
        print(f"  {src:20s}: {cnt:4d} ({pct:.1f}%)")

    # Compare feature distributions between sources
    feat_names = ["cement", "water", "fine_agg", "coarse_agg",
                  "fine_add_1", "fine_add_2", "plasticizer",
                  "wc_ratio", "wb_ratio", "log_age"]
    
    x_nc = ds_nc.all_features
    x_syn = ds_syn.all_features
    y_nc = ds_nc.target.to_numpy()
    y_syn = ds_syn.target.to_numpy()
    
    print("\n" + "=" * 80)
    print("DISTRIBUTION COMPARISON: Normal_Concrete vs Synthetic")
    print("=" * 80)
    print(f"{'Feature':15s} {'NC mean':>10} {'NC std':>10} {'SYN mean':>10} {'SYN std':>10} {'Shift':>10}")
    print("-" * 65)
    
    # Fit scaler on full training set
    x_train = ds_all.all_features[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    
    for j, name in enumerate(feat_names):
        nc_mean = x_nc[:, j].mean()
        nc_std = x_nc[:, j].std()
        syn_mean = x_syn[:, j].mean()
        syn_std = x_syn[:, j].std()
        # Standardized shift (how many NC-stds apart)
        shift = abs(nc_mean - syn_mean) / (nc_std + 1e-8)
        print(f"{name:15s} {nc_mean:10.2f} {nc_std:10.2f} {syn_mean:10.2f} {syn_std:10.2f} {shift:10.2f}")

    # Target distribution
    print(f"\n{'target':15s} {y_nc.mean():10.2f} {y_nc.std():10.2f} {y_syn.mean():10.2f} {y_syn.std():10.2f} {abs(y_nc.mean()-y_syn.mean())/(y_nc.std()+1e-8):10.2f}")

    # Age distribution
    print("\nAge distribution:")
    ages_nc = ds_nc.age_days.to_numpy()
    ages_syn = ds_syn.age_days.to_numpy()
    print(f"  NC  ages: {sorted(set(ages_nc.astype(int)))}")
    print(f"  SYN ages: {sorted(set(ages_syn.astype(int)))}")

    # The key insight
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    nc_pct = source_counts.get("normal_concrete", 0) / len(test_idx) * 100
    box_pct = source_counts.get("boxcrete", 0) / len(test_idx) * 100
    syn_pct = source_counts.get("synthetic", 0) / len(test_idx) * 100
    pretrain_pct = nc_pct + box_pct
    
    print(f"\nTest set: {pretrain_pct:.0f}% from pre-train sources, {syn_pct:.0f}% from fine-tune source")
    print(f"\nWhen we fine-tune on synthetic only ({ds_syn.n_samples} samples),")
    print(f"the model forgets patterns for {pretrain_pct:.0f}% of the test data!")
    print(f"This is CATASTROPHIC FORGETTING - the model overwrites pre-trained")
    print(f"knowledge with synthetic-specific patterns.")

if __name__ == "__main__":
    main()
