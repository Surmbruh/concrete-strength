"""Verify grouped split has no leakage + compare split sizes."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from materialgen.data_preparation import (
    load_and_unify_datasets, stratified_split, grouped_stratified_split,
    COMPOSITION_COLUMNS
)
import numpy as np

ds = load_and_unify_datasets("data")

# Old split
old = stratified_split(ds, seed=42)
# New split
new = grouped_stratified_split(ds, seed=42)

print("=" * 60)
print("SPLIT SIZES")
print("=" * 60)
print(f"{'':15s} {'Old':>8s} {'New (grouped)':>14s}")
for k in ['train', 'val', 'test']:
    print(f"  {k:12s} {len(old[k]):8d} {len(new[k]):14d}")

# Check leakage in new split
feat = ds.features.copy()
feat['source'] = ds.source
feat['age_days'] = ds.age_days

feat['split'] = 'unknown'
feat.loc[new['train'], 'split'] = 'train'
feat.loc[new['val'], 'split'] = 'val'
feat.loc[new['test'], 'split'] = 'test'

comp_cols = COMPOSITION_COLUMNS
feat['comp_key'] = feat[comp_cols].apply(lambda r: tuple(round(x, 2) for x in r), axis=1)

# Find compositions in multiple splits
multi_age = feat.groupby('comp_key').filter(lambda g: g['age_days'].nunique() > 1)
leaked = multi_age.groupby('comp_key').filter(lambda g: g['split'].nunique() > 1)

print(f"\n{'=' * 60}")
print(f"LEAKAGE CHECK (grouped split)")
print(f"{'=' * 60}")
print(f"Compositions with multiple ages: {multi_age['comp_key'].nunique()}")
print(f"Compositions leaked across splits: {leaked['comp_key'].nunique()}")

test_comps = set(feat.loc[feat['split'] == 'test', 'comp_key'])
train_comps = set(feat.loc[feat['split'] == 'train', 'comp_key'])
overlap = test_comps & train_comps
test_with_leak = feat[(feat['split'] == 'test') & (feat['comp_key'].isin(overlap))]
print(f"Test samples with same composition in train: {len(test_with_leak)} / {len(new['test'])}")

# Distribution check
print(f"\n{'=' * 60}")
print(f"TARGET DISTRIBUTION (mean ± std)")
print(f"{'=' * 60}")
targets = ds.target.to_numpy()
for name, idx_arr in [("Old train", old['train']), ("Old test", old['test']),
                       ("New train", new['train']), ("New test", new['test'])]:
    t = targets[idx_arr]
    print(f"  {name:12s}: {t.mean():.2f} ± {t.std():.2f}  (n={len(t)})")
