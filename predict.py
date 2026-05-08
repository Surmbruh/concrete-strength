"""Инференс модели прогнозирования прочности бетона.

Этот скрипт позволяет запустить обученную модель на ВАШИХ данных.

Формат входных данных (CSV):
    Обязательные колонки (порядок не важен):
    - cement       (кг/м³) — цемент
    - water        (кг/м³) — вода
    - sand         (кг/м³) — мелкий заполнитель (песок)
    - coarse_agg   (кг/м³) — крупный заполнитель
    - fly_ash      (кг/м³) — зола-унос (0 если нет)
    - blast_furnace_slag (кг/м³) — шлак (0 если нет)
    - superplasticizer   (кг/м³) — суперпластификатор (0 если нет)
    - age_days     (дни)   — возраст бетона (1, 3, 7, 28, ...)

Пример CSV:
    cement,water,sand,coarse_agg,fly_ash,blast_furnace_slag,superplasticizer,age_days
    350,180,750,1050,0,0,5,28
    300,190,800,1000,50,0,8,7
    400,160,700,1100,0,100,10,28

Использование:
    python predict.py --input your_data.csv --checkpoint_dir experiments/checkpoints

    # С выбором метода:
    python predict.py --input data.csv --checkpoint_dir checkpoints --method single
    python predict.py --input data.csv --checkpoint_dir checkpoints --method ensemble

Выход: CSV с колонками
    - predicted_strength (МПа) — предсказание прочности
    - uncertainty (МПа)        — 95% доверительный интервал (±)
    - lower_bound (МПа)        — нижняя граница
    - upper_bound (МПа)        — верхняя граница
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig


# ── Derived features (те же, что при обучении) ────────────────────────
COMPOSITION_COLS = [
    "cement", "water", "sand", "coarse_agg",
    "fly_ash", "blast_furnace_slag", "superplasticizer",
]

def compute_derived_features(df):
    """Вычисляет производные признаки из состава и возраста."""
    cement = df["cement"].values
    water = df["water"].values
    bfs = df["blast_furnace_slag"].values
    fa = df["fly_ash"].values
    age = df["age_days"].values.astype(float)

    wc_ratio = np.where(cement > 0, water / cement, 0)
    total_binder = cement + bfs + fa
    log_age = np.log1p(age)

    return np.column_stack([
        df[COMPOSITION_COLS].values,
        wc_ratio,
        total_binder,
        log_age,
    ])


def load_scalers(data_dir="data", seed=42):
    """Загружает скейлеры, обученные на training data."""
    ds = load_and_unify_datasets(data_dir)
    split = stratified_split(ds, seed=seed)
    x_train = ds.all_features[split["train"]]
    y_train = ds.target.to_numpy()[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
    return feat_scaler, tgt_scaler


def predict_single(df, checkpoint_dir, data_dir="data", mc_samples=30):
    """Предсказание лучшей одиночной GAN моделью."""
    feat_scaler, tgt_scaler = load_scalers(data_dir)

    # Prepare features
    x_raw = compute_derived_features(df)
    x_scaled = feat_scaler.transform(x_raw)

    # Load best GAN checkpoint
    ckpts = sorted(Path(checkpoint_dir).glob("gan_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No GAN checkpoints in {checkpoint_dir}")

    best_gen = None
    best_ckpt = None
    for ckpt in ckpts:
        for hidden in [[256, 128, 64], [256, 128, 64, 32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=x_scaled.shape[1], hidden_dims=hidden,
                    dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu",
                                              weights_only=True))
                best_gen = gen
                best_ckpt = ckpt.name
                break
            except Exception:
                continue
        if best_gen:
            break

    if not best_gen:
        raise RuntimeError("Could not load any GAN checkpoint")

    print(f"Loaded model: {best_ckpt}")
    mu_scaled, sigma_scaled = best_gen.predict(x_scaled, mc_samples=mc_samples)

    mu = tgt_scaler.inverse_transform(mu_scaled).ravel()
    sigma = sigma_scaled.ravel() * tgt_scaler.scale[0]
    ci95 = 1.96 * sigma

    return mu, ci95


def predict_ensemble(df, checkpoint_dir, data_dir="data", mc_samples=20):
    """Предсказание ансамблем всех GAN моделей (усреднение)."""
    feat_scaler, tgt_scaler = load_scalers(data_dir)

    x_raw = compute_derived_features(df)
    x_scaled = feat_scaler.transform(x_raw)

    ckpts = sorted(Path(checkpoint_dir).glob("gan_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No GAN checkpoints in {checkpoint_dir}")

    all_mu = []
    all_sigma = []
    loaded = 0

    for ckpt in ckpts:
        for hidden in [[256, 128, 64], [256, 128, 64, 32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=x_scaled.shape[1], hidden_dims=hidden,
                    dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu",
                                              weights_only=True))

                mu_s, sig_s = gen.predict(x_scaled, mc_samples=mc_samples)
                all_mu.append(tgt_scaler.inverse_transform(mu_s).ravel())
                all_sigma.append(sig_s.ravel() * tgt_scaler.scale[0])
                loaded += 1
                print(f"  Loaded: {ckpt.name}")
                break
            except Exception:
                continue

    print(f"Ensemble: {loaded} models")
    all_mu = np.stack(all_mu, axis=0)
    all_sigma = np.stack(all_sigma, axis=0)

    mu = all_mu.mean(axis=0)
    # Total uncertainty = model disagreement + average aleatoric
    sigma = np.sqrt(all_mu.std(axis=0)**2 + all_sigma.mean(axis=0)**2)
    ci95 = 1.96 * sigma

    return mu, ci95


def main():
    parser = argparse.ArgumentParser(
        description="Прогноз прочности бетона по составу",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--input", required=True,
                       help="CSV с данными (см. формат выше)")
    parser.add_argument("--output", default=None,
                       help="Выходной CSV (по умолчанию: input_predictions.csv)")
    parser.add_argument("--checkpoint_dir",
                       default="experiments/checkpoints",
                       help="Папка с .pt чекпоинтами")
    parser.add_argument("--data_dir", default="data",
                       help="Папка с обучающими данными (для скейлеров)")
    parser.add_argument("--method", choices=["single", "ensemble"],
                       default="ensemble",
                       help="single = лучшая модель, ensemble = все модели")
    parser.add_argument("--mc_samples", type=int, default=30,
                       help="MC-dropout сэмплы для оценки неопределённости")
    args = parser.parse_args()

    # Read input
    df = pd.read_csv(args.input)
    print(f"Input: {len(df)} samples from {args.input}")

    # Check columns
    required = COMPOSITION_COLS + ["age_days"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        print(f"Required: {required}")
        print(f"Found:    {list(df.columns)}")
        sys.exit(1)

    # Fill NaN in optional columns
    for col in ["fly_ash", "blast_furnace_slag", "superplasticizer"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Predict
    if args.method == "single":
        mu, ci95 = predict_single(df, args.checkpoint_dir, args.data_dir,
                                  args.mc_samples)
    else:
        mu, ci95 = predict_ensemble(df, args.checkpoint_dir, args.data_dir,
                                    args.mc_samples)

    # Build output
    result = df.copy()
    result["predicted_strength"] = np.round(mu, 2)
    result["uncertainty_95ci"] = np.round(ci95, 2)
    result["lower_bound"] = np.round(mu - ci95, 2)
    result["upper_bound"] = np.round(mu + ci95, 2)

    # Output
    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print(f"\nPreview:")
    preview_cols = ["cement", "water", "age_days", "predicted_strength",
                    "uncertainty_95ci"]
    preview_cols = [c for c in preview_cols if c in result.columns]
    print(result[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
