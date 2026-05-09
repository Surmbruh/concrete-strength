"""Инференс модели прогнозирования прочности бетона.

Этот скрипт позволяет запустить обученную модель на ВАШИХ данных.

Формат входных данных (CSV):
    Обязательные колонки (порядок не важен):
    - cement       (кг/м³) — цемент
    - water        (кг/м³) — вода
    - sand         (кг/м³) — мелкий заполнитель (песок)
    - coarse_agg   (кг/м³) — крупный заполнитель
    - fine_add_1   (кг/м³) — зола-унос / шлак (0 если нет)
    - fine_add_2   (кг/м³) — микрокремнезём / доп. добавка (0 если нет)
    - plasticizer  (кг/м³) — пластификатор (0 если нет)
    - age_days     (дни)   — возраст бетона (1, 3, 7, 28, ...)

Пример CSV:
    cement,water,sand,coarse_agg,fine_add_1,fine_add_2,plasticizer,age_days
    350,180,750,1050,0,0,5,28
    300,190,800,1000,50,0,8,7

Использование:
    python predict.py --input your_data.csv --checkpoint_dir experiments/checkpoints
    python predict.py --input data.csv --checkpoint_dir checkpoints --method single

Выход: CSV с колонками predicted_strength, uncertainty_95ci, lower_bound, upper_bound
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from materialgen.data_preparation import (
    load_and_unify_datasets, stratified_split,
    COMPOSITION_COLUMNS, DERIVED_COLUMNS,
)
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig


def build_features(df):
    """Строит полный вектор признаков из пользовательского CSV.

    Принимает DataFrame с composition колонками + age_days.
    Возвращает numpy array [n_samples, 10] с теми же признаками,
    что использовались при обучении.
    """
    cement = df["cement"].values.astype(float)
    water = df["water"].values.astype(float)
    fine1 = df["fine_add_1"].values.astype(float)
    fine2 = df["fine_add_2"].values.astype(float)
    age = df["age_days"].values.astype(float)

    wc_ratio = np.where(cement > 0, water / cement, 0.0)
    wb_ratio = np.where(
        (cement + fine1 + fine2) > 0,
        water / (cement + fine1 + fine2),
        0.0)
    log_age = np.log1p(age)

    composition = df[list(COMPOSITION_COLUMNS)].values.astype(float)
    derived = np.column_stack([wc_ratio, wb_ratio, log_age])

    return np.hstack([composition, derived])


def load_scalers(data_dir="data", seed=42):
    """Загружает скейлеры, обученные на training data."""
    ds = load_and_unify_datasets(data_dir)
    split = stratified_split(ds, seed=seed)
    x_train = ds.all_features[split["train"]]
    y_train = ds.target.to_numpy()[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
    return feat_scaler, tgt_scaler


def load_model(checkpoint_path, input_dim):
    """Пробует загрузить модель с разными архитектурами."""
    for hidden in [[256, 128, 64], [256, 128, 64, 32]]:
        try:
            gen = ConcreteGenerator(GeneratorConfig(
                input_dim=input_dim, hidden_dims=hidden,
                dropout=0.1, seed=42))
            gen.load_state_dict(torch.load(
                checkpoint_path, map_location="cpu", weights_only=True))
            return gen
        except Exception:
            continue
    return None


def predict_single(x_scaled, checkpoint_dir, tgt_scaler, mc_samples=30):
    """Предсказание лучшей одиночной GAN моделью."""
    ckpts = sorted(Path(checkpoint_dir).glob("gan_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No GAN checkpoints in {checkpoint_dir}")

    for ckpt in ckpts:
        gen = load_model(ckpt, x_scaled.shape[1])
        if gen:
            print(f"Loaded: {ckpt.name}")
            mu_s, sig_s = gen.predict(x_scaled, mc_samples=mc_samples)
            mu = tgt_scaler.inverse_transform(mu_s).ravel()
            sigma = sig_s.ravel() * tgt_scaler.scale[0]
            return mu, 1.96 * sigma

    raise RuntimeError("Could not load any GAN checkpoint")


def predict_ensemble(x_scaled, checkpoint_dir, tgt_scaler, mc_samples=20):
    """Предсказание ансамблем всех GAN моделей."""
    ckpts = sorted(Path(checkpoint_dir).glob("gan_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No GAN checkpoints in {checkpoint_dir}")

    all_mu, all_sigma = [], []
    for ckpt in ckpts:
        gen = load_model(ckpt, x_scaled.shape[1])
        if gen:
            mu_s, sig_s = gen.predict(x_scaled, mc_samples=mc_samples)
            all_mu.append(tgt_scaler.inverse_transform(mu_s).ravel())
            all_sigma.append(sig_s.ravel() * tgt_scaler.scale[0])
            print(f"  Loaded: {ckpt.name}")

    if not all_mu:
        raise RuntimeError("Could not load any GAN checkpoint")

    print(f"Ensemble: {len(all_mu)} models")
    all_mu = np.stack(all_mu)
    all_sigma = np.stack(all_sigma)

    mu = all_mu.mean(axis=0)
    sigma = np.sqrt(all_mu.std(axis=0)**2 + all_sigma.mean(axis=0)**2)
    return mu, 1.96 * sigma


def main():
    parser = argparse.ArgumentParser(
        description="Прогноз прочности бетона по составу",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV с данными")
    parser.add_argument("--output", default=None, help="Выходной CSV")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                       help="Папка с .pt чекпоинтами (по умолчанию: checkpoints/)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--method", choices=["single", "ensemble"],
                       default="ensemble")
    parser.add_argument("--mc_samples", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Input: {len(df)} samples from {args.input}")

    # Check required columns
    required = list(COMPOSITION_COLUMNS) + ["age_days"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        print(f"Required: {required}")
        print(f"Found:    {list(df.columns)}")
        sys.exit(1)

    # Fill NaN
    for col in ["fine_add_1", "fine_add_2", "plasticizer"]:
        df[col] = df[col].fillna(0)

    # Build features & scale
    feat_scaler, tgt_scaler = load_scalers(args.data_dir)
    x_raw = build_features(df)
    x_scaled = feat_scaler.transform(x_raw)

    # Predict
    if args.method == "single":
        mu, ci95 = predict_single(x_scaled, args.checkpoint_dir,
                                  tgt_scaler, args.mc_samples)
    else:
        mu, ci95 = predict_ensemble(x_scaled, args.checkpoint_dir,
                                    tgt_scaler, args.mc_samples)

    # Output
    result = df.copy()
    result["predicted_strength"] = np.round(mu, 2)
    result["uncertainty_95ci"] = np.round(ci95, 2)
    result["lower_bound"] = np.round(mu - ci95, 2)
    result["upper_bound"] = np.round(mu + ci95, 2)

    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
    print(result[["cement", "water", "age_days",
                   "predicted_strength", "uncertainty_95ci"]].head(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()
