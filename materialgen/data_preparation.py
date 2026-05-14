"""Подготовка данных: унификация датасетов, feature engineering, split.

Объединяет три источника данных (Normal_Concrete_DB, boxcrete_data,
synthetic_training_data) в единый формат для обучения GAN. Реализует
feature engineering (w/c ratio, log(t), взаимодействия) и
стратифицированное разбиение на train/val/test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import read_dataset_frame


# =============================================================================
# Единая схема данных
# =============================================================================

# Канонические имена столбцов после унификации
COMPOSITION_COLUMNS = [
    "cement",
    "water",
    "sand",
    "coarse_agg",
    "fine_add_1",   # fly ash, BFS, mineral powder, etc.
    "fine_add_2",   # microsilica, дополнительная добавка
    "plasticizer",
]
DERIVED_COLUMNS = [
    "w_c_ratio",         # water / cement (закон Абрамса)
    "w_b_ratio",         # water / (cement + fine_add_1 + fine_add_2)
    "log_age",           # log(age_days)
    "cement_x_logage",   # cement * log(age) — взаимодействие
    "wc_x_logage",       # w_c_ratio * log(age) — Абрамс во времени
    "binder_agg_ratio",  # (cement + fine_add_1) / (sand + coarse_agg)
]
TARGET_COLUMN = "strength_mpa"
AGE_COLUMN = "age_days"

ALL_FEATURE_COLUMNS = COMPOSITION_COLUMNS + DERIVED_COLUMNS


@dataclass
class UnifiedDataset:
    """Результат объединения и предобработки всех источников данных."""

    features: pd.DataFrame          # все признаки (composition + derived)
    target: pd.Series               # strength_mpa
    age_days: pd.Series             # возраст образца (дни)
    source: pd.Series               # источник каждой записи
    composition_columns: list[str] = field(default_factory=lambda: list(COMPOSITION_COLUMNS))
    derived_columns: list[str] = field(default_factory=lambda: list(DERIVED_COLUMNS))
    target_column: str = TARGET_COLUMN

    @property
    def n_samples(self) -> int:
        return len(self.features)

    @property
    def composition(self) -> np.ndarray:
        """Матрица составов [n_samples, n_components]."""
        return self.features[self.composition_columns].to_numpy(dtype=float)

    @property
    def all_features(self) -> np.ndarray:
        """Все признаки включая derived [n_samples, n_features]."""
        return self.features.to_numpy(dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "composition_columns": self.composition_columns,
            "derived_columns": self.derived_columns,
            "target_column": self.target_column,
            "sources": dict(self.source.value_counts()),
            "age_days_unique": sorted(self.age_days.unique().tolist()),
        }


# =============================================================================
# Маппинг колонок из исходных датасетов
# =============================================================================

def _unify_normal_concrete_db(csv_path: str | Path) -> pd.DataFrame:
    """Привести Normal_Concrete_DB.csv к единой схеме.

    Исходные колонки (sep=';'):
        Цем (кг/м³), BFS (кг/м³), Зола-унос (кг/м³), Пластификатор (кг/м³),
        Вода (кг/м³), Суперпластификатор (кг/м³), Крупный заполнитель (кг/м³),
        Мелкий заполнитель (кг/м³), Возраст (дни), Осадка конуса (мм), CS_28d (МПа)

    В этом датасете есть Возраст — используем его (обычно 28 дней, но не всегда).
    """
    frame = read_dataset_frame(csv_path)

    # Автоматический маппинг по порядку (CSV с BOM может иметь проблемы с именами)
    cols = list(frame.columns)
    # Порядок колонок фиксирован в описании датасета
    mapping = {}
    if len(cols) >= 11:
        mapping = {
            cols[0]: "cement",
            cols[1]: "fine_add_1",       # BFS
            cols[2]: "fine_add_2",       # Зола-унос
            cols[3]: "plasticizer",
            cols[4]: "water",
            cols[5]: "_superplasticizer",
            cols[6]: "coarse_agg",
            cols[7]: "sand",
            cols[8]: "age_days",
            cols[9]: "_slump",
            cols[10]: "strength_mpa",
        }
    else:
        raise ValueError(
            f"Normal_Concrete_DB: expected ≥11 columns, got {len(cols)}: {cols}"
        )

    frame = frame.rename(columns=mapping)

    # Числовая коррекция — десятичный разделитель ','
    for col in ["cement", "fine_add_1", "fine_add_2", "plasticizer",
                "water", "coarse_agg", "sand", "age_days", "strength_mpa",
                "_superplasticizer"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(
                frame[col].astype(str).str.replace(",", "."), errors="coerce",
            )

    # Пластификатор = сумма обоих
    if "_superplasticizer" in frame.columns:
        frame["plasticizer"] = frame["plasticizer"].fillna(0) + frame["_superplasticizer"].fillna(0)

    frame["source"] = "normal_concrete_db"

    # Отбираем только нужные столбцы
    result_cols = COMPOSITION_COLUMNS + [AGE_COLUMN, TARGET_COLUMN, "source"]
    for col in result_cols:
        if col not in frame.columns:
            frame[col] = 0.0

    return frame[result_cols].dropna(subset=[TARGET_COLUMN])


def _unify_boxcrete(csv_path: str | Path) -> pd.DataFrame:
    """Привести boxcrete_data.csv к единой схеме.

    Прочность в psi → конвертация в МПа (×0.00689476).
    Содержит временные ряды: Time = 1, 3, 5, 28 дней.
    """
    frame = read_dataset_frame(csv_path)

    result = pd.DataFrame()
    result["cement"] = pd.to_numeric(frame.get("Cement (kg/m3)", 0), errors="coerce").fillna(0)
    result["water"] = pd.to_numeric(frame.get("Water (kg/m3)", 0), errors="coerce").fillna(0)
    result["sand"] = pd.to_numeric(frame.get("Fine Aggregate (kg/m3)", 0), errors="coerce").fillna(0)
    result["coarse_agg"] = pd.to_numeric(frame.get("Coarse Aggregates (kg/m3)", 0), errors="coerce").fillna(0)
    result["fine_add_1"] = (
        pd.to_numeric(frame.get("Fly Ash (kg/m3)", 0), errors="coerce").fillna(0)
        + pd.to_numeric(frame.get("Slag (kg/m3)", 0), errors="coerce").fillna(0)
    )
    result["fine_add_2"] = 0.0
    result["plasticizer"] = (
        pd.to_numeric(frame.get("HRWR (kg/m3)", 0), errors="coerce").fillna(0)
        + pd.to_numeric(frame.get("MRWR (kg/m3)", 0), errors="coerce").fillna(0)
    )
    result["age_days"] = pd.to_numeric(frame.get("Time", 28), errors="coerce").fillna(28)

    # Конвертация psi → MPa
    strength_psi = pd.to_numeric(
        frame.get("strength(mean) (MPa)", frame.get("Strength (Mean)", 0)),
        errors="coerce",
    ).fillna(0)
    # Если колонка уже в MPa, используем как есть; иначе конвертируем
    if "strength(mean) (MPa)" in frame.columns:
        result["strength_mpa"] = strength_psi
    else:
        result["strength_mpa"] = strength_psi * 0.00689476

    result["source"] = "boxcrete"
    return result.dropna(subset=[TARGET_COLUMN])


def _unify_synthetic(csv_path: str | Path) -> pd.DataFrame:
    """Привести synthetic_training_data.csv к единой схеме.

    Содержит strength_1, strength_3, strength_7, strength_28 →
    «разворачиваем» в 4 строки на каждый состав.
    """
    frame = read_dataset_frame(csv_path)

    # Маппинг столбцов
    base = pd.DataFrame()
    base["cement"] = pd.to_numeric(frame.get("cement", 0), errors="coerce").fillna(0)
    base["water"] = pd.to_numeric(frame.get("water", 0), errors="coerce").fillna(0)
    base["sand"] = pd.to_numeric(frame.get("sand", 0), errors="coerce").fillna(0)
    base["coarse_agg"] = pd.to_numeric(frame.get("gravel", 0), errors="coerce").fillna(0)
    base["fine_add_1"] = (
        pd.to_numeric(frame.get("fly_ash", 0), errors="coerce").fillna(0)
        + pd.to_numeric(frame.get("mineral_powder", 0), errors="coerce").fillna(0)
    )
    base["fine_add_2"] = pd.to_numeric(frame.get("microsilica_kg", 0), errors="coerce").fillna(0)
    base["plasticizer"] = pd.to_numeric(frame.get("plasticizer_kg", 0), errors="coerce").fillna(0)

    # Разворачиваем по времени: strength_1, strength_3, strength_7, strength_28
    time_columns = {1: "strength_1", 3: "strength_3", 7: "strength_7", 28: "strength_28"}
    rows = []
    for idx, row in base.iterrows():
        for age, col_name in time_columns.items():
            if col_name in frame.columns:
                strength = pd.to_numeric(frame.at[idx, col_name], errors="coerce")
                if pd.notna(strength) and strength > 0:
                    new_row = row.to_dict()
                    new_row["age_days"] = float(age)
                    new_row["strength_mpa"] = float(strength)
                    new_row["source"] = "synthetic"
                    rows.append(new_row)

    return pd.DataFrame(rows)


# =============================================================================
# Feature Engineering
# =============================================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить производные признаки к единому DataFrame.

    Создаёт: w_c_ratio, w_b_ratio, log_age, cement_x_logage,
    wc_x_logage, binder_agg_ratio.
    """
    df = df.copy()

    # Водоцементное отношение (закон Абрамса)
    cement_safe = df["cement"].clip(lower=1.0)
    df["w_c_ratio"] = df["water"] / cement_safe

    # Водо-вяжущее отношение
    binder = df["cement"] + df["fine_add_1"] + df["fine_add_2"]
    binder_safe = binder.clip(lower=1.0)
    df["w_b_ratio"] = df["water"] / binder_safe

    # Логарифм возраста
    age_safe = df["age_days"].clip(lower=0.5)
    df["log_age"] = np.log(age_safe)

    # Взаимодействие цемента и времени (набор прочности)
    df["cement_x_logage"] = df["cement"] * df["log_age"]

    # Абрамс во времени: w/c ratio × log(age)
    df["wc_x_logage"] = df["w_c_ratio"] * df["log_age"]

    # Отношение вяжущее / заполнитель
    aggregate = (df["sand"] + df["coarse_agg"]).clip(lower=1.0)
    df["binder_agg_ratio"] = (df["cement"] + df["fine_add_1"]) / aggregate

    return df


# =============================================================================
# Объединение + Split
# =============================================================================

def load_and_unify_datasets(
    data_dir: str | Path,
    *,
    normal_concrete_file: str = "Normal_Concrete_DB.csv",
    boxcrete_file: str = "boxcrete_data.csv",
    synthetic_file: str = "synthetic_training_data.csv",
    include_normal: bool = True,
    include_boxcrete: bool = True,
    include_synthetic: bool = True,
) -> UnifiedDataset:
    """Загрузить, объединить и обогатить все датасеты.

    Parameters
    ----------
    data_dir : путь к папке с CSV файлами
    include_* : какие датасеты включать
    """
    data_path = Path(data_dir)
    parts: list[pd.DataFrame] = []

    if include_normal:
        path = data_path / normal_concrete_file
        if path.exists():
            parts.append(_unify_normal_concrete_db(path))

    if include_boxcrete:
        path = data_path / boxcrete_file
        if path.exists():
            parts.append(_unify_boxcrete(path))

    if include_synthetic:
        path = data_path / synthetic_file
        if path.exists():
            parts.append(_unify_synthetic(path))

    if not parts:
        raise FileNotFoundError(f"No datasets found in {data_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = add_derived_features(combined)

    # Удаляем строки с NaN в ключевых столбцах
    combined = combined.dropna(subset=[TARGET_COLUMN, "cement", "water"])

    # Заполняем оставшиеся NaN нулями (добавки, пластификатор и т.д.)
    feature_cols = COMPOSITION_COLUMNS + DERIVED_COLUMNS
    combined[feature_cols] = combined[feature_cols].fillna(0.0)

    return UnifiedDataset(
        features=combined[feature_cols].reset_index(drop=True),
        target=combined[TARGET_COLUMN].reset_index(drop=True),
        age_days=combined[AGE_COLUMN].reset_index(drop=True),
        source=combined["source"].reset_index(drop=True),
    )


def stratified_split(
    dataset: UnifiedDataset,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_bins: int = 5,
) -> dict[str, np.ndarray]:
    """Стратифицированное разбиение по уровням прочности.

    Возвращает dict с ключами 'train', 'val', 'test' → массивы индексов.
    Стратификация выполняется по квантилям прочности для сохранения
    баланса распределений.
    """
    rng = np.random.default_rng(seed)
    n = dataset.n_samples

    # Бинируем прочность по квантилям
    strengths = dataset.target.to_numpy()
    bins = np.quantile(strengths, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6  # чтобы max попал в последний бин
    bin_assignments = np.digitize(strengths, bins) - 1

    train_idx, val_idx, test_idx = [], [], []
    for b in range(n_bins):
        indices = np.where(bin_assignments == b)[0]
        rng.shuffle(indices)
        n_bin = len(indices)
        n_test = max(1, int(n_bin * test_ratio))
        n_val = max(1, int(n_bin * val_ratio))
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return {
        "train": np.array(train_idx, dtype=int),
        "val": np.array(val_idx, dtype=int),
        "test": np.array(test_idx, dtype=int),
    }


def grouped_stratified_split(
    dataset: UnifiedDataset,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_bins: int = 5,
) -> dict[str, np.ndarray]:
    """Стратифицированное разбиение по группам составов (без утечки данных).

    Все строки с одинаковым составом (7 композиционных признаков)
    гарантированно попадают в один и тот же split.  Это предотвращает
    ситуацию, когда модель видит состав при t=3 в train и тестируется
    на том же составе при t=28 — что является group leakage.

    Стратификация — по медианной прочности группы (по квантилям).
    """
    rng = np.random.default_rng(seed)

    features = dataset.features[dataset.composition_columns]
    strengths = dataset.target.to_numpy()

    # Создаём ключ группы из 7 композиционных столбцов (округление до 0.01)
    comp_keys = features.apply(
        lambda r: tuple(round(float(x), 2) for x in r), axis=1
    )
    # Назначаем group_id каждой уникальной композиции
    unique_keys = list(comp_keys.unique())
    key_to_gid = {k: i for i, k in enumerate(unique_keys)}
    group_ids = comp_keys.map(key_to_gid).to_numpy()

    n_groups = len(unique_keys)

    # Медианная прочность каждой группы — для стратификации
    group_median_strength = np.zeros(n_groups)
    group_row_indices: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for idx in range(len(strengths)):
        gid = group_ids[idx]
        group_row_indices[gid].append(idx)
    for gid in range(n_groups):
        group_median_strength[gid] = np.median(strengths[group_row_indices[gid]])

    # Бинируем ГРУППЫ по квантилям медианной прочности
    bins = np.quantile(group_median_strength, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    group_bin = np.digitize(group_median_strength, bins) - 1

    train_idx, val_idx, test_idx = [], [], []
    for b in range(n_bins):
        groups_in_bin = np.where(group_bin == b)[0]
        rng.shuffle(groups_in_bin)
        n_grp = len(groups_in_bin)
        n_test = max(1, int(n_grp * test_ratio))
        n_val = max(1, int(n_grp * val_ratio))

        for gid in groups_in_bin[:n_test]:
            test_idx.extend(group_row_indices[gid])
        for gid in groups_in_bin[n_test:n_test + n_val]:
            val_idx.extend(group_row_indices[gid])
        for gid in groups_in_bin[n_test + n_val:]:
            train_idx.extend(group_row_indices[gid])

    return {
        "train": np.array(train_idx, dtype=int),
        "val": np.array(val_idx, dtype=int),
        "test": np.array(test_idx, dtype=int),
    }
