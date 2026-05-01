from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PreparedDataset:
    """Numeric training matrices together with dataset-derived statistics."""

    components: np.ndarray
    properties: np.ndarray
    component_bounds: dict[str, list[float]]
    property_ranges: dict[str, dict[str, float]]


def read_dataset_frame(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame with pandas' delimiter inference."""

    path = Path(csv_path)
    try:
        frame = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True, decimal=",")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset {csv_path} is empty") from exc

    if frame.empty:
        raise ValueError(f"Dataset {csv_path} is empty")
    return frame


def _select_numeric_columns(frame: pd.DataFrame, columns: list[str], csv_path: str | Path) -> pd.DataFrame:
    """Select columns from the DataFrame and coerce them to numeric values."""

    missing_columns = [name for name in columns if name not in frame.columns]
    if missing_columns:
        names = ", ".join(repr(name) for name in missing_columns)
        raise KeyError(f"Columns {names} not found in {csv_path}")

    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric.isna()
    if invalid_mask.any().any():
        row_position, column_position = next(zip(*np.where(invalid_mask.to_numpy())))
        row_number = row_position + 2
        column_name = columns[column_position]
        raise ValueError(f"Invalid numeric value at row {row_number}, column {column_name!r}")
    return numeric


def prepare_dataset(
    csv_path: str | Path,
    component_columns: list[str],
    property_columns: list[str],
    component_aliases: dict[str, str] | None = None,
    min_time: float | None = None,
) -> PreparedDataset:
    """Load the training CSV and compute matrices plus dataset-driven ranges.

    When *component_aliases* is provided, CSV columns are renamed before
    extraction so the returned dataset uses canonical names.  The mapping
    is ``{csv_column_name: canonical_name}``.
    """

    frame = read_dataset_frame(csv_path)
    if component_aliases:
        frame = frame.rename(columns=component_aliases)
    if min_time is not None:
        time_col = next((col for col in frame.columns if col.lower() == "time"), None)
        if time_col:
            frame = frame[frame[time_col] >= min_time]
    component_frame = _select_numeric_columns(frame, component_columns, csv_path)
    property_frame = _select_numeric_columns(frame, property_columns, csv_path)

    component_stats = component_frame.agg(["min", "max"]).transpose()
    property_stats = property_frame.agg(["min", "max", "mean"]).transpose()

    return PreparedDataset(
        components=component_frame.to_numpy(dtype=float),
        properties=property_frame.to_numpy(dtype=float),
        component_bounds={
            name: [
                float(component_stats.loc[name, "min"]),
                float(component_stats.loc[name, "max"]),
            ]
            for name in component_columns
        },
        property_ranges={
            name: {
                "min": float(property_stats.loc[name, "min"]),
                "max": float(property_stats.loc[name, "max"]),
                "mean": float(property_stats.loc[name, "mean"]),
            }
            for name in property_columns
        },
    )


def load_dataset(
    csv_path: str | Path,
    component_columns: list[str],
    property_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the training CSV and split it into component and property matrices."""

    dataset = prepare_dataset(csv_path, component_columns, property_columns)
    return dataset.components, dataset.properties


def load_targets(payload: str | Path) -> dict[str, float]:
    """Load target properties either from a JSON file or from a JSON string."""

    path = Path(payload)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raw = json.loads(str(payload))
    return {str(key): float(value) for key, value in raw.items()}
