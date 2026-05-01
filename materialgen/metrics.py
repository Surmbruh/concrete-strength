"""Метрики оценки качества прогноза и калибровки неопределённости.

Предоставляет как поточечные метрики (MAE, RMSE, MAPE, R²), так и
метрики калибровки доверительных интервалов (PICP, MPIW, sharpness).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np


# =============================================================================
# Dataclass для результатов
# =============================================================================

@dataclass
class RegressionMetrics:
    """Набор стандартных метрик регрессии."""

    mae: float
    rmse: float
    mape: float
    r2: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "R2": self.r2,
            "n_samples": self.n_samples,
        }


@dataclass
class CalibrationMetrics:
    """Метрики калибровки доверительных интервалов."""

    picp: float          # Prediction Interval Coverage Probability
    mpiw: float          # Mean Prediction Interval Width
    sharpness: float     # = MPIW / (y_max - y_min)
    confidence_level: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "PICP": self.picp,
            "MPIW": self.mpiw,
            "sharpness": self.sharpness,
            "confidence_level": self.confidence_level,
        }


@dataclass
class FullEvaluation:
    """Совокупный результат оценки модели."""

    regression: RegressionMetrics
    calibration: CalibrationMetrics | None = None
    per_time: dict[int, RegressionMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"regression": self.regression.to_dict()}
        if self.calibration is not None:
            result["calibration"] = self.calibration.to_dict()
        if self.per_time:
            result["per_time"] = {
                str(t): m.to_dict() for t, m in self.per_time.items()
            }
        return result


# =============================================================================
# Поточечные метрики
# =============================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка (МПа)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Корень из среднеквадратичной ошибки."""
    return float(sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Средняя абсолютная процентная ошибка (%).

    Значения ``y_true``, близкие к нулю, защищены ``epsilon`` от деления
    на ноль.
    """
    safe_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации R²."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Вычислить все метрики регрессии разом."""

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return RegressionMetrics(
        mae=mean_absolute_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        mape=mean_absolute_percentage_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        n_samples=len(y_true),
    )


# =============================================================================
# Калибровка неопределённости
# =============================================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    confidence_level: float = 0.95,
) -> CalibrationMetrics:
    """Оценить калибровку доверительных интервалов.

    Parameters
    ----------
    y_true : массив реальных значений
    y_mean : массив средних предсказаний
    y_std  : массив стандартных отклонений предсказаний
    confidence_level : уровень доверия (по умолчанию 0.95)
    """
    from scipy import stats  # lazy import — может не понадобиться

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_mean = np.asarray(y_mean, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()

    z = stats.norm.ppf((1.0 + confidence_level) / 2.0)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std

    covered = np.logical_and(y_true >= lower, y_true <= upper)
    picp = float(np.mean(covered))

    widths = upper - lower
    mpiw = float(np.mean(widths))

    y_range = float(np.max(y_true) - np.min(y_true))
    sharpness = mpiw / y_range if y_range > 1e-8 else float("nan")

    return CalibrationMetrics(
        picp=picp,
        mpiw=mpiw,
        sharpness=sharpness,
        confidence_level=confidence_level,
    )


# =============================================================================
# Полная оценка
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
    age_days: np.ndarray | None = None,
    confidence_level: float = 0.95,
) -> FullEvaluation:
    """Комплексная оценка модели прогноза прочности.

    Parameters
    ----------
    y_true : реальные значения прочности (МПа)
    y_pred : предсказанные средние значения прочности
    y_std  : если есть — стандартное отклонение предсказаний
    age_days : если есть — возраст для per-time breakdown
    confidence_level : уровень доверия для калибровки CI
    """

    regression = compute_regression_metrics(y_true, y_pred)

    calibration = None
    if y_std is not None:
        calibration = compute_calibration_metrics(
            y_true, y_pred, y_std, confidence_level,
        )

    per_time: dict[int, RegressionMetrics] = {}
    if age_days is not None:
        age_arr = np.asarray(age_days, dtype=int).ravel()
        for t in sorted(set(age_arr)):
            mask = age_arr == t
            if mask.sum() > 0:
                per_time[int(t)] = compute_regression_metrics(
                    np.asarray(y_true).ravel()[mask],
                    np.asarray(y_pred).ravel()[mask],
                )

    return FullEvaluation(
        regression=regression,
        calibration=calibration,
        per_time=per_time,
    )
