"""Оценка неопределённости: объединение aleatoric + epistemic.

Агрегирует неопределённость из трёх источников:
1. Aleatoric (шум данных) — из σ-головы генератора
2. Epistemic (незнание модели) — из MC-dropout генератора + BNN posterior
3. Discriminator confidence — дополнительный сигнал от дискриминатора
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .generator import ConcreteGenerator
from .discriminator import NeatBNNDiscriminator


# =============================================================================
# Результат оценки
# =============================================================================

@dataclass
class UncertaintyResult:
    """Результат предсказания с оценкой неопределённости."""

    strength_mean: np.ndarray       # [n] — средний прогноз (МПа)
    strength_std: np.ndarray        # [n] — полная неопределённость
    aleatoric_std: np.ndarray       # [n] — aleatoric component
    epistemic_std: np.ndarray       # [n] — epistemic component
    ci_lower: np.ndarray            # [n] — нижняя граница CI
    ci_upper: np.ndarray            # [n] — верхняя граница CI
    confidence_level: float         # уровень доверия (0.95)
    discriminator_score: np.ndarray | None = None  # [n] — скор D
    discriminator_std: np.ndarray | None = None     # [n] — uncertainty D

    def to_dict(self) -> dict[str, Any]:
        result = {
            "strength_mean": self.strength_mean.tolist(),
            "strength_std": self.strength_std.tolist(),
            "aleatoric_std": self.aleatoric_std.tolist(),
            "epistemic_std": self.epistemic_std.tolist(),
            "ci_lower": self.ci_lower.tolist(),
            "ci_upper": self.ci_upper.tolist(),
            "confidence_level": self.confidence_level,
        }
        if self.discriminator_score is not None:
            result["discriminator_score"] = self.discriminator_score.tolist()
        if self.discriminator_std is not None:
            result["discriminator_std"] = self.discriminator_std.tolist()
        return result


# =============================================================================
# Estimator
# =============================================================================

class UncertaintyEstimator:
    """Агрегатор неопределённости из генератора и дискриминатора.

    Usage:
        estimator = UncertaintyEstimator(generator, discriminator)
        result = estimator.predict(x_features, confidence_level=0.95)
    """

    def __init__(
        self,
        generator: ConcreteGenerator,
        discriminator: NeatBNNDiscriminator | None = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def predict(
        self,
        x: np.ndarray,
        *,
        mc_samples: int = 50,
        confidence_level: float = 0.95,
    ) -> UncertaintyResult:
        """Предсказание с полной оценкой неопределённости.

        Parameters
        ----------
        x : входные признаки [n, input_dim]
        mc_samples : количество MC-прогонов для epistemic uncertainty
        confidence_level : уровень доверительного интервала

        Returns
        -------
        UncertaintyResult с разложением неопределённости
        """
        from scipy import stats

        x_tensor = torch.as_tensor(x, dtype=torch.float32)

        # === MC-Dropout через генератор ===
        self.generator.train()  # включаем dropout
        mus_list: list[np.ndarray] = []
        sigmas_list: list[np.ndarray] = []

        with torch.no_grad():
            for _ in range(mc_samples):
                mu, sigma = self.generator(x_tensor)
                mus_list.append(mu.numpy().ravel())
                sigmas_list.append(sigma.numpy().ravel())

        self.generator.eval()

        mus = np.stack(mus_list, axis=0)       # [mc, n]
        sigmas = np.stack(sigmas_list, axis=0)  # [mc, n]

        # Epistemic = std от MC-dropout прогонов
        epistemic_std = mus.std(axis=0)

        # Aleatoric = среднее предсказанной σ
        aleatoric_std = sigmas.mean(axis=0)

        # Total
        total_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

        # Mean prediction
        strength_mean = mus.mean(axis=0)

        # Confidence interval
        z = stats.norm.ppf((1.0 + confidence_level) / 2.0)
        ci_lower = strength_mean - z * total_std
        ci_upper = strength_mean + z * total_std

        # Прочность не может быть отрицательной
        ci_lower = np.maximum(ci_lower, 0.0)

        # === Discriminator score (если доступен) ===
        disc_score = None
        disc_std = None
        if self.discriminator is not None and self.discriminator.bnn is not None:
            pairs = np.concatenate(
                [x, strength_mean.reshape(-1, 1)], axis=1,
            )
            disc_score, disc_std = self.discriminator.score(pairs)
            disc_score = disc_score.ravel()
            disc_std = disc_std.ravel()

        return UncertaintyResult(
            strength_mean=strength_mean,
            strength_std=total_std,
            aleatoric_std=aleatoric_std,
            epistemic_std=epistemic_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            discriminator_score=disc_score,
            discriminator_std=disc_std,
        )
