from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Store per-column mean and standard deviation for simple normalization.

    A small in-project scaler keeps the code lightweight and avoids an extra
    dependency just for standardization.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        """Estimate scaling parameters from a 2D array."""

        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Convert raw values into standardized z-scores."""

        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Return standardized values back to the original units."""

        return values * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize scaler state so trained models can be saved and reloaded."""

        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "StandardScaler":
        """Rebuild a scaler from saved metadata."""

        return cls(
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
        )
