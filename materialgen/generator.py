"""Генератор: нейросеть прямого прогноза (состав + время → прочность).

Реализует архитектуру FC + Residual + Heteroscedastic output:
- Вход: состав смеси (7 компонентов) + производные (w/c, w/b, log_age) = 10 dim
- Выход: (μ, σ) — среднее предсказание прочности и aleatoric uncertainty
- Ограничения: SoftPlus на выходе (strength ≥ 0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Конфигурация генератора
# =============================================================================

@dataclass
class GeneratorConfig:
    """Гиперпараметры архитектуры и обучения генератора."""

    input_dim: int = 10           # 7 компонентов + w_c_ratio + w_b_ratio + log_age
    hidden_dims: list[int] | None = None  # по умолчанию [128, 64, 32]
    dropout: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = True
    use_spectral_norm: bool = False  # для стабилизации GAN

    # Обучение (standalone, до GAN)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]

    @classmethod
    def from_dict(cls, payload: dict) -> GeneratorConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "use_residual": self.use_residual,
            "use_spectral_norm": self.use_spectral_norm,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


# =============================================================================
# Строительные блоки
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual FC block: out = F(x) + x.

    Если входная и выходная размерности совпадают, skip connection
    работает напрямую. Иначе используется линейная проекция.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        self.block = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class FCBlock(nn.Module):
    """Обычный FC блок (Linear + BN + LeakyReLU + Dropout)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        linear = nn.Linear(in_dim, out_dim)
        if use_spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# Генератор
# =============================================================================

class ConcreteGenerator(nn.Module):
    """Генератор: состав бетона + время → прогноз прочности (μ, σ).

    Архитектура:
        Input(10) → FC(128) → ResBlock(128) → FC(64) → ResBlock(64)
        → FC(32) → OutputHead(2) → SoftPlus → (μ, σ)

    Два выхода:
        μ — среднее предсказание прочности (МПа)
        σ — aleatoric uncertainty (оценка шума данных)
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GeneratorConfig()
        self.config = config

        hidden_dims = config.hidden_dims
        assert hidden_dims and len(hidden_dims) >= 1

        # Входной слой
        modules: list[nn.Module] = [
            FCBlock(
                config.input_dim,
                hidden_dims[0],
                dropout=config.dropout,
                use_batch_norm=config.use_batch_norm,
                use_spectral_norm=config.use_spectral_norm,
            )
        ]

        # Скрытые слои с опциональными Residual blocks
        for i in range(len(hidden_dims)):
            if config.use_residual:
                modules.append(ResidualBlock(
                    hidden_dims[i],
                    dropout=config.dropout,
                    use_batch_norm=config.use_batch_norm,
                ))
            if i + 1 < len(hidden_dims):
                modules.append(FCBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout=config.dropout,
                    use_batch_norm=config.use_batch_norm,
                    use_spectral_norm=config.use_spectral_norm,
                ))

        self.backbone = nn.Sequential(*modules)

        # Output head: 2 выхода (mu, log_sigma)
        self.output_head = nn.Linear(hidden_dims[-1], 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход.

        Parameters
        ----------
        x : тензор входных признаков [batch, input_dim]
            Содержит: composition (7) + w_c_ratio + w_b_ratio + log_age

        Returns
        -------
        (mu, sigma) — средняя прочность и aleatoric uncertainty
            mu    : [batch, 1] — прочность (МПа), ≥ 0
            sigma : [batch, 1] — стандартное отклонение, > 0
        """
        h = self.backbone(x)
        raw = self.output_head(h)

        mu_raw, log_sigma = raw[:, 0:1], raw[:, 1:2]

        # SoftPlus гарантирует strength ≥ 0
        mu = F.softplus(mu_raw)
        # sigma > 0
        sigma = F.softplus(log_sigma) + 1e-6

        return mu, sigma

    def predict(
        self,
        x: np.ndarray,
        *,
        mc_samples: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Предсказание с опциональным MC-dropout.

        Parameters
        ----------
        x : входные признаки [n_samples, input_dim]
        mc_samples : количество MC-прогонов (>1 для epistemic uncertainty)

        Returns
        -------
        (mean_mu, total_std) — средний прогноз и полная неопределённость
        """
        device = next(self.parameters()).device
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)

        if mc_samples <= 1:
            self.eval()
            with torch.no_grad():
                mu, sigma = self.forward(x_tensor)
            return mu.cpu().numpy(), sigma.cpu().numpy()

        # MC-dropout: оставляем dropout включённым
        self.train()
        mus, sigmas = [], []
        with torch.no_grad():
            for _ in range(mc_samples):
                mu, sigma = self.forward(x_tensor)
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())

        mus = np.stack(mus, axis=0)       # [mc, batch, 1]
        sigmas = np.stack(sigmas, axis=0)

        mean_mu = mus.mean(axis=0)
        epistemic_std = mus.std(axis=0)
        aleatoric_std = sigmas.mean(axis=0)
        total_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

        return mean_mu, total_std


# =============================================================================
# Standalone supervised обучение (baseline)
# =============================================================================

def train_generator_supervised(
    generator: ConcreteGenerator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    config: GeneratorConfig | None = None,
) -> dict[str, Any]:
    """Обучить генератор как обычный регрессор (supervised baseline).

    Использует NLL-loss с heteroscedastic output (Gaussian likelihood).

    Returns
    -------
    dict с историей обучения: train_losses, val_losses, epochs_run
    """
    if config is None:
        config = generator.config

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)

    x_v, y_v = None, None
    if x_val is not None and y_val is not None:
        x_v = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        y_v = torch.as_tensor(y_val, dtype=torch.float32, device=device).reshape(-1, 1)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 30

    for epoch in range(config.epochs):
        generator.train()

        # Mini-batch training
        perm = torch.randperm(x_t.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, x_t.shape[0], config.batch_size):
            idx = perm[start:start + config.batch_size]
            if len(idx) < 2:
                continue  # BatchNorm requires batch > 1
            xb, yb = x_t[idx], y_t[idx]

            mu, sigma = generator(xb)

            # NLL loss (Gaussian)
            nll = 0.5 * (torch.log(sigma ** 2) + (yb - mu) ** 2 / (sigma ** 2))
            loss = nll.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        if x_v is not None:
            generator.eval()
            with torch.no_grad():
                mu_v, sigma_v = generator(x_v)
                val_nll = 0.5 * (torch.log(sigma_v ** 2) + (y_v - mu_v) ** 2 / (sigma_v ** 2))
                val_loss = val_nll.mean().item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in generator.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        generator.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_run": len(train_losses),
        "best_val_loss": best_val_loss if x_v is not None else None,
    }
