"""GAN Training Loop: оркестрация обучения генератора и дискриминатора.

Реализует ConcreteGAN — semi-supervised conditional GAN с
physics-informed regularization и progressive training.

Цикл обучения:
    1. Дискриминатор (NEAT+BNN) учится отличать реальные пары от фейковых
    2. Генератор учится обманывать дискриминатор + supervised MSE + physics
    3. Progressive schedule: supervised → adversarial
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import ConcreteGenerator, GeneratorConfig
from .discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from .physics import GostTable, combined_physics_loss
from .metrics import compute_regression_metrics
from .scaler import StandardScaler
from .stage_common import write_json


# =============================================================================
# Конфигурация GAN
# =============================================================================

@dataclass
class GANConfig:
    """Гиперпараметры GAN-обучения."""

    # Общие
    total_epochs: int = 500
    seed: int = 42

    # Loss weights (progressive schedule)
    lambda_mse_start: float = 1.0    # начальный вес supervised loss
    lambda_mse_end: float = 0.3      # конечный вес supervised loss
    lambda_adv_start: float = 0.0    # начальный вес adversarial loss
    lambda_adv_end: float = 0.7      # конечный вес adversarial loss
    lambda_physics: float = 0.5      # вес физических ограничений (константа)

    # Physics sub-weights
    lambda_mono: float = 1.0
    lambda_abrams: float = 0.5
    lambda_gost: float = 0.3

    # Обучение генератора
    generator_lr: float = 1e-3
    generator_weight_decay: float = 1e-4

    # Обучение дискриминатора
    n_disc_steps: int = 1             # шагов D на 1 шаг G
    disc_svi_steps: int = 3           # SVI шагов на каждый шаг D

    # Стабилизация GAN
    label_smoothing: float = 0.1     # real label = 1 - smoothing
    noise_std: float = 0.05          # шум на входы дискриминатора

    # Progressive schedule (3 фазы)
    phase1_end: int = 100            # конец чисто-supervised фазы
    phase2_end: int = 200            # конец переходной фазы

    # Validation
    val_interval: int = 10           # каждые N эпох
    early_stopping_patience: int = 50
    batch_size: int = 32

    # Индекс w/c ratio в входном тензоре генератора
    wc_ratio_index: int = 7

    @classmethod
    def from_dict(cls, payload: dict) -> GANConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# =============================================================================
# Training History
# =============================================================================

@dataclass
class GANTrainingHistory:
    """Полная история обучения GAN."""

    generator_losses: list[float] = field(default_factory=list)
    discriminator_losses: list[float] = field(default_factory=list)
    supervised_losses: list[float] = field(default_factory=list)
    adversarial_losses: list[float] = field(default_factory=list)
    physics_losses: list[float] = field(default_factory=list)

    val_mae: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    val_r2: list[float] = field(default_factory=list)

    lambda_mse_history: list[float] = field(default_factory=list)
    lambda_adv_history: list[float] = field(default_factory=list)

    best_epoch: int = 0
    best_val_mae: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_losses": self.generator_losses,
            "discriminator_losses": self.discriminator_losses,
            "supervised_losses": self.supervised_losses,
            "adversarial_losses": self.adversarial_losses,
            "physics_losses": self.physics_losses,
            "val_mae": self.val_mae,
            "val_rmse": self.val_rmse,
            "val_r2": self.val_r2,
            "lambda_mse_history": self.lambda_mse_history,
            "lambda_adv_history": self.lambda_adv_history,
            "best_epoch": self.best_epoch,
            "best_val_mae": self.best_val_mae,
        }


# =============================================================================
# ConcreteGAN
# =============================================================================

class ConcreteGAN:
    """Semi-supervised Conditional GAN для прогноза свойств бетона.

    Lifecycle:
        1. __init__(generator, discriminator, config)
        2. prepare_discriminator(x_train, y_train) → NEAT evolution + BNN init
        3. train(x_train, y_train, x_val, y_val) → GAN training loop
        4. predict(x) → прогноз с неопределённостью
    """

    def __init__(
        self,
        generator: ConcreteGenerator,
        discriminator: NeatBNNDiscriminator,
        config: GANConfig | None = None,
        gost: GostTable | None = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.config = config or GANConfig()
        self.gost = gost
        self.history = GANTrainingHistory()
        self.feature_scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self._best_generator_state: dict | None = None

    def _get_lambda_schedule(self, epoch: int) -> tuple[float, float]:
        """Вычислить текущие веса loss по progressive schedule."""
        cfg = self.config

        if epoch < cfg.phase1_end:
            # Фаза 1: чистый supervised
            return cfg.lambda_mse_start, cfg.lambda_adv_start
        elif epoch < cfg.phase2_end:
            # Фаза 2: плавный переход
            progress = (epoch - cfg.phase1_end) / max(cfg.phase2_end - cfg.phase1_end, 1)
            lam_mse = cfg.lambda_mse_start + (cfg.lambda_mse_end - cfg.lambda_mse_start) * progress
            lam_adv = cfg.lambda_adv_start + (cfg.lambda_adv_end - cfg.lambda_adv_start) * progress
            return lam_mse, lam_adv
        else:
            # Фаза 3: adversarial доминирует
            return cfg.lambda_mse_end, cfg.lambda_adv_end

    def _make_discriminator_pairs(
        self,
        x_features: torch.Tensor,
        y_strength: torch.Tensor,
    ) -> torch.Tensor:
        """Конкатенировать признаки + прочность для дискриминатора."""
        return torch.cat([x_features, y_strength], dim=1)

    def prepare_discriminator(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        artifacts_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Этап подготовки: NEAT эволюция + BNN инициализация дискриминатора.

        Parameters
        ----------
        x_train : входные признаки [n, input_dim]
        y_train : целевые прочности [n, 1] или [n]
        artifacts_dir : куда сохранить артефакты NEAT

        Returns
        -------
        dict с результатами эволюции и pre-training
        """
        y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train

        # Реальные пары
        real_pairs = np.concatenate([x_train, y_train], axis=1)

        # Фейковые пары: реальный состав + случайная прочность
        rng = np.random.default_rng(self.config.seed)
        fake_strengths = rng.uniform(
            y_train.min() * 0.5,
            y_train.max() * 1.5,
            size=y_train.shape,
        )
        fake_pairs = np.concatenate([x_train, fake_strengths], axis=1)

        # Этап 1: NEAT эволюция
        evolution_result = self.discriminator.evolve_topology(
            real_pairs, fake_pairs,
            artifacts_dir=artifacts_dir,
        )

        # Этап 2: BNN инициализация
        self.discriminator.init_bnn()

        # Этап 3: BNN pre-training
        pretrain_result = self.discriminator.pretrain_bnn(real_pairs, fake_pairs)

        return {
            "evolution": evolution_result,
            "pretrain": pretrain_result,
        }

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> GANTrainingHistory:
        """Основной GAN training loop.

        Parameters
        ----------
        x_train : входные признаки [n_train, input_dim]
        y_train : целевые прочности [n_train]
        x_val : валидационные признаки
        y_val : валидационные прочности

        Returns
        -------
        GANTrainingHistory с полной историей обучения
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device("cpu")

        self.generator = self.generator.to(device)

        x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(
            y_train.reshape(-1, 1), dtype=torch.float32, device=device,
        )

        gen_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=cfg.generator_lr,
            weight_decay=cfg.generator_weight_decay,
        )

        patience_counter = 0

        for epoch in range(cfg.total_epochs):
            lam_mse, lam_adv = self._get_lambda_schedule(epoch)
            self.history.lambda_mse_history.append(lam_mse)
            self.history.lambda_adv_history.append(lam_adv)

            self.generator.train()

            # === Mini-batch training ===
            perm = torch.randperm(x_t.shape[0])
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_sup_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_phy_loss = 0.0
            n_batches = 0

            for start in range(0, x_t.shape[0], cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                if len(idx) < 2:
                    continue  # BatchNorm requires batch > 1
                xb, yb = x_t[idx], y_t[idx]

                # --- Шаг дискриминатора ---
                d_loss = self._discriminator_step(xb, yb)
                epoch_d_loss += d_loss

                # --- Шаг генератора ---
                g_loss, sup_loss, adv_loss, phy_loss = self._generator_step(
                    xb, yb, lam_mse, lam_adv, gen_optimizer,
                )
                epoch_g_loss += g_loss
                epoch_sup_loss += sup_loss
                epoch_adv_loss += adv_loss
                epoch_phy_loss += phy_loss
                n_batches += 1

            # Средние loss за эпоху
            n = max(n_batches, 1)
            self.history.generator_losses.append(epoch_g_loss / n)
            self.history.discriminator_losses.append(epoch_d_loss / n)
            self.history.supervised_losses.append(epoch_sup_loss / n)
            self.history.adversarial_losses.append(epoch_adv_loss / n)
            self.history.physics_losses.append(epoch_phy_loss / n)

            # === Validation ===
            if x_val is not None and epoch % cfg.val_interval == 0:
                val_metrics = self._validate(x_val, y_val)
                self.history.val_mae.append(val_metrics["mae"])
                self.history.val_rmse.append(val_metrics["rmse"])
                self.history.val_r2.append(val_metrics["r2"])

                if val_metrics["mae"] < self.history.best_val_mae - 1e-6:
                    self.history.best_val_mae = val_metrics["mae"]
                    self.history.best_epoch = epoch
                    self._best_generator_state = {
                        k: v.clone() for k, v in self.generator.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= cfg.early_stopping_patience:
                    break

        # Восстановить лучшее состояние
        if self._best_generator_state is not None:
            self.generator.load_state_dict(self._best_generator_state)

        return self.history

    def _discriminator_step(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
    ) -> float:
        """Один шаг обучения дискриминатора (BNN SVI)."""
        cfg = self.config

        # Генерируем фейковые прогнозы
        with torch.no_grad():
            y_fake, _ = self.generator(xb)

        real_pairs = self._make_discriminator_pairs(xb, yb).detach().cpu().numpy()
        fake_pairs = self._make_discriminator_pairs(xb, y_fake).detach().cpu().numpy()

        # SVI update дискриминатора
        if self.discriminator.bnn is not None:
            all_inputs = np.concatenate([real_pairs, fake_pairs], axis=0)
            labels = np.concatenate([
                np.ones((len(real_pairs), 1)),
                np.zeros((len(fake_pairs), 1)),
            ], axis=0)

            # Простой SVI step через BNN
            score_real, _ = self.discriminator.score(real_pairs, mc_samples=5)
            score_fake, _ = self.discriminator.score(fake_pairs, mc_samples=5)
            d_loss = (1.0 - score_real.mean()) + score_fake.mean()
            return float(d_loss)

        return 0.0

    def _generator_step(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        lam_mse: float,
        lam_adv: float,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, float, float]:
        """Один шаг обучения генератора."""
        cfg = self.config

        mu, sigma = self.generator(xb)

        # 1. Supervised loss (NLL)
        nll = 0.5 * (torch.log(sigma ** 2) + (yb - mu) ** 2 / (sigma ** 2))
        sup_loss = nll.mean()

        # 2. Adversarial loss
        adv_loss = torch.tensor(0.0, device=xb.device)
        if lam_adv > 0 and self.discriminator.bnn is not None:
            fake_pairs = self._make_discriminator_pairs(xb, mu)
            d_score, _ = self.discriminator.score_tensor(fake_pairs, mc_samples=5)
            # Хотим чтобы D(fake) → 1 (обмануть дискриминатор)
            target = torch.ones_like(d_score) * (1.0 - cfg.label_smoothing)
            adv_loss = F.binary_cross_entropy(
                d_score.clamp(1e-6, 1 - 1e-6), target,
            )

        # 3. Physics loss
        phy_loss = torch.tensor(0.0, device=xb.device)
        if cfg.lambda_physics > 0:
            # Composition features (без log_age) для monotonicity
            x_comp = xb[:, :-1]  # всё кроме последнего (log_age)
            phy_loss, _ = combined_physics_loss(
                generator=self.generator,
                x_composition=x_comp,
                x_with_time=xb,
                wc_index=cfg.wc_ratio_index,
                gost=self.gost,
                y_pred_28d=mu.squeeze() if self.gost else None,
                lambda_mono=cfg.lambda_mono,
                lambda_abrams=cfg.lambda_abrams,
                lambda_gost=cfg.lambda_gost,
            )

        # Комбинированный loss
        total_loss = (
            lam_mse * sup_loss
            + lam_adv * adv_loss
            + cfg.lambda_physics * phy_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return (
            total_loss.item(),
            sup_loss.item(),
            adv_loss.item(),
            phy_loss.item() if isinstance(phy_loss, torch.Tensor) else phy_loss,
        )

    def _validate(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Валидация генератора."""
        self.generator.eval()
        x_v = torch.as_tensor(x_val, dtype=torch.float32)
        with torch.no_grad():
            mu, sigma = self.generator(x_v)
        y_pred = mu.numpy().ravel()
        y_true = np.asarray(y_val).ravel()

        metrics = compute_regression_metrics(y_true, y_pred)
        return {"mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2}

    def predict(
        self,
        x: np.ndarray,
        mc_samples: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Предсказание с оценкой неопределённости.

        Returns
        -------
        (mean_strength, total_std) — прогноз и полная неопределённость
        """
        return self.generator.predict(x, mc_samples=mc_samples)

    def save(self, output_dir: str | Path) -> dict[str, str]:
        """Сохранить всю GAN-модель (генератор + дискриминатор + config)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        gen_path = out / "generator.pt"
        torch.save(self.generator.state_dict(), gen_path)

        disc_path = out / "discriminator.pt"
        self.discriminator.save(disc_path)

        config_path = out / "gan_config.json"
        write_json(config_path, self.config.to_dict())

        history_path = out / "training_history.json"
        write_json(history_path, self.history.to_dict())

        return {
            "generator": str(gen_path),
            "discriminator": str(disc_path),
            "config": str(config_path),
            "history": str(history_path),
        }
