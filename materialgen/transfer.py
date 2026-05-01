"""Transfer Learning: pre-train на больших данных → fine-tune на малых.

Двухступенчатая стратегия:
1. Pre-train генератор + дискриминатор на Normal_Concrete_DB + boxcrete
2. Fine-tune на малой лабораторной выборке (synthetic_training_data)
   с заморозкой ранних слоёв и informative BNN prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from .discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from .gan_trainer import ConcreteGAN, GANConfig, GANTrainingHistory
from .data_preparation import (
    UnifiedDataset,
    load_and_unify_datasets,
    stratified_split,
)
from .physics import GostTable, load_gost_table
from .metrics import evaluate_model, FullEvaluation
from .stage_common import write_json


# =============================================================================
# Конфигурация
# =============================================================================

@dataclass
class TransferConfig:
    """Параметры transfer learning pipeline."""

    # Pre-training
    pretrain_epochs: int = 200
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 64

    # Fine-tuning
    finetune_epochs: int = 300
    finetune_lr: float = 1e-4        # в 10× меньше
    finetune_batch_size: int = 16
    freeze_layers: int = 2            # заморозить первые N слоёв генератора
    lambda_physics_boost: float = 1.0 # увеличенный вес физики при fine-tuning

    # Данные
    data_dir: str = "data"
    pretrain_sources: list[str] | None = None  # ['normal_concrete_db', 'boxcrete']
    finetune_source: str = "synthetic"

    seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict) -> TransferConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# =============================================================================
# TransferLearner
# =============================================================================

class TransferLearner:
    """Оркестратор transfer learning для прогноза прочности бетона.

    Usage:
        learner = TransferLearner(config)
        learner.pretrain()
        learner.finetune()
        result = learner.evaluate()
    """

    def __init__(
        self,
        config: TransferConfig | None = None,
        generator_config: GeneratorConfig | None = None,
        discriminator_config: DiscriminatorConfig | None = None,
        gan_config: GANConfig | None = None,
    ) -> None:
        self.config = config or TransferConfig()
        self.gen_config = generator_config or GeneratorConfig()
        self.disc_config = discriminator_config or DiscriminatorConfig()
        self.gan_config = gan_config or GANConfig()

        self.generator: ConcreteGenerator | None = None
        self.discriminator: NeatBNNDiscriminator | None = None
        self.gan: ConcreteGAN | None = None
        self.gost: GostTable | None = None

        self.pretrain_dataset: UnifiedDataset | None = None
        self.finetune_dataset: UnifiedDataset | None = None
        self.pretrain_history: dict[str, Any] | None = None
        self.finetune_history: GANTrainingHistory | None = None

    def load_data(self) -> None:
        """Загрузить и подготовить данные для обоих этапов."""
        # Pre-training данные (большие)
        self.pretrain_dataset = load_and_unify_datasets(
            self.config.data_dir,
            include_normal=True,
            include_boxcrete=True,
            include_synthetic=False,
        )

        # Fine-tuning данные (малые)
        self.finetune_dataset = load_and_unify_datasets(
            self.config.data_dir,
            include_normal=False,
            include_boxcrete=False,
            include_synthetic=True,
        )

        # ГОСТ
        gost_path = Path(self.config.data_dir) / "ГОСТы.csv"
        if gost_path.exists():
            self.gost = load_gost_table(gost_path)

    def pretrain(
        self,
        artifacts_dir: str | Path = "artifacts/pretrain",
    ) -> dict[str, Any]:
        """Этап 1: Pre-training на больших данных.

        Обучает генератор как supervised регрессор и эволюционирует
        топологию дискриминатора.

        Returns
        -------
        dict с метриками и путями к артефактам
        """
        if self.pretrain_dataset is None:
            self.load_data()

        art_dir = Path(artifacts_dir)
        art_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.pretrain_dataset
        split = stratified_split(dataset, seed=self.config.seed)

        x_all = dataset.all_features
        y_all = dataset.target.to_numpy()

        x_train, y_train = x_all[split["train"]], y_all[split["train"]]
        x_val, y_val = x_all[split["val"]], y_all[split["val"]]

        # 1. Supervised pre-training генератора
        self.generator = ConcreteGenerator(self.gen_config)
        self.pretrain_history = train_generator_supervised(
            self.generator, x_train, y_train, x_val, y_val,
            config=GeneratorConfig(
                input_dim=self.gen_config.input_dim,
                learning_rate=self.config.pretrain_lr,
                epochs=self.config.pretrain_epochs,
                batch_size=self.config.pretrain_batch_size,
                seed=self.config.seed,
            ),
        )

        # Сохраняем checkpoint генератора
        gen_path = art_dir / "pretrained_generator.pt"
        torch.save(self.generator.state_dict(), gen_path)

        # 2. Эволюция дискриминатора
        self.discriminator = NeatBNNDiscriminator(self.disc_config)

        summary = {
            "stage": "pretrain",
            "generator_epochs": self.pretrain_history["epochs_run"],
            "generator_best_val_loss": self.pretrain_history.get("best_val_loss"),
            "generator_checkpoint": str(gen_path),
        }
        write_json(art_dir / "pretrain_summary.json", summary)
        return summary

    def finetune(
        self,
        artifacts_dir: str | Path = "artifacts/finetune",
    ) -> GANTrainingHistory:
        """Этап 2: Fine-tuning на малой лабораторной выборке.

        Замораживает ранние слои генератора, снижает lr, увеличивает
        вес физических ограничений. Запускает полный GAN training loop.

        Returns
        -------
        GANTrainingHistory
        """
        if self.generator is None:
            raise RuntimeError("Run pretrain() first")
        if self.finetune_dataset is None:
            self.load_data()

        art_dir = Path(artifacts_dir)

        dataset = self.finetune_dataset
        split = stratified_split(dataset, seed=self.config.seed)

        x_all = dataset.all_features
        y_all = dataset.target.to_numpy()

        x_train, y_train = x_all[split["train"]], y_all[split["train"]]
        x_val, y_val = x_all[split["val"]], y_all[split["val"]]

        # Заморозить ранние слои генератора
        self._freeze_early_layers(self.config.freeze_layers)

        # Модифицированный GAN config для fine-tuning
        ft_gan_config = GANConfig(
            total_epochs=self.config.finetune_epochs,
            generator_lr=self.config.finetune_lr,
            batch_size=self.config.finetune_batch_size,
            lambda_physics=self.config.lambda_physics_boost,
            seed=self.config.seed,
        )

        self.gan = ConcreteGAN(
            generator=self.generator,
            discriminator=self.discriminator,
            config=ft_gan_config,
            gost=self.gost,
        )

        # Подготовка дискриминатора
        self.gan.prepare_discriminator(
            x_train, y_train, artifacts_dir=str(art_dir),
        )

        # GAN training
        self.finetune_history = self.gan.train(
            x_train, y_train, x_val, y_val,
        )

        # Сохраняем результаты
        self.gan.save(str(art_dir))

        return self.finetune_history

    def evaluate(
        self,
        x_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        age_test: np.ndarray | None = None,
    ) -> FullEvaluation:
        """Оценить итоговую модель на тестовых данных.

        Если x_test/y_test не переданы, использует test split
        из finetune_dataset.
        """
        if self.generator is None:
            raise RuntimeError("Model not trained")

        if x_test is None:
            dataset = self.finetune_dataset
            split = stratified_split(dataset, seed=self.config.seed)
            x_test = dataset.all_features[split["test"]]
            y_test = dataset.target.to_numpy()[split["test"]]
            age_test = dataset.age_days.to_numpy()[split["test"]]

        mu, sigma = self.generator.predict(x_test, mc_samples=30)
        return evaluate_model(
            y_true=y_test,
            y_pred=mu.ravel(),
            y_std=sigma.ravel(),
            age_days=age_test,
        )

    def _freeze_early_layers(self, n_layers: int) -> None:
        """Заморозить первые n_layers слоёв backbone генератора."""
        if self.generator is None:
            return

        frozen = 0
        for child in self.generator.backbone.children():
            if frozen >= n_layers:
                break
            for param in child.parameters():
                param.requires_grad = False
            frozen += 1
