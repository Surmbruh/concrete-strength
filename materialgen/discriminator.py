"""Дискриминатор на базе NEAT+BNN для GAN прогноза свойств бетона.

Двухэтапная стратегия:
1. Эволюция топологии через NEAT (python-neat / neatest / bneatest)
2. Bayesian fine-tuning весов через Pyro SVI

Дискриминатор получает конкатенацию (состав, прогнозированная прочность)
и выдаёт скор реалистичности [0, 1] + оценку неопределённости через
BNN posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO

from .config import OptimizerConfig
from .neat_bnn import NeatBNNRegressor, _extract_topology, build_regressor_from_genome
from .neat_optimizer import NEATOptimizer
from .neatest import Agent as NeatestAgent
from .bneatest import Agent as BneatestAgent
from .scaler import StandardScaler


# =============================================================================
# Конфигурация
# =============================================================================

@dataclass
class DiscriminatorConfig:
    """Гиперпараметры дискриминатора."""

    # NEAT эволюция
    algorithm: str = "bneatest"  # python-neat | neatest | bneatest
    neat_generations: int = 10
    pop_size: int = 100
    top_k: int = 2
    max_eval_samples: int = 200  # subsample for fast NEAT evaluation

    # BNN fine-tuning
    prior_std: float = 0.5
    posterior_scale_init: float = 0.05
    kl_weight: float = 0.01
    svi_lr: float = 0.005
    svi_epochs: int = 50
    mc_samples: int = 30

    # GAN-loop SVI updates
    gan_svi_steps_per_epoch: int = 5

    seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict) -> DiscriminatorConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# =============================================================================
# NEAT Agents для эволюции дискриминатора
# =============================================================================

class DiscriminatorNeatestAgent(NeatestAgent):
    """NEAT-агент: эволюционирует топологию дискриминатора.

    Fitness = насколько хорошо сеть отличает реальные пары
    (состав, прочность) от фейковых (случайных/нефизичных).
    """

    def __init__(
        self,
        real_pairs: np.ndarray,
        fake_pairs: np.ndarray,
    ) -> None:
        self.real_pairs = real_pairs  # [n, input_dim] — (composition, strength)
        self.fake_pairs = fake_pairs  # [n, input_dim]

    def rollout(self, genome) -> float:
        """Оценка генома: accuracy бинарной классификации real/fake."""
        correct = 0
        total = 0

        # Real → должно быть > 0.5 (tanh > 0)
        for sample in self.real_pairs:
            output = genome(sample.tolist())
            if len(output) > 0 and output[0] > 0.0:
                correct += 1
            total += 1

        # Fake → должно быть < 0.5 (tanh < 0)
        for sample in self.fake_pairs:
            output = genome(sample.tolist())
            if len(output) > 0 and output[0] <= 0.0:
                correct += 1
            total += 1

        return correct / max(total, 1)


class DiscriminatorBneatestAgent(BneatestAgent):
    """Bayesian NEAT-агент для дискриминатора."""

    def __init__(
        self,
        real_pairs: np.ndarray,
        fake_pairs: np.ndarray,
    ) -> None:
        self.real_pairs = real_pairs
        self.fake_pairs = fake_pairs

    def rollout(self, genome) -> float:
        correct = 0
        total = 0
        for sample in self.real_pairs:
            output = genome(sample.tolist())
            if len(output) > 0 and output[0] > 0.0:
                correct += 1
            total += 1
        for sample in self.fake_pairs:
            output = genome(sample.tolist())
            if len(output) > 0 and output[0] <= 0.0:
                correct += 1
            total += 1
        return correct / max(total, 1)


# =============================================================================
# Дискриминатор
# =============================================================================

class NeatBNNDiscriminator:
    """Дискриминатор на базе NEAT+BNN.

    Lifecycle:
        1. evolve_topology() → NEAT эволюция архитектуры
        2. init_bnn() → конвертация лучшего генома в BNN
        3. update() → SVI шаги в GAN-цикле
        4. score() → MC-sampling для скора + uncertainty

    Вход : concat(composition_features, strength, log_age) → dim = ~12
    Выход: score ∈ [0, 1] + σ_score (BNN uncertainty)
    """

    def __init__(self, config: DiscriminatorConfig | None = None) -> None:
        if config is None:
            config = DiscriminatorConfig()
        self.config = config
        self.input_dim: int | None = None
        self.genome = None
        self.neat_config = None
        self.bnn: NeatBNNRegressor | None = None
        self.input_scaler: StandardScaler | None = None
        self.svi: SVI | None = None
        self._topology: dict | None = None

    def evolve_topology(
        self,
        real_pairs: np.ndarray,
        fake_pairs: np.ndarray,
        *,
        artifacts_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Этап 1: эволюция топологии NEAT.

        Запускает NEAT для поиска оптимальной архитектуры дискриминатора.

        Parameters
        ----------
        real_pairs : реальные пары (composition + strength) [n, dim]
        fake_pairs : фейковые пары [n, dim]
        artifacts_dir : директория для артефактов (опционально)

        Returns
        -------
        dict с результатами эволюции (fitness, genome path, etc.)
        """
        import configparser, tempfile

        self.input_dim = real_pairs.shape[1]

        # Fit scaler on full data
        all_data = np.concatenate([real_pairs, fake_pairs], axis=0)
        self.input_scaler = StandardScaler.fit(all_data)
        real_scaled = self.input_scaler.transform(real_pairs)
        fake_scaled = self.input_scaler.transform(fake_pairs)

        # Subsample for fast NEAT evaluation
        max_s = self.config.max_eval_samples
        rng = np.random.default_rng(self.config.seed)
        if len(real_scaled) > max_s:
            idx_r = rng.choice(len(real_scaled), max_s, replace=False)
            idx_f = rng.choice(len(fake_scaled), max_s, replace=False)
            real_scaled = real_scaled[idx_r]
            fake_scaled = fake_scaled[idx_f]

        # Write temporary neat.ini with our pop_size override
        if artifacts_dir is not None:
            art_path = Path(artifacts_dir) / "discriminator_neat"
            art_path.mkdir(parents=True, exist_ok=True)
        else:
            art_path = Path("artifacts") / "discriminator_neat"
            art_path.mkdir(parents=True, exist_ok=True)

        temp_ini = art_path / "neat_override.ini"
        ini_parser = configparser.ConfigParser()
        ini_parser.optionxform = str
        ini_parser.read(str(Path(__file__).resolve().with_name("neat.ini")))
        if "NEAT" not in ini_parser:
            ini_parser["NEAT"] = {}
        ini_parser["NEAT"]["pop_size"] = str(self.config.pop_size)
        ini_parser["NEAT"]["algorithm"] = self.config.algorithm
        with open(temp_ini, "w") as f:
            ini_parser.write(f)

        # Создаём optimizer
        optimizer_config = OptimizerConfig(
            algorithm=self.config.algorithm,
            pop_size=self.config.pop_size,
            limit_generations=self.config.neat_generations,
            top_k=self.config.top_k,
            seed=self.config.seed,
        )
        bounds_lower = np.zeros(1)  # output: score [0, 1]
        bounds_upper = np.ones(1)

        optimizer = NEATOptimizer(
            input_size=self.input_dim,
            output_size=1,
            config=optimizer_config,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
        )

        # Для NEAT нужны пары (input → target). Мы превращаем
        # дискриминацию в задачу: input = pair, target = label (0 или 1)
        all_inputs = np.concatenate([real_scaled, fake_scaled], axis=0)
        labels = np.concatenate([
            np.ones((len(real_scaled), 1)),
            np.zeros((len(fake_scaled), 1)),
        ], axis=0)

        result = optimizer.optimize(
            properties_scaled=all_inputs,
            target_components=labels,
            top_k=self.config.top_k,
            artifacts_dir=str(art_path),
            neat_config_path=str(temp_ini),
        )

        # Сохраняем лучший геном
        if result.get("candidates"):
            best = result["candidates"][0]
            genome_path = best.get("network_artifact", {}).get("genome")
            if genome_path and Path(genome_path).exists():
                with open(genome_path, "rb") as f:
                    self.genome = cloudpickle.load(f)

        self._topology = None  # будет извлечена в init_bnn
        return result

    def init_bnn(self) -> None:
        """Этап 2: конвертация NEAT-генома в Bayesian NN.

        Берёт лучшую топологию из evolve_topology() и создаёт
        NeatBNNRegressor для дальнейшего SVI fine-tuning.
        """
        if self.genome is None:
            raise RuntimeError(
                "No genome available. Run evolve_topology() first."
            )

        self._topology = _extract_topology(
            self.genome, neat_config=self.neat_config,
        )

        self.bnn = NeatBNNRegressor(
            topology=self._topology,
            prior_std=self.config.prior_std,
            posterior_scale_init=self.config.posterior_scale_init,
            kl_weight=self.config.kl_weight,
            seed=self.config.seed,
            bounds_lower=np.zeros(1),
            bounds_upper=np.ones(1),
        )

    def pretrain_bnn(
        self,
        real_pairs: np.ndarray,
        fake_pairs: np.ndarray,
    ) -> dict[str, Any]:
        """Начальное обучение BNN на реальных/фейковых парах.

        Parameters
        ----------
        real_pairs : реальные пары [n, dim]
        fake_pairs : фейковые пары [n, dim]

        Returns
        -------
        dict с результатами обучения
        """
        if self.bnn is None:
            raise RuntimeError("BNN not initialized. Run init_bnn() first.")

        all_inputs = np.concatenate([real_pairs, fake_pairs], axis=0)
        labels = np.concatenate([
            np.ones((len(real_pairs), 1)),
            np.zeros((len(fake_pairs), 1)),
        ], axis=0)

        result = self.bnn.fit(
            properties=all_inputs,
            components=labels,
            learning_rate=self.config.svi_lr,
            epochs=self.config.svi_epochs,
            mc_samples=self.config.mc_samples,
            seed=self.config.seed,
        )
        return result

    def score(
        self,
        pairs: np.ndarray,
        mc_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Оценить реалистичность пар (состав, прочность).

        Parameters
        ----------
        pairs : массив пар [n, dim]
        mc_samples : количество MC-прогонов (по умолчанию из config)

        Returns
        -------
        (mean_score, std_score) — средний скор и BNN uncertainty
            mean_score : [n, 1], ∈ [0, 1]
            std_score  : [n, 1], > 0
        """
        if self.bnn is None:
            raise RuntimeError("BNN not initialized")

        if mc_samples is None:
            mc_samples = self.config.mc_samples

        mean, std = self.bnn.predict_components(
            pairs, mc_samples=mc_samples,
        )
        # Клипаем в [0, 1]
        mean = np.clip(mean, 0.0, 1.0)
        return mean, std

    def score_tensor(
        self,
        pairs: torch.Tensor,
        mc_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score с поддержкой PyTorch tensors (для GAN backprop).

        NOTE: Градиенты через BNN MC sampling не проходят напрямую.
        Используем straight-through estimator или REINFORCE для
        обучения генератора через дискриминатор.
        """
        pairs_np = pairs.detach().cpu().numpy()
        mean_np, std_np = self.score(pairs_np, mc_samples=mc_samples)
        mean_t = torch.as_tensor(mean_np, dtype=torch.float32, device=pairs.device)
        std_t = torch.as_tensor(std_np, dtype=torch.float32, device=pairs.device)
        return mean_t, std_t

    def save(self, path: str | Path) -> None:
        """Сохранить состояние дискриминатора."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config.to_dict(),
            "input_dim": self.input_dim,
            "input_scaler": self.input_scaler.to_dict() if self.input_scaler else None,
        }

        # Сохраняем геном
        if self.genome is not None:
            genome_path = path.with_name(f"{path.stem}_genome.pkl")
            with open(genome_path, "wb") as f:
                cloudpickle.dump(self.genome, f)
            checkpoint["genome_path"] = str(genome_path)

        # Сохраняем BNN
        if self.bnn is not None:
            bnn_path = path.with_name(f"{path.stem}_bnn.pt")
            self.bnn.save(bnn_path)
            checkpoint["bnn_path"] = str(bnn_path)

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> NeatBNNDiscriminator:
        """Загрузить дискриминатор из чекпоинта."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        disc = cls(config=DiscriminatorConfig.from_dict(checkpoint["config"]))
        disc.input_dim = checkpoint.get("input_dim")

        if checkpoint.get("input_scaler"):
            disc.input_scaler = StandardScaler.from_dict(checkpoint["input_scaler"])

        if checkpoint.get("genome_path"):
            genome_path = Path(checkpoint["genome_path"])
            if genome_path.exists():
                with open(genome_path, "rb") as f:
                    disc.genome = cloudpickle.load(f)

        if checkpoint.get("bnn_path"):
            bnn_path = Path(checkpoint["bnn_path"])
            if bnn_path.exists():
                disc.bnn = NeatBNNRegressor.load(bnn_path)

        return disc
