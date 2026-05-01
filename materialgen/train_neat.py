"""Стадия 2 — обучение обратной NEAT-сети.

Содержит конфиг (`BackwardStageConfig`, `load_backward_config`) и
публичную точку входа `run_train_neat`.

В этой ветке NEAT обучается напрямую на датасете пар (свойства → компоненты)
без использования суррогата.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DatasetInputConfig, OptimizerConfig, _resolve_config_path
from .data import prepare_dataset
from .neat_optimizer import NEATOptimizer
from .scaler import StandardScaler
from .stage_common import resolve_artifacts_layout, write_json
from .visualization import write_fitness_history_plot, write_predictions_plot, write_residuals_plot


# =============================================================================
# Конфигурация стадии
# =============================================================================

@dataclass
class BackwardStageConfig:
    """JSON configuration used by the inverse NEAT-training stage."""

    dataset: DatasetInputConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    top_k: int = 1
    neat_config_path: str | None = None

    @property
    def component_columns(self) -> list[str]:
        return self.dataset.components

    @property
    def property_columns(self) -> list[str]:
        return self.dataset.properties

    @classmethod
    def from_dict(cls, payload: dict) -> "BackwardStageConfig":
        dataset_payload = payload.get("backward_input", payload.get("dataset", payload.get("neat_input", {})))
        legacy_data_path = payload.get("data_path")
        components = payload.get("components", payload.get("component_columns"))
        properties = payload.get("properties", payload.get("property_columns"))
        dataset = DatasetInputConfig.from_dict(
            dataset_payload,
            legacy_data_path=legacy_data_path,
            legacy_components=components,
            legacy_properties=properties,
        )
        config = cls(
            dataset=dataset,
            optimizer=OptimizerConfig.from_dict(payload.get("optimizer", {})),
            top_k=int(payload.get("top_k", 1)),
            neat_config_path=None if payload.get("neat_config_path") in (None, "") else str(payload["neat_config_path"]),
        )
        config.validate()
        return config

    def resolve_paths(self, base_dir: Path) -> None:
        self.dataset.resolve_paths(base_dir)
        self.neat_config_path = _resolve_config_path(base_dir, self.neat_config_path)

    def validate(self) -> None:
        self.dataset.validate("backward_input")
        if self.top_k < 1:
            raise ValueError("backward.top_k must be at least 1")

    def to_dict(self) -> dict:
        payload = {
            "data_path": self.dataset.data_path,
            "components": self.dataset.components,
            "properties": self.dataset.properties,
            "optimizer": self.optimizer.__dict__,
            "top_k": self.top_k,
        }
        if self.neat_config_path:
            payload["neat_config_path"] = self.neat_config_path
        return payload


def load_backward_config(path: str | Path) -> BackwardStageConfig:
    """Read a backward-stage JSON file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = BackwardStageConfig.from_dict(payload)
    config.resolve_paths(config_path.resolve().parent)
    return config


# =============================================================================
# Реализация стадии
# =============================================================================

def _augment_metadata(
    metadata_path: Path,
    *,
    component_bounds: dict[str, list[float]],
    input_names: list[str],
    output_names: list[str],
) -> None:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["component_bounds"] = component_bounds
    payload["input_names"] = input_names
    payload["output_names"] = output_names
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Публичная точка входа
# =============================================================================

def run_train_neat(
    *,
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    inverse_dir: str | Path | None = None,
) -> dict:
    """Train the inverse NEAT network directly on the dataset (no surrogate)."""

    config = load_backward_config(config_path)
    layout = resolve_artifacts_layout(artifacts_dir, inverse_dir=inverse_dir)

    dataset = prepare_dataset(
        config.dataset.data_path,
        config.component_columns,
        config.property_columns,
    )
    training_components = np.asarray(dataset.components, dtype=float)
    training_properties = np.asarray(dataset.properties, dtype=float)

    property_scaler = StandardScaler.fit(training_properties)
    properties_scaled = np.asarray(property_scaler.transform(training_properties), dtype=float)

    lower_bounds = np.asarray(
        [dataset.component_bounds[name][0] for name in config.component_columns], dtype=float,
    )
    upper_bounds = np.asarray(
        [dataset.component_bounds[name][1] for name in config.component_columns], dtype=float,
    )

    optimizer = NEATOptimizer(
        input_size=len(config.property_columns),
        output_size=len(config.component_columns),
        config=config.optimizer,
        bounds_lower=lower_bounds,
        bounds_upper=upper_bounds,
        input_names=config.property_columns,
        output_names=config.component_columns,
    )

    output_dir = layout.inverse_dir

    result = optimizer.optimize(
        properties_scaled=properties_scaled,
        target_components=training_components,
        top_k=config.top_k,
        artifacts_dir=str(output_dir),
        neat_config_path=config.neat_config_path,
    )

    candidates = result["candidates"]
    statistics = result["statistics"]
    visualizations = result["visualizations"]
    network_artifacts = result["network_artifacts"]

    for artifact in network_artifacts:
        if "metadata" in artifact:
            _augment_metadata(
                Path(artifact["metadata"]),
                component_bounds=dataset.component_bounds,
                input_names=config.property_columns,
                output_names=config.component_columns,
            )

    best_components = None
    if candidates and "network_artifact" in candidates[0]:
        import cloudpickle
        genome_path = Path(candidates[0]["network_artifact"]["genome"])
        if genome_path.exists():
            with genome_path.open("rb") as handle:
                best_genome = cloudpickle.load(handle)
            if not callable(best_genome):
                import neat
                config_ini_path = Path(candidates[0]["network_artifact"]["config"])
                neat_config = neat.Config(
                    neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    str(config_ini_path),
                )
                net = neat.nn.FeedForwardNetwork.create(best_genome, neat_config)
                best_genome = lambda row: net.activate(row)
            best_components = np.asarray(
                [optimizer._decode_components(best_genome(row.tolist()))
                 for row in properties_scaled],
                dtype=float,
            )

    plot_fitness = write_fitness_history_plot(
        best_fitness=statistics["best_fitness_history"],
        mean_fitness=statistics["mean_fitness_history"],
        median_fitness=statistics["median_fitness_history"],
        stdev_fitness=statistics.get("stdev_fitness_history"),
        output_path=output_dir / "fitness_history.png",
    )
    if best_components is not None:
        plot_predictions = write_predictions_plot(
            y_true=training_components,
            y_pred=best_components,
            y_std=None,
            property_names=config.component_columns,
            output_path=output_dir / "predictions.png",
            title="Predicted vs Actual Components (best candidate)",
        )
        plot_residuals = write_residuals_plot(
            y_true=training_components,
            y_pred=best_components,
            property_names=config.component_columns,
            output_path=output_dir / "residuals.png",
            title="Component Residuals (best candidate)",
        )
    else:
        plot_predictions = None
        plot_residuals = None

    saved_config_path = write_json(output_dir / "backward_config.json", config.to_dict())
    summary = {
        "stage": "train_neat",
        "artifacts_dir": str(output_dir),
        "statistics": statistics,
        "plots": {
            "fitness_history": plot_fitness,
            "predictions": plot_predictions,
            "residuals": plot_residuals,
        },
        "visualizations": visualizations,
        "network_artifacts": network_artifacts,
        "best_candidate": candidates[0] if candidates else None,
        "backward_config": str(saved_config_path),
    }
    summary_path = write_json(output_dir / "training_summary.json", summary)

    manifest = {
        "stage": "train_neat",
        "backward_config": str(saved_config_path),
        "summary_file": str(summary_path),
        "selected_rank": 1 if candidates else None,
        "selected_network_artifact": network_artifacts[0] if network_artifacts else None,
        "selected_visualization": visualizations[0] if visualizations else None,
        "input_names": config.property_columns,
        "output_names": config.component_columns,
        "component_bounds": dataset.component_bounds,
        "property_scaler": {
            "mean": property_scaler.mean.tolist(),
            "scale": property_scaler.scale.tolist(),
        },
    }
    manifest_path = write_json(output_dir / "inverse_model_manifest.json", manifest)
    summary["manifest_file"] = str(manifest_path)
    write_json(summary_path, summary)
    return summary


__all__ = [
    "BackwardStageConfig",
    "load_backward_config",
    "run_train_neat",
]
