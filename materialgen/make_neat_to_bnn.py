"""Стадия 3 — конвертация NEAT-генома в байесовскую сеть (NEAT→BNN).

Содержит конфиг (`BNNStageConfig`, `load_neat_to_bnn_config`) и публичную
точку входа `run_make_neat_to_bnn`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import numpy as np

from .config import _resolve_config_path
from .data import prepare_dataset
from .neat_bnn import build_regressor_from_genome
from .neatest.node import NodeType as _LocalNodeType
from .stage_common import resolve_artifacts_layout, write_json
from .visualization import (
    write_bnn_topology,
    write_predictions_plot,
    write_residuals_plot,
    write_training_plot,
    write_tsne_plot,
)


# =============================================================================
# Конфигурация стадии
# =============================================================================

@dataclass
class BNNStageConfig:
    """JSON configuration for the NEAT-to-BNN conversion and training stage."""

    candidate_index: int = 1
    data_path: str = ""
    components: list[str] | None = None
    properties: list[str] | None = None
    learning_rate: float = 0.005
    epochs: int = 300
    batch_size: int = 32
    validation_split: float = 0.2
    mc_samples: int = 30
    early_stopping_rounds: int = 30
    seed: int = 42
    prior_std: float = 0.5
    posterior_scale_init: float = 0.05
    kl_weight: float = 0.01

    @classmethod
    def from_dict(cls, payload: dict) -> "BNNStageConfig":
        config = cls(
            candidate_index=int(payload.get("candidate_index", 1)),
            data_path=str(payload.get("data_path", "")),
            components=payload.get("components"),
            properties=payload.get("properties"),
            learning_rate=float(payload.get("learning_rate", 0.005)),
            epochs=int(payload.get("epochs", 300)),
            batch_size=int(payload.get("batch_size", 32)),
            validation_split=float(payload.get("validation_split", 0.2)),
            mc_samples=int(payload.get("mc_samples", 30)),
            early_stopping_rounds=int(payload.get("early_stopping_rounds", 30)),
            seed=int(payload.get("seed", 42)),
            prior_std=float(payload.get("prior_std", 0.5)),
            posterior_scale_init=float(payload.get("posterior_scale_init", 0.05)),
            kl_weight=float(payload.get("kl_weight", 0.01)),
        )
        config.validate()
        return config

    def resolve_paths(self, base_dir: Path) -> None:
        resolved = _resolve_config_path(base_dir, self.data_path or None)
        self.data_path = "" if resolved is None else resolved

    def validate(self) -> None:
        if self.candidate_index < 1:
            raise ValueError("candidate_index must be at least 1")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    def to_dict(self) -> dict:
        return {
            "candidate_index": self.candidate_index,
            "data_path": self.data_path,
            "components": self.components,
            "properties": self.properties,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "mc_samples": self.mc_samples,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "prior_std": self.prior_std,
            "posterior_scale_init": self.posterior_scale_init,
            "kl_weight": self.kl_weight,
        }


def load_neat_to_bnn_config(path: str | Path) -> BNNStageConfig:
    """Read a NEAT-to-BNN stage JSON config file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = BNNStageConfig.from_dict(payload)
    config.resolve_paths(config_path.resolve().parent)
    return config


# =============================================================================
# Реализация стадии
# =============================================================================

def _fix_node_types(genome) -> None:
    """Re-bind node.type to the local NodeType enum (fixes cross-module pickle mismatch)."""
    _val_map = {t.value: t for t in _LocalNodeType}
    for node in genome.nodes:
        node.type = _val_map.get(node.type.value, node.type)


def _load_genome_and_config(inverse_dir: Path, candidate_index: int):
    """Load a genome for a given candidate.

    Returns (genome, input_names, output_names, component_bounds, neat_config_or_none).
    """

    manifest_path = inverse_dir / "inverse_model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing inverse model manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_names = list(manifest.get("input_names", []))
    output_names = list(manifest.get("output_names", []))
    component_bounds = manifest.get("component_bounds", {})

    selected_artifact = manifest.get("selected_network_artifact")
    if candidate_index == 1 and selected_artifact:
        artifact = selected_artifact
    else:
        genome_name = f"candidate_{candidate_index}_genome.pkl"
        metadata_name = f"candidate_{candidate_index}_network.json"

        genome_path = inverse_dir / genome_name
        if not genome_path.exists():
            raise FileNotFoundError(f"Candidate {candidate_index} genome not found: {genome_path}")

        metadata_path = inverse_dir / metadata_name
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            input_names = metadata.get("input_names", input_names)
            output_names = metadata.get("output_names", output_names)
            component_bounds = metadata.get("component_bounds", component_bounds)

        artifact = {
            "genome": str(genome_path),
            "config": str(inverse_dir / "neat_config.ini"),
        }

    genome_path = Path(artifact["genome"])
    if not genome_path.exists():
        genome_path = inverse_dir / genome_path.name
    with genome_path.open("rb") as handle:
        genome = cloudpickle.load(handle)
    _fix_node_types(genome)

    neat_config = None
    metadata_path = inverse_dir / f"candidate_{candidate_index}_network.json"
    if selected_artifact and candidate_index == 1:
        meta_file = selected_artifact.get("metadata")
        if meta_file:
            metadata_path = Path(meta_file)
            if not metadata_path.exists():
                metadata_path = inverse_dir / metadata_path.name
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("algorithm") == "python-neat":
            import neat
            config_ini = Path(artifact.get("config", ""))
            if not config_ini.exists():
                config_ini = inverse_dir / config_ini.name
            neat_config = neat.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                str(config_ini),
            )

    return genome, input_names, output_names, component_bounds, neat_config


def _load_training_data(
    config,
    inverse_dir: Path,
    input_names: list[str],
    output_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load property/component pairs for training (returns (properties, components))."""

    component_columns = config.components or output_names
    property_columns = config.properties or input_names

    if config.data_path:
        dataset = prepare_dataset(config.data_path, component_columns, property_columns)
        return np.asarray(dataset.properties, dtype=float), np.asarray(dataset.components, dtype=float)

    backward_config_path = inverse_dir / "backward_config.json"
    if backward_config_path.exists():
        bw = json.loads(backward_config_path.read_text(encoding="utf-8"))
        data_path = bw.get("data_path", "")
        if data_path:
            dataset = prepare_dataset(data_path, component_columns, property_columns)
            return np.asarray(dataset.properties, dtype=float), np.asarray(dataset.components, dtype=float)

    raise FileNotFoundError(
        "No data_path in config and no data_path in train_neat's backward_config.json. "
        "Please specify data_path in the BNN stage config."
    )


def run_make_neat_to_bnn(
    *,
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    inverse_dir: str | Path | None = None,
    bnn_dir: str | Path | None = None,
) -> dict:
    """Convert the best NEAT genome into a Bayesian NN, train it, and save."""

    config = load_neat_to_bnn_config(config_path)
    layout = resolve_artifacts_layout(artifacts_dir, inverse_dir=inverse_dir, bnn_dir=bnn_dir)

    genome, input_names, output_names, component_bounds, neat_config = _load_genome_and_config(
        layout.inverse_dir, config.candidate_index,
    )

    component_columns = config.components or output_names
    property_columns = config.properties or input_names

    lower_bounds = np.asarray([component_bounds[name][0] for name in component_columns], dtype=float)
    upper_bounds = np.asarray([component_bounds[name][1] for name in component_columns], dtype=float)

    properties, components = _load_training_data(
        config, layout.inverse_dir, input_names, output_names,
    )

    regressor = build_regressor_from_genome(
        genome,
        bounds_lower=lower_bounds,
        bounds_upper=upper_bounds,
        input_names=property_columns,
        output_names=component_columns,
        prior_std=config.prior_std,
        posterior_scale_init=config.posterior_scale_init,
        kl_weight=config.kl_weight,
        seed=config.seed,
        neat_config=neat_config,
    )

    training_result = regressor.fit(
        properties,
        components,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        mc_samples=config.mc_samples,
        early_stopping_rounds=config.early_stopping_rounds,
        seed=config.seed,
    )

    output_dir = layout.bnn_dir
    model_path = output_dir / "bnn_model.pt"
    regressor.save(model_path)

    _rng = np.random.default_rng(config.seed)
    _idx = _rng.permutation(len(components))
    _val_size = max(1, int(len(components) * config.validation_split))
    _val_idx = _idx[:_val_size]
    _train_idx = _idx[_val_size:] if _val_size < len(components) else _idx

    plot_loss = write_training_plot(
        epoch_train_losses=training_result["epoch_train_losses"],
        epoch_val_losses=training_result["epoch_val_losses"],
        output_path=output_dir / "training_loss.png",
        title="NEAT-BNN Training Progress",
    )

    pred_mean, pred_std = regressor.predict_components(properties, mc_samples=config.mc_samples)
    plot_predictions = write_predictions_plot(
        y_true=components,
        y_pred=pred_mean,
        y_std=pred_std,
        property_names=component_columns,
        output_path=output_dir / "predictions.png",
        title="Predicted vs Actual Components (NEAT-BNN)",
    )
    plot_residuals = write_residuals_plot(
        y_true=components,
        y_pred=pred_mean,
        property_names=component_columns,
        output_path=output_dir / "residuals.png",
        title="Component Residuals (NEAT-BNN)",
    )

    plot_tsne = write_tsne_plot(
        components, _train_idx, _val_idx, component_columns,
        output_dir / "tsne.png",
        "T-SNE компонентов (NEAT-BNN)",
    )

    topology_artifacts = write_bnn_topology(regressor, output_dir)

    write_json(output_dir / "neat_to_bnn_config.json", config.to_dict())

    manifest = {
        "stage": "make_neat_to_bnn",
        "model_filename": "bnn_model.pt",
        "inverse_dir": str(layout.inverse_dir),
        "candidate_index": config.candidate_index,
        "input_names": property_columns,
        "output_names": component_columns,
        "component_bounds": {name: [float(lower_bounds[i]), float(upper_bounds[i])] for i, name in enumerate(component_columns)},
    }
    write_json(output_dir / "bnn_model_manifest.json", manifest)

    summary = {
        "stage": "make_neat_to_bnn",
        "artifacts_dir": str(output_dir),
        "inverse_dir": str(layout.inverse_dir),
        "candidate_index": config.candidate_index,
        "training": training_result,
        "plots": {
            "training_loss": plot_loss,
            "predictions": plot_predictions,
            "residuals": plot_residuals,
            "tsne": plot_tsne,
            "topology": topology_artifacts,
        },
        "model_path": str(model_path),
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "BNNStageConfig",
    "load_neat_to_bnn_config",
    "run_make_neat_to_bnn",
    "_load_genome_and_config",
    "_load_training_data",
]
