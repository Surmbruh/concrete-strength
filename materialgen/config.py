from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path


def _resolve_config_path(base_dir: Path, raw_path: str | None) -> str | None:
    """Resolve config-local paths relative to the file they came from."""

    if raw_path in (None, ""):
        return raw_path
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


@dataclass
class DatasetInputConfig:
    """Common dataset description shared by BNN and NEAT stages."""

    data_path: str
    components: list[str]
    properties: list[str]

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        legacy_data_path: str | None = None,
        legacy_components: list[str] | None = None,
        legacy_properties: list[str] | None = None,
    ) -> "DatasetInputConfig":
        """Build a dataset input block from new or legacy config keys."""

        data_path = payload.get("data_path", legacy_data_path)
        components = payload.get("components", legacy_components)
        properties = payload.get("properties", legacy_properties)
        if data_path is None:
            raise ValueError(f"{cls.__name__}.data_path is required")
        if components is None:
            raise ValueError(f"{cls.__name__}.components is required")
        if properties is None:
            raise ValueError(f"{cls.__name__}.properties is required")
        return cls(
            data_path=str(data_path),
            components=[str(name) for name in components],
            properties=[str(name) for name in properties],
        )

    def resolve_paths(self, base_dir: Path) -> None:
        """Resolve any relative file paths inside this block."""

        resolved = _resolve_config_path(base_dir, self.data_path)
        self.data_path = "" if resolved is None else resolved

    def validate(self, label: str) -> None:
        """Fail early when the dataset block is incomplete."""

        if not self.data_path:
            raise ValueError(f"{label}.data_path must not be empty")
        if not self.components:
            raise ValueError(f"{label}.components must contain at least one column")
        if not self.properties:
            raise ValueError(f"{label}.properties must contain at least one column")

    def to_dict(self) -> dict:
        """Serialize the dataset block into plain JSON-compatible values."""

        return {
            "data_path": self.data_path,
            "components": self.components,
            "properties": self.properties,
        }


@dataclass
class OptimizerConfig:
    """Hyperparameters for NEATEST/BNEATEST and network visualization."""

    visualization_samples: int = 96
    visualization_input_sigma: float = 0.12
    mc_samples: int = 30
    uncertainty_weight: float = 0.1
    isolated_output_penalty: float = 1.0
    limit_generations: int = 5
    top_k: int = 2
    seed: int = None

    # NEATEST parameters (loaded from neat.ini)
    algorithm: str = None
    pop_size: int = 250
    es_population: int = 512
    sigma: float = 0.01
    elite_rate: float = 0.1
    use_bias: bool = True
    optimizer_lr: float = 0.01
    node_mutation_rate: float = 0.4
    connection_mutation_rate: float = 0.3
    disable_connection_mutation_rate: float = 0.05
    dominant_gene_rate: float = 0.5
    dominant_gene_delta: float = 0.01
    hidden_activation: str = "passthrough"
    output_activation: str = "tanh"

    # BNEATEST-specific parameters
    sigma_prior: float = 1.0
    kl_weight: float = 0.01
    kl_warmup_steps: int = 5
    initial_rho: float = -3.0
    n_eval_samples: int = 1
    risk_aversion: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict) -> "OptimizerConfig":
        """Create an optimizer config from JSON data."""

        supported_fields = {item.name for item in dataclass_fields(cls)}
        filtered_payload = {key: value for key, value in payload.items() if key in supported_fields}
        return cls(**filtered_payload)


