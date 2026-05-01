from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

INVERSE_STAGE_DIR = "train_neat"
BNN_STAGE_DIR = "make_neat_to_bnn"


@dataclass(frozen=True)
class ArtifactsLayout:
    """Common folder layout used by all CLI stages."""

    root: Path
    inverse_dir: Path
    bnn_dir: Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory when it does not exist yet."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_artifacts_layout(
    artifacts_dir: str | Path,
    *,
    inverse_dir: str | Path | None = None,
    bnn_dir: str | Path | None = None,
) -> ArtifactsLayout:
    """Resolve the canonical artifacts folders used by the stage pipeline."""

    root = ensure_dir(artifacts_dir)
    return ArtifactsLayout(
        root=root,
        inverse_dir=ensure_dir(inverse_dir or root / INVERSE_STAGE_DIR),
        bnn_dir=ensure_dir(bnn_dir or root / BNN_STAGE_DIR),
    )


def write_json(path: str | Path, payload: dict) -> Path:
    """Write one JSON document with UTF-8 encoding and pretty indentation."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _format_property_names(names: list[str]) -> str:
    return ", ".join(repr(name) for name in names)


def validate_target_keys(property_columns: list[str], targets: dict[str, float]) -> None:
    """Fail fast when provided targets do not match the trained model."""

    expected = list(property_columns)
    received = list(targets)
    missing = [name for name in expected if name not in targets]
    unexpected = [name for name in received if name not in property_columns]
    if not missing and not unexpected:
        return

    details: list[str] = []
    if missing:
        details.append(f"missing required properties: {_format_property_names(missing)}")
    if unexpected:
        details.append(f"unexpected properties: {_format_property_names(unexpected)}")
    joined = "; ".join(details)
    raise ValueError(
        "Target properties do not match the trained model: "
        f"{joined}. Expected exactly: {_format_property_names(expected)}."
    )


def validate_column_sets(
    *,
    expected: list[str],
    actual: list[str],
    expected_label: str,
    actual_label: str,
) -> None:
    """Require both stages to describe the same semantic columns."""

    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    if not missing and not unexpected:
        return

    details: list[str] = []
    if missing:
        details.append(f"missing from {actual_label}: {_format_property_names(missing)}")
    if unexpected:
        details.append(f"unexpected in {actual_label}: {_format_property_names(unexpected)}")
    raise ValueError(
        f"{expected_label} and {actual_label} must describe the same columns: {'; '.join(details)}."
    )


def collect_extrapolation_warnings(
    model_dir: str | Path,
    property_columns: list[str],
    targets: dict[str, float],
) -> list[str]:
    """Warn when requested target properties lie outside the observed training range."""

    warnings: list[str] = []
    data_profile_path = Path(model_dir) / "data_profile.json"
    if not data_profile_path.exists():
        return warnings

    with data_profile_path.open("r", encoding="utf-8") as handle:
        data_profile = json.load(handle)
    ranges = data_profile.get("property_ranges", {})
    for name in property_columns:
        if name not in ranges or name not in targets:
            continue
        lower = float(ranges[name]["min"])
        upper = float(ranges[name]["max"])
        target_value = float(targets[name])
        if target_value < lower or target_value > upper:
            warnings.append(
                f"Target '{name}'={target_value:.4f} is outside the training range [{lower:.4f}, {upper:.4f}]. "
                "The result is an extrapolation."
            )
    return warnings
