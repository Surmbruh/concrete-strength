"""Lightweight experiment tracker.

Logs every experiment run to a JSONL file with config, metrics, and timing.
Works on both CPU and GPU without external dependencies.

Usage:
    tracker = ExperimentTracker("experiments/")
    with tracker.run("supervised_grid", config={"lr": 1e-3, ...}) as run:
        # ... train model ...
        run.log_metrics({"mae": 9.51, "r2": 0.504})
        run.log_artifact("model.pt", "/path/to/model.pt")
"""
import json
import time
import platform
import torch
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    """Single experiment run."""
    experiment: str
    config: dict
    metrics: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    timestamp: str = ""
    duration_sec: float = 0.0
    device: str = ""
    hostname: str = ""
    status: str = "running"
    notes: str = ""

    def log_metrics(self, metrics: dict):
        self.metrics.update(metrics)

    def log_artifact(self, name: str, path: str):
        self.artifacts.append({"name": name, "path": path})

    def add_tag(self, tag: str):
        self.tags.append(tag)


class ExperimentTracker:
    """JSON-based experiment tracker."""

    def __init__(self, log_dir: str | Path = "experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiment_log.jsonl"

    def run(self, experiment: str, config: dict = None, tags: list = None):
        """Context manager for an experiment run."""
        return _RunContext(
            tracker=self,
            experiment=experiment,
            config=config or {},
            tags=tags or [],
        )

    def log_run(self, record: RunRecord):
        """Append a completed run to the log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")

    def load_runs(self, experiment: str = None) -> list[dict]:
        """Load all runs, optionally filtered by experiment name."""
        if not self.log_file.exists():
            return []
        runs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if experiment is None or rec.get("experiment") == experiment:
                    runs.append(rec)
        return runs

    def best_run(self, experiment: str = None, metric: str = "mae",
                 minimize: bool = True) -> dict | None:
        """Find the best run by a given metric."""
        runs = [r for r in self.load_runs(experiment)
                if r.get("status") == "completed" and metric in r.get("metrics", {})]
        if not runs:
            return None
        return min(runs, key=lambda r: r["metrics"][metric]) if minimize \
            else max(runs, key=lambda r: r["metrics"][metric])

    def summary_table(self, experiment: str = None) -> str:
        """Print a summary table of all completed runs."""
        runs = [r for r in self.load_runs(experiment) if r.get("status") == "completed"]
        if not runs:
            return "No completed runs."

        # Collect all metric keys
        all_metrics = set()
        for r in runs:
            all_metrics.update(r.get("metrics", {}).keys())
        metric_keys = sorted(all_metrics)

        lines = []
        header = f"{'Experiment':<30} {'Duration':>10} " + " ".join(f"{k:>10}" for k in metric_keys)
        lines.append(header)
        lines.append("-" * len(header))
        for r in runs:
            dur = f"{r.get('duration_sec', 0):.0f}s"
            vals = " ".join(
                f"{r.get('metrics', {}).get(k, float('nan')):10.4f}" for k in metric_keys
            )
            lines.append(f"{r['experiment']:<30} {dur:>10} {vals}")
        return "\n".join(lines)


class _RunContext:
    """Context manager for a single experiment run."""

    def __init__(self, tracker: ExperimentTracker, experiment: str,
                 config: dict, tags: list):
        self.tracker = tracker
        self.record = RunRecord(
            experiment=experiment,
            config=config,
            tags=tags,
            timestamp=datetime.now().isoformat(),
            device=_detect_device(),
            hostname=platform.node(),
        )
        self._start_time = None

    def __enter__(self) -> RunRecord:
        self._start_time = time.time()
        print(f"[Tracker] Starting: {self.record.experiment} "
              f"on {self.record.device}", flush=True)
        return self.record

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record.duration_sec = time.time() - self._start_time
        if exc_type is not None:
            self.record.status = "failed"
            self.record.notes = f"{exc_type.__name__}: {exc_val}"
        else:
            self.record.status = "completed"
        self.tracker.log_run(self.record)
        print(f"[Tracker] {self.record.status}: {self.record.experiment} "
              f"({self.record.duration_sec:.1f}s)", flush=True)
        return False  # don't suppress exceptions


def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"cuda:{name}"
    return "cpu"


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
