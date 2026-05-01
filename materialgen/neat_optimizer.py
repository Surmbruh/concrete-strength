from __future__ import annotations

import configparser
import copy
import html
import io
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np

from .config import OptimizerConfig
from .neatest import NEATEST, Adam, Agent as NeatestAgent, StatisticsReporter, StdOutReporter
from .neatest.node import NodeType, group_nodes
from .bneatest import BNEATEST, Agent as BneatestAgent

_VALID_ALGORITHMS = ("python-neat", "neatest", "bneatest")


def _resolve_activation(name: str):
    """Resolve an activation name to the callable from neatest.node."""
    from .neatest import node as _node_mod
    func = getattr(_node_mod, name, None)
    if func is None:
        raise ValueError(f"Unknown activation: {name}")
    return func


# ---------------------------------------------------------------------------
# NEATEST / BNEATEST agent adapters
# ---------------------------------------------------------------------------

class DatasetAgent(NeatestAgent):
    """Evaluates a NEATEST genome directly against known (property, component) pairs."""

    def __init__(
        self,
        optimizer: NEATOptimizer,
        properties_scaled: np.ndarray,
        target_components: np.ndarray,
    ) -> None:
        self.optimizer = optimizer
        self.properties_scaled = properties_scaled
        self.target_components = target_components

    def rollout(self, genome) -> float:
        diagnostics = self.optimizer._evaluate_candidate(
            genome,
            properties_scaled=self.properties_scaled,
            target_components=self.target_components,
        )
        output_ids = {node.id for node in genome.outputs}
        connected = {c.out_node.id for c in genome.connections if c.enabled}
        n_isolated = sum(1 for oid in output_ids if oid not in connected)
        if n_isolated > 0:
            penalty = self.optimizer.config.isolated_output_penalty * n_isolated / self.optimizer.output_size
            return 1.0 / (1.0 + diagnostics["objective"] + penalty)
        return float(diagnostics["fitness"])


class BayesianDatasetAgent(BneatestAgent):
    """Evaluates a BNEATEST genome directly against known (property, component) pairs."""

    def __init__(
        self,
        optimizer: NEATOptimizer,
        properties_scaled: np.ndarray,
        target_components: np.ndarray,
    ) -> None:
        self.optimizer = optimizer
        self.properties_scaled = properties_scaled
        self.target_components = target_components

    def rollout(self, genome) -> float:
        diagnostics = self.optimizer._evaluate_candidate(
            genome,
            properties_scaled=self.properties_scaled,
            target_components=self.target_components,
        )
        output_ids = {node.id for node in genome.outputs}
        connected = {c.out_node.id for c in genome.connections if c.enabled}
        n_isolated = sum(1 for oid in output_ids if oid not in connected)
        if n_isolated > 0:
            penalty = self.optimizer.config.isolated_output_penalty * n_isolated / self.optimizer.output_size
            return 1.0 / (1.0 + diagnostics["objective"] + penalty)
        return float(diagnostics["fitness"])


# ---------------------------------------------------------------------------
# python-neat wrapper for callable interface
# ---------------------------------------------------------------------------

class _PythonNeatCallableNetwork:
    """Wrapper that makes a python-neat FeedForwardNetwork callable like a NEATEST genome."""

    def __init__(self, network):
        self.network = network

    def __call__(self, inputs):
        return self.network.activate(inputs)


class NEATOptimizer:
    """Run python-neat / NEATEST / BNEATEST over candidate concrete recipes.

    The evolutionary algorithm evolves small feed-forward networks that map
    target properties to component values. Fitness is measured directly
    against known (property, component) pairs from the training dataset.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: OptimizerConfig,
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        """Store everything needed to score genomes and render their graphs."""
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.config = config
        self.bounds_lower = bounds_lower.astype(float)
        self.bounds_upper = bounds_upper.astype(float)
        self.component_scales = np.where((self.bounds_upper - self.bounds_lower) < 1e-8, 1.0, self.bounds_upper - self.bounds_lower)
        self.input_names = input_names or [f"input_{index}" for index in range(self.input_size)]
        self.output_names = output_names or [f"output_{index}" for index in range(self.output_size)]

    def _decode_components(self, outputs: list[float] | np.ndarray) -> np.ndarray:
        """Convert NEAT outputs from [-1, 1] into real component bounds."""

        outputs_array = np.clip(np.asarray(outputs, dtype=float), -1.0, 1.0)
        scaled = (outputs_array + 1.0) / 2.0
        return self.bounds_lower + scaled * (self.bounds_upper - self.bounds_lower)

    def _load_neat_ini(self, neat_config_path: str | Path | None = None) -> OptimizerConfig:
        """Load parameters from neat.ini and merge into a copy of self.config."""

        parser = configparser.ConfigParser()
        parser.optionxform = str
        default_config_path = Path(__file__).resolve().with_name("neat.ini")
        config_files = [str(default_config_path)]
        if neat_config_path is not None:
            config_files.append(str(Path(neat_config_path)))
        parser.read(config_files)

        ini_values: dict[str, str] = {}
        for section in parser.sections():
            for key, value in parser.items(section):
                ini_values[key] = value

        from dataclasses import fields as dataclass_fields, asdict
        merged = asdict(self.config)
        type_map = {f.name: f.type for f in dataclass_fields(OptimizerConfig)}
        for key, raw_value in ini_values.items():
            if key in type_map:
                field_type = type_map[key]
                if field_type == "bool" or field_type is bool:
                    merged[key] = raw_value.lower() in ("true", "1", "yes")
                elif field_type == "int" or field_type is int:
                    merged[key] = int(raw_value)
                elif field_type == "float" or field_type is float:
                    merged[key] = float(raw_value)
                else:
                    merged[key] = raw_value

        return OptimizerConfig(**merged)

    def _resolve_neat_config(self, neat_config_path: str | Path | None = None) -> str:
        """Read neat.ini and return its content as a string for saving."""

        parser = configparser.ConfigParser()
        parser.optionxform = str
        default_config_path = Path(__file__).resolve().with_name("neat.ini")
        config_files = [str(default_config_path)]
        if neat_config_path is not None:
            config_files.append(str(Path(neat_config_path)))
        parser.read(config_files)

        buffer = io.StringIO()
        parser.write(buffer, space_around_delimiters=True)
        return buffer.getvalue().strip()

    # ------------------------------------------------------------------
    # Candidate evaluation (shared by all algorithms)
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        network,
        properties_scaled: np.ndarray,
        target_components: np.ndarray,
    ) -> dict[str, Any]:
        """Score one network by how well its predicted components match the dataset.

        ``network`` must be callable: ``network(inputs) -> list[float]``.
        For python-neat this is a wrapped FeedForwardNetwork; for NEATEST
        the genome itself is callable.
        """

        properties_batch = np.asarray(properties_scaled, dtype=float)
        target_components_batch = np.asarray(target_components, dtype=float)
        if properties_batch.ndim == 1:
            properties_batch = properties_batch[None, :]
        if target_components_batch.ndim == 1:
            target_components_batch = target_components_batch[None, :]

        components = np.asarray(
            [self._decode_components(network(sample.tolist())) for sample in properties_batch],
            dtype=float,
        )

        component_residual = (components - target_components_batch) / self.component_scales
        objective_values = np.mean(component_residual ** 2, axis=1)

        representative_index = int(np.argmin(objective_values))
        objective = float(np.mean(objective_values))
        return {
            "components": components[representative_index].tolist(),
            "target_components": target_components_batch[representative_index].tolist(),
            "objective": objective,
            "fitness": 1.0 / (1.0 + objective),
        }

    # ------------------------------------------------------------------
    # Visualization helpers (shared labels / formatting)
    # ------------------------------------------------------------------

    @staticmethod
    def _distribution_text(mean: float, std: float, prefix: str = "a") -> str:
        return f"{prefix}~N(mu={mean:.2f}, sigma={std:.2f})"

    @staticmethod
    def _node_penwidth(std: float) -> float:
        return 1.2 + min(2.8, max(0.0, std) * 3.0)

    @staticmethod
    def _html_table_label(title: str, subtitle: str, details: list[str]) -> str:
        rows = [
            f"<TR><TD><B>{html.escape(title)}</B></TD></TR>",
            f"<TR><TD><FONT POINT-SIZE=\"10\">{html.escape(subtitle)}</FONT></TD></TR>",
        ]
        rows.extend(f"<TR><TD><FONT POINT-SIZE=\"10\">{html.escape(detail)}</FONT></TD></TR>" for detail in details)
        return "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLPADDING=\"2\">" + "".join(rows) + "</TABLE>>"

    def _summary_label(self, title: str, candidate: dict[str, Any]) -> str:
        lines = [
            title,
            f"objective={float(candidate['objective']):.4f}",
            f"fitness={float(candidate['fitness']):.4f}",
            f"activation_mc_samples={int(self.config.visualization_samples)}",
            f"input_sigma={float(self.config.visualization_input_sigma):.2f}",
        ]
        predicted = candidate.get("components", [])
        target = candidate.get("target_components", [])
        for name, pred, tgt in zip(self.output_names, predicted, target):
            lines.append(f"{name}: pred={float(pred):.2f}, target={float(tgt):.2f}")
        return "\\n".join(lines)

    @staticmethod
    def _stats_for(node_id: int, activation_stats: dict[int, dict[str, float]]) -> dict[str, float]:
        return activation_stats.get(int(node_id), {"mean": 0.0, "std": 0.0})

    def _output_component_distribution(self, output_index: int, activation_mean: float, activation_std: float) -> str:
        component_span = float(self.bounds_upper[output_index] - self.bounds_lower[output_index])
        component_mean = float(
            self.bounds_lower[output_index] + ((np.clip(activation_mean, -1.0, 1.0) + 1.0) / 2.0) * component_span
        )
        component_std = float(0.5 * max(0.0, activation_std) * component_span)
        return self._distribution_text(component_mean, component_std, prefix="mix")

    def _build_input_node_line(self, node_id, name, target_value, activation_stats):
        stats = self._stats_for(node_id, activation_stats)
        label = self._html_table_label(
            title=name, subtitle="target input",
            details=[self._distribution_text(stats["mean"], stats["std"]), f"value={target_value:.2f}"],
        )
        return (
            f'  "{node_id}" [label={label}, shape=box, style="filled,rounded", '
            f'fillcolor="#dbeafe", color="#1d4ed8", penwidth={self._node_penwidth(stats["std"]):.2f}];'
        )

    def _build_output_node_line(self, node_id, name, output_index, activation_stats):
        stats = self._stats_for(node_id, activation_stats)
        label = self._html_table_label(
            title=name, subtitle="mix output",
            details=[
                self._distribution_text(stats["mean"], stats["std"]),
                self._output_component_distribution(output_index, stats["mean"], stats["std"]),
            ],
        )
        return (
            f'  "{node_id}" [label={label}, shape=doublecircle, style="filled", fillcolor="#dcfce7", '
            f'color="#15803d", penwidth={self._node_penwidth(stats["std"]):.2f}];'
        )

    def _build_hidden_node_line(self, node_id, activation_stats):
        stats = self._stats_for(node_id, activation_stats)
        label = self._html_table_label(
            title=f"h{node_id}", subtitle="hidden node",
            details=[self._distribution_text(stats["mean"], stats["std"])],
        )
        return (
            f'  "{node_id}" [label={label}, shape=ellipse, style="filled", fillcolor="#f3f4f6", '
            f'color="#4b5563", penwidth={self._node_penwidth(stats["std"]):.2f}];'
        )

    def _write_graphviz(self, dot_source: str, output_prefix: Path) -> dict[str, str]:
        dot_path = output_prefix.with_suffix(".dot")
        png_path = output_prefix.with_suffix(".png")
        svg_path = output_prefix.with_suffix(".svg")
        dot_path.write_text(dot_source, encoding="utf-8")
        artifacts: dict[str, str] = {"dot": str(dot_path)}
        try:
            subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
            artifacts["png"] = str(png_path)
        except Exception:
            pass
        try:
            subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
            artifacts["svg"] = str(svg_path)
        except Exception:
            pass
        return artifacts

    @staticmethod
    def _component_signature(components: list[float]) -> tuple[float, ...]:
        return tuple(round(value, 4) for value in components)

    def _select_unique_candidates(self, hall_of_fame: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()
        for item in hall_of_fame:
            signature = self._component_signature(item["components"])
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    # ======================================================================
    # python-neat path
    # ======================================================================

    def _resolve_python_neat_config(self, neat_config_path: str | Path | None = None) -> str:
        """Generate a neat-python .ini config with correct input/output sizes."""

        parser = configparser.ConfigParser()
        parser.optionxform = str
        default_config_path = Path(__file__).resolve().with_name("neat_python.ini")
        config_files: list[str] = []
        if default_config_path.exists():
            config_files.append(str(default_config_path))
        if neat_config_path is not None:
            config_files.append(str(Path(neat_config_path)))
        parser.read(config_files)

        # Ensure correct I/O sizes
        if not parser.has_section("DefaultGenome"):
            parser.add_section("DefaultGenome")
        parser.set("DefaultGenome", "num_inputs", str(self.input_size))
        parser.set("DefaultGenome", "num_outputs", str(self.output_size))
        parser.set("DefaultGenome", "feed_forward", "True")

        buffer = io.StringIO()
        parser.write(buffer, space_around_delimiters=True)
        return buffer.getvalue().strip()

    def _count_isolated_outputs_pyneat(self, genome, neat_config) -> int:
        output_keys = set(neat_config.genome_config.output_keys)
        reachable = {conn.key[1] for conn in genome.connections.values() if conn.enabled}
        return sum(1 for key in output_keys if key not in reachable)

    def _estimate_activation_statistics_pyneat(self, genome, neat_config, target_scaled):
        import neat
        network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        ordered_nodes = [*network.input_nodes, *[node for node, *_ in network.node_evals]]
        sample_count = max(1, int(self.config.visualization_samples))
        sigma = max(0.0, float(self.config.visualization_input_sigma))
        rng = np.random.default_rng(self.config.seed)
        sampled_inputs = np.repeat(target_scaled[None, :], sample_count, axis=0)
        if sigma > 0.0:
            sampled_inputs += rng.normal(0.0, sigma, size=sampled_inputs.shape)
        activations: dict[int, list[float]] = {int(nid): [] for nid in ordered_nodes}
        for sample in sampled_inputs:
            network.activate(sample.tolist())
            for nid in ordered_nodes:
                activations[int(nid)].append(float(network.values.get(nid, 0.0)))
        return {
            nid: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for nid, vals in activations.items()
        }

    def _build_edge_line_pyneat(self, connection) -> str:
        color = "#15803d" if connection.weight >= 0 else "#b91c1c"
        style = "solid" if connection.enabled else "dashed"
        penwidth = 0.8 + min(3.5, abs(connection.weight))
        return (
            f'  "{connection.key[0]}" -> "{connection.key[1]}" '
            f'[label="{connection.weight:.2f}", color="{color}", style="{style}", penwidth={penwidth:.2f}];'
        )

    def _dot_source_pyneat(self, genome, neat_config, title, target_scaled, target_original, candidate):
        input_keys = list(neat_config.genome_config.input_keys)
        output_keys = list(neat_config.genome_config.output_keys)
        activation_stats = self._estimate_activation_statistics_pyneat(genome, neat_config, target_scaled)
        lines = [
            "digraph neat_network {",
            '  graph [rankdir=LR, labelloc="t", labeljust="l"];',
            f'  label="{title}";',
            '  node [fontname="Helvetica", fontsize=10];',
            f'  "summary" [label="{self._summary_label(title, candidate)}", '
            'shape=note, style="filled", fillcolor="#fff7ed", color="#c2410c"];',
        ]
        for index, nid in enumerate(input_keys):
            lines.append(self._build_input_node_line(
                nid, self.input_names[index], float(target_original[index]), activation_stats))
        for index, nid in enumerate(output_keys):
            lines.append(self._build_output_node_line(
                nid, self.output_names[index], index, activation_stats))
        for nid in sorted(genome.nodes):
            if nid in output_keys:
                continue
            lines.append(self._build_hidden_node_line(nid, activation_stats))
        for conn in genome.connections.values():
            lines.append(self._build_edge_line_pyneat(conn))
        lines.append("}")
        return "\n".join(lines)

    def _write_visualization_pyneat(self, genome, neat_config, output_prefix, title, target_scaled, target_original, candidate):
        dot = self._dot_source_pyneat(genome, neat_config, title, target_scaled, target_original, candidate)
        return self._write_graphviz(dot, output_prefix)

    def _write_network_artifact_pyneat(self, genome, neat_config, output_prefix, config_path, candidate):
        import neat
        genome_path = output_prefix.with_name(f"{output_prefix.name}_genome.pkl")
        network_path = output_prefix.with_name(f"{output_prefix.name}_network.pkl")
        metadata_path = output_prefix.with_name(f"{output_prefix.name}_network.json")
        with genome_path.open("wb") as handle:
            pickle.dump(genome, handle)
        artifacts = {"genome": str(genome_path), "config": str(config_path)}
        try:
            network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            with network_path.open("wb") as handle:
                pickle.dump(network, handle)
            artifacts["network"] = str(network_path)
        except Exception:
            pass
        metadata = {
            "algorithm": "python-neat",
            "objective": float(candidate["objective"]),
            "fitness": float(candidate["fitness"]),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "config": str(config_path),
            "files": artifacts,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifacts["metadata"] = str(metadata_path)
        return artifacts

    def _write_statistics_pyneat(self, statistics_reporter, artifacts_path):
        fitness_history_path = artifacts_path / "fitness_history.csv"
        speciation_path = artifacts_path / "speciation.csv"
        species_fitness_path = artifacts_path / "species_fitness.csv"
        summary_path = artifacts_path / "neat_statistics.json"
        statistics_reporter.save_genome_fitness(delimiter=",", filename=str(fitness_history_path))
        statistics_reporter.save_species_count(delimiter=",", filename=str(speciation_path))
        statistics_reporter.save_species_fitness(delimiter=",", filename=str(species_fitness_path))
        best_genome = statistics_reporter.best_genome() if statistics_reporter.most_fit_genomes else None
        species_sizes = (
            [[int(s) for s in gen] for gen in statistics_reporter.get_species_sizes()]
            if statistics_reporter.generation_statistics else []
        )
        summary = {
            "generation_count": len(statistics_reporter.most_fit_genomes),
            "best_fitness": float(best_genome.fitness) if best_genome is not None else None,
            "best_fitness_history": [float(g.fitness) for g in statistics_reporter.most_fit_genomes],
            "mean_fitness_history": [float(v) for v in statistics_reporter.get_fitness_mean()],
            "median_fitness_history": [float(v) for v in statistics_reporter.get_fitness_median()],
            "stdev_fitness_history": [float(v) for v in statistics_reporter.get_fitness_stdev()],
            "species_sizes": species_sizes,
            "files": {
                "fitness_history": str(fitness_history_path),
                "speciation": str(speciation_path),
                "species_fitness": str(species_fitness_path),
                "summary": str(summary_path),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _optimize_python_neat(
        self,
        properties_scaled: np.ndarray,
        target_components: np.ndarray,
        top_k: int,
        artifacts_path: Path,
        neat_config_path: str | Path | None = None,
    ) -> dict[str, object]:
        """Run the original neat-python algorithm."""

        import neat

        config_path = artifacts_path / "neat_config.ini"
        config_path.write_text(
            self._resolve_python_neat_config(neat_config_path=neat_config_path),
            encoding="utf-8",
        )

        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        population = neat.Population(neat_config)
        population.add_reporter(neat.StdOutReporter(True))
        statistics_reporter = neat.StatisticsReporter()
        population.add_reporter(statistics_reporter)
        population.add_reporter(neat.Checkpointer(generation_interval=10))

        hall_of_fame: list[dict[str, Any]] = []

        def eval_genomes(genomes, runtime_config) -> None:
            for genome_id, genome in genomes:
                network = neat.nn.FeedForwardNetwork.create(genome, runtime_config)
                wrapped = _PythonNeatCallableNetwork(network)
                diagnostics = self._evaluate_candidate(
                    wrapped,
                    properties_scaled=properties_scaled,
                    target_components=target_components,
                )
                n_isolated = self._count_isolated_outputs_pyneat(genome, runtime_config)
                if n_isolated > 0:
                    penalty = self.config.isolated_output_penalty * n_isolated / self.output_size
                    genome.fitness = 1.0 / (1.0 + diagnostics["objective"] + penalty)
                else:
                    genome.fitness = float(diagnostics["fitness"])
                hall_of_fame.append({
                    "genome_id": genome_id,
                    "genome": copy.deepcopy(genome),
                    **diagnostics,
                })

        population.run(eval_genomes, self.config.limit_generations)

        # Re-rank based on most recent evaluation (already populated above)
        hall_of_fame.sort(key=lambda item: float(item["objective"]))

        statistics = self._write_statistics_pyneat(statistics_reporter, artifacts_path)
        candidates = self._select_unique_candidates(hall_of_fame, top_k=top_k)

        vis_props_original = np.mean(properties_scaled, axis=0)
        vis_props_scaled = np.mean(properties_scaled, axis=0)
        visualizations: list[dict[str, str]] = []
        network_artifacts: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates, start=1):
            viz = self._write_visualization_pyneat(
                genome=candidate["genome"], neat_config=neat_config,
                output_prefix=artifacts_path / f"candidate_{index}",
                title=f"NEAT Candidate {index}",
                target_scaled=vis_props_scaled, target_original=vis_props_original,
                candidate=candidate,
            )
            candidate["visualization"] = viz
            visualizations.append(viz)

            artifact = self._write_network_artifact_pyneat(
                genome=candidate["genome"], neat_config=neat_config,
                output_prefix=artifacts_path / f"candidate_{index}",
                config_path=config_path, candidate=candidate,
            )
            candidate["network_artifact"] = artifact
            candidate.pop("genome", None)
            network_artifacts.append(artifact)

        return {
            "candidates": candidates,
            "statistics": statistics,
            "visualization_dir": str(artifacts_path),
            "visualizations": visualizations,
            "network_artifacts": network_artifacts,
        }

    # ======================================================================
    # NEATEST / BNEATEST path
    # ======================================================================

    def _estimate_activation_statistics_neatest(self, genome, target_scaled):
        sample_count = max(1, int(self.config.visualization_samples))
        sigma = max(0.0, float(self.config.visualization_input_sigma))
        rng = np.random.default_rng(self.config.seed)
        sampled_inputs = np.repeat(target_scaled[None, :], sample_count, axis=0)
        if sigma > 0.0:
            sampled_inputs += rng.normal(0.0, sigma, size=sampled_inputs.shape)
        genome_copy = genome.deepcopy()
        ordered_nodes = sorted(genome_copy.nodes, key=lambda n: n.depth)
        activations: dict[int, list[float]] = {int(n.id): [] for n in ordered_nodes}
        for sample in sampled_inputs:
            genome_copy(sample.tolist())
            for node in ordered_nodes:
                activations[int(node.id)].append(float(node.value))
        return {
            nid: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for nid, vals in activations.items()
        }

    @staticmethod
    def _build_edge_line_neatest(connection) -> str:
        weight_value = connection.weight.value
        color = "#15803d" if weight_value >= 0 else "#b91c1c"
        style = "solid" if connection.enabled else "dashed"
        penwidth = 0.8 + min(3.5, abs(weight_value))
        return (
            f'  "{connection.in_node.id}" -> "{connection.out_node.id}" '
            f'[label="{weight_value:.2f}", color="{color}", style="{style}", penwidth={penwidth:.2f}];'
        )

    def _dot_source_neatest(self, genome, title, target_scaled, target_original, candidate):
        input_nodes = [n for n in genome.nodes if n.type == NodeType.INPUT]
        output_nodes = list(genome.outputs)
        hidden_nodes = [n for n in genome.nodes if n.type == NodeType.HIDDEN]
        bias_nodes = [n for n in genome.nodes if n.type == NodeType.BIAS]
        activation_stats = self._estimate_activation_statistics_neatest(genome, target_scaled)

        lines = [
            "digraph neat_network {",
            '  graph [rankdir=LR, labelloc="t", labeljust="l"];',
            f'  label="{title}";',
            '  node [fontname="Helvetica", fontsize=10];',
            f'  "summary" [label="{self._summary_label(title, candidate)}", '
            'shape=note, style="filled", fillcolor="#fff7ed", color="#c2410c"];',
        ]
        for i, node in enumerate(input_nodes):
            name = self.input_names[i] if i < len(self.input_names) else f"input_{i}"
            val = float(target_original[i]) if i < len(target_original) else 0.0
            lines.append(self._build_input_node_line(node.id, name, val, activation_stats))

        for node in bias_nodes:
            stats = self._stats_for(node.id, activation_stats)
            label = self._html_table_label(
                title=f"bias_{node.id}", subtitle="bias=1.0",
                details=[self._distribution_text(stats["mean"], stats["std"])],
            )
            lines.append(
                f'  "{node.id}" [label={label}, shape=diamond, style="filled", fillcolor="#fef3c7", '
                f'color="#92400e", penwidth={self._node_penwidth(stats["std"]):.2f}];'
            )

        for i, node in enumerate(output_nodes):
            name = self.output_names[i] if i < len(self.output_names) else f"output_{i}"
            lines.append(self._build_output_node_line(node.id, name, i, activation_stats))

        for node in hidden_nodes:
            lines.append(self._build_hidden_node_line(node.id, activation_stats))

        for conn in genome.connections:
            lines.append(self._build_edge_line_neatest(conn))

        lines.append("}")
        return "\n".join(lines)

    def _write_visualization_neatest(self, genome, output_prefix, title, target_scaled, target_original, candidate):
        dot = self._dot_source_neatest(genome, title, target_scaled, target_original, candidate)
        return self._write_graphviz(dot, output_prefix)

    def _write_network_artifact_neatest(self, genome, output_prefix, config_path, candidate):
        genome_path = output_prefix.with_name(f"{output_prefix.name}_genome.pkl")
        metadata_path = output_prefix.with_name(f"{output_prefix.name}_network.json")
        with genome_path.open("wb") as handle:
            cloudpickle.dump(genome, handle)
        artifacts = {"genome": str(genome_path), "config": str(config_path)}
        metadata = {
            "algorithm": self.config.algorithm,
            "objective": float(candidate["objective"]),
            "fitness": float(candidate["fitness"]),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "config": str(config_path),
            "files": artifacts,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifacts["metadata"] = str(metadata_path)
        return artifacts

    def _write_statistics_neatest(self, statistics_reporter, artifacts_path):
        summary_path = artifacts_path / "neat_statistics.json"
        fitness_history_path = artifacts_path / "fitness_history.csv"
        generation_stats = statistics_reporter.generation_stats
        best_fitness_history = [gs.fitness_best for gs in generation_stats]
        mean_fitness_history = [gs.fitness_mean for gs in generation_stats]
        median_fitness_history = [gs.fitness_median for gs in generation_stats]
        stdev_fitness_history = [gs.fitness_stdev for gs in generation_stats]
        with fitness_history_path.open("w", encoding="utf-8") as f:
            f.write("generation,best,mean,median,stdev\n")
            for i, gs in enumerate(generation_stats):
                f.write(f"{i+1},{gs.fitness_best},{gs.fitness_mean},{gs.fitness_median},{gs.fitness_stdev}\n")
        best_fitness = max(best_fitness_history) if best_fitness_history else None
        summary = {
            "generation_count": len(generation_stats),
            "best_fitness": best_fitness,
            "best_fitness_history": best_fitness_history,
            "mean_fitness_history": mean_fitness_history,
            "median_fitness_history": median_fitness_history,
            "stdev_fitness_history": stdev_fitness_history,
            "files": {"fitness_history": str(fitness_history_path), "summary": str(summary_path)},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _build_neatest(self, agent, effective_config):
        hidden_activation = _resolve_activation(effective_config.hidden_activation)
        output_activation = _resolve_activation(effective_config.output_activation)

        if effective_config.algorithm == "bneatest":
            engine = BNEATEST(
                agent=agent,
                n_networks=effective_config.pop_size,
                es_population=effective_config.es_population,
                input_size=self.input_size,
                output_size=self.output_size,
                bias=effective_config.use_bias,
                sigma=effective_config.sigma,
                node_mutation_rate=effective_config.node_mutation_rate,
                connection_mutation_rate=effective_config.connection_mutation_rate,
                disable_connection_mutation_rate=effective_config.disable_connection_mutation_rate,
                dominant_gene_rate=effective_config.dominant_gene_rate,
                dominant_gene_delta=effective_config.dominant_gene_delta,
                seed=effective_config.seed,
                elite_rate=effective_config.elite_rate,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                optimizer_kwargs={"lr": effective_config.optimizer_lr},
                sigma_prior=effective_config.sigma_prior,
                kl_weight=effective_config.kl_weight,
                kl_warmup_steps=effective_config.kl_warmup_steps,
                initial_rho=effective_config.initial_rho,
                n_eval_samples=effective_config.n_eval_samples,
                risk_aversion=effective_config.risk_aversion,
            )
        else:
            engine = NEATEST(
                agent=agent,
                optimizer=Adam,
                n_networks=effective_config.pop_size,
                es_population=effective_config.es_population,
                input_size=self.input_size,
                output_size=self.output_size,
                bias=effective_config.use_bias,
                sigma=effective_config.sigma,
                node_mutation_rate=effective_config.node_mutation_rate,
                connection_mutation_rate=effective_config.connection_mutation_rate,
                disable_connection_mutation_rate=effective_config.disable_connection_mutation_rate,
                dominant_gene_rate=effective_config.dominant_gene_rate,
                dominant_gene_delta=effective_config.dominant_gene_delta,
                seed=effective_config.seed,
                elite_rate=effective_config.elite_rate,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                optimizer_kwargs={"lr": effective_config.optimizer_lr},
            )

        statistics_reporter = StatisticsReporter()
        engine.add_reporter(StdOutReporter())
        engine.add_reporter(statistics_reporter)
        return engine, statistics_reporter

    def _optimize_neatest(
        self,
        properties_scaled: np.ndarray,
        target_components: np.ndarray,
        top_k: int,
        artifacts_path: Path,
        effective_config: OptimizerConfig,
        neat_config_path: str | Path | None = None,
    ) -> dict[str, object]:
        """Run NEATEST or BNEATEST algorithm."""

        config_path = artifacts_path / "neat_config.ini"
        config_path.write_text(self._resolve_neat_config(neat_config_path=neat_config_path), encoding="utf-8")

        if effective_config.algorithm == "bneatest":
            agent = BayesianDatasetAgent(
                optimizer=self, properties_scaled=properties_scaled,
                target_components=target_components,
            )
        else:
            agent = DatasetAgent(
                optimizer=self, properties_scaled=properties_scaled,
                target_components=target_components,
            )

        engine, statistics_reporter = self._build_neatest(agent, effective_config)
        engine.train(n_steps=self.config.limit_generations)

        # Build hall of fame from final population + best genome
        hall_of_fame: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        all_genomes = list(engine.population)
        if hasattr(engine, 'best_genome') and engine.best_genome is not None:
            all_genomes.append(engine.best_genome)
        for genome in all_genomes:
            gid = id(genome)
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            diagnostics = self._evaluate_candidate(
                genome,
                properties_scaled=properties_scaled,
                target_components=target_components,
            )
            hall_of_fame.append({"genome_id": gid, "genome": genome.deepcopy(), **diagnostics})
        hall_of_fame.sort(key=lambda item: float(item["objective"]))

        statistics = self._write_statistics_neatest(statistics_reporter, artifacts_path)
        candidates = self._select_unique_candidates(hall_of_fame, top_k=top_k)

        vis_props_original = np.mean(properties_scaled, axis=0)
        vis_props_scaled = np.mean(properties_scaled, axis=0)
        visualizations: list[dict[str, str]] = []
        network_artifacts: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates, start=1):
            viz = self._write_visualization_neatest(
                genome=candidate["genome"],
                output_prefix=artifacts_path / f"candidate_{index}",
                title=f"NEAT Candidate {index}",
                target_scaled=vis_props_scaled, target_original=vis_props_original,
                candidate=candidate,
            )
            candidate["visualization"] = viz
            visualizations.append(viz)

            artifact = self._write_network_artifact_neatest(
                genome=candidate["genome"],
                output_prefix=artifacts_path / f"candidate_{index}",
                config_path=config_path, candidate=candidate,
            )
            candidate["network_artifact"] = artifact
            candidate.pop("genome", None)
            network_artifacts.append(artifact)

        return {
            "candidates": candidates,
            "statistics": statistics,
            "visualization_dir": str(artifacts_path),
            "visualizations": visualizations,
            "network_artifacts": network_artifacts,
        }

    # ======================================================================
    # Public entry point
    # ======================================================================

    def optimize(
        self,
        properties_scaled: np.ndarray,
        target_components: np.ndarray,
        top_k: int,
        artifacts_dir: str | Path,
        neat_config_path: str | Path | None = None,
    ) -> dict[str, object]:
        """Run the full evolutionary search and return ranked candidates plus artifacts."""

        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        # Determine effective algorithm from ini + json config
        effective_config = self._load_neat_ini(neat_config_path=neat_config_path)
        self.config = effective_config

        algorithm = effective_config.algorithm
        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Valid options: {', '.join(_VALID_ALGORITHMS)}"
            )

        if algorithm == "python-neat":
            return self._optimize_python_neat(
                properties_scaled, target_components, top_k,
                artifacts_path, neat_config_path=neat_config_path,
            )

        return self._optimize_neatest(
            properties_scaled, target_components, top_k,
            artifacts_path, effective_config, neat_config_path=neat_config_path,
        )
