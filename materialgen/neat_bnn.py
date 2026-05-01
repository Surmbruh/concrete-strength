from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.poutine as poutine
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.nn import PyroModule

from .neatest.node import NodeType, group_nodes
from .scaler import StandardScaler


def _is_python_neat_genome(genome) -> bool:
    """Detect whether the genome is from python-neat (vs NEATEST/BNEATEST)."""
    return hasattr(genome, 'connections') and isinstance(genome.connections, dict)


def _extract_topology_python_neat(genome, neat_config) -> dict[str, Any]:
    """Convert a python-neat genome into a layered topology description.

    ``neat_config`` is a ``neat.Config`` object needed to identify input/output keys
    and compute feed-forward layers.
    """
    import neat

    input_keys = list(neat_config.genome_config.input_keys)
    output_keys = list(neat_config.genome_config.output_keys)

    # Compute feed-forward layers using neat's graph utilities
    connections = {
        cg.key for cg in genome.connections.values() if cg.enabled
    }
    ff_result = neat.graphs.feed_forward_layers(input_keys, output_keys, connections)
    # feed_forward_layers may return (layers, required) tuple or just layers
    if isinstance(ff_result, tuple):
        ff_layers = ff_result[0]
    else:
        ff_layers = ff_result

    # Build full layer structure: layer 0 = inputs, then hidden layers, last = outputs
    all_layers: list[list[int]] = [list(input_keys)]
    for layer in ff_layers:
        all_layers.append(sorted(layer))

    # Ensure outputs are in the last layer
    if output_keys and set(output_keys) != set(all_layers[-1]):
        # Separate outputs from hidden nodes in the last layer
        last_hidden = [n for n in all_layers[-1] if n not in output_keys]
        last_output = [n for n in all_layers[-1] if n in output_keys]
        if last_hidden:
            all_layers[-1] = last_hidden
            all_layers.append(last_output)
        else:
            all_layers[-1] = last_output

    # Build enabled connection lookup
    connection_weights: dict[tuple[int, int], float] = {}
    connection_responses: dict[int, float] = {}
    connection_biases: dict[int, float] = {}
    for cg in genome.connections.values():
        if cg.enabled:
            connection_weights[cg.key] = float(cg.weight)
    for node_id, ng in genome.nodes.items():
        connection_responses[node_id] = float(ng.response)
        connection_biases[node_id] = float(ng.bias)

    weight_inits: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    bias_inits: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []

    for layer_idx in range(1, len(all_layers)):
        target_nodes = all_layers[layer_idx]
        source_pool: list[int] = []
        for prev_idx in range(layer_idx):
            source_pool.extend(all_layers[prev_idx])

        source_index = {nid: i for i, nid in enumerate(source_pool)}
        n_targets = len(target_nodes)
        n_sources = len(source_pool)

        w_init = torch.zeros(n_targets, n_sources)
        mask = torch.zeros(n_targets, n_sources)
        b_init = torch.zeros(n_targets)
        resp = torch.ones(n_targets)

        for t_idx, t_node in enumerate(target_nodes):
            b_init[t_idx] = connection_biases.get(t_node, 0.0)
            resp[t_idx] = connection_responses.get(t_node, 1.0)
            for s_node in source_pool:
                key = (s_node, t_node)
                if key in connection_weights:
                    s_idx = source_index[s_node]
                    w_init[t_idx, s_idx] = float(connection_weights[key])
                    mask[t_idx, s_idx] = 1.0

        weight_inits.append(w_init)
        masks.append(mask)
        bias_inits.append(b_init)
        responses.append(resp)

    return {
        "input_keys": input_keys,
        "output_keys": output_keys,
        "layers": all_layers,
        "weight_inits": weight_inits,
        "masks": masks,
        "bias_inits": bias_inits,
        "responses": responses,
    }


def _extract_topology(genome, neat_config=None) -> dict[str, Any]:
    """Convert a genome into a layered topology description.

    Supports both NEATEST/BNEATEST genomes and python-neat genomes.
    For python-neat genomes, ``neat_config`` (a ``neat.Config`` object) is required.

    Returns a dict with keys:
        input_keys:  ordered list of input node IDs (INPUT nodes only, not BIAS)
        output_keys: ordered list of output node IDs
        layers:      list of lists of node IDs (layer 0 = inputs+bias, then hidden, last = outputs)
        weight_inits: per-layer weight init tensors
        masks:        per-layer binary mask tensors
        bias_inits:   per-layer bias init tensors
        responses:    per-layer response multiplier tensors
    """
    if _is_python_neat_genome(genome):
        if neat_config is None:
            raise ValueError(
                "neat_config is required for python-neat genomes. "
                "Pass the neat.Config object to _extract_topology."
            )
        return _extract_topology_python_neat(genome, neat_config)

    # NEATEST / BNEATEST path
    # Group nodes by depth to get layers
    node_groups = group_nodes(genome.nodes, 'depth')

    # NOTE: bneatest defines its own NodeType enum separate from neatest.node.NodeType.
    # Direct enum comparison (n.type == NodeType.BIAS) fails across modules.
    # Compare by .name to handle both variants.
    def _is_bias(n) -> bool:
        return getattr(n.type, 'name', '') == 'BIAS'

    def _is_input(n) -> bool:
        return getattr(n.type, 'name', '') == 'INPUT'

    # Identify node types
    input_keys = [n.id for n in genome.nodes if _is_input(n)]
    output_keys = [n.id for n in genome.outputs]

    # Build layer structure: all nodes grouped by depth, excluding BIAS nodes
    all_layers: list[list[int]] = []
    for group in node_groups:
        non_bias = [n.id for n in group if not _is_bias(n)]
        if non_bias:
            all_layers.append(non_bias)

    # Map node IDs to layer index
    node_to_layer: dict[int, int] = {}
    for layer_idx, nodes in enumerate(all_layers):
        for node_id in nodes:
            node_to_layer[node_id] = layer_idx

    # Build enabled connection lookup; fold BIAS node connections into bias_contributions
    connection_weights: dict[tuple[int, int], float] = {}
    bias_contributions: dict[int, float] = {}  # target node id -> accumulated bias weight
    for conn in genome.connections:
        if conn.enabled:
            if _is_bias(conn.in_node):
                t_id = conn.out_node.id
                bias_contributions[t_id] = bias_contributions.get(t_id, 0.0) + conn.weight.value
            else:
                key = (conn.in_node.id, conn.out_node.id)
                connection_weights[key] = conn.weight.value

    weight_inits: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    bias_inits: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []

    for layer_idx in range(1, len(all_layers)):
        target_nodes = all_layers[layer_idx]
        source_pool: list[int] = []
        for prev_idx in range(layer_idx):
            source_pool.extend(all_layers[prev_idx])

        source_index = {nid: i for i, nid in enumerate(source_pool)}
        n_targets = len(target_nodes)
        n_sources = len(source_pool)

        w_init = torch.zeros(n_targets, n_sources)
        mask = torch.zeros(n_targets, n_sources)
        b_init = torch.zeros(n_targets)
        resp = torch.ones(n_targets)

        for t_idx, t_node in enumerate(target_nodes):
            # Bias comes from BIAS node connections folded into bias_contributions.
            b_init[t_idx] = bias_contributions.get(t_node, 0.0)
            for s_node in source_pool:
                key = (s_node, t_node)
                if key in connection_weights:
                    s_idx = source_index[s_node]
                    w_init[t_idx, s_idx] = float(connection_weights[key])
                    mask[t_idx, s_idx] = 1.0

        weight_inits.append(w_init)
        masks.append(mask)
        bias_inits.append(b_init)
        responses.append(resp)

    return {
        "input_keys": input_keys,
        "output_keys": output_keys,
        "layers": all_layers,
        "weight_inits": weight_inits,
        "masks": masks,
        "bias_inits": bias_inits,
        "responses": responses,
    }


class _NeatBayesianNetwork(PyroModule):
    """Pyro module whose topology mirrors a NEAT feed-forward genome.

    Each layer is a masked dense linear transformation with Bayesian weights.
    Non-existent connections are zeroed out via a fixed binary mask so they
    never contribute to the output regardless of the sampled weight value.
    """

    def __init__(
        self,
        topology: dict[str, Any],
        prior_std: float,
        kl_weight: float,
        bounds_lower: torch.Tensor,
        bounds_upper: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_layers = len(topology["weight_inits"])
        self.prior_std = float(prior_std)
        self.kl_weight = float(kl_weight)
        self.output_dim = len(topology["output_keys"])

        self.register_buffer("bounds_lower", bounds_lower.float())
        self.register_buffer("bounds_upper", bounds_upper.float())

        all_node_order: list[int] = []
        for layer in topology["layers"]:
            all_node_order.extend(layer)
        output_indices = [all_node_order.index(ok) for ok in topology["output_keys"]]
        self.register_buffer("output_indices", torch.tensor(output_indices, dtype=torch.long))

        for k in range(self.n_layers):
            self.register_buffer(f"weight_init_{k}", topology["weight_inits"][k].float())
            self.register_buffer(f"mask_{k}", topology["masks"][k].float())
            self.register_buffer(f"bias_init_{k}", topology["bias_inits"][k].float())
            self.register_buffer(f"response_{k}", topology["responses"][k].float())

    def _decode_components(self, tanh_out: torch.Tensor) -> torch.Tensor:
        """Convert tanh outputs from [-1, 1] to component space."""

        scaled = (tanh_out + 1.0) / 2.0
        return self.bounds_lower + scaled * (self.bounds_upper - self.bounds_lower)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Run the forward pass and optionally attach the observation likelihood."""

        layers_data = []
        for k in range(self.n_layers):
            layers_data.append({
                "weight_init": getattr(self, f"weight_init_{k}"),
                "mask": getattr(self, f"mask_{k}"),
                "bias_init": getattr(self, f"bias_init_{k}"),
                "response": getattr(self, f"response_{k}"),
            })

        all_activations = x

        with poutine.scale(scale=self.kl_weight):
            for k, ld in enumerate(layers_data):
                weight = pyro.sample(
                    f"layer_{k}.weight",
                    dist.Normal(ld["weight_init"], self.prior_std).to_event(2),
                )
                bias = pyro.sample(
                    f"layer_{k}.bias",
                    dist.Normal(ld["bias_init"], self.prior_std).to_event(1),
                )
                masked_weight = weight * ld["mask"]
                pre_act = bias + ld["response"] * (all_activations @ masked_weight.T)
                layer_out = torch.tanh(pre_act)
                all_activations = torch.cat([all_activations, layer_out], dim=1)

        output_activations = all_activations[:, self.output_indices]
        decoded = self._decode_components(output_activations)

        obs_scale = pyro.param(
            "obs_scale",
            0.1 * torch.ones(self.output_dim, dtype=decoded.dtype, device=decoded.device),
            constraint=constraints.positive,
        )
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(decoded, obs_scale).to_event(1), obs=y)

        return decoded


class NeatBNNRegressor:
    """Wrapper around the NEAT-topology Bayesian network.

    Handles scaling, training (SVI), Monte Carlo prediction, and persistence.
    """

    def __init__(
        self,
        topology: dict[str, Any],
        prior_std: float = 0.5,
        posterior_scale_init: float = 0.05,
        kl_weight: float = 0.01,
        seed: int = 420,
        bounds_lower: np.ndarray | None = None,
        bounds_upper: np.ndarray | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        self.topology = topology
        self.prior_std = float(prior_std)
        self.posterior_scale_init = float(posterior_scale_init)
        self.kl_weight = float(kl_weight)
        self.seed = int(seed)
        self.device = torch.device("cpu")

        self.bounds_lower = np.asarray(bounds_lower, dtype=float) if bounds_lower is not None else np.zeros(0)
        self.bounds_upper = np.asarray(bounds_upper, dtype=float) if bounds_upper is not None else np.zeros(0)
        self.input_names = input_names or []
        self.output_names = output_names or []

        self.property_scaler: StandardScaler | None = None

        self.model = _NeatBayesianNetwork(
            topology=topology,
            prior_std=self.prior_std,
            kl_weight=self.kl_weight,
            bounds_lower=torch.as_tensor(self.bounds_lower, dtype=torch.float32),
            bounds_upper=torch.as_tensor(self.bounds_upper, dtype=torch.float32),
        ).to(self.device)

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def guide(self, x: torch.Tensor, y: torch.Tensor | None = None) -> None:
        """Variational posterior: Normal(loc, scale) for each weight and bias."""

        init_scale = max(self.posterior_scale_init, 1e-4)
        with poutine.scale(scale=self.kl_weight):
            for k in range(self.model.n_layers):
                w_init = getattr(self.model, f"weight_init_{k}")
                b_init = getattr(self.model, f"bias_init_{k}")

                weight_loc = pyro.param(f"layer_{k}.weight_loc", w_init.clone())
                weight_scale = pyro.param(
                    f"layer_{k}.weight_scale",
                    torch.full_like(w_init, init_scale),
                    constraint=constraints.positive,
                )
                bias_loc = pyro.param(f"layer_{k}.bias_loc", b_init.clone())
                bias_scale = pyro.param(
                    f"layer_{k}.bias_scale",
                    torch.full_like(b_init, init_scale),
                    constraint=constraints.positive,
                )
                pyro.sample(f"layer_{k}.weight", dist.Normal(weight_loc, weight_scale).to_event(2))
                pyro.sample(f"layer_{k}.bias", dist.Normal(bias_loc, bias_scale).to_event(1))

    def fit(
        self,
        properties: np.ndarray,
        components: np.ndarray,
        *,
        learning_rate: float = 0.005,
        epochs: int = 300,
        batch_size: int = 32,
        validation_split: float = 0.2,
        mc_samples: int = 30,
        early_stopping_rounds: int = 30,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Train the BNN using SVI with Trace_ELBO."""

        used_seed = seed if seed is not None else self.seed
        pyro.set_rng_seed(used_seed)
        pyro.clear_param_store()

        self.model = _NeatBayesianNetwork(
            topology=self.topology,
            prior_std=self.prior_std,
            kl_weight=self.kl_weight,
            bounds_lower=torch.as_tensor(self.bounds_lower, dtype=torch.float32),
            bounds_upper=torch.as_tensor(self.bounds_upper, dtype=torch.float32),
        ).to(self.device)

        self.property_scaler = StandardScaler.fit(properties)
        x_scaled = self.property_scaler.transform(properties)

        rng = np.random.default_rng(used_seed)
        indices = rng.permutation(x_scaled.shape[0])
        val_size = max(1, int(x_scaled.shape[0] * validation_split))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        if train_idx.size == 0:
            train_idx = val_idx

        x_train = x_scaled[train_idx]
        y_train = components[train_idx]
        x_val = x_scaled[val_idx]
        y_val = components[val_idx]

        optimizer = pyro.optim.Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        epoch_train_losses: list[float] = []
        epoch_val_losses: list[float] = []

        component_scale = np.maximum(self.bounds_upper - self.bounds_lower, 1.0)

        for epoch in range(epochs):
            perm = rng.permutation(x_train.shape[0])
            x_epoch = x_train[perm]
            y_epoch = y_train[perm]

            for start in range(0, x_epoch.shape[0], batch_size):
                stop = start + batch_size
                svi.step(self._to_tensor(x_epoch[start:stop]), self._to_tensor(y_epoch[start:stop]))

            train_pred, _ = self.predict_components(properties[train_idx], mc_samples=mc_samples)
            train_loss = float(np.mean(((train_pred - components[train_idx]) / component_scale) ** 2))
            epoch_train_losses.append(train_loss)

            val_pred, _ = self.predict_components(properties[val_idx], mc_samples=mc_samples)
            val_loss = float(np.mean(((val_pred - components[val_idx]) / component_scale) ** 2))
            epoch_val_losses.append(val_loss)

            if val_loss + 1e-8 < best_val_loss:
                best_val_loss = val_loss
                best_state = pyro.get_param_store().get_state()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_rounds:
                    break

        if best_state is not None:
            pyro.get_param_store().set_state(best_state)

        return {
            "train_loss": epoch_train_losses[-1] if epoch_train_losses else 0.0,
            "validation_loss": float(best_val_loss),
            "epochs_run": len(epoch_train_losses),
            "epoch_train_losses": epoch_train_losses,
            "epoch_val_losses": epoch_val_losses,
        }

    def predict_components(
        self,
        properties: np.ndarray,
        mc_samples: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_components, std_components) in real component space."""

        samples = self.sample_components(properties, mc_samples=mc_samples)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return mean, std

    def sample_components(
        self,
        properties: np.ndarray,
        mc_samples: int = 30,
    ) -> np.ndarray:
        """Return raw MC component samples in real component space.

        Returns array of shape ``(mc_samples, n_points, n_components)``.
        """

        if self.property_scaler is None:
            raise RuntimeError("Model must be fitted before prediction")

        x_scaled = self.property_scaler.transform(np.asarray(properties, dtype=float))
        if x_scaled.ndim == 1:
            x_scaled = x_scaled[None, :]

        x_tensor = self._to_tensor(x_scaled)
        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=mc_samples,
            return_sites=["_RETURN"],
        )
        with torch.no_grad():
            samples = predictive(x_tensor)["_RETURN"]

        return samples.cpu().numpy()

    def save(self, path: str | Path) -> None:
        """Persist the trained model to a Torch checkpoint."""

        if self.property_scaler is None:
            raise RuntimeError("Cannot save an unfitted model")

        serializable_topology = {
            "input_keys": self.topology["input_keys"],
            "output_keys": self.topology["output_keys"],
            "layers": self.topology["layers"],
            "weight_inits": [t.cpu() for t in self.topology["weight_inits"]],
            "masks": [t.cpu() for t in self.topology["masks"]],
            "bias_inits": [t.cpu() for t in self.topology["bias_inits"]],
            "responses": [t.cpu() for t in self.topology["responses"]],
        }

        checkpoint = {
            "topology": serializable_topology,
            "prior_std": self.prior_std,
            "posterior_scale_init": self.posterior_scale_init,
            "kl_weight": self.kl_weight,
            "seed": self.seed,
            "bounds_lower": self.bounds_lower.tolist(),
            "bounds_upper": self.bounds_upper.tolist(),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "property_scaler": self.property_scaler.to_dict(),
            "param_store_state": pyro.get_param_store().get_state(),
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "NeatBNNRegressor":
        """Reconstruct a trained model from a checkpoint."""

        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        topology = checkpoint["topology"]

        regressor = cls(
            topology=topology,
            prior_std=float(checkpoint["prior_std"]),
            posterior_scale_init=float(checkpoint["posterior_scale_init"]),
            kl_weight=float(checkpoint["kl_weight"]),
            seed=int(checkpoint.get("seed", 42)),
            bounds_lower=np.asarray(checkpoint["bounds_lower"], dtype=float),
            bounds_upper=np.asarray(checkpoint["bounds_upper"], dtype=float),
            input_names=checkpoint.get("input_names", []),
            output_names=checkpoint.get("output_names", []),
        )
        regressor.property_scaler = StandardScaler.from_dict(checkpoint["property_scaler"])

        pyro.clear_param_store()
        regressor.model = _NeatBayesianNetwork(
            topology=topology,
            prior_std=regressor.prior_std,
            kl_weight=regressor.kl_weight,
            bounds_lower=torch.as_tensor(regressor.bounds_lower, dtype=torch.float32),
            bounds_upper=torch.as_tensor(regressor.bounds_upper, dtype=torch.float32),
        ).to(regressor.device)
        pyro.get_param_store().set_state(checkpoint["param_store_state"])
        return regressor


def build_regressor_from_genome(
    genome,
    *,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    prior_std: float = 0.5,
    posterior_scale_init: float = 0.05,
    kl_weight: float = 0.01,
    seed: int = 42,
    neat_config=None,
) -> NeatBNNRegressor:
    """Convenience factory: genome -> ready-to-train regressor.

    For python-neat genomes, pass ``neat_config`` (a ``neat.Config`` object).
    """

    topology = _extract_topology(genome, neat_config=neat_config)
    return NeatBNNRegressor(
        topology=topology,
        prior_std=prior_std,
        posterior_scale_init=posterior_scale_init,
        kl_weight=kl_weight,
        seed=seed,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        input_names=input_names,
        output_names=output_names,
    )
