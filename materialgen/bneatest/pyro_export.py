"""Export a trained BayesianGenome to a Pyro probabilistic model."""
from typing import List

import torch
import pyro
import pyro.distributions as dist
import pyro.nn as pnn

from .genome import Genome
from .node import NodeType, group_nodes


class BayesianNetModel(pnn.PyroModule):
    """Pyro probabilistic model converted from a trained BNEATEST genome.

    Each connection weight is represented as a PyroSample with a Normal prior
    parameterized by the trained mu and sigma from the genome.
    """

    def __init__(self, genome: Genome):
        super().__init__()
        self.genome_nodes = []
        self.genome_structure = []

        # Store node info
        sorted_nodes = sorted(genome.nodes, key=lambda n: n.depth)
        for node in sorted_nodes:
            self.genome_nodes.append({
                'id': node.id,
                'type': node.type,
                'activation': node.activation,
                'depth': node.depth,
            })

        # Store connection structure and register weight priors
        self._weight_names = []
        for conn in genome.connections:
            if conn.enabled:
                name = f'w_{conn.innovation}'
                self._weight_names.append(name)
                self.genome_structure.append({
                    'in_node_id': conn.in_node.id,
                    'out_node_id': conn.out_node.id,
                    'innovation': conn.innovation,
                    'weight_name': name,
                })
                # Register as PyroSample with learned posterior as prior
                prior = dist.Normal(
                    torch.tensor(conn.weight.mu.item()),
                    torch.tensor(conn.weight.sigma.item()))
                setattr(self, name, pnn.PyroSample(prior))

        # Build adjacency: out_node_id -> list of (in_node_id, weight_name)
        self._adjacency = {}
        for s in self.genome_structure:
            out_id = s['out_node_id']
            if out_id not in self._adjacency:
                self._adjacency[out_id] = []
            self._adjacency[out_id].append((s['in_node_id'], s['weight_name']))

        # Identify input and output nodes
        grouped = group_nodes(genome.nodes, 'type')
        self._input_ids = [n.id for n in grouped[0]]
        self._output_ids = [n.id for n in grouped[-1]]
        self._bias_ids = [n.id for n in genome.nodes if n.type == NodeType.BIAS]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the evolved Bayesian network.

        Args:
            inputs: Tensor of shape (input_size,)

        Returns:
            Tensor of shape (output_size,)
        """
        values = {}

        # Set input values
        for i, node_id in enumerate(self._input_ids):
            values[node_id] = inputs[i]

        # Set bias values
        for node_id in self._bias_ids:
            values[node_id] = torch.tensor(1.0)

        # Forward pass in topological order
        for node_info in self.genome_nodes:
            nid = node_info['id']
            if node_info['type'] in (NodeType.INPUT, NodeType.BIAS):
                continue

            value = torch.tensor(0.0)
            if nid in self._adjacency:
                for in_id, weight_name in self._adjacency[nid]:
                    w = getattr(self, weight_name)
                    if in_id in values:
                        value = value + values[in_id] * w
            values[nid] = node_info['activation'](value.item())

        outputs = torch.stack([
            torch.tensor(values[nid]) if not isinstance(values.get(nid), torch.Tensor)
            else values[nid]
            for nid in self._output_ids
        ])
        return outputs


def genome_to_pyro_model(genome: Genome) -> BayesianNetModel:
    """Convert a trained BayesianGenome to a Pyro probabilistic model.

    The resulting model can be used with Pyro's inference tools
    (Predictive, MCMC, SVI, etc.) for downstream Bayesian analysis.

    Args:
        genome: A trained BayesianGenome from BNEATEST.

    Returns:
        A PyroModule that can be used with Pyro inference.
    """
    return BayesianNetModel(genome)
