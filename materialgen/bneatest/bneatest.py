#!/usr/bin/env python3
import random
from typing import Union, List, Callable, Tuple, NewType, Dict, Type, Sequence, Optional
import itertools
import functools
from abc import ABC, abstractmethod
import operator
import os
import sys
import time
import pathlib
import inspect

import torch
from torch.distributions import Normal, kl_divergence

try:
    disable_mpi = os.environ.get('BNEATEST_DISABLE_MPI')
    if disable_mpi and disable_mpi != '0':
        raise ImportError
    from mpi4py import MPI  # type: ignore
except ImportError:
    from .MPI import MPI
    MPI = MPI()
import numpy as np
import cloudpickle  # type: ignore

from .genome import Genome
from .node import Node, NodeType, passthrough
from .connection import Connection, GeneRate, DummyConnection, align_connections
from .weight import BayesianWeight
from .optimizers import BayesianOptimizer
from .reporting import ReporterSet, GenerationStats, BaseReporter
from .version import VERSION
from tqdm import tqdm



@functools.lru_cache(maxsize=1)
def _center_function(population_size: int) -> np.ndarray:
    centers = np.arange(0, population_size)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers


def _compute_ranks(rewards: Union[List[float], np.ndarray]) -> np.ndarray:
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks


def rank_transformation(rewards: Union[List[float], np.ndarray]) -> np.ndarray:
    ranks = _compute_ranks(rewards)
    values = _center_function(len(rewards))
    return values[ranks]  # type: ignore


class ContextGenome(Genome):
    '''Genome class that holds data which depends on the context.'''
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.fitness: float = 0.0
        self.generation: int = 1
        super().__init__(nodes, connections)

    def copy(self) -> 'ContextGenome':  # type: ignore
        new_genome = super(ContextGenome, self).copy()
        return ContextGenome(new_genome.nodes, new_genome.connections)

    def deepcopy(self) -> 'ContextGenome':  # type: ignore
        new_genome = super(ContextGenome, self).deepcopy()
        return ContextGenome(new_genome.nodes, new_genome.connections)


SortedContextGenomes = NewType('SortedContextGenomes', List[ContextGenome])


class Agent(ABC):
    @abstractmethod
    def rollout(self, genome: Genome) -> float:
        ...


class BNEATEST(object):
    def __init__(self,
                 agent: Agent,
                 n_networks: int,
                 es_population: int,
                 input_size: int,
                 output_size: int,
                 bias: bool,
                 node_mutation_rate: float,
                 connection_mutation_rate: float,
                 disable_connection_mutation_rate: float,
                 dominant_gene_rate: float,
                 dominant_gene_delta: float,
                 seed: int,
                 save_checkpoint_n: int = 50,
                 logdir: str = None,
                 optimizer_kwargs: dict = {},
                 hidden_layers: List[int] = [],
                 elite_rate: float = 0.0,
                 sigma: float = 0.01,
                 hidden_activation: Callable[[float], float] = passthrough,
                 output_activation: Callable[[float], float] = passthrough,
                 # Bayesian-specific parameters
                 sigma_prior: float = 1.0,
                 kl_weight: float = 0.01,
                 kl_warmup_steps: int = 0,
                 initial_rho: float = -3.0,
                 n_eval_samples: int = 1,
                 risk_aversion: float = 0.0):

        comm = MPI.COMM_WORLD
        self.version = VERSION
        n_proc = comm.Get_size()
        self.n_proc = n_proc
        self.seed = seed
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.logdir = logdir
        self.agent = agent
        self.es_population = es_population
        self.n_networks = n_networks
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.bias = bias
        self.sigma = sigma
        self.save_checkpoint_n = save_checkpoint_n
        self.elite_rate = elite_rate
        self.disable_connection_mutation_rate = disable_connection_mutation_rate
        self.node_mutation_rate = node_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.dominant_gene_rate = dominant_gene_rate
        self.dominant_gene_delta = dominant_gene_delta
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Bayesian parameters
        self.sigma_prior = sigma_prior
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.initial_rho = initial_rho
        self.n_eval_samples = n_eval_samples
        self.risk_aversion = risk_aversion

        self.weights: List[BayesianWeight] = []
        self.gene_rates: List[GeneRate] = []
        self.connections: Dict[Connection, int] = {}
        self.node_id_generator = itertools.count(0, 1)
        self.connection_id_generator = itertools.count(0, 1)

        self.optimizer: BayesianOptimizer = BayesianOptimizer(
            self.weights, **optimizer_kwargs)

        self.generation: int = 1
        self.best_fitness: float = -float('inf')
        self.best_genome: ContextGenome
        self.population: List[ContextGenome]

        self.data: List[Tuple[int, str, int, float]] = []
        self.reporters = ReporterSet()

        if not comm.rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f
        else:
            if self.logdir:
                from pathlib import Path
                Path(self.logdir).mkdir(parents=True, exist_ok=True)

        self.population = self.create_population()

    def _effective_kl_weight(self) -> float:
        """KL weight with optional linear warmup."""
        if self.kl_warmup_steps <= 0:
            return self.kl_weight
        progress = min(1.0, self.generation / self.kl_warmup_steps)
        return self.kl_weight * progress

    def add_reporter(self, reporter: BaseReporter) -> None:
        """Add a reporter for training logging/statistics."""
        self.reporters.add(reporter)

    def _compute_bayesian_stats(self) -> Tuple[float, float]:
        """Compute mean and max sigma across all weights."""
        if not self.weights:
            return 0.0, 0.0
        sigmas = [w.sigma.item() for w in self.weights]
        return float(np.mean(sigmas)), float(np.max(sigmas))

    def random_genome(self) -> ContextGenome:
        '''Create fc neural network with random Bayesian weights.'''
        layers = []
        connections: List[Connection] = []

        # Input Nodes
        input_nodes = [Node(next(self.node_id_generator), NodeType.INPUT, depth=0.0)
                       for _ in range(self.input_size)]
        if self.bias:
            input_nodes.append(Node(next(self.node_id_generator), NodeType.BIAS,
                                    value=1.0, depth=0.0))
        layers.append(input_nodes)

        # Hidden Nodes
        for idx, hidden_layer in enumerate(self.hidden_layers):
            depth = 1 / (len(self.hidden_layers) + 1) * (idx + 1)
            hidden_nodes = [Node(next(self.node_id_generator),
                                 NodeType.HIDDEN,
                                 self.hidden_activation,
                                 depth=depth)
                            for _ in range(hidden_layer)]
            if self.bias:
                hidden_nodes.append(
                    Node(next(self.node_id_generator),
                         NodeType.BIAS, value=1.0, depth=depth))
            layers.append(hidden_nodes)

        # Output Nodes
        output_nodes = [Node(next(self.node_id_generator), NodeType.OUTPUT,
                             self.output_activation, depth=1.0)
                        for _ in range(self.output_size)]
        layers.append(output_nodes)

        for i in range(1, len(layers)):
            for j in range(len(layers[i-1])):
                input_node = layers[i-1][j]
                for k in range(len(layers[i])):
                    output_node = layers[i][k]
                    dummy_connection = DummyConnection(input_node, output_node)
                    innovation, weight, dominant_gene_rate = self.register_connection(
                        dummy_connection)

                    connections += [Connection(
                        in_node=input_node, out_node=output_node,
                        innovation=innovation,
                        dominant_gene_rate=dominant_gene_rate,
                        weight=weight)]

        nodes: List[Node] = functools.reduce(operator.iconcat, layers, [])
        return ContextGenome(nodes, connections)

    def create_population(self) -> List[ContextGenome]:
        population: List[ContextGenome] = [self.random_genome()]
        for _ in range(self.n_networks - 1):
            population.append(population[0].copy())
        return population

    def register_connection(self, dummy_connection: DummyConnection):
        if dummy_connection in self.connections:
            innovation = self.connections[dummy_connection]
            weight = self.weights[innovation]
            dominant_gene_rate = self.gene_rates[innovation]
        else:
            innovation = next(self.connection_id_generator)
            weight = BayesianWeight(
                mu=self.random.gauss(0.0, self.sigma),
                rho=self.initial_rho)
            dominant_gene_rate = GeneRate(self.dominant_gene_rate)
            self.weights.append(weight)
            self.gene_rates.append(dominant_gene_rate)
            self.connections[dummy_connection] = innovation
            self.optimizer.add_weight(weight)
        return innovation, weight, dominant_gene_rate

    def add_connection_mutation(self, genome: Genome) -> None:
        '''Create new connection between two random non-connected nodes.'''
        def _add_connection_mutation(depth=0):
            if depth > 20:
                return
            in_idx = self.random.randint(0, len(genome.nodes) - 1)
            in_node = genome.nodes[in_idx]
            out_idx = self.random.randint(0, len(genome.nodes) - 1)
            out_node = genome.nodes[out_idx]
            dummy_connection = DummyConnection(in_node, out_node)
            try:
                index = genome.connections.index(dummy_connection)
                if not genome.connections[index].enabled:
                    if (self.random.random() <=
                            genome.connections[index].dominant_gene_rate.value):
                        genome.connections[index].enabled = True
                        return
                else:
                    _add_connection_mutation(depth+1)
                    return
            except ValueError:
                pass

            if (out_node.type == NodeType.BIAS or
                    out_node.type == NodeType.INPUT or
                    in_node.type == NodeType.OUTPUT):
                _add_connection_mutation(depth+1)
                return
            innovation, weight, dominant_gene_rate = self.register_connection(
                dummy_connection)

            connection = Connection(in_node=in_node, out_node=out_node,
                                    innovation=innovation,
                                    dominant_gene_rate=dominant_gene_rate,
                                    weight=weight)
            genome.connections.append(connection)
        _add_connection_mutation()

    def add_node_mutation(self, genome: Genome,
                          activation: Callable[[float], float] = lambda x: x) -> None:
        '''Add a node to a random connection and split the connection.'''
        idx = self.random.randint(0, len(genome.connections)-1)
        genome.connections[idx].enabled = False

        new_node = Node(max(genome.nodes) + 1, NodeType.HIDDEN, activation)

        # First connection: in -> new_node with near-identity weight
        first_weight = BayesianWeight(mu=0.0, rho=self.initial_rho)
        first_gene_rate = GeneRate(self.dominant_gene_rate)
        first_innovation = next(self.connection_id_generator)
        first_connection = Connection(in_node=genome.connections[idx].in_node,
                                      out_node=new_node,
                                      innovation=first_innovation,
                                      dominant_gene_rate=first_gene_rate,
                                      weight=first_weight)
        self.weights.append(first_weight)
        self.gene_rates.append(first_gene_rate)
        self.connections[first_connection] = first_innovation
        self.optimizer.add_weight(first_weight)

        # Second connection: new_node -> out with mu from split connection
        second_weight = BayesianWeight(
            mu=genome.connections[idx].weight.mu.item(),
            rho=self.initial_rho)
        second_gene_rate = GeneRate(self.dominant_gene_rate)
        second_innovation = next(self.connection_id_generator)
        second_connection = Connection(in_node=new_node,
                                       out_node=genome.connections[idx].out_node,
                                       innovation=second_innovation,
                                       dominant_gene_rate=second_gene_rate,
                                       weight=second_weight)
        self.weights.append(second_weight)
        self.gene_rates.append(second_gene_rate)
        self.connections[second_connection] = second_innovation
        self.optimizer.add_weight(second_weight)

        genome.connections.append(first_connection)
        genome.connections.append(second_connection)
        new_node.depth = (first_connection.in_node.depth +
                          second_connection.out_node.depth) / 2
        genome.nodes.append(new_node)

    def disable_connection_mutation(self, genome: Genome) -> None:
        def _disable_connection_mutation(depth=0):
            if depth > 20:
                return
            idx = self.random.randint(0, len(genome.connections)-1)
            if (genome.connections[idx].out_node.type == NodeType.OUTPUT or
                    genome.connections[idx].in_node.type == NodeType.INPUT or
                    genome.connections[idx].in_node.type == NodeType.BIAS):
                _disable_connection_mutation(depth + 1)
                return
            else:
                if not genome.connections[idx].enabled:
                    _disable_connection_mutation(depth + 1)
                    return
                else:
                    genome.connections[idx].enabled = False
                    return
        _disable_connection_mutation()

    def next_generation(self, sorted_population: SortedContextGenomes):
        population: List[ContextGenome]
        if self.elite_rate > 0.0:
            population = sorted_population[0:int(self.n_networks * self.elite_rate)]
        else:
            population = []

        self.generation += 1
        while len(population) < self.n_networks:
            genome_1 = self.get_random_genome()
            genome_2 = self.get_random_genome()
            new_genome = self.crossover(genome_1, genome_2)
            if self.random.random() < self.disable_connection_mutation_rate:
                self.disable_connection_mutation(new_genome)
            if self.random.random() < self.node_mutation_rate:
                self.add_node_mutation(
                    new_genome,
                    activation=self.hidden_activation)
            if self.random.random() < self.connection_mutation_rate:
                self.add_connection_mutation(
                    new_genome)
            new_genome.generation = self.generation
            population.append(new_genome)

        self.population = population

    def calculate_grads(self, genome: ContextGenome):
        """Hybrid gradient estimation: NES for reward + analytical KL divergence."""
        comm = MPI.COMM_WORLD

        # Deep copy genome for perturbation evaluation
        cp_genome: ContextGenome = genome.deepcopy()  # type: ignore

        # Remove disabled connections
        for i in reversed(range(len(cp_genome.connections))):
            if not cp_genome.connections[i].enabled:
                del cp_genome.connections[i]

        n_conns = len(cp_genome.connections)
        if n_conns == 0:
            return

        # Extract current mu and rho values
        mus = np.array([c.weight.mu.item() for c in cp_genome.connections])
        rhos = np.array([c.weight.rho.item() for c in cp_genome.connections])

        # Generate perturbations for both mu and rho (antithetic sampling)
        eps_mu = self.np_random.normal(0.0, self.sigma, (self.es_population // 2, n_conns))
        eps_rho = self.np_random.normal(0.0, self.sigma, (self.es_population // 2, n_conns))
        eps_mu = np.concatenate([eps_mu, -eps_mu])
        eps_rho = np.concatenate([eps_rho, -eps_rho])

        # Evaluate perturbed networks (distributed via MPI)
        n_jobs = self.es_population // comm.Get_size()
        rewards: List[float] = []
        rewards_array: np.ndarray = np.zeros(self.es_population, dtype='d')

        for i in range(comm.rank * n_jobs, n_jobs * (comm.rank + 1)):
            for j, c in enumerate(cp_genome.connections):
                c.weight.mu.data.fill_(mus[j] + eps_mu[i, j])
                c.weight.rho.data.fill_(rhos[j] + eps_rho[i, j])
            rewards.append(self.agent.rollout(cp_genome))

        comm.Allgatherv([np.array(rewards, dtype=np.float64), MPI.DOUBLE],
                        [rewards_array, MPI.DOUBLE])

        # Rank-transform rewards and compute NES gradients
        ranked_rewards = rank_transformation(rewards_array)
        mu_grads = np.dot(ranked_rewards, eps_mu) / (self.es_population * self.sigma)
        rho_grads = np.dot(ranked_rewards, eps_rho) / (self.es_population * self.sigma)
        mu_grads = np.clip(mu_grads, -1.0, 1.0)
        rho_grads = np.clip(rho_grads, -1.0, 1.0)

        # Compute KL divergence gradient via PyTorch autograd
        effective_kl = self._effective_kl_weight()
        if effective_kl > 0:
            prior = Normal(torch.tensor(0.0), torch.tensor(self.sigma_prior))
            kl_total = torch.tensor(0.0)
            enabled_weights = []
            for c in genome.connections:
                if c.enabled:
                    enabled_weights.append(c.weight)
                    kl_total = kl_total + kl_divergence(c.weight.distribution, prior)
            kl_total.backward()

        # Update gene rates
        for gene_rate in self.gene_rates:
            gene_rate.value -= self.dominant_gene_delta

        # Combine NES + KL gradients and assign to weight parameters
        idx = 0
        for connection in genome.connections:
            if connection.enabled:
                w = connection.weight
                # NES gradient (negative because we maximize reward)
                nes_mu_grad = -mu_grads[idx]
                nes_rho_grad = -rho_grads[idx]

                # Add KL gradient if applicable
                if effective_kl > 0 and w.mu.grad is not None:
                    kl_mu_grad = w.mu.grad.item()
                    kl_rho_grad = w.rho.grad.item() if w.rho.grad is not None else 0.0
                    w.mu.grad = torch.tensor(
                        nes_mu_grad + effective_kl * kl_mu_grad,
                        dtype=torch.float32)
                    w.rho.grad = torch.tensor(
                        nes_rho_grad + effective_kl * kl_rho_grad,
                        dtype=torch.float32)
                else:
                    w.mu.grad = torch.tensor(nes_mu_grad, dtype=torch.float32)
                    w.rho.grad = torch.tensor(nes_rho_grad, dtype=torch.float32)

                connection.dominant_gene_rate.value += 2 * self.dominant_gene_delta
                idx += 1

        for gene_rate in self.gene_rates:
            gene_rate.value = min(0.6, max(0.4, gene_rate.value))

    def train(self, n_steps: int, fitness_threshold: Optional[float] = None) -> None:
        comm = MPI.COMM_WORLD
        n_jobs = self.n_networks // comm.Get_size()
        for step in range(n_steps):
            gen_start_time = time.time()
            self.reporters.start_generation(self.generation)

            rewards = []
            for genome in tqdm(self.population[
                    comm.rank*n_jobs: n_jobs * (comm.rank + 1)], desc='population training'):
                if self.n_eval_samples > 1 and self.risk_aversion > 0:
                    evals = [self.agent.rollout(genome)
                             for _ in range(self.n_eval_samples)]
                    mean_r = np.mean(evals)
                    std_r = np.std(evals)
                    reward = mean_r - self.risk_aversion * std_r
                else:
                    reward = self.agent.rollout(genome)
                rewards.append(reward)
            rewards = functools.reduce(
                operator.iconcat, comm.allgather(rewards), [])

            for idx, reward in enumerate(rewards):
                self.population[idx].fitness = reward

            self.train_genome(self.get_random_genome())

            sorted_population: SortedContextGenomes = self.sort_population(
                self.population)

            reward = self.agent.rollout(sorted_population[0])
            self.data.append((int(self.generation), 'BNEATEST', self.seed, reward))
            if reward >= self.best_fitness:
                self.best_fitness = reward
                self.best_genome = sorted_population[0].deepcopy()

            self.reporters.post_evaluate(sorted_population[0], self.population)

            # Compute generation statistics
            rewards_arr = np.array(rewards)
            mean_sigma, max_sigma = self._compute_bayesian_stats()
            n_nodes_list = [len(g.nodes) for g in self.population]
            n_conns_list = [len([c for c in g.connections if c.enabled])
                           for g in self.population]
            best_genome = sorted_population[0]

            gen_stats = GenerationStats(
                generation=self.generation,
                fitness_mean=float(np.mean(rewards_arr)),
                fitness_stdev=float(np.std(rewards_arr)),
                fitness_median=float(np.median(rewards_arr)),
                fitness_best=float(np.max(rewards_arr)),
                fitness_worst=float(np.min(rewards_arr)),
                population_size=len(self.population),
                n_connections_mean=float(np.mean(n_conns_list)),
                n_connections_best=float(n_conns_list[0]),
                n_nodes_mean=float(np.mean(n_nodes_list)),
                n_nodes_best=float(n_nodes_list[0]),
                best_genome_id=id(best_genome),
                elapsed_time=time.time() - gen_start_time,
                mean_weight_sigma=mean_sigma,
                max_weight_sigma=max_sigma,
                kl_weight=self._effective_kl_weight(),
            )
            self.reporters.end_generation(gen_stats)

            if (fitness_threshold is not None
                    and self.best_fitness >= fitness_threshold):
                self.reporters.found_solution(self.generation, self.best_genome)
                break

            if self.generation and not self.generation % self.save_checkpoint_n:
                self.save_checkpoint()

            self.next_generation(sorted_population)

        self.reporters.training_complete()
        if self.logdir:
            if MPI.COMM_WORLD.rank == 0:
                self.save_logs()

    def train_genome(self, genome: ContextGenome, n_steps: int = 1):
        for _ in range(n_steps):
            self.optimizer.zero_grad()
            self.calculate_grads(genome)
            self.optimizer.step()

    def crossover(self, genome_1: Genome, genome_2: Genome) -> ContextGenome:
        '''Crossover two genomes by aligning their innovation numbers.'''
        connections: List[Connection] = []
        nodes: List[Node] = []
        connections_1, connections_2 = align_connections(
            genome_1.connections, genome_2.connections)

        for idx in range(len(connections_1)):
            connection_1 = connections_1[idx]
            connection_2 = connections_2[idx]
            enabled: bool
            connection: Connection
            if connection_1.dummy or connection_2.dummy:
                if connection_1.dummy:
                    connection = connection_2
                else:
                    connection = connection_1
                if connection.enabled:
                    if self.random.random() <= connection.dominant_gene_rate.value:
                        enabled = True
                    else:
                        enabled = False
                else:
                    continue
            else:
                connection = connection_1
                if connection_1.enabled and connection_2.enabled:
                    enabled = True
                elif connection_1.enabled ^ connection_2.enabled:
                    enabled = (self.random.random() <=
                               connection.dominant_gene_rate.value)
                else:
                    if self.random.random() <= connection.dominant_gene_rate.value:
                        enabled = False
                    else:
                        continue

            in_node = Node(connection.in_node.id, connection.in_node.type,
                           connection.in_node.activation,
                           depth=connection.in_node.depth)
            out_node = Node(connection.out_node.id, connection.out_node.type,
                            connection.out_node.activation,
                            depth=connection.out_node.depth)

            nodes_dict = dict(zip(nodes, range(len(nodes))))
            if in_node not in nodes_dict:
                nodes.append(in_node)
                nodes_dict[in_node] = len(nodes)-1
            if out_node not in nodes_dict:
                nodes.append(out_node)
                nodes_dict[out_node] = len(nodes)-1
            connection = Connection(nodes[nodes_dict[in_node]],
                                    nodes[nodes_dict[out_node]],
                                    innovation=connection.innovation,
                                    weight=connection.weight,
                                    dominant_gene_rate=connection.dominant_gene_rate)
            connection.enabled = enabled
            connections.append(connection)
        # Preserve all structural nodes (inputs, bias, outputs) from parent
        nodes_ids = {n.id for n in nodes}
        for node in genome_1.nodes:
            if node.type in (NodeType.INPUT, NodeType.BIAS, NodeType.OUTPUT) and node.id not in nodes_ids:
                nodes.append(Node(node.id, node.type, node.activation, depth=node.depth))
                nodes_ids.add(node.id)

        new_genome = ContextGenome(nodes, connections)
        return new_genome

    def save_logs(self):
        import pandas as pd  # type: ignore
        data = pd.DataFrame(
            self.data, columns=['Generation', 'Algorithm', 'Seed', 'Reward'])
        data = data.astype(
            {"Generation": int, "Algorithm": str, 'Seed': int, 'Reward': float})
        file = os.path.join(self.logdir, f'{time.strftime("%Y%m%d-%H%M%S")}.csv')
        data.to_csv(file, index=False)

    @staticmethod
    def sort_population(population: List[ContextGenome]) -> SortedContextGenomes:
        return SortedContextGenomes(sorted(population, key=lambda x: x.fitness,
                                           reverse=True))

    def get_random_genome(self) -> ContextGenome:
        """Return random genome from a sorted population."""
        rewards: np.ndarray = np.array([genome.fitness for genome in self.population])
        eps = np.finfo(float).eps
        normalized_rewards: np.ndarray = rewards - rewards.min() + eps
        probabilities = normalized_rewards / np.sum(normalized_rewards)
        return self.np_random.choice(self.population, p=probabilities)

    def save_checkpoint(self) -> None:
        if MPI.COMM_WORLD.rank == 0:
            frame = inspect.stack()[2]
            module = inspect.getmodule(frame[0])
            if module is not None:
                folder = pathlib.Path(
                    f"{os.path.join(os.path.dirname(module.__file__), 'checkpoints')}")
                folder.mkdir(parents=True, exist_ok=True)
                filename = f'{int(time.time())}.checkpoint'
                save_path = os.path.abspath(os.path.join(folder, filename))
                print(f"\033[33;1mCheckpoint: {save_path}\033[0m")
                with open(save_path, 'wb') as output:
                    cloudpickle.dump(self, output)

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'BNEATEST':
        comm = MPI.COMM_WORLD
        if not comm.rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f
        print(f"\033[33;1mLoading: {filename}\033[0m")
        with open(filename, 'rb') as checkpoint:
            bneatest = cloudpickle.load(checkpoint)
        if comm.rank == 0:
            if bneatest.logdir:
                from pathlib import Path
                Path(bneatest.logdir).mkdir(parents=True, exist_ok=True)
        n_proc = comm.Get_size()
        if bneatest.n_proc != n_proc:
            print("\033[31;1mWarning: Number of process mismatch\033[0m")
        if bneatest.version != VERSION:
            print("\033[31;1mWarning: Checkpoint version mismatch!\n"
                  f"Current Version: {VERSION.major}.{VERSION.minor}.{VERSION.patch}\n"
                  "Checkpoint Version:"
                  f" {bneatest.version.major}.{bneatest.version.minor}."
                  f"{bneatest.version.patch}\n\033[0m")
        return bneatest
