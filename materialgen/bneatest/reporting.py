"""Reporting and statistics collection for BNEATEST, inspired by neat-python."""
import time
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    fitness_mean: float
    fitness_stdev: float
    fitness_median: float
    fitness_best: float
    fitness_worst: float
    population_size: int
    n_connections_mean: float
    n_connections_best: float
    n_nodes_mean: float
    n_nodes_best: float
    best_genome_id: int
    elapsed_time: float
    # Bayesian-specific
    mean_weight_sigma: float = 0.0
    max_weight_sigma: float = 0.0
    kl_weight: float = 0.0


class BaseReporter(ABC):
    """Abstract reporter interface (mirrors neat-python)."""

    @abstractmethod
    def start_generation(self, generation: int) -> None:
        ...

    @abstractmethod
    def end_generation(self, stats: GenerationStats) -> None:
        ...

    @abstractmethod
    def post_evaluate(self, best_genome, population) -> None:
        ...

    @abstractmethod
    def found_solution(self, generation: int, best_genome) -> None:
        ...

    @abstractmethod
    def training_complete(self) -> None:
        ...


class StdOutReporter(BaseReporter):
    """Prints generation statistics to stdout, neat-python style."""

    def __init__(self, show_bayesian_stats: bool = True):
        self.show_bayesian_stats = show_bayesian_stats
        self._generation_start_time: float = 0.0
        self._generation_times: List[float] = []

    def start_generation(self, generation: int) -> None:
        self._generation_start_time = time.time()
        print(f'\n ****** Running generation {generation} ****** \n')

    def end_generation(self, stats: GenerationStats) -> None:
        elapsed = stats.elapsed_time
        self._generation_times.append(elapsed)

        print(f'Population size: {stats.population_size}')
        print(f'Fitness - mean: {stats.fitness_mean:.3f}  '
              f'stdev: {stats.fitness_stdev:.3f}  '
              f'median: {stats.fitness_median:.3f}')
        print(f'Fitness - best: {stats.fitness_best:.3f}  '
              f'worst: {stats.fitness_worst:.3f}')
        print(f'Complexity - '
              f'nodes (mean/best): {stats.n_nodes_mean:.1f}/{stats.n_nodes_best:.0f}  '
              f'connections (mean/best): {stats.n_connections_mean:.1f}/'
              f'{stats.n_connections_best:.0f}')

        if self.show_bayesian_stats:
            print(f'Weight uncertainty - '
                  f'mean sigma: {stats.mean_weight_sigma:.4f}  '
                  f'max sigma: {stats.max_weight_sigma:.4f}  '
                  f'KL weight: {stats.kl_weight:.4f}')

        print(f'Elapsed: {elapsed:.3f}s', end='')
        if len(self._generation_times) > 1:
            avg = np.mean(self._generation_times[-10:])
            print(f'  (10-gen avg: {avg:.3f}s)', end='')
        print()

    def post_evaluate(self, best_genome, population) -> None:
        pass

    def found_solution(self, generation: int, best_genome) -> None:
        print(f'\nBest fitness threshold reached in generation {generation}!')

    def training_complete(self) -> None:
        total = sum(self._generation_times)
        print(f'\nTraining complete. Total time: {total:.1f}s '
              f'({len(self._generation_times)} generations)')


class StatisticsReporter(BaseReporter):
    """Collects per-generation statistics for later analysis and plotting."""

    def __init__(self):
        self.generation_stats: List[GenerationStats] = []
        self.best_fitnesses: List[float] = []
        self.mean_fitnesses: List[float] = []
        self.stdev_fitnesses: List[float] = []
        self.median_fitnesses: List[float] = []
        self.mean_sigmas: List[float] = []
        self.max_sigmas: List[float] = []
        self.n_connections: List[float] = []
        self.n_nodes: List[float] = []
        self.generations: List[int] = []
        self._best_fitness_ever: float = -float('inf')
        self._best_genome = None

    def start_generation(self, generation: int) -> None:
        pass

    def end_generation(self, stats: GenerationStats) -> None:
        self.generation_stats.append(stats)
        self.generations.append(stats.generation)
        self.best_fitnesses.append(stats.fitness_best)
        self.mean_fitnesses.append(stats.fitness_mean)
        self.stdev_fitnesses.append(stats.fitness_stdev)
        self.median_fitnesses.append(stats.fitness_median)
        self.mean_sigmas.append(stats.mean_weight_sigma)
        self.max_sigmas.append(stats.max_weight_sigma)
        self.n_connections.append(stats.n_connections_mean)
        self.n_nodes.append(stats.n_nodes_mean)

    def post_evaluate(self, best_genome, population) -> None:
        fitness = best_genome.fitness
        if fitness > self._best_fitness_ever:
            self._best_fitness_ever = fitness
            self._best_genome = best_genome

    def found_solution(self, generation: int, best_genome) -> None:
        pass

    def training_complete(self) -> None:
        pass

    @property
    def best_genome(self):
        return self._best_genome

    def get_fitness_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (generations, means, stdevs) as numpy arrays."""
        return (np.array(self.generations),
                np.array(self.mean_fitnesses),
                np.array(self.stdev_fitnesses))

    def get_sigma_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (generations, mean_sigmas, max_sigmas) as numpy arrays."""
        return (np.array(self.generations),
                np.array(self.mean_sigmas),
                np.array(self.max_sigmas))


class ReporterSet:
    """Manages multiple reporters."""

    def __init__(self):
        self.reporters: List[BaseReporter] = []

    def add(self, reporter: BaseReporter) -> None:
        self.reporters.append(reporter)

    def start_generation(self, generation: int) -> None:
        for r in self.reporters:
            r.start_generation(generation)

    def end_generation(self, stats: GenerationStats) -> None:
        for r in self.reporters:
            r.end_generation(stats)

    def post_evaluate(self, best_genome, population) -> None:
        for r in self.reporters:
            r.post_evaluate(best_genome, population)

    def found_solution(self, generation: int, best_genome) -> None:
        for r in self.reporters:
            r.found_solution(generation, best_genome)

    def training_complete(self) -> None:
        for r in self.reporters:
            r.training_complete()
