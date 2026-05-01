import math
import copy
from dataclasses import dataclass, field

import torch
from torch.distributions import Normal


class BayesianWeight:
    """Bayesian weight represented as a Normal distribution N(mu, sigma).

    sigma = softplus(rho) ensures sigma > 0.
    Supports reparameterization trick for autograd-compatible sampling.
    """

    def __init__(self, mu: float, rho: float = -3.0):
        self.mu = torch.tensor(mu, dtype=torch.float32, requires_grad=True)
        self.rho = torch.tensor(rho, dtype=torch.float32, requires_grad=True)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rho)

    @property
    def distribution(self) -> Normal:
        return Normal(self.mu, self.sigma)

    def sample(self) -> torch.Tensor:
        """Sample a weight value using the reparameterization trick."""
        epsilon = torch.randn(1)
        return self.mu + self.sigma * epsilon

    @property
    def value(self) -> float:
        """Mean value (for compatibility and deterministic evaluation)."""
        return self.mu.item()

    def copy(self) -> 'BayesianWeight':
        """Shallow copy — returns same object (shared weight semantics)."""
        return self

    def deepcopy(self) -> 'BayesianWeight':
        """Deep copy — new independent BayesianWeight with same parameter values."""
        return BayesianWeight(self.mu.item(), self.rho.item())

    def __repr__(self) -> str:
        return f'BayesianWeight(mu={self.mu.item():.4f}, sigma={self.sigma.item():.4f})'
