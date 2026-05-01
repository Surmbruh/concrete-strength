from typing import List

import torch

from .weight import BayesianWeight


class BayesianOptimizer:
    """Optimizer wrapper over torch.optim.Adam for BayesianWeight parameters.

    Manages both mu and rho parameters of each BayesianWeight.
    Supports dynamically adding new weights (from mutations).
    """

    def __init__(self, weights: List[BayesianWeight], lr: float = 0.01, **kwargs):
        self.weights = weights
        self.lr = lr
        self.kwargs = kwargs
        params = []
        for w in weights:
            params.extend([w.mu, w.rho])
        if params:
            self.optimizer = torch.optim.Adam(params, lr=lr, **kwargs)
        else:
            self.optimizer = None

    def _ensure_optimizer(self):
        """Create optimizer if it doesn't exist yet (e.g., initially empty weights)."""
        if self.optimizer is None and self.weights:
            params = []
            for w in self.weights:
                params.extend([w.mu, w.rho])
            self.optimizer = torch.optim.Adam(params, lr=self.lr, **self.kwargs)

    def step(self) -> None:
        self._ensure_optimizer()
        if self.optimizer is not None:
            self.optimizer.step()

    def zero_grad(self) -> None:
        self._ensure_optimizer()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def add_weight(self, weight: BayesianWeight) -> None:
        """Register a new BayesianWeight's parameters with the optimizer."""
        if self.optimizer is None:
            # Create optimizer with just this weight
            self.optimizer = torch.optim.Adam(
                [weight.mu, weight.rho], lr=self.lr, **self.kwargs)
        else:
            self.optimizer.add_param_group({'params': [weight.mu, weight.rho]})
