from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from gradient_estimate import gradient_estimate
from policy_network import PolicyNetwork


def simple_sga_update(policy: PolicyNetwork, grads: List[torch.Tensor], learning_rate: float) -> None:
    """Apply theta <- theta + eta * gradient_estimate."""
    with torch.no_grad():
        for parameter, grad in zip(policy.parameters(), grads):
            parameter.add_(learning_rate * grad)


def simple_SGA(
    policy: PolicyNetwork,
    iterations: int = 150,
    learning_rate: float = 0.02,
    N: int = 4,
    episodes_per_update: int = 8,
    horizon: int = 80,
    gamma: float = 0.99,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Train the policy using simple stochastic gradient ascent."""
    history: List[Dict[str, float]] = []
    for iteration in range(1, iterations + 1):
        grads, diagnostics = gradient_estimate(
            policy=policy,
            N=N,
            episodes=episodes_per_update,
            horizon=horizon,
            gamma=gamma,
            seed=seed + iteration,
        )
        simple_sga_update(policy, grads, learning_rate)

        history.append({
            "iteration": float(iteration),
            "avg_total_reward": diagnostics["avg_total_reward"],
            "avg_discounted_return": diagnostics["avg_discounted_return"],
        })
    return history
