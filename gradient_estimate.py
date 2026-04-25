from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random

import torch

from environment import ID_TO_ACTION, Position, simulator
from policy_network import PolicyNetwork, action_distribution


@dataclass
class TrajectoryStats:
    total_reward: float
    discounted_return: float


def discounted_returns_to_go(rewards: Sequence[float], gamma: float) -> List[float]:
    """Return G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k for every t."""
    returns = [0.0 for _ in rewards]
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


def sample_trajectory(
    policy: PolicyNetwork,
    N: int = 4,
    horizon: int = 80,
    gamma: float = 0.99,
    rng: random.Random | None = None,
) -> Tuple[List[torch.Tensor], List[int], TrajectoryStats]:
    """Sample one finite-horizon trajectory using the simulator and current policy."""
    rng = rng or random.Random()
    predator_pos: Position = (1, 1)
    prey_pos: Position = (N, N)

    log_probs: List[torch.Tensor] = []
    rewards: List[int] = []

    for _ in range(horizon):
        dist = action_distribution(policy, N, predator_pos, prey_pos)
        action_id_tensor = dist.sample()
        action_id = int(action_id_tensor.item())
        log_probs.append(dist.log_prob(action_id_tensor))

        predator_pos, prey_pos, reward = simulator(
            N=N,
            predator_pos=predator_pos,
            prey_pos=prey_pos,
            predator_action=ID_TO_ACTION[action_id],
            rng=rng,
        )
        rewards.append(int(reward))

    discounted = sum((gamma ** t) * r for t, r in enumerate(rewards))
    return log_probs, rewards, TrajectoryStats(
        total_reward=float(sum(rewards)),
        discounted_return=float(discounted),
    )


def policy_gradient_objective(
    policy: PolicyNetwork,
    N: int = 4,
    episodes: int = 8,
    horizon: int = 80,
    gamma: float = 0.99,
    seed: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Monte Carlo score-function objective whose gradient is an unbiased estimate
    of the finite-horizon discounted policy gradient.

    Objective estimate:
        (1/M) sum_i sum_t gamma^t G_t^{(i)} log pi_theta(A_t | S_t)
    where G_t = sum_{k=t}^{T-1} gamma^{k-t} R_k.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    base_rng = random.Random(seed)
    terms: List[torch.Tensor] = []
    total_rewards: List[float] = []
    discounted_returns: List[float] = []

    for episode_idx in range(episodes):
        episode_seed = base_rng.randint(0, 2**31 - 1)
        rng = random.Random(episode_seed)
        log_probs, rewards, stats = sample_trajectory(
            policy=policy,
            N=N,
            horizon=horizon,
            gamma=gamma,
            rng=rng,
        )
        returns = discounted_returns_to_go(rewards, gamma)

        device = log_probs[0].device
        weights = torch.tensor(
            [(gamma ** t) * returns[t] for t in range(horizon)],
            dtype=torch.float32,
            device=device,
        )
        log_prob_tensor = torch.stack(log_probs)
        terms.append(torch.sum(log_prob_tensor * weights.detach()))

        total_rewards.append(stats.total_reward)
        discounted_returns.append(stats.discounted_return)

    objective = torch.stack(terms).mean()
    diagnostics = {
        "avg_total_reward": float(sum(total_rewards) / len(total_rewards)),
        "avg_discounted_return": float(sum(discounted_returns) / len(discounted_returns)),
    }
    return objective, diagnostics


def gradient_estimate(
    policy: PolicyNetwork,
    N: int = 4,
    episodes: int = 8,
    horizon: int = 80,
    gamma: float = 0.99,
    seed: int | None = None,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Return an unbiased Monte Carlo estimate of the policy gradient.

    PyTorch's torch.autograd.grad is used on the REINFORCE surrogate objective;
    the score function itself is obtained from Categorical.log_prob(action).
    """
    objective, diagnostics = policy_gradient_objective(
        policy=policy,
        N=N,
        episodes=episodes,
        horizon=horizon,
        gamma=gamma,
        seed=seed,
    )
    grads = torch.autograd.grad(objective, list(policy.parameters()))
    return list(grads), diagnostics
