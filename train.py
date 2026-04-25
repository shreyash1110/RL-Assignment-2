from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gradient_estimate import policy_gradient_objective
from policy_network import PolicyNetwork
from simple_sga import simple_SGA


def moving_average(values: List[float], window: int = 10) -> List[float]:
    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(sum(values[start:idx + 1]) / (idx - start + 1))
    return out


def save_history_csv(history: List[Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "iteration", "avg_total_reward", "avg_discounted_return"])
        writer.writeheader()
        writer.writerows(history)


def plot_curve(history: List[Dict[str, float]], method: str, path: str, window: int = 10) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xs = [int(row["iteration"]) for row in history]
    ys = [float(row["avg_total_reward"]) for row in history]
    smooth = moving_average(ys, window=window)

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(xs, ys, linewidth=1.0, alpha=0.45, label="batch average total reward")
    plt.plot(xs, smooth, linewidth=2.0, label=f"moving average, window={window}")
    plt.xlabel("Iteration")
    plt.ylabel("Total reward over one horizon")
    plt.title(f"Learning curve: {method}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def train_adam(
    policy: PolicyNetwork,
    iterations: int = 150,
    learning_rate: float = 0.01,
    N: int = 4,
    episodes_per_update: int = 8,
    horizon: int = 80,
    gamma: float = 0.99,
    seed: int = 1000,
) -> List[Dict[str, float]]:
    """Train policy by maximizing the same REINFORCE objective using Adam."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: List[Dict[str, float]] = []

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        objective, diagnostics = policy_gradient_objective(
            policy=policy,
            N=N,
            episodes=episodes_per_update,
            horizon=horizon,
            gamma=gamma,
            seed=seed + iteration,
        )
        loss = -objective
        loss.backward()
        optimizer.step()

        history.append({
            "iteration": float(iteration),
            "avg_total_reward": diagnostics["avg_total_reward"],
            "avg_discounted_return": diagnostics["avg_discounted_return"],
        })
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="EE675 Assignment 2: Policy-gradient predator-prey training")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=4, help="Monte Carlo trajectories per policy update")
    parser.add_argument("--horizon", type=int, default=60, help="Finite truncation horizon")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sga-lr", type=float, default=0.02)
    parser.add_argument("--adam-lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    sga_policy = PolicyNetwork(hidden_dim=64)
    sga_history = simple_SGA(
        policy=sga_policy,
        iterations=args.iterations,
        learning_rate=args.sga_lr,
        N=4,
        episodes_per_update=args.episodes,
        horizon=args.horizon,
        gamma=args.gamma,
        seed=args.seed,
    )
    for row in sga_history:
        row["method"] = "simple_SGA"

    torch.manual_seed(args.seed)
    adam_policy = PolicyNetwork(hidden_dim=64)
    adam_history = train_adam(
        policy=adam_policy,
        iterations=args.iterations,
        learning_rate=args.adam_lr,
        N=4,
        episodes_per_update=args.episodes,
        horizon=args.horizon,
        gamma=args.gamma,
        seed=args.seed + 10000,
    )
    for row in adam_history:
        row["method"] = "Adam"

    all_history = sga_history + adam_history
    save_history_csv(all_history, os.path.join(args.out_dir, "results", "training_log.csv"))
    plot_curve(sga_history, "simple SGA", os.path.join(args.out_dir, "plots", "simple_sga_learning_curve.png"))
    plot_curve(adam_history, "Adam", os.path.join(args.out_dir, "plots", "adam_learning_curve.png"))

    print("Saved:")
    print(os.path.join(args.out_dir, "results", "training_log.csv"))
    print(os.path.join(args.out_dir, "plots", "simple_sga_learning_curve.png"))
    print(os.path.join(args.out_dir, "plots", "adam_learning_curve.png"))


if __name__ == "__main__":
    main()
