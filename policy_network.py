from __future__ import annotations

from typing import Tuple
import torch
from torch import nn
from torch.distributions import Categorical

from environment import ACTIONS, Position, get_valid_action_ids


class PolicyNetwork(nn.Module):
    """
    Neural policy pi_theta(a | s) for N=4 predator-prey.

    Input  : [pred_x, pred_y, prey_x, prey_y], each normalized to [0, 1]
    Output : 5 action logits corresponding to stay/up/down/left/right
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(ACTIONS)),
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(state_tensor)


def encode_state(
    N: int,
    predator_pos: Position,
    prey_pos: Position,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode a state as normalized coordinates."""
    denom = float(N - 1)
    values = [
        (predator_pos[0] - 1) / denom,
        (predator_pos[1] - 1) / denom,
        (prey_pos[0] - 1) / denom,
        (prey_pos[1] - 1) / denom,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def valid_action_mask(
    N: int,
    predator_pos: Position,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean mask over the 5 global actions. Invalid boundary moves are masked out."""
    mask = torch.zeros(len(ACTIONS), dtype=torch.bool, device=device)
    for idx in get_valid_action_ids(N, predator_pos):
        mask[idx] = True
    return mask


def action_distribution(
    policy: PolicyNetwork,
    N: int,
    predator_pos: Position,
    prey_pos: Position,
) -> Categorical:
    """Return a Categorical distribution pi_theta(. | state), after invalid-action masking."""
    device = next(policy.parameters()).device
    state_tensor = encode_state(N, predator_pos, prey_pos, device=device)
    logits = policy(state_tensor)
    mask = valid_action_mask(N, predator_pos, device=device)
    masked_logits = logits.masked_fill(~mask, -1.0e9)
    return Categorical(logits=masked_logits)
