from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import random

Position = Tuple[int, int]
Action = str

ACTIONS: Tuple[Action, ...] = ("stay", "up", "down", "left", "right")
ACTION_TO_ID: Dict[Action, int] = {action: idx for idx, action in enumerate(ACTIONS)}
ID_TO_ACTION: Dict[int, Action] = {idx: action for idx, action in enumerate(ACTIONS)}

# x increases to the right, y increases upward.
ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    "stay": (0, 0),
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def is_valid_position(N: int, pos: Position) -> bool:
    """Return True iff pos is inside the N x N grid."""
    x, y = pos
    return 1 <= x <= N and 1 <= y <= N


def apply_action(N: int, pos: Position, action: Action) -> Position:
    """Apply action; invalid boundary moves keep the agent at the same position."""
    if action not in ACTION_DELTAS:
        raise ValueError(f"Unknown action {action!r}. Valid actions are {ACTIONS}.")

    dx, dy = ACTION_DELTAS[action]
    candidate = (pos[0] + dx, pos[1] + dy)
    return candidate if is_valid_position(N, candidate) else pos


def get_valid_actions(N: int, pos: Position) -> List[Action]:
    """Return valid actions at pos. 'stay' is always valid."""
    valid: List[Action] = []
    for action in ACTIONS:
        next_pos = apply_action(N, pos, action)
        if action == "stay" or next_pos != pos:
            valid.append(action)
    return valid


def get_valid_action_ids(N: int, pos: Position) -> List[int]:
    """Return action indices corresponding to get_valid_actions."""
    return [ACTION_TO_ID[a] for a in get_valid_actions(N, pos)]


def get_valid_next_positions(N: int, pos: Position) -> List[Position]:
    """Distinct valid next locations under the same move set, including staying."""
    candidates: List[Position] = []
    seen = set()
    for action in get_valid_actions(N, pos):
        nxt = apply_action(N, pos, action)
        if nxt not in seen:
            seen.add(nxt)
            candidates.append(nxt)
    return candidates


def respawn_prey(N: int, predator_pos: Position, rng: Optional[random.Random] = None) -> Position:
    """Respawn prey uniformly over all cells except the predator's cell."""
    rng = rng or random
    cells = [(x, y) for x in range(1, N + 1) for y in range(1, N + 1)]
    allowed = [cell for cell in cells if cell != predator_pos]
    return rng.choice(allowed)


def simulator(
    N: int,
    predator_pos: Position,
    prey_pos: Position,
    predator_action: Action,
    rng: Optional[random.Random] = None,
) -> Tuple[Position, Position, int]:
    """
    One-step simulator for the predator-prey environment.

    The predator first acts. If it reaches the prey, reward is 1 and the prey
    respawns uniformly outside the predator cell. Otherwise, the prey moves
    uniformly over its valid next cells, and a catch is checked again.
    """
    rng = rng or random

    if N <= 1:
        raise ValueError("N must be at least 2 for a predator-prey game.")
    if not is_valid_position(N, predator_pos):
        raise ValueError(f"Invalid predator position: {predator_pos}")
    if not is_valid_position(N, prey_pos):
        raise ValueError(f"Invalid prey position: {prey_pos}")
    if predator_pos == prey_pos:
        raise ValueError("Predator and prey should not occupy the same cell before a step.")

    next_predator_pos = apply_action(N, predator_pos, predator_action)

    if next_predator_pos == prey_pos:
        return next_predator_pos, respawn_prey(N, next_predator_pos, rng), 1

    prey_candidates = get_valid_next_positions(N, prey_pos)
    next_prey_pos = rng.choice(prey_candidates)

    if next_predator_pos == next_prey_pos:
        return next_predator_pos, respawn_prey(N, next_predator_pos, rng), 1

    return next_predator_pos, next_prey_pos, 0
