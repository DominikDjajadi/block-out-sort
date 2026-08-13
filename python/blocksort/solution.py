"""Serializable solution actions, replay, and verification.

Solver solutions must survive serialization, so actions are stored using a
stable *locator* (block color + the block's cells before the move) rather than
the transient :attr:`~blocksort.actions.Action.block_index`. Within any state
sharing a canonical key the occupied cells are identical, so the locator
uniquely identifies the block in any equivalent state.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .actions import Action
from .conformance import action_to_normalized, normalized_to_action
from .environment import Environment
from .state import State


def serialize_action(state: State, action: Action) -> dict[str, Any]:
    """A stable, JSON-serializable form of ``action`` in ``state``.

    Shape: ``{"color", "cells", "dir", "distance", "exit"}``.
    """
    return action_to_normalized(state, action)


def deserialize_action(state: State, data: dict[str, Any]) -> Action:
    """Resolve a serialized action back to an :class:`Action` for ``state``."""
    return normalized_to_action(state, data)


def serialize_solution(
    env: Environment, state: State, actions: Iterable[Action]
) -> list[dict[str, Any]]:
    """Serialize a whole action path, resolving each action in its own state."""
    out: list[dict[str, Any]] = []
    cur = state
    for action in actions:
        out.append(serialize_action(cur, action))
        cur = env.apply_action(cur, action)
    return out


def deserialize_solution(
    env: Environment, state: State, data: Sequence[dict[str, Any]]
) -> tuple[Action, ...]:
    """Resolve a serialized path to concrete actions by replaying from ``state``."""
    actions: list[Action] = []
    cur = state
    for entry in data:
        action = deserialize_action(cur, entry)
        actions.append(action)
        cur = env.apply_action(cur, action)
    return tuple(actions)


def replay_solution(
    env: Environment, state: State, actions: Iterable[Action]
) -> State:
    """Apply ``actions`` in order from ``state``; return the final state.

    Raises :class:`~blocksort.environment.IllegalActionError` on any illegal
    step (each action is validated by the environment).
    """
    cur = state
    for action in actions:
        cur = env.apply_action(cur, action)
    return cur


def verify_solution(
    env: Environment,
    state: State,
    actions: Sequence[Action],
    expected_move_count: int | None = None,
) -> bool:
    """Confirm a solution is legal end-to-end and reaches a terminal state.

    Checks that every action is legal in the state where it occurs, the final
    state is terminal, and (if given) the path length equals
    ``expected_move_count``.
    """
    if expected_move_count is not None and len(actions) != expected_move_count:
        return False
    try:
        final = replay_solution(env, state, actions)
    except Exception:
        return False
    return env.is_terminal(final)


def describe_solution(
    actions: Sequence[Action],
    env: Environment | None = None,
    state: State | None = None,
) -> str:
    """Human-readable, one-line-per-move description.

    If ``env`` and ``state`` are supplied, block colors are resolved by replay;
    otherwise the transient block index is shown.
    """
    lines: list[str] = []
    cur = state
    for i, action in enumerate(actions, start=1):
        if env is not None and cur is not None:
            block = cur.blocks[action.block_index]
            who = block.color
        else:
            who = f"block#{action.block_index}"
        tail = "EXIT" if action.exit else str(action.distance)
        lines.append(f"{i}. {who} {action.direction.value} {tail}")
        if env is not None and cur is not None:
            cur = env.apply_action(cur, action)
    return "\n".join(lines)
