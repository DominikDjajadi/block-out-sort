"""Versioned supervised policy-value dataset record format (JSON Lines).

A record describes one state with exact labels. It is versioned, deterministic,
JSON-serializable, replayable, and independent of Python object repr. See the
README "Dataset" section for the full schema documentation.
"""

from __future__ import annotations

from typing import Any, Optional

from ..conformance import _normalized_hashable, action_to_normalized, block_from_spec
from ..environment import Environment
from ..oracle import ActionAnalysis, StateAnalysis
from ..schema import Cell, Level
from ..serialization import level_to_dict
from ..signature import static_level_signature
from ..solver import SolveResult
from ..state import State
from .targets import (
    DEFAULT_TEMPERATURE,
    DEFAULT_VALUE_NORM_CONSTANT,
    soft_regret_policy,
    uniform_optimal_policy,
    value_target,
)

DATASET_VERSION = 1

POLICY_UNIFORM_OPTIMAL = "uniform-optimal"
POLICY_SOFT_REGRET = "soft-regret"
POLICY_SINGLE_VERIFIED_OPTIMAL = "single-verified-optimal"

LABEL_FULL_EXACT = "full-exact"
LABEL_EXACT_PATH_POLICY = "exact-path-policy"
# Expert-iteration replay extends the exact supervised schema with this
# explicitly approximate root visit-policy label.
LABEL_SEARCH_VISIT_POLICY = "search-visit-policy"


def serialize_state(state: State) -> dict[str, Any]:
    """Self-contained, replayable serialization of a dynamic state."""
    blocks = []
    for block in state.blocks:
        entry: dict[str, Any] = {
            "color": block.color,
            "cells": [[c.r, c.c] for c in block.sorted_cells()],
        }
        if block.unlock_at:
            entry["unlockAt"] = block.unlock_at
        blocks.append(entry)
    return {"blocks": blocks, "total_blocks": state.total_blocks}


def deserialize_state(level: Level, state_data: dict[str, Any]) -> State:
    """Reconstruct a :class:`State` from a record's level + state block."""
    blocks = tuple(block_from_spec(b) for b in state_data["blocks"])
    total = int(state_data["total_blocks"])
    return State(level=level, blocks=blocks, total_blocks=total)


def _sorted_action_analyses(
    analysis: StateAnalysis,
) -> list[ActionAnalysis]:
    return sorted(analysis.actions, key=lambda a: _normalized_hashable(a.serialized))


def build_record(
    analysis: StateAnalysis,
    state: State,
    *,
    level_id: str,
    policy_type: str = POLICY_UNIFORM_OPTIMAL,
    temperature: float = DEFAULT_TEMPERATURE,
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
    provenance: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build one dataset record from an exact analysis.

    Returns ``None`` (caller should skip) unless the state is non-terminal,
    solvable, and *fully exact* (its value and every successor value proven).
    This guarantees every exported record has an exact value, exact action
    costs, and exact regrets.
    """
    if analysis.terminal:
        return None
    if not (analysis.exact and analysis.solvable and analysis.all_successors_exact):
        return None
    if analysis.value is None:
        return None

    ordered = _sorted_action_analyses(analysis)
    legal_actions = [a.serialized for a in ordered]
    action_costs = [a.cost for a in ordered]          # int or None (infinite)
    action_regrets = [a.regret for a in ordered]      # int or None (infinite)
    optimal_actions = [a.serialized for a in ordered if a.optimal]

    if policy_type == POLICY_UNIFORM_OPTIMAL:
        policy = uniform_optimal_policy(ordered)
        policy_meta = {"type": POLICY_UNIFORM_OPTIMAL, "temperature": None}
    elif policy_type == POLICY_SOFT_REGRET:
        policy = soft_regret_policy(ordered, temperature)
        policy_meta = {"type": POLICY_SOFT_REGRET, "temperature": temperature}
    else:
        raise ValueError(f"unknown policy target type: {policy_type}")

    record: dict[str, Any] = {
        "version": DATASET_VERSION,
        "label_kind": LABEL_FULL_EXACT,
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": True,
        "action_values_complete": True,
        "level_id": level_id,
        "static_level_signature": analysis.static_signature,
        "level": level_to_dict(state.level),
        "state": serialize_state(state),
        "state_key": analysis.state_key,
        "remaining_blocks": state.remaining,
        "cleared_blocks": state.cleared,
        "legal_actions": legal_actions,
        "optimal_remaining_moves": analysis.value,
        "optimal_actions": optimal_actions,
        "action_costs": action_costs,
        "action_regrets": action_regrets,
        "policy": policy_meta,
        "policy_target": policy,
        "value_target": value_target(
            analysis.value, constant=value_norm_constant
        ),
        "provenance": [provenance] if provenance is not None else [],
    }
    return record


def build_exact_path_record(
    result: SolveResult,
    state: State,
    env: Environment,
    *,
    level_id: str,
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
    provenance: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build an exact-value, single-verified-optimal-action record.

    Unlike :func:`build_record`, this label requires only one exact root A*
    proof.  The returned optimal path proves that its first action is optimal,
    but it deliberately leaves every other action's cost/regret unknown.  The
    completeness flags prevent downstream evaluation from treating that known
    optimal action as the exhaustive set of optimal actions.
    """
    if env.is_terminal(state):
        return None
    if not (
        result.solvable is True
        and result.optimal
        and result.move_count is not None
        and result.move_count > 0
        and result.serialized_actions
    ):
        return None

    proof_actions = [dict(action) for action in result.serialized_actions]
    selected_key = _normalized_hashable(proof_actions[0])
    legal_actions = sorted(
        (action_to_normalized(state, action) for action in env.legal_actions(state)),
        key=_normalized_hashable,
    )
    selected_indices = [
        index for index, action in enumerate(legal_actions)
        if _normalized_hashable(action) == selected_key
    ]
    if len(selected_indices) != 1:
        raise ValueError(
            "verified optimal path's first action is not uniquely legal at root")
    selected_index = selected_indices[0]
    action_costs: list[Optional[int]] = [None] * len(legal_actions)
    action_regrets: list[Optional[int]] = [None] * len(legal_actions)
    policy_target = [0.0] * len(legal_actions)
    # Q*(s, a_first) = V*(s) because the stored path is optimal.
    action_costs[selected_index] = result.move_count
    action_regrets[selected_index] = 0
    policy_target[selected_index] = 1.0

    return {
        "version": DATASET_VERSION,
        "label_kind": LABEL_EXACT_PATH_POLICY,
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": False,
        "action_values_complete": False,
        "level_id": level_id,
        "static_level_signature": static_level_signature(state.level),
        "level": level_to_dict(state.level),
        "state": serialize_state(state),
        "state_key": env.canonical_key(state),
        "remaining_blocks": state.remaining,
        "cleared_blocks": state.cleared,
        "legal_actions": legal_actions,
        "optimal_remaining_moves": result.move_count,
        # This is a proven-optimal subset, not an exhaustive set.
        "optimal_actions": [legal_actions[selected_index]],
        "action_costs": action_costs,
        "action_regrets": action_regrets,
        "policy": {
            "type": POLICY_SINGLE_VERIFIED_OPTIMAL,
            "temperature": None,
        },
        "policy_target": policy_target,
        "policy_proof": {
            "type": "verified-optimal-path",
            "length": result.move_count,
            "actions": proof_actions,
        },
        "value_target": value_target(
            result.move_count, constant=value_norm_constant),
        "target_source": "exact_astar_path",
        "provenance": [provenance] if provenance is not None else [],
    }
