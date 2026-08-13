"""Expert-iteration example records.

A record is a superset of the supervised dataset record (so the encoders can
consume it) plus expert-iteration metadata that keeps exact and approximate
labels distinguishable:

  target_source       : "exact_oracle" | "exact_astar_path" | "graph_search"
  value_exact         : bool
  teacher_checkpoint  : str | None
  generation_iteration: int
  state_provenance    : dict
  astar               : {"max_nodes", "termination_reason"}
  search (search only): {"simulations", "visit_counts", "policy_temperature",
                         "search_value_cost", "solved", "solution_length"}

Full-exact records carry exhaustive action labels. Exact-path records carry an
exact value plus one proven optimal action. Search records leave exact values
unknown (``None``).
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..dataset.schema import (
    DATASET_VERSION,
    LABEL_EXACT_PATH_POLICY,
    LABEL_FULL_EXACT,
    LABEL_SEARCH_VISIT_POLICY,
    build_exact_path_record,
    build_record,
    serialize_state,
)
from ..dataset.targets import DEFAULT_VALUE_NORM_CONSTANT
from ..oracle import StateAnalysis
from ..serialization import level_to_dict
from ..state import State, canonical_key
from ..training.validation import validate_positive_finite

SOURCE_EXACT = "exact_oracle"
SOURCE_EXACT_PATH = "exact_astar_path"
SOURCE_SEARCH = "graph_search"


_LABEL_METADATA_BY_SOURCE = {
    SOURCE_EXACT: {
        "label_kind": LABEL_FULL_EXACT,
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": True,
        "action_values_complete": True,
    },
    SOURCE_EXACT_PATH: {
        "label_kind": LABEL_EXACT_PATH_POLICY,
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    },
    SOURCE_SEARCH: {
        "label_kind": LABEL_SEARCH_VISIT_POLICY,
        "value_exact": False,
        "policy_exact": False,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    },
}


def ensure_record_label_metadata(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Fill legacy label metadata and reject contradictory source claims.

    Older replay snapshots predate the explicit completeness fields. They are
    migrated in memory from ``target_source``. A record that already claims
    incompatible metadata is rejected rather than silently being interpreted
    as stronger supervision.
    """
    source = record.get("target_source")
    expected = _LABEL_METADATA_BY_SOURCE.get(source)
    if expected is None:
        return record
    for field, value in expected.items():
        if field in record and record[field] != value:
            raise ValueError(
                f"{source} record has inconsistent {field}: "
                f"expected {value!r}, got {record[field]!r}")
        record.setdefault(field, value)
    return record


def dedup_key(record: dict[str, Any]) -> tuple[str, str]:
    return (record["static_level_signature"], record["state_key"])


def difficulty(record: dict[str, Any]) -> float:
    """A difficulty proxy used for replay preservation (higher = harder)."""
    if record.get("optimal_remaining_moves") is not None:
        return float(record["optimal_remaining_moves"])
    search = record.get("search") or {}
    if search.get("search_value_cost") is not None:
        return max(
            float(record.get("remaining_blocks", 0)),
            float(search["search_value_cost"]),
        )
    return float(record.get("remaining_blocks", 0))


def build_exact_example(
    analysis: StateAnalysis,
    state: State,
    *,
    level_id: str,
    iteration: int,
    astar_max_nodes: int,
    teacher_checkpoint: Optional[str],
    provenance: dict[str, Any],
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
) -> Optional[dict[str, Any]]:
    """An exact-oracle example, or ``None`` if the state is not fully exact."""
    record = build_record(
        analysis, state, level_id=level_id, policy_type="uniform-optimal",
        value_norm_constant=value_norm_constant, provenance=provenance,
    )
    if record is None:
        return None
    record["target_source"] = SOURCE_EXACT
    record["value_exact"] = True
    record["teacher_checkpoint"] = teacher_checkpoint
    record["generation_iteration"] = iteration
    record["state_provenance"] = provenance
    record["astar"] = {"max_nodes": astar_max_nodes, "termination_reason": "exact"}
    record["search"] = None
    return record


def build_exact_path_example(
    result,
    state: State,
    env: Environment,
    *,
    level_id: str,
    iteration: int,
    astar_max_nodes: int,
    teacher_checkpoint: Optional[str],
    provenance: dict[str, Any],
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
) -> Optional[dict[str, Any]]:
    """An exact-value example backed by one verified optimal A* path."""
    fallback_provenance = {
        **provenance,
        "labeling": {
            "strategy": "full_exact_then_exact_path",
            "fallback_reason": "successor_proof_incomplete",
        },
    }
    record = build_exact_path_record(
        result,
        state,
        env,
        level_id=level_id,
        value_norm_constant=value_norm_constant,
        provenance=fallback_provenance,
    )
    if record is None:
        return None
    record["target_source"] = SOURCE_EXACT_PATH
    record["value_exact"] = True
    record["teacher_checkpoint"] = teacher_checkpoint
    record["generation_iteration"] = iteration
    record["state_provenance"] = fallback_provenance
    record["astar"] = {
        "max_nodes": astar_max_nodes,
        "termination_reason": "exact_root_successor_incomplete",
    }
    record["search"] = None
    return record


def build_search_example(
    result,
    state: State,
    *,
    level_id: str,
    static_signature: str,
    iteration: int,
    teacher_checkpoint: Optional[str],
    simulations: int,
    policy_temperature: float,
    provenance: dict[str, Any],
    astar_max_nodes: int,
    astar_reason: str,
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
) -> Optional[dict[str, Any]]:
    """A graph-search example built from a :class:`SearchResult`.

    The policy target is the normalized root visit distribution; the value target
    is the (approximate) search cost estimate. Returns ``None`` for a state with
    no legal actions (nothing to learn).
    """
    legal = list(result.legal_action_locators)
    if not legal:
        return None
    policy = [float(p) for p in result.visit_policy]
    reported_cost = float(result.search_value_cost)
    const = validate_positive_finite(
        "search-example value normalization constant", value_norm_constant)
    if not math.isfinite(reported_cost):
        raise ValueError(
            f"search value cost must be finite; got {reported_cost!r}")

    # Search Q-values are estimates, not exact distances.  Nevertheless every
    # remaining block requires its own removal action, so no valid remaining-
    # move estimate may fall below ``state.remaining``.  A replay-verified
    # solution length, when present, is a valid upper bound.  Persist both the
    # original estimate and the bounds so uncertainty is never hidden.
    lower_bound = int(state.remaining)
    solved = bool(result.solved)
    solution_verified = bool(getattr(result, "solution_verified", solved))
    solution_length = result.solution_length
    upper_bound = (
        int(solution_length)
        if solved and solution_verified and solution_length is not None
        else None
    )
    if upper_bound is not None and upper_bound < lower_bound:
        raise ValueError(
            "verified search solution length cannot be below remaining blocks: "
            f"{upper_bound} < {lower_bound}")
    bounded_cost = max(float(lower_bound), reported_cost)
    if upper_bound is not None:
        bounded_cost = min(bounded_cost, float(upper_bound))

    return {
        "version": DATASET_VERSION,
        "label_kind": LABEL_SEARCH_VISIT_POLICY,
        "value_exact": False,
        "policy_exact": False,
        "optimal_actions_complete": False,
        "action_values_complete": False,
        "level_id": level_id,
        "static_level_signature": static_signature,
        "level": level_to_dict(state.level),
        "state": serialize_state(state),
        "state_key": canonical_key(state),
        "remaining_blocks": state.remaining,
        "cleared_blocks": state.cleared,
        "legal_actions": legal,
        "policy_target": policy,
        # Exact labels are unknown for search examples.
        "optimal_remaining_moves": None,
        "optimal_actions": [],
        "action_costs": [None] * len(legal),
        "action_regrets": [None] * len(legal),
        "policy": {"type": "search-visits", "temperature": policy_temperature},
        "value_target": {
            # Kept under the legacy field name for encoder compatibility; the
            # estimate_kind and bounds make clear that this is not exact V*.
            "raw_optimal_moves": bounded_cost,
            "normalized_value": -bounded_cost / const,
            "normalization": {"scheme": "neg_over_constant", "constant": const},
            "estimate_kind": "bounded_search_estimate",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
        "target_source": SOURCE_SEARCH,
        "teacher_checkpoint": teacher_checkpoint,
        "generation_iteration": iteration,
        "state_provenance": provenance,
        "astar": {"max_nodes": astar_max_nodes, "termination_reason": astar_reason},
        "search": {
            "simulations": int(simulations),
            "visit_counts": [int(n) for n in result.visit_counts],
            "policy_temperature": policy_temperature,
            "search_value_cost": bounded_cost,
            "reported_search_value_cost": reported_cost,
            "value_lower_bound": lower_bound,
            "value_upper_bound": upper_bound,
            "solved": solved,
            "solution_verified": solution_verified,
            "solution_length": solution_length,
        },
        "provenance": [provenance],
    }


def tag_historical(record: dict[str, Any]) -> dict[str, Any]:
    """Tag an existing supervised dataset record as historical exact replay."""
    record = dict(record)
    record.setdefault("target_source", SOURCE_EXACT)
    record.setdefault("value_exact", True)
    record.setdefault("teacher_checkpoint", None)
    record.setdefault("generation_iteration", 0)
    record.setdefault("state_provenance", {"sampling": "historical"})
    record.setdefault("astar", {"max_nodes": None, "termination_reason": "exact"})
    record.setdefault("search", None)
    return ensure_record_label_metadata(record)
