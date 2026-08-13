from __future__ import annotations

import json
from types import SimpleNamespace

from blocksort.cotraining.trace_ranking_pool import (
    _build_preference_record,
    _load_training_levels,
    _preference_rejection_reason,
    _reconstruct_divergence_state,
)
from blocksort.environment import Environment
from blocksort.oracle import Oracle
from blocksort.serialization import level_from_dict
from blocksort.signature import static_level_signature


def _two_exit_level() -> dict:
    return {
        "name": "two-optimal-exits",
        "rows": 1,
        "cols": 1,
        "blocks": [{"color": "red", "cells": [[0, 0]]}],
        "exits": [
            {"edge": "left", "start": 0, "length": 1, "color": "red"},
            {"edge": "right", "start": 0, "length": 1, "color": "red"},
        ],
    }


def test_training_level_loader_deduplicates_states_by_level(tmp_path) -> None:
    raw_level = _two_exit_level()
    signature = static_level_signature(level_from_dict(raw_level))
    records = [
        {
            "level": raw_level,
            "level_id": "level",
            "static_level_signature": signature,
            "state_key": state_key,
        }
        for state_key in ("a", "b")
    ]
    path = tmp_path / "pool.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    levels, states = _load_training_levels(path)

    assert set(levels) == {signature}
    assert set(states) == {(signature, "a"), (signature, "b")}


def test_preference_requires_both_actions_to_be_oracle_optimal() -> None:
    optimal = SimpleNamespace(serialized={
        "color": "red", "cells": [[0, 0]], "dir": "left",
        "distance": 0, "exit": True,
    }, optimal=True)
    nonoptimal = SimpleNamespace(serialized={
        "color": "red", "cells": [[0, 0]], "dir": "right",
        "distance": 0, "exit": True,
    }, optimal=False)
    analysis = SimpleNamespace(
        exact=True,
        solvable=True,
        all_successors_exact=True,
        actions=(optimal, nonoptimal),
    )

    assert _preference_rejection_reason(
        analysis, optimal.serialized, nonoptimal.serialized,
    ) == "competing_branch_not_oracle_optimal"


def test_preference_record_is_symmetric_for_candidate_success() -> None:
    env = Environment()
    level = level_from_dict(_two_exit_level())
    state = env.initial_state(level)
    analysis = Oracle(env, max_nodes=100).analyze(state)
    actions = [item.serialized for item in analysis.actions if item.optimal]
    assert len(actions) == 2
    divergence = {
        "simulation": 3,
        "divergence_depth": 0,
        "kind": "root_selection",
        "shared_selection_node": {
            "prior_l1_distance": 0.02,
            "max_prior_absolute_delta": 0.01,
            "node_value_cost_delta": 0.0,
        },
    }
    outcomes = {
        "incumbent": {
            "solved": False,
            "first_solution_simulation": None,
        },
        "candidate": {
            "solved": True,
            "first_solution_simulation": 90,
        },
    }

    record = _build_preference_record(
        analysis=analysis,
        state=state,
        source_level_id="level",
        direction="candidate_only",
        preferred_action=actions[1],
        competing_action=actions[0],
        divergence=divergence,
        outcomes=outcomes,
        checkpoint_sha256={"incumbent": "old", "candidate": "new"},
        budget=95,
        trace_seed=7,
        difficulty_stratum="band",
    )

    preference = record["trace_preference"]
    assert preference["successful_role"] == "candidate"
    assert preference["unsuccessful_role"] == "incumbent"
    assert preference["successful_checkpoint_sha256"] == "new"
    assert preference["both_actions_full_exact_oracle_optimal"] is True
    assert preference["preferred_action_index"] \
        != preference["competing_action_index"]


def test_reconstruct_divergence_accepts_in_memory_tuple_node_key() -> None:
    env = Environment()
    level = level_from_dict(_two_exit_level())
    state = env.initial_state(level)
    signature = static_level_signature(level)
    trace = {"timeline": [{"path_locators": []}]}
    divergence = {
        "simulation": 1,
        "divergence_depth": 0,
        "shared_selection_node": {
            "node_key": (signature, env.canonical_key(state)),
        },
    }

    reconstructed = _reconstruct_divergence_state(
        env, state, trace, divergence)

    assert env.canonical_key(reconstructed) == env.canonical_key(state)
