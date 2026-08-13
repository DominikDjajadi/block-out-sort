from __future__ import annotations

import math

import pytest

from blocksort.cotraining import successor_ranking as ranking
from blocksort.dataset.schema import build_record
from blocksort.environment import Environment
from blocksort.oracle import Oracle
from blocksort.serialization import level_from_dict
from blocksort.training.config import EncodingConfig


def _example(costs):
    return {
        "level_id": "level",
        "static_level_signature": "signature",
        "state_key": "state",
        "optimal_remaining_moves": 3.0,
        "depth_bucket": "1_to_3",
        "remaining_blocks": 2,
        "action_costs": costs,
    }


def test_state_metrics_reports_correct_value_ranking():
    row = ranking._state_metrics(
        _example([3.0, 5.0, None]),
        [2.5, 4.0, 7.0],
    )

    assert row["decision_state"]
    assert row["selected_action_is_optimal"]
    assert row["selected_action_regret"] == 0.0
    assert row["optimal_pair_score"] == 2.0
    assert row["optimal_pair_count"] == 2
    assert row["all_pair_score"] == 3.0
    assert row["all_pair_count"] == 3
    assert row["best_optimal_vs_suboptimal_margin_raw_moves"] == 1.5


def test_state_metrics_counts_ties_and_reversals():
    row = ranking._state_metrics(
        _example([3.0, 4.0, 6.0]),
        [5.0, 5.0, 4.0],
    )

    assert not row["selected_action_is_optimal"]
    assert row["selected_action_regret"] == 3.0
    assert row["optimal_pair_score"] == 0.5
    assert row["optimal_pair_count"] == 2
    assert row["best_optimal_vs_suboptimal_margin_raw_moves"] == -1.0


def test_aggregate_excludes_forced_states_from_decision_rate():
    ranked = ranking._state_metrics(
        _example([3.0, 4.0]),
        [2.0, 3.0],
    )
    forced = ranking._state_metrics(
        _example([3.0]),
        [2.0],
    )

    result = ranking._aggregate([ranked, forced])

    assert result["states"] == 2
    assert result["decision_states"] == 1
    assert result["forced_or_all_optimal_states"] == 1
    assert result["top1_optimal_rate"] == 1.0


def test_aggregate_tracks_infinite_cost_choices():
    failed = ranking._state_metrics(
        _example([3.0, None]),
        [5.0, 2.0],
    )

    result = ranking._aggregate([failed])

    assert result["infinite_cost_choice_count"] == 1
    assert result["infinite_cost_choice_rate"] == 1.0
    assert result["mean_selected_oracle_regret"] is None


def test_comparison_uses_candidate_minus_reference():
    reference = ranking._model_report(
        [_example([3.0, 4.0])], [[2.0, 3.0]])
    candidate = ranking._model_report(
        [_example([3.0, 4.0])], [[3.0, 2.0]])

    result = ranking._comparison(reference, candidate)

    assert result["top1_optimal_rate_delta"] == -1.0
    assert result["optimal_vs_suboptimal_pair_accuracy_delta"] == -1.0
    assert result["mean_selected_oracle_regret_delta"] == 1.0


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (1, "1_to_3"),
        (3, "1_to_3"),
        (4, "4_to_6"),
        (6, "4_to_6"),
        (7, "7_to_9"),
        (9.9, "7_to_9"),
        (10, "10_plus"),
    ],
)
def test_depth_bucket(depth, expected):
    assert ranking._depth_bucket(depth) == expected


def test_ordering_score_treats_equal_predictions_as_half_credit():
    assert ranking._ordering_score(1.0, 2.0) == 1.0
    assert ranking._ordering_score(2.0, 1.0) == 0.0
    assert ranking._ordering_score(1.0, 1.0) == 0.5
    assert math.isclose(
        ranking._ordering_score(1.0, 1.0 + 1e-10), 0.5)


def test_prepare_examples_replays_exact_actions_and_successors():
    level = level_from_dict({
        "name": "ranking-smoke",
        "cols": 5,
        "rows": 5,
        "blocks": [
            {"color": "red", "cells": [[0, 0]]},
            {"color": "blue", "cells": [[4, 4]]},
        ],
        "exits": [
            {"edge": "top", "start": 0, "length": 1, "color": "red"},
            {"edge": "bottom", "start": 4, "length": 1, "color": "blue"},
        ],
    })
    env = Environment()
    state = env.initial_state(level)
    analysis = Oracle(env).analyze(state)
    record = build_record(
        analysis, state, level_id="ranking-smoke")
    assert record is not None

    examples, boards, global_features, locations = (
        ranking._prepare_examples(
            [record], EncodingConfig(), max_cost=1_000.0))

    assert len(examples) == 1
    assert len(examples[0]["action_costs"]) == len(record["legal_actions"])
    assert len(boards) == len(global_features) == len(locations)
    assert all(example_index == 0 for example_index, _ in locations)
    fixed = examples[0]["predicted_action_costs"]
    assert all(value is None for value in fixed)
    assert len(locations) == len(record["legal_actions"])
