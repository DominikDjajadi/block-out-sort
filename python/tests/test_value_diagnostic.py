from __future__ import annotations

import argparse
import math

import pytest

from blocksort.cotraining import value_diagnostic as diagnostic


def _metadata():
    return [
        {
            "target": 1.0,
            "exact": True,
            "source": "exact_oracle",
            "iteration": 0,
            "remaining_blocks": 1,
            "static_level_signature": "a",
            "state_key": "a0",
        },
        {
            "target": 2.0,
            "exact": True,
            "source": "exact_oracle",
            "iteration": 2,
            "remaining_blocks": 2,
            "static_level_signature": "b",
            "state_key": "b0",
        },
        {
            "target": 4.0,
            "exact": False,
            "source": "graph_search",
            "iteration": 2,
            "remaining_blocks": 3,
            "static_level_signature": "c",
            "state_key": "c0",
        },
    ]


def test_metric_summary_reports_signed_bias_and_error():
    result = diagnostic._metric_summary(
        _metadata()[:2], [2.0, 2.0])

    assert result["count"] == 2
    assert result["bias_raw_moves"] == pytest.approx(0.5)
    assert result["mae_raw_moves"] == pytest.approx(0.5)
    assert result["rmse_raw_moves"] == pytest.approx(math.sqrt(0.5))
    assert result["within_one_move_rate"] == 1.0
    assert result["pearson"] is None


def test_model_report_keeps_exact_and_search_labels_separate():
    result = diagnostic._model_report(
        _metadata(), [1.0, 3.0, 8.0])

    assert result["exact_labels"]["count"] == 2
    assert result["exact_labels"]["mae_raw_moves"] == pytest.approx(0.5)
    assert result["search_labels"]["count"] == 1
    assert result["search_labels"]["mae_raw_moves"] == pytest.approx(4.0)
    assert set(result["by_source_and_iteration"]) == {
        "exact_oracle:iteration_0",
        "exact_oracle:iteration_2",
        "graph_search:iteration_2",
    }


def test_comparison_reports_candidate_minus_reference():
    reference = diagnostic._model_report(
        _metadata(), [1.0, 2.0, 4.0])
    candidate = diagnostic._model_report(
        _metadata(), [2.0, 4.0, 7.0])

    result = diagnostic._comparison(reference, candidate)

    assert result["exact_labels"]["mae_delta_raw_moves"] == pytest.approx(1.5)
    assert result["search_labels"]["bias_delta_raw_moves"] == pytest.approx(3.0)


def test_parse_checkpoint_rejects_invalid_syntax():
    with pytest.raises(argparse.ArgumentTypeError):
        diagnostic._parse_checkpoint("missing-path")
