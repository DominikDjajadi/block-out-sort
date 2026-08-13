"""Tests for the operational graph-search latency benchmark."""

from __future__ import annotations

import pytest

from blocksort.cotraining.latency_benchmark import (
    _percentile, _summarize_rows)


def test_percentile_interpolates() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_summarize_rows_reports_latency_and_throughput() -> None:
    rows = [
        {
            "elapsed_seconds": 0.1,
            "solved": True,
            "budget": 8,
            "level_index": 1,
            "repeat": 0,
        },
        {
            "elapsed_seconds": 0.3,
            "solved": False,
            "budget": 8,
            "level_index": 2,
            "repeat": 0,
        },
    ]

    result = _summarize_rows(rows)

    assert result["requests"] == 2
    assert result["unique_levels"] == 2
    assert result["solve_rate"] == 0.5
    assert result["latency_ms"]["mean"] == pytest.approx(200.0)
    assert result["latency_ms"]["p50"] == pytest.approx(200.0)
    assert result["throughput"]["requests_per_second"] == pytest.approx(5.0)
    assert result["throughput"]["nominal_simulations_per_second"] == \
        pytest.approx(40.0)
