from __future__ import annotations

import json

import pytest

from blocksort.cotraining.midbudget_dev_pool import STRATA
from blocksort.cotraining.replay_anchor_pool import (
    _annotated_records,
    _base_holdout_signatures,
    _load_replay_by_level,
    _verify_pool,
)
from blocksort.serialization import level_from_dict
from blocksort.signature import static_level_signature


def _level(name: str) -> dict:
    return {
        "name": name, "rows": 1, "cols": 1,
        "blocks": [{"color": name, "cells": [[0, 0]]}],
        "exits": [{
            "edge": "left", "start": 0, "length": 1, "color": name}],
    }


def _record(name: str, *, source: str = "exact_oracle") -> dict:
    level = _level(name)
    signature = static_level_signature(level_from_dict(level))
    return {
        "level": level, "static_level_signature": signature,
        "state_key": name, "target_source": source,
    }


def test_replay_loader_excludes_holdouts_and_nonexact_records(tmp_path):
    exact = _record("exact")
    excluded = _record("excluded")
    search = _record("search", source="graph_search")
    path = tmp_path / "replay.jsonl"
    path.write_text("".join(
        json.dumps(record) + "\n" for record in (exact, excluded, search)),
        encoding="utf-8")
    by_level, summary = _load_replay_by_level(
        path, excluded={excluded["static_level_signature"]})
    assert set(by_level) == {exact["static_level_signature"]}
    assert summary["source_records"] == 3
    assert summary["excluded_records"] == 1
    assert summary["nonexact_records"] == 1


def test_annotation_and_verification_balance_by_level_not_record():
    by_level = {}
    assignments = {}
    for index, band in enumerate(STRATA):
        first = _record(f"{band}-a")
        second = dict(first)
        second["state_key"] = f"{band}-second-state"
        signature = first["static_level_signature"]
        by_level[signature] = [first, second]
        assignments[signature] = {
            "difficulty_stratum": band,
            "champion_checkpoint_sha256": "champion",
            "solved": [None] * 9,
            "screening_seed": 7,
        }
    records = _annotated_records(by_level, assignments)
    report = _verify_pool(records, excluded=set(), levels_per_band=1)
    assert report["unique_level_count"] == 4
    assert report["record_count"] == 8
    assert report["level_counts_by_band"] == {band: 1 for band in STRATA}
    assert all("training_anchor" in record for record in records)


def test_anchor_verification_rejects_evaluation_overlap():
    record = _record("overlap")
    record["training_anchor"] = {"difficulty_stratum": STRATA[0]}
    with pytest.raises(RuntimeError, match="overlaps"):
        _verify_pool(
            [record], excluded={record["static_level_signature"]},
            levels_per_band=1)


def test_base_split_excludes_only_validation_and_test(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "train_levels": ["train"],
        "validation_levels": ["validation"],
        "test_levels": ["test"],
    }), encoding="utf-8")
    assert _base_holdout_signatures(path) == {"validation", "test"}
