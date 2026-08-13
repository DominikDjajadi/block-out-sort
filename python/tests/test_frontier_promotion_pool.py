"""Candidate-blind frontier promotion-pool contracts."""

from __future__ import annotations

import json

import pytest

from blocksort.cotraining.frontier_promotion_pool import (
    STRATA, build_exclusion_manifest, classify_frontier_stratum,
    load_exclusion_manifest)
from blocksort.serialization import level_from_dict, level_to_dict


def _tiny_level():
    return level_from_dict({
        "name": "tiny",
        "cols": 3,
        "rows": 3,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [
            {"edge": "left", "start": 1, "length": 1, "color": "red"}
        ],
    })


@pytest.mark.parametrize(("solved", "expected"), [
    ([True, True, True, True, True], STRATA[0]),
    ([False, True, True, True, True], STRATA[1]),
    ([False, False, True, True, True], STRATA[2]),
    ([False, False, False, True, True], STRATA[3]),
    ([False, False, False, False, True], STRATA[4]),
    ([False, False, False, False, False], STRATA[4]),
])
def test_classify_frontier_stratum(solved, expected) -> None:
    assert classify_frontier_stratum([20, 34, 57, 95, 160], solved) == expected


def test_exclusion_manifest_freezes_signatures(tmp_path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({
        "level": level_to_dict(_tiny_level()),
    }) + "\n", encoding="utf-8")
    output = tmp_path / "exclusions.json"

    built = build_exclusion_manifest(
        source_globs=[str(source)], output_path=output)
    loaded, signatures = load_exclusion_manifest(output)

    assert built["manifest_sha256"] == loaded["manifest_sha256"]
    assert len(signatures) == 1


def test_exclusion_manifest_detects_source_change(tmp_path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({
        "level": level_to_dict(_tiny_level()),
    }) + "\n", encoding="utf-8")
    output = tmp_path / "exclusions.json"
    build_exclusion_manifest(source_globs=[str(source)], output_path=output)
    source.write_text(source.read_text(encoding="utf-8") + "\n",
                      encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        load_exclusion_manifest(output)
