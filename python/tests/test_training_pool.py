from __future__ import annotations

import json
from pathlib import Path

import pytest

from blocksort.dataset.training_pool import (
    DEFAULT_STRATA,
    TrainingPoolConfig,
    _verify_generated_pool,
    generate_training_pool,
    load_excluded_signatures,
)
from blocksort.serialization import level_from_dict
from blocksort.signature import static_level_signature


def _simple_level(name: str = "frozen") -> dict:
    return {
        "name": name,
        "rows": 1,
        "cols": 1,
        "blocks": [{"color": "red", "cells": [[0, 0]]}],
        "exits": [
            {"edge": "left", "start": 0, "length": 1, "color": "red"}
        ],
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_training_pool_requires_frozen_exclusion_source(tmp_path):
    cfg = TrainingPoolConfig(
        output=str(tmp_path / "levels.json"),
        exclude_level_files=(),
        per_stratum_count=1,
    )
    with pytest.raises(ValueError, match="frozen level source"):
        generate_training_pool(cfg)


def test_exclusions_accept_direct_wrapped_and_manifest_identities(tmp_path):
    level = _simple_level()
    signature = static_level_signature(level_from_dict(level))
    direct = tmp_path / "direct.json"
    wrapped = tmp_path / "wrapped.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_json(direct, [level])
    wrapped.write_text(
        json.dumps({"level": level, "static_level_signature": signature}) + "\n",
        encoding="utf-8",
    )
    _write_json(manifest, {
        "promotion_validation": [{"signature": signature}],
        "final_test": [{"signature": "sealed-signature"}],
    })

    signatures, sources = load_excluded_signatures(
        (str(direct), str(wrapped), str(manifest)))

    assert signatures == {signature, "sealed-signature"}
    assert len(sources) == 3
    assert all(source["signature_count"] >= 1 for source in sources)


def test_generation_is_balanced_disjoint_and_reproducible(tmp_path):
    frozen = tmp_path / "frozen.json"
    _write_json(frozen, [_simple_level()])

    reports = []
    output_bytes = []
    for run in ("a", "b"):
        output = tmp_path / f"levels_{run}.json"
        report = generate_training_pool(TrainingPoolConfig(
            output=str(output),
            report=str(tmp_path / f"report_{run}.json"),
            exclude_level_files=(str(frozen),),
            per_stratum_count=1,
            seed=8765,
            density_min=0.35,
            density_max=0.45,
            mutation_budget_min=2,
            mutation_budget_max=4,
        ))
        reports.append(report)
        output_bytes.append(output.read_bytes())

    assert output_bytes[0] == output_bytes[1]
    expected = {f"{rows}x{cols}_c{colors}": 1
                for rows, cols, colors in DEFAULT_STRATA}
    assert reports[0]["verification"]["observed_stratum_counts"] == expected
    assert reports[0]["verification"]["frozen_signature_overlap_count"] == 0
    assert reports[0]["output"]["level_count"] == 4
    assert reports[0]["output"]["unique_static_signature_count"] == 4

    generated = json.loads((tmp_path / "levels_a.json").read_text(encoding="utf-8"))
    frozen_signature = static_level_signature(level_from_dict(generated[0]))
    with pytest.raises(RuntimeError, match="overlaps frozen"):
        _verify_generated_pool(generated, {frozen_signature}, per_stratum_count=1)


def test_generation_refuses_to_overwrite_artifacts(tmp_path):
    frozen = tmp_path / "frozen.json"
    output = tmp_path / "levels.json"
    _write_json(frozen, [_simple_level()])
    output.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_training_pool(TrainingPoolConfig(
            output=str(output),
            exclude_level_files=(str(frozen),),
            per_stratum_count=1,
        ))
