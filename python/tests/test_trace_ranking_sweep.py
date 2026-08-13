from __future__ import annotations

import json

import pytest

from blocksort.cotraining.trace_ranking_sweep import (
    TraceRankingSweepConfig,
    _parse_args,
    _preregistered_contract,
)
from blocksort.training.experiment_identity import hash_canonical_value
from blocksort.training.transaction import sha256_file


def _config_and_preregistration(tmp_path):
    paths = {}
    for name in (
            "champion.pt", "replay.jsonl", "trace.jsonl",
            "retention.jsonl"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    prereg = {
        "schema_version": 1,
        "semantics": "matched_trace_pair_ranking_sweep_preregistration_v1",
        "status": "frozen_before_matched_training",
        "fixed_inputs": {
            "checkpoint_sha256": sha256_file(paths["champion.pt"]),
            "trace_dataset_sha256": sha256_file(paths["trace.jsonl"]),
            "source_config_sha256": sha256_file(source / "config.json"),
            "replay_snapshot_sha256": sha256_file(paths["replay.jsonl"]),
        },
        "fixed_training": {"trace_margin": 0.05},
        "arms": {
            "control": {"trace_ranking_weight": 0.0},
            "light": {"trace_ranking_weight": 0.001},
            "moderate": {"trace_ranking_weight": 0.004},
        },
    }
    prereg["fingerprint"] = hash_canonical_value(prereg)
    prereg_path = tmp_path / "preregistration.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    cfg = TraceRankingSweepConfig(
        champion_checkpoint=str(paths["champion.pt"]),
        source_run=str(source),
        replay_snapshot=str(paths["replay.jsonl"]),
        trace_dataset=str(paths["trace.jsonl"]),
        retention_dataset=str(paths["retention.jsonl"]),
        preregistration=str(prereg_path),
        output_dir=str(tmp_path / "output"),
    )
    return cfg, prereg_path


def test_preregistered_contract_binds_inputs_and_coefficients(tmp_path) -> None:
    cfg, _ = _config_and_preregistration(tmp_path)

    prereg, arms, margin = _preregistered_contract(cfg)

    assert prereg["status"] == "frozen_before_matched_training"
    assert arms == {"control": 0.0, "light": 0.001, "moderate": 0.004}
    assert margin == 0.05


def test_preregistered_contract_rejects_tampering(tmp_path) -> None:
    cfg, path = _config_and_preregistration(tmp_path)
    prereg = json.loads(path.read_text(encoding="utf-8"))
    prereg["arms"]["light"]["trace_ranking_weight"] = 1.0
    path.write_text(json.dumps(prereg), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        _preregistered_contract(cfg)


def test_execution_controls_do_not_enter_sweep_config() -> None:
    cfg, arms, stop = _parse_args([
        "--champion-checkpoint", "champion.pt",
        "--source-run", "source",
        "--replay-snapshot", "replay.jsonl",
        "--trace-dataset", "trace.jsonl",
        "--retention-dataset", "retention.jsonl",
        "--preregistration", "prereg.json",
        "--output-dir", "output",
        "--arms", "control",
        "--stop-after-round", "1",
    ])

    assert isinstance(cfg, TraceRankingSweepConfig)
    assert arms == ("control",)
    assert stop == 1
