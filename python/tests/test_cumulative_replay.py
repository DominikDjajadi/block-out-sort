from __future__ import annotations

import json

import pytest

from blocksort.cotraining import cumulative_replay
from blocksort.training.transaction import sha256_file


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_round(root, round_number, champion_sha256):
    round_dir = root / f"round_{round_number:03d}"
    round_dir.mkdir()
    files = {
        "source_sample": ("training_sample_source.jsonl", "{}\n"),
        "sample": ("training_sample.jsonl", "{}\n"),
        "policy_weights": ("training_policy_weights.json", "[1.0]"),
        "value_weights": ("training_value_weights.json", "[1.0]"),
        "effective_value_weights": (
            "training_effective_value_weights.json", "[1.0]"),
    }
    manifest_items = {}
    for key, (name, text) in files.items():
        path = round_dir / name
        path.write_text(text, encoding="utf-8")
        manifest_items[key] = {"path": name, "sha256": sha256_file(path)}
    target_summary = round_dir / "training_policy_target_summary.json"
    _write_json(target_summary, {"round": round_number})
    manifest = {
        "round": round_number,
        "record_count": 1,
        **manifest_items,
        "policy_targets": {
            "profile": "incumbent_optimal",
            "policy_target_sha256": f"target-{round_number}",
            "incumbent_checkpoint_sha256": champion_sha256,
            "summary": {
                "path": target_summary.name,
                "sha256": sha256_file(target_summary),
            },
        },
        "training_seed": 100 + round_number,
    }
    _write_json(round_dir / "training_sample_manifest.json", manifest)
    candidate = round_dir / "candidate.pt"
    candidate.write_bytes(f"candidate-{round_number}".encode("ascii"))
    _write_json(round_dir / "report.json", {
        "protagonist": {
            "training_performed": True,
            "candidate_checkpoint": f"round_{round_number:03d}/candidate.pt",
            "candidate_checkpoint_sha256": sha256_file(candidate),
            "promotion_score_candidate": 0.1,
            "promoted": False,
        },
    })


def test_validate_weights_rejects_bad_length_and_nonfinite():
    with pytest.raises(ValueError, match="one value"):
        cumulative_replay._validate_weights(
            [1.0], expected_length=2, label="policy weights")
    with pytest.raises(ValueError, match="finite"):
        cumulative_replay._validate_weights(
            [float("nan")], expected_length=1, label="policy weights")


def test_build_identity_verifies_ordered_persisted_rounds(tmp_path):
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"champion")
    champion_sha256 = sha256_file(champion)
    source = tmp_path / "source"
    source.mkdir()
    _write_json(source / "config.json", {
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "trainable_part": "policy_adapter",
        "search_value_loss_weight": 0.0,
        "policy_target_profile": "incumbent_optimal",
    })
    _write_json(source / "run_state.json", {"completed_rounds": [1, 2]})
    _write_round(source, 1, champion_sha256)
    _write_round(source, 2, champion_sha256)
    cfg = cumulative_replay.CumulativeReplayConfig(
        champion_checkpoint=str(champion),
        source_run=str(source),
        output_dir=str(tmp_path / "output"),
        milestones=(1, 2),
    )

    identity, source_config = cumulative_replay.build_experiment_identity(cfg)

    assert identity["semantics"] == \
        "offline_cumulative_persisted_replay_v1"
    assert [item["round"] for item in identity["inputs"]["rounds"]] == [1, 2]
    assert [item["training_seed"] for item in identity["inputs"]["rounds"]] == [
        101, 102]
    assert identity["inputs"]["champion_checkpoint_sha256"] == \
        champion_sha256
    assert source_config["trainable_part"] == "policy_adapter"

    (source / "round_002" / "training_policy_weights.json").write_text(
        "[2.0]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        cumulative_replay.build_experiment_identity(cfg)


def test_build_identity_rejects_targets_from_another_incumbent(tmp_path):
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"champion")
    source = tmp_path / "source"
    source.mkdir()
    _write_json(source / "config.json", {
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "trainable_part": "policy_adapter",
        "search_value_loss_weight": 0.0,
        "policy_target_profile": "incumbent_optimal",
    })
    _write_json(source / "run_state.json", {"completed_rounds": [1]})
    _write_round(source, 1, "different-champion")
    cfg = cumulative_replay.CumulativeReplayConfig(
        champion_checkpoint=str(champion),
        source_run=str(source),
        output_dir=str(tmp_path / "output"),
        milestones=(1,),
    )

    with pytest.raises(ValueError, match="different incumbent"):
        cumulative_replay.build_experiment_identity(cfg)
