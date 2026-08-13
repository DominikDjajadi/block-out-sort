"""Replay persisted co-training updates cumulatively on one offline learner.

This diagnostic keeps the official champion immutable.  It consumes each
round's exact persisted sample, policy/value weights, and training seed in
order, resetting the optimizer between rounds just as live co-training does.
The only intended difference from the reset control is that round N starts
from cumulative learner N-1 instead of the unchanged champion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..expert_iteration.train import (
    configure_trainable_part,
    train_expert,
    value_supervision_weights_for,
)
from ..model_identity import model_state_sha256
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from ..training.dataset import load_records
from ..training.transaction import atomic_write_json, sha256_file


SCHEMA_VERSION = 1
SEMANTICS = "offline_cumulative_persisted_replay_v1"


@dataclass(frozen=True)
class CumulativeReplayConfig:
    champion_checkpoint: str
    source_run: str
    output_dir: str
    milestones: tuple[int, ...] = (5, 10, 20, 35)
    device: str = "cpu"

    def validate(self) -> None:
        if not Path(self.champion_checkpoint).is_file():
            raise ValueError("champion checkpoint does not exist")
        source = Path(self.source_run)
        if not source.is_dir():
            raise ValueError("source co-training run does not exist")
        if not (source / "config.json").is_file():
            raise ValueError("source co-training run lacks config.json")
        if not (source / "run_state.json").is_file():
            raise ValueError("source co-training run lacks run_state.json")
        if (not self.milestones
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in self.milestones)
                or tuple(sorted(set(self.milestones))) != self.milestones):
            raise ValueError(
                "milestones must be unique, increasing positive integers")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate_weights(
    values: Any,
    *,
    expected_length: int,
    label: str,
) -> list[float]:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(
            f"{label} must contain one value per persisted training record")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _referenced_artifact(
    round_dir: Path,
    manifest: dict[str, Any],
    key: str,
) -> tuple[Path, str]:
    item = manifest.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"training sample manifest lacks {key}")
    path = round_dir / str(item.get("path", ""))
    expected = str(item.get("sha256", ""))
    if not path.is_file() or not expected:
        raise ValueError(f"training sample manifest has invalid {key}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"persisted {key} hash mismatch: {path}")
    return path, observed


def _round_identity(
    source: Path,
    round_number: int,
    *,
    champion_sha256: str,
) -> dict[str, Any]:
    round_dir = source / f"round_{round_number:03d}"
    manifest_path = round_dir / "training_sample_manifest.json"
    report_path = round_dir / "report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError(
            f"source round {round_number} lacks persisted training evidence")
    manifest = _load_json(manifest_path)
    if int(manifest.get("round", -1)) != round_number:
        raise ValueError(f"source round manifest number differs: {round_number}")
    if int(manifest.get("record_count", 0)) <= 0:
        raise ValueError(f"source round has no training records: {round_number}")
    targets = manifest.get("policy_targets", {})
    if targets.get("incumbent_checkpoint_sha256") != champion_sha256:
        raise ValueError(
            f"source round {round_number} targets a different incumbent")
    artifacts = {}
    for key in (
            "source_sample", "sample", "policy_weights", "value_weights",
            "effective_value_weights"):
        _path, digest = _referenced_artifact(round_dir, manifest, key)
        artifacts[key] = digest
    summary_item = targets.get("summary", {})
    summary_path = round_dir / str(summary_item.get("path", ""))
    if (not summary_path.is_file()
            or sha256_file(summary_path) != summary_item.get("sha256")):
        raise RuntimeError(
            f"source round {round_number} policy-target summary mismatch")
    report = _load_json(report_path)
    protagonist = report.get("protagonist", {})
    if not protagonist.get("training_performed"):
        raise ValueError(f"source round {round_number} did not train")
    reset_checkpoint = source / str(
        protagonist.get("candidate_checkpoint", ""))
    reset_checkpoint_sha256 = str(
        protagonist.get("candidate_checkpoint_sha256", ""))
    if (not reset_checkpoint.is_file() or not reset_checkpoint_sha256
            or sha256_file(reset_checkpoint) != reset_checkpoint_sha256):
        raise RuntimeError(
            f"source round {round_number} reset candidate integrity failure")
    return {
        "round": round_number,
        "record_count": int(manifest["record_count"]),
        "training_seed": int(manifest["training_seed"]),
        "policy_target_profile": targets.get("profile"),
        "policy_target_sha256": targets.get("policy_target_sha256"),
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "artifacts": artifacts,
        "reset_candidate_checkpoint": str(
            protagonist.get("candidate_checkpoint")),
        "reset_candidate_checkpoint_sha256": reset_checkpoint_sha256,
        "reset_promotion_score": protagonist.get("promotion_score_candidate"),
        "reset_promoted": bool(protagonist.get("promoted", False)),
    }


def build_experiment_identity(
    cfg: CumulativeReplayConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate source artifacts and return immutable experiment identity."""
    cfg.validate()
    source = Path(cfg.source_run)
    champion_sha256 = sha256_file(cfg.champion_checkpoint)
    source_config = _load_json(source / "config.json")
    source_state = _load_json(source / "run_state.json")
    completed = tuple(int(value) for value in source_state["completed_rounds"])
    last_round = cfg.milestones[-1]
    if completed != tuple(range(1, max(completed) + 1)):
        raise ValueError("source run completed rounds are not contiguous")
    if last_round > max(completed):
        raise ValueError("milestone exceeds completed source rounds")
    if source_config.get("trainable_part") != "policy_adapter":
        raise ValueError("source run did not use policy_adapter training")
    if source_config.get("policy_target_profile") != "incumbent_optimal":
        raise ValueError("source run did not use incumbent_optimal targets")
    if float(source_config.get("search_value_loss_weight", -1)) != 0.0:
        raise ValueError("source run enabled approximate search value loss")
    rounds = [
        _round_identity(
            source, round_number, champion_sha256=champion_sha256)
        for round_number in range(1, last_round + 1)
    ]
    semantic_config = {
        "champion_checkpoint": cfg.champion_checkpoint,
        "source_run": cfg.source_run,
        "milestones": list(cfg.milestones),
        "device": cfg.device,
        "training": {
            key: source_config[key]
            for key in (
                "epochs", "batch_size", "learning_rate", "weight_decay",
                "grad_clip", "trainable_part", "search_value_loss_weight",
                "policy_target_profile")
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": semantic_config,
        "inputs": {
            "champion_checkpoint_sha256": champion_sha256,
            "source_config_sha256": sha256_file(source / "config.json"),
            "source_run_state_sha256": sha256_file(source / "run_state.json"),
            "rounds": rounds,
        },
    }
    result["fingerprint"] = _canonical_sha256(result)
    return result, source_config


def _load_round_training_inputs(
    source: Path,
    round_identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    round_dir = source / f"round_{round_identity['round']:03d}"
    manifest = _load_json(round_dir / "training_sample_manifest.json")
    sample_path, _ = _referenced_artifact(round_dir, manifest, "sample")
    policy_path, _ = _referenced_artifact(
        round_dir, manifest, "policy_weights")
    value_path, _ = _referenced_artifact(
        round_dir, manifest, "value_weights")
    effective_path, _ = _referenced_artifact(
        round_dir, manifest, "effective_value_weights")
    records = load_records(sample_path)
    expected = int(round_identity["record_count"])
    if len(records) != expected:
        raise RuntimeError("persisted sample record count changed")
    policy_weights = _validate_weights(
        _load_json(policy_path), expected_length=expected,
        label="policy weights")
    value_weights = _validate_weights(
        _load_json(value_path), expected_length=expected,
        label="value weights")
    effective = _validate_weights(
        _load_json(effective_path), expected_length=expected,
        label="effective value weights")
    recomputed = value_supervision_weights_for(
        records, value_weights, search_value_loss_weight=0.0)
    if any(not math.isclose(left, right, abs_tol=1e-12)
           for left, right in zip(effective, recomputed)):
        raise RuntimeError("persisted effective value weights do not reproduce")
    return records, policy_weights, value_weights


def _valid_cached_round(
    checkpoint_path: Path,
    summary_path: Path,
    *,
    fingerprint: str,
    round_identity: dict[str, Any],
    predecessor_sha256: str,
) -> dict[str, Any] | None:
    if not checkpoint_path.exists() and not summary_path.exists():
        return None
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise RuntimeError("partial cumulative round output exists")
    summary = _load_json(summary_path)
    expected = {
        "experiment_fingerprint": fingerprint,
        "round": round_identity["round"],
        "source_manifest_sha256": round_identity["manifest_sha256"],
        "predecessor_checkpoint_sha256": predecessor_sha256,
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError("cached cumulative round has incompatible identity")
    if sha256_file(checkpoint_path) != summary.get("checkpoint_sha256"):
        raise RuntimeError("cached cumulative checkpoint integrity failure")
    return summary


def run_cumulative_replay(cfg: CumulativeReplayConfig) -> dict[str, Any]:
    identity, source_config = build_experiment_identity(cfg)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "experiment.json"
    if identity_path.exists():
        if _load_json(identity_path) != identity:
            raise RuntimeError("output directory belongs to another experiment")
    else:
        atomic_write_json(identity_path, identity)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = root / "rounds"
    summary_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    champion = load_checkpoint(cfg.champion_checkpoint, map_location="cpu")
    encoding, model_config, value_norm = configs_from_checkpoint(champion)
    model = model_from_checkpoint(champion, map_location=device)
    predecessor_path = Path(cfg.champion_checkpoint)
    predecessor_sha256 = identity["inputs"]["champion_checkpoint_sha256"]
    round_summaries = []
    source = Path(cfg.source_run)

    for round_identity in identity["inputs"]["rounds"]:
        round_number = int(round_identity["round"])
        checkpoint_path = checkpoint_dir / f"round_{round_number:03d}.pt"
        summary_path = summary_dir / f"round_{round_number:03d}.json"
        cached = _valid_cached_round(
            checkpoint_path,
            summary_path,
            fingerprint=identity["fingerprint"],
            round_identity=round_identity,
            predecessor_sha256=predecessor_sha256,
        )
        if cached is not None:
            print(f"reusing cumulative round {round_number}", flush=True)
            model = model_from_checkpoint(
                load_checkpoint(checkpoint_path, map_location="cpu"),
                map_location=device,
            )
            predecessor_path = checkpoint_path
            predecessor_sha256 = cached["checkpoint_sha256"]
            round_summaries.append(cached)
            continue

        print(
            f"training cumulative round {round_number}/"
            f"{cfg.milestones[-1]}", flush=True)
        records, policy_weights, value_weights = _load_round_training_inputs(
            source, round_identity)
        trainable_info = configure_trainable_part(
            model, source_config["trainable_part"])
        train_info = train_expert(
            model,
            records,
            policy_weights,
            value_weights=value_weights,
            encoding_config=encoding,
            value_norm=value_norm,
            epochs=int(source_config["epochs"]),
            batch_size=int(source_config["batch_size"]),
            learning_rate=float(source_config["learning_rate"]),
            weight_decay=float(source_config["weight_decay"]),
            grad_clip=float(source_config["grad_clip"]),
            device=device,
            seed=int(round_identity["training_seed"]),
            search_value_loss_weight=float(
                source_config["search_value_loss_weight"]),
        )
        train_info = {
            **train_info,
            **trainable_info,
            "source_round": round_number,
            "source_manifest_sha256": round_identity["manifest_sha256"],
            "policy_target_profile":
                round_identity["policy_target_profile"],
            "policy_target_sha256":
                round_identity["policy_target_sha256"],
            "predecessor_checkpoint": str(predecessor_path),
            "predecessor_checkpoint_sha256": predecessor_sha256,
        }
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=round_number,
            best_val_metric=None,
            encoding_config=encoding,
            model_config=model_config,
            value_norm=value_norm,
            seed=int(round_identity["training_seed"]),
            dataset_version=champion.get("dataset_version", 1),
            split_identity={
                "kind": SEMANTICS,
                "source_run": cfg.source_run,
                "source_round": round_number,
                "source_manifest_sha256":
                    round_identity["manifest_sha256"],
                "predecessor_checkpoint_sha256": predecessor_sha256,
            },
            metrics={"cumulative_replay": train_info},
            experiment_fingerprint=identity["fingerprint"],
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        cumulative_model_state_sha256 = model_state_sha256(model)
        reset_model_state_sha256 = None
        if round_number == 1 or round_number in cfg.milestones:
            reset_checkpoint_path = (
                source / round_identity["reset_candidate_checkpoint"])
            reset_checkpoint = load_checkpoint(
                reset_checkpoint_path, map_location="cpu")
            reset_model_state_sha256 = model_state_sha256(
                reset_checkpoint["model_state"])
        if (round_number == 1
                and cumulative_model_state_sha256
                != reset_model_state_sha256):
            raise RuntimeError(
                "cumulative round 1 did not reproduce the reset control")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "semantics": SEMANTICS,
            "experiment_fingerprint": identity["fingerprint"],
            "round": round_number,
            "milestone": round_number in cfg.milestones,
            "source_manifest_sha256": round_identity["manifest_sha256"],
            "source_reset_candidate_checkpoint_sha256":
                round_identity["reset_candidate_checkpoint_sha256"],
            "source_reset_model_state_sha256": reset_model_state_sha256,
            "source_reset_promotion_score":
                round_identity["reset_promotion_score"],
            "predecessor_checkpoint": str(predecessor_path),
            "predecessor_checkpoint_sha256": predecessor_sha256,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "model_state_sha256": cumulative_model_state_sha256,
            "train": train_info,
        }
        atomic_write_json(summary_path, summary)
        predecessor_path = checkpoint_path
        predecessor_sha256 = checkpoint_sha256
        round_summaries.append(summary)

    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "champion_checkpoint": cfg.champion_checkpoint,
        "champion_checkpoint_sha256":
            identity["inputs"]["champion_checkpoint_sha256"],
        "source_run": cfg.source_run,
        "rounds_replayed": len(round_summaries),
        "milestones": {
            str(summary["round"]): {
                "checkpoint": summary["checkpoint"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "model_state_sha256": summary["model_state_sha256"],
            }
            for summary in round_summaries
            if summary["milestone"]
        },
        "final_checkpoint": round_summaries[-1]["checkpoint"],
        "final_checkpoint_sha256":
            round_summaries[-1]["checkpoint_sha256"],
        "final_model_state_sha256":
            round_summaries[-1]["model_state_sha256"],
        "final_test_status": "sealed_not_evaluated",
    }
    atomic_write_json(root / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay persisted co-training samples cumulatively")
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=[5, 10, 20, 35])
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = CumulativeReplayConfig(
        champion_checkpoint=args.champion_checkpoint,
        source_run=args.source_run,
        output_dir=args.output_dir,
        milestones=tuple(args.milestones),
        device=args.device,
    )
    result = run_cumulative_replay(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
