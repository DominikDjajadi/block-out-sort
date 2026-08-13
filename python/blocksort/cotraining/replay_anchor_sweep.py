"""Matched offline sweep for baseline-difficulty replay anchoring.

The experiment reconstructs each persisted co-training sample from a terminal
replay snapshot, verifies it against the original per-round manifest, and then
replays the 30 updates in three arms.  Current and recent records are identical
in every arm.  The light and moderate arms replace only 200 or 400 records from
the 800-record historical quota with a level-balanced, training-only anchor
pool.

At the original five-round milestones, the existing held-out retention guard
is evaluated and enforced.  The control arm is additionally required to
reproduce the source run's pre-rollback candidate model state at every round.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ..environment import Environment
from ..expert_iteration.records import dedup_key
from ..expert_iteration.replay import (
    REPLAY_AGE_BUCKETS,
    REPLAY_AGE_CURRENT,
    REPLAY_AGE_HISTORICAL,
    REPLAY_AGE_RECENT,
    ReplayBuffer,
    replay_age_bucket,
)
from ..expert_iteration.train import (
    configure_trainable_part,
    source_weights_for,
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
from ..training.experiment_identity import hash_canonical_value
from ..training.transaction import atomic_write_json, sha256_file
from .policy_targets import (
    condition_policy_targets,
    incumbent_legal_probabilities,
)
from .retention import (
    evaluate_retention,
    load_retention_pool,
    summarize_retention,
)


SCHEMA_VERSION = 1
SEMANTICS = "matched_baseline_difficulty_replay_anchor_sweep_v1"
ARM_ANCHOR_COUNTS = {"control": 0, "light": 200, "moderate": 400}


@dataclass(frozen=True)
class ReplayAnchorSweepConfig:
    champion_checkpoint: str
    source_run: str
    replay_snapshot: str
    anchor_pool: str
    retention_dataset: str
    output_dir: str
    rounds: int = 30
    milestone_interval: int = 5
    device: str = "auto"

    def validate(self) -> None:
        for label, value in (
                ("champion checkpoint", self.champion_checkpoint),
                ("replay snapshot", self.replay_snapshot),
                ("anchor pool", self.anchor_pool),
                ("retention dataset", self.retention_dataset)):
            if not Path(value).is_file():
                raise ValueError(f"{label} does not exist: {value}")
        if not Path(self.source_run).is_dir():
            raise ValueError("source run does not exist")
        if self.rounds <= 0 or self.milestone_interval <= 0:
            raise ValueError("rounds and milestone interval must be positive")
        if self.rounds % self.milestone_interval:
            raise ValueError("rounds must end on a retention milestone")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate_source_config(source_config: dict[str, Any]) -> None:
    expected = {
        "train_sample_size": 2000,
        "replay_current_fraction": 0.35,
        "replay_recent_fraction": 0.25,
        "replay_historical_fraction": 0.40,
        "replay_recent_window": 2,
        "replay_sample_with_replacement": False,
        "trainable_part": "policy_adapter",
        "policy_target_profile": "incumbent_optimal",
        "search_value_loss_weight": 0.0,
        "learner_retention_enforce": True,
    }
    for key, value in expected.items():
        if source_config.get(key) != value:
            raise ValueError(
                f"source config {key}={source_config.get(key)!r}; "
                f"expected {value!r}")


def reconstruct_source_sample(
    replay_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the exact raw age-quota sample described by a manifest."""
    sampling = manifest["sampling"]
    round_number = int(manifest["round"])
    records = [
        row for row in replay_records
        if int(row.get("generation_iteration", 0)) <= round_number
    ]
    buckets = {name: [] for name in REPLAY_AGE_BUCKETS}
    for row in records:
        buckets[replay_age_bucket(
            row,
            current_iteration=round_number,
            recent_window=int(sampling["recent_window"]),
        )].append(row)
    realized = sampling["realized"]
    available = {
        name: len(buckets[name]) for name in REPLAY_AGE_BUCKETS}
    if available != realized["available_records"]:
        raise RuntimeError(
            f"round {round_number} replay buckets do not reproduce: "
            f"{available} != {realized['available_records']}")
    weighting = manifest["weighting"]
    seed = int(sampling["seed"])
    sampled: list[dict[str, Any]] = []
    for index, name in enumerate(REPLAY_AGE_BUCKETS):
        count = int(realized["target_counts"][name])
        sampled.extend(ReplayBuffer._weighted_sample(
            buckets[name],
            count,
            current_iteration=round_number,
            weight_exact_historical=float(
                weighting["weight_exact_historical"]),
            weight_exact_new=float(weighting["weight_exact_new"]),
            weight_search=float(weighting["weight_search"]),
            seed=(seed * 1_000_003 + index) & 0xFFFFFFFF,
            with_replacement=bool(sampling["with_replacement"]),
        ))
    observed = _sha_bytes(_canonical_jsonl(sampled))
    expected = manifest["source_sample"]["sha256"]
    if observed != expected:
        raise RuntimeError(
            f"round {round_number} raw sample hash mismatch: "
            f"{observed} != {expected}")
    return sampled


def _anchor_groups(
    anchor_records: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for record in anchor_records:
        metadata = record.get("training_anchor")
        band = metadata.get("difficulty_stratum") \
            if isinstance(metadata, dict) else None
        signature = record.get("static_level_signature")
        if not isinstance(band, str) or not isinstance(signature, str):
            raise ValueError("anchor records require band and level metadata")
        groups[band][signature].append(record)
    if len(groups) != 4:
        raise ValueError("anchor pool must contain exactly four bands")
    for band, levels in groups.items():
        if len(levels) != 50:
            raise ValueError(
                f"anchor band {band!r} must contain exactly 50 levels")
        for rows in levels.values():
            rows.sort(key=lambda row: tuple(
                str(part) for part in dedup_key(row)))
    return {band: dict(levels) for band, levels in groups.items()}


def select_level_balanced_anchors(
    groups: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    count: int,
    round_number: int,
) -> list[dict[str, Any]]:
    """Select equal band/level coverage, rotating states between rounds."""
    if count == 0:
        return []
    if count % len(groups):
        raise ValueError("anchor count must divide evenly across bands")
    per_band = count // len(groups)
    selected: list[dict[str, Any]] = []
    for band in sorted(groups):
        levels = groups[band]
        signatures = sorted(levels)
        if per_band % len(signatures):
            raise ValueError("anchor quota must divide evenly across levels")
        per_level = per_band // len(signatures)
        for signature in signatures:
            rows = levels[signature]
            offset = (round_number - 1) % len(rows)
            for index in range(per_level):
                selected.append(rows[(offset + index) % len(rows)])
    if len(selected) != count:
        raise RuntimeError("anchor selection count mismatch")
    return selected


def compose_arm_sample(
    conditioned_source: list[dict[str, Any]],
    conditioned_anchors: list[dict[str, Any]],
    *,
    round_number: int,
    recent_window: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replace only the tail of the ordinary historical quota."""
    buckets = {name: [] for name in REPLAY_AGE_BUCKETS}
    for record in conditioned_source:
        buckets[replay_age_bucket(
            record,
            current_iteration=round_number,
            recent_window=recent_window,
        )].append(record)
    observed = {name: len(rows) for name, rows in buckets.items()}
    if sum(observed.values()) != 2000:
        raise RuntimeError(f"unexpected source sample size: {observed}")
    anchor_count = len(conditioned_anchors)
    if anchor_count not in ARM_ANCHOR_COUNTS.values():
        raise ValueError("unsupported anchor count")
    historical_count = observed[REPLAY_AGE_HISTORICAL]
    if historical_count < anchor_count:
        raise RuntimeError(
            "source historical quota is smaller than anchor replacement")
    ordinary_historical = historical_count - anchor_count
    records = [
        *buckets[REPLAY_AGE_CURRENT],
        *buckets[REPLAY_AGE_RECENT],
        *buckets[REPLAY_AGE_HISTORICAL][:ordinary_historical],
        *conditioned_anchors,
    ]
    return records, {
        "current": observed[REPLAY_AGE_CURRENT],
        "recent": observed[REPLAY_AGE_RECENT],
        "ordinary_historical": ordinary_historical,
        "difficulty_anchor": anchor_count,
    }


def _load_champion_retention_cache(
    source: Path,
    *,
    champion_sha256: str,
    selected: list[dict[str, Any]],
    budgets: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    expected_selected = [{
        "static_level_signature": row["static_level_signature"],
        "difficulty_stratum": row["difficulty_stratum"],
    } for row in selected]
    for path in (source / "retention_cache").glob("*.json"):
        payload = _load_json(path)
        metadata = payload.get("metadata", {})
        if (metadata.get("checkpoint_content_sha256") == champion_sha256
                and metadata.get("selected") == expected_selected
                and metadata.get("budgets") == budgets
                and metadata.get("seed") == seed
                and payload.get("rows_sha256")
                == hash_canonical_value(payload.get("rows"))):
            return payload["rows"]
    raise RuntimeError("matching champion retention cache was not found")


def _write_gzip_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    data = gzip.compress(_canonical_jsonl(records), compresslevel=6, mtime=0)
    digest = _sha_bytes(data)
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"cached sample differs: {path}")
    else:
        path.write_bytes(data)
    return digest


def _save_model(
    path: Path,
    *,
    model,
    champion: dict[str, Any],
    encoding,
    model_config,
    value_norm,
    round_number: int,
    seed: int,
    arm: str,
    fingerprint: str,
    semantics: str = SEMANTICS,
) -> str:
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=round_number,
        best_val_metric=None,
        encoding_config=encoding,
        model_config=model_config,
        value_norm=value_norm,
        seed=seed,
        dataset_version=champion.get("dataset_version", 1),
        split_identity={
            "kind": semantics,
            "arm": arm,
            "round": round_number,
        },
        metrics={"replay_anchor_sweep": {
            "arm": arm, "round": round_number}},
        experiment_fingerprint=fingerprint,
    )
    return sha256_file(path)


def _source_candidate_state_sha(source: Path, round_number: int) -> str:
    report = _load_json(source / f"round_{round_number:03d}" / "report.json")
    relative = report["protagonist"]["candidate_checkpoint"]
    checkpoint = load_checkpoint(source / relative, map_location="cpu")
    return model_state_sha256(checkpoint["model_state"])


def build_experiment_identity(
    cfg: ReplayAnchorSweepConfig,
    source_config: dict[str, Any],
) -> dict[str, Any]:
    source = Path(cfg.source_run)
    manifests = []
    for round_number in range(1, cfg.rounds + 1):
        path = source / f"round_{round_number:03d}" \
            / "training_sample_manifest.json"
        if not path.is_file():
            raise ValueError(f"source round {round_number} lacks manifest")
        manifests.append({
            "round": round_number,
            "sha256": sha256_file(path),
        })
    identity = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": asdict(cfg),
        "arms": ARM_ANCHOR_COUNTS,
        "inputs": {
            "champion_checkpoint_sha256": sha256_file(
                cfg.champion_checkpoint),
            "source_config_sha256": sha256_file(source / "config.json"),
            "replay_snapshot_sha256": sha256_file(cfg.replay_snapshot),
            "anchor_pool_sha256": sha256_file(cfg.anchor_pool),
            "retention_dataset_sha256": sha256_file(
                cfg.retention_dataset),
            "round_manifests": manifests,
        },
        "training": {
            key: source_config[key]
            for key in (
                "epochs", "batch_size", "learning_rate", "weight_decay",
                "grad_clip", "trainable_part", "policy_target_profile",
                "weight_exact_historical", "weight_exact_new",
                "weight_search", "exact_path_policy_confidence",
                "search_value_loss_weight", "seed")
        },
    }
    identity["fingerprint"] = hash_canonical_value(identity)
    return identity


def _condition_records(
    records: list[dict[str, Any]],
    *,
    champion_model,
    encoding,
    value_norm,
    device: torch.device,
    batch_size: int,
    champion_sha256: str,
) -> list[dict[str, Any]]:
    probabilities = incumbent_legal_probabilities(
        records,
        model=champion_model,
        encoding_config=encoding,
        value_norm=value_norm,
        device=device,
        batch_size=batch_size,
    )
    conditioned, _summary = condition_policy_targets(
        records,
        probabilities,
        profile="incumbent_optimal",
        incumbent_checkpoint_sha256=champion_sha256,
    )
    return conditioned


def run_replay_anchor_sweep(cfg: ReplayAnchorSweepConfig) -> dict[str, Any]:
    cfg.validate()
    source = Path(cfg.source_run)
    source_config = _load_json(source / "config.json")
    _validate_source_config(source_config)
    if int(source_config["learner_milestone_interval"]) \
            != cfg.milestone_interval:
        raise ValueError("milestone interval differs from source run")
    identity = build_experiment_identity(cfg, source_config)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiment_path = root / "experiment.json"
    if experiment_path.exists():
        if _load_json(experiment_path) != identity:
            raise RuntimeError("output directory belongs to another experiment")
    else:
        atomic_write_json(experiment_path, identity)

    device = _resolve_device(cfg.device)
    champion = load_checkpoint(cfg.champion_checkpoint, map_location="cpu")
    encoding, model_config, value_norm = configs_from_checkpoint(champion)
    champion_model = model_from_checkpoint(champion, map_location=device)
    champion_sha256 = identity["inputs"]["champion_checkpoint_sha256"]
    replay_records = load_records(cfg.replay_snapshot)
    anchor_records = load_records(cfg.anchor_pool)
    anchor_groups = _anchor_groups(anchor_records)

    retention_records, _signatures, retention_manifest = load_retention_pool(
        cfg.retention_dataset,
        per_band=int(source_config["learner_retention_per_band"]),
    )
    budgets = [int(value) for value in
               source_config["learner_retention_budgets"]]
    retention_seed = int(source_config["seed"])
    reference_rows = _load_champion_retention_cache(
        source,
        champion_sha256=champion_sha256,
        selected=retention_records,
        budgets=budgets,
        seed=retention_seed,
    )
    atomic_write_json(root / "retention_manifest.json", {
        **retention_manifest,
        "budgets": budgets,
        "seed": retention_seed,
        "maximum_regression": source_config[
            "learner_retention_max_regression"],
        "reference_checkpoint_sha256": champion_sha256,
        "reference_rows_sha256": hash_canonical_value(reference_rows),
    })

    # Condition the sealed training-only anchors once; the champion target
    # model is fixed in the source experiment and in every sweep arm.
    conditioned_anchor_pool = _condition_records(
        anchor_records,
        champion_model=champion_model,
        encoding=encoding,
        value_norm=value_norm,
        device=device,
        batch_size=int(source_config["batch_size"]),
        champion_sha256=champion_sha256,
    )
    conditioned_groups = _anchor_groups(conditioned_anchor_pool)

    results: dict[str, Any] = {}
    env = Environment()
    for arm, anchor_count in ARM_ANCHOR_COUNTS.items():
        print(
            f"=== replay anchor arm {arm} ({anchor_count}/2000) ===",
            flush=True,
        )
        arm_root = root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        state_path = arm_root / "run_state.json"
        if state_path.exists():
            state = _load_json(state_path)
            if state.get("experiment_fingerprint") != identity["fingerprint"]:
                raise RuntimeError(f"{arm} cached state has wrong identity")
            completed = int(state["completed_round"])
            model = model_from_checkpoint(
                load_checkpoint(arm_root / "latest.pt", map_location="cpu"),
                map_location=device,
            )
        else:
            completed = 0
            state = {
                "schema_version": 1,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "anchor_count": anchor_count,
                "completed_round": 0,
                "milestones": [],
                "retention_rollbacks": 0,
            }
            model = model_from_checkpoint(champion, map_location=device)
            _save_model(
                arm_root / "latest.pt",
                model=model,
                champion=champion,
                encoding=encoding,
                model_config=model_config,
                value_norm=value_norm,
                round_number=0,
                seed=int(source_config["seed"]),
                arm=arm,
                fingerprint=identity["fingerprint"],
            )
            _save_model(
                arm_root / "anchor.pt",
                model=model,
                champion=champion,
                encoding=encoding,
                model_config=model_config,
                value_norm=value_norm,
                round_number=0,
                seed=int(source_config["seed"]),
                arm=arm,
                fingerprint=identity["fingerprint"],
            )
            atomic_write_json(state_path, state)

        for round_number in range(completed + 1, cfg.rounds + 1):
            round_dir = arm_root / f"round_{round_number:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            source_round = source / f"round_{round_number:03d}"
            manifest = _load_json(
                source_round / "training_sample_manifest.json")
            raw_source = reconstruct_source_sample(replay_records, manifest)
            conditioned_source = _condition_records(
                raw_source,
                champion_model=champion_model,
                encoding=encoding,
                value_norm=value_norm,
                device=device,
                batch_size=int(source_config["batch_size"]),
                champion_sha256=champion_sha256,
            )
            conditioned_sha = _sha_bytes(_canonical_jsonl(conditioned_source))
            if conditioned_sha != manifest["sample"]["sha256"]:
                raise RuntimeError(
                    f"round {round_number} conditioned sample did not "
                    "reproduce source manifest")
            anchors = select_level_balanced_anchors(
                conditioned_groups,
                count=anchor_count,
                round_number=round_number,
            )
            records, composition = compose_arm_sample(
                conditioned_source,
                anchors,
                round_number=round_number,
                recent_window=int(source_config["replay_recent_window"]),
            )
            weights = source_weights_for(
                records,
                round_number,
                weight_exact_historical=float(
                    source_config["weight_exact_historical"]),
                weight_exact_new=float(source_config["weight_exact_new"]),
                weight_search=float(source_config["weight_search"]),
                exact_path_policy_confidence=float(
                    source_config["exact_path_policy_confidence"]),
            )
            value_weights = list(weights)
            effective = value_supervision_weights_for(
                records,
                value_weights,
                search_value_loss_weight=float(
                    source_config["search_value_loss_weight"]),
            )
            if arm == "control":
                persisted_weights = _load_json(
                    source_round / "training_policy_weights.json")
                persisted_effective = _load_json(
                    source_round / "training_effective_value_weights.json")
                if weights != persisted_weights or effective != persisted_effective:
                    raise RuntimeError(
                        f"round {round_number} control weights do not reproduce")

            sample_sha = _write_gzip_jsonl(
                round_dir / "training_sample.jsonl.gz", records)
            sample_manifest = {
                "schema_version": 1,
                "semantics": SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "round": round_number,
                "source_manifest_sha256": sha256_file(
                    source_round / "training_sample_manifest.json"),
                "source_sample_sha256": manifest["sample"]["sha256"],
                "compressed_sample_sha256": sample_sha,
                "record_count": len(records),
                "composition": composition,
                "anchor_selection_sha256": hash_canonical_value([
                    list(dedup_key(record)) for record in anchors]),
                "policy_weights_sha256": hash_canonical_value(weights),
                "value_weights_sha256": hash_canonical_value(value_weights),
                "effective_value_weights_sha256": hash_canonical_value(
                    effective),
                "training_seed": int(manifest["training_seed"]),
            }
            atomic_write_json(round_dir / "sample_manifest.json", sample_manifest)

            print(
                f"{arm}: training round {round_number}/{cfg.rounds}",
                flush=True,
            )
            trainable = configure_trainable_part(
                model, source_config["trainable_part"])
            train_info = train_expert(
                model,
                records,
                weights,
                value_weights=value_weights,
                encoding_config=encoding,
                value_norm=value_norm,
                epochs=int(source_config["epochs"]),
                batch_size=int(source_config["batch_size"]),
                learning_rate=float(source_config["learning_rate"]),
                weight_decay=float(source_config["weight_decay"]),
                grad_clip=float(source_config["grad_clip"]),
                device=device,
                seed=int(manifest["training_seed"]),
                search_value_loss_weight=float(
                    source_config["search_value_loss_weight"]),
            )
            candidate_state_sha = model_state_sha256(model)
            source_candidate_sha = None
            if arm == "control":
                source_candidate_sha = _source_candidate_state_sha(
                    source, round_number)
                if candidate_state_sha != source_candidate_sha:
                    raise RuntimeError(
                        f"control round {round_number} did not reproduce "
                        "the source candidate")

            milestone = round_number % cfg.milestone_interval == 0
            retention = None
            rolled_back = False
            if milestone:
                candidate_path = round_dir / "candidate.pt"
                candidate_checkpoint_sha = _save_model(
                    candidate_path,
                    model=model,
                    champion=champion,
                    encoding=encoding,
                    model_config=model_config,
                    value_norm=value_norm,
                    round_number=round_number,
                    seed=int(manifest["training_seed"]),
                    arm=arm,
                    fingerprint=identity["fingerprint"],
                )
                candidate_rows = evaluate_retention(
                    env,
                    model,
                    encoding,
                    value_norm,
                    retention_records,
                    budgets=budgets,
                    device=device,
                    seed=retention_seed,
                )
                retention = summarize_retention(
                    reference_rows,
                    candidate_rows,
                    budgets=budgets,
                    max_regression=float(source_config[
                        "learner_retention_max_regression"]),
                )
                atomic_write_json(round_dir / "retention_rows.json", {
                    "checkpoint_sha256": candidate_checkpoint_sha,
                    "rows": candidate_rows,
                    "rows_sha256": hash_canonical_value(candidate_rows),
                })
                atomic_write_json(round_dir / "retention.json", retention)
                if retention["passed"]:
                    _save_model(
                        arm_root / "anchor.pt",
                        model=model,
                        champion=champion,
                        encoding=encoding,
                        model_config=model_config,
                        value_norm=value_norm,
                        round_number=round_number,
                        seed=int(manifest["training_seed"]),
                        arm=arm,
                        fingerprint=identity["fingerprint"],
                    )
                else:
                    model = model_from_checkpoint(
                        load_checkpoint(
                            arm_root / "anchor.pt", map_location="cpu"),
                        map_location=device,
                    )
                    rolled_back = True
                    state["retention_rollbacks"] += 1
                state["milestones"].append({
                    "round": round_number,
                    "passed": bool(retention["passed"]),
                    "rolled_back": rolled_back,
                    "failures": retention["failures"],
                    "candidate_model_state_sha256": candidate_state_sha,
                })
                print(
                    f"{arm}: retention round {round_number}: "
                    f"passed={retention['passed']}; "
                    f"failures={len(retention['failures'])}",
                    flush=True,
                )

            latest_sha = _save_model(
                arm_root / "latest.pt",
                model=model,
                champion=champion,
                encoding=encoding,
                model_config=model_config,
                value_norm=value_norm,
                round_number=round_number,
                seed=int(manifest["training_seed"]),
                arm=arm,
                fingerprint=identity["fingerprint"],
            )
            round_summary = {
                "schema_version": 1,
                "semantics": SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "round": round_number,
                "milestone": milestone,
                "composition": composition,
                "candidate_model_state_sha256": candidate_state_sha,
                "source_candidate_model_state_sha256": source_candidate_sha,
                "retention": retention,
                "rolled_back": rolled_back,
                "active_checkpoint_sha256": latest_sha,
                "active_model_state_sha256": model_state_sha256(model),
                "train": {**train_info, **trainable},
            }
            atomic_write_json(round_dir / "summary.json", round_summary)
            state["completed_round"] = round_number
            state["active_checkpoint_sha256"] = latest_sha
            state["active_model_state_sha256"] = model_state_sha256(model)
            atomic_write_json(state_path, state)

        results[arm] = state

    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "arms": results,
    }
    atomic_write_json(root / "result.json", result)
    return result


def _parse_args(argv: list[str] | None = None) -> ReplayAnchorSweepConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--replay-snapshot", required=True)
    parser.add_argument("--anchor-pool", required=True)
    parser.add_argument("--retention-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--milestone-interval", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    return ReplayAnchorSweepConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    result = run_replay_anchor_sweep(_parse_args(argv))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
