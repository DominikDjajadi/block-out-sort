"""Matched fixed-champion policy-distillation retention sweep.

Every arm replays the source run's exact 2,000-record samples and training
seeds.  A separate 200-state, level-balanced training-only anchor batch is
present in all arms but contributes no policy-target or value loss.  The sole
experimental variable is the coefficient on KL(champion || learner) over
legal actions: 0.0, 0.1, or 0.5.

The source retention guard is evaluated and enforced every five rounds.  The
zero-KL control must reproduce every source candidate model state exactly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..environment import Environment
from ..expert_iteration.records import dedup_key
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
)
from ..training.dataset import load_records
from ..training.experiment_identity import hash_canonical_value
from ..training.transaction import atomic_write_json, sha256_file
from .replay_anchor_sweep import (
    _anchor_groups,
    _canonical_jsonl,
    _condition_records,
    _load_champion_retention_cache,
    _load_json,
    _resolve_device,
    _save_model,
    _sha_bytes,
    _source_candidate_state_sha,
    _validate_source_config,
    reconstruct_source_sample,
    select_level_balanced_anchors,
)
from .retention import (
    evaluate_retention,
    load_retention_pool,
    summarize_retention,
)


SCHEMA_VERSION = 1
SEMANTICS = "matched_fixed_champion_anchor_distillation_sweep_v1"
ARM_KL_WEIGHTS = {"control": 0.0, "light": 0.1, "moderate": 0.5}
ANCHOR_RECORDS_PER_ROUND = 200


@dataclass(frozen=True)
class PolicyDistillationSweepConfig:
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


def build_experiment_identity(
    cfg: PolicyDistillationSweepConfig,
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
        "arms": ARM_KL_WEIGHTS,
        "anchor_records_per_round": ANCHOR_RECORDS_PER_ROUND,
        "anchor_loss": {
            "direction": "kl_fixed_champion_to_learner_on_legal_actions",
            "target_loss_weight": 0.0,
            "value_loss_weight": 0.0,
            "batching": "one_level_balanced_anchor_pass_per_epoch_v1",
        },
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


def run_policy_distillation_sweep(
    cfg: PolicyDistillationSweepConfig,
) -> dict[str, Any]:
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

    results: dict[str, Any] = {}
    env = Environment()
    for arm, kl_weight in ARM_KL_WEIGHTS.items():
        print(
            f"=== fixed champion distillation arm {arm} "
            f"(KL={kl_weight:g}) ===",
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
                "policy_anchor_weight": kl_weight,
                "completed_round": 0,
                "milestones": [],
                "retention_rollbacks": 0,
            }
            model = model_from_checkpoint(champion, map_location=device)
            for name in ("latest.pt", "anchor.pt"):
                _save_model(
                    arm_root / name,
                    model=model,
                    champion=champion,
                    encoding=encoding,
                    model_config=model_config,
                    value_norm=value_norm,
                    round_number=0,
                    seed=int(source_config["seed"]),
                    arm=arm,
                    fingerprint=identity["fingerprint"],
                    semantics=SEMANTICS,
                )
            atomic_write_json(state_path, state)

        for round_number in range(completed + 1, cfg.rounds + 1):
            round_dir = arm_root / f"round_{round_number:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            source_round = source / f"round_{round_number:03d}"
            manifest_path = source_round / "training_sample_manifest.json"
            manifest = _load_json(manifest_path)
            raw_source = reconstruct_source_sample(replay_records, manifest)
            records = _condition_records(
                raw_source,
                champion_model=champion_model,
                encoding=encoding,
                value_norm=value_norm,
                device=device,
                batch_size=int(source_config["batch_size"]),
                champion_sha256=champion_sha256,
            )
            if _sha_bytes(_canonical_jsonl(records)) \
                    != manifest["sample"]["sha256"]:
                raise RuntimeError(
                    f"round {round_number} conditioned source hash mismatch")
            anchors = select_level_balanced_anchors(
                anchor_groups,
                count=ANCHOR_RECORDS_PER_ROUND,
                round_number=round_number,
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
            persisted_weights = _load_json(
                source_round / "training_policy_weights.json")
            persisted_effective = _load_json(
                source_round / "training_effective_value_weights.json")
            if weights != persisted_weights or effective != persisted_effective:
                raise RuntimeError(
                    f"round {round_number} source weights do not reproduce")
            atomic_write_json(round_dir / "sample_manifest.json", {
                "schema_version": 1,
                "semantics": SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "round": round_number,
                "source_manifest_sha256": sha256_file(manifest_path),
                "source_sample_sha256": manifest["sample"]["sha256"],
                "record_count": len(records),
                "anchor_record_count": len(anchors),
                "anchor_selection_sha256": hash_canonical_value([
                    list(dedup_key(record)) for record in anchors]),
                "anchor_target_loss_weight": 0.0,
                "anchor_value_loss_weight": 0.0,
                "policy_anchor_weight": kl_weight,
                "policy_weights_sha256": hash_canonical_value(weights),
                "effective_value_weights_sha256": hash_canonical_value(
                    effective),
                "training_seed": int(manifest["training_seed"]),
            })

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
                policy_anchor_model=champion_model,
                policy_anchor_weight=kl_weight,
                policy_anchor_records=anchors,
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
                    semantics=SEMANTICS,
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
                        semantics=SEMANTICS,
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
                semantics=SEMANTICS,
            )
            atomic_write_json(round_dir / "summary.json", {
                "schema_version": 1,
                "semantics": SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "round": round_number,
                "policy_anchor_weight": kl_weight,
                "milestone": milestone,
                "candidate_model_state_sha256": candidate_state_sha,
                "source_candidate_model_state_sha256": source_candidate_sha,
                "retention": retention,
                "rolled_back": rolled_back,
                "active_checkpoint_sha256": latest_sha,
                "active_model_state_sha256": model_state_sha256(model),
                "train": {**train_info, **trainable},
            })
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


def _parse_args(argv: list[str] | None = None) -> PolicyDistillationSweepConfig:
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
    return PolicyDistillationSweepConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    result = run_policy_distillation_sweep(_parse_args(argv))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
