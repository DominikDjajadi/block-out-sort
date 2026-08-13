"""Run a preregistered matched trace-pair retention sweep.

Every arm replays the source run's exact persisted samples, policy/value
weights, training seeds, cumulative learner ancestry, and retention rollback
semantics.  The sole experimental variable is the preregistered coefficient on
the frozen training-only first-divergence pairwise hinge loss.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..environment import Environment
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
)
from .retention import (
    evaluate_retention,
    load_retention_pool,
    summarize_retention,
)


SCHEMA_VERSION = 1
SEMANTICS = "matched_trace_pair_ranking_retention_sweep_v1"
PREREGISTRATION_SEMANTICS = \
    "matched_trace_pair_ranking_sweep_preregistration_v1"
ARM_ORDER = ("control", "light", "moderate")


@dataclass(frozen=True)
class TraceRankingSweepConfig:
    champion_checkpoint: str
    source_run: str
    replay_snapshot: str
    trace_dataset: str
    retention_dataset: str
    preregistration: str
    output_dir: str
    rounds: int = 30
    milestone_interval: int = 5
    device: str = "auto"

    def validate(self) -> None:
        for label, raw in (
                ("champion checkpoint", self.champion_checkpoint),
                ("replay snapshot", self.replay_snapshot),
                ("trace dataset", self.trace_dataset),
                ("retention dataset", self.retention_dataset),
                ("preregistration", self.preregistration)):
            if not Path(raw).is_file():
                raise ValueError(f"{label} does not exist: {raw}")
        if not Path(self.source_run).is_dir():
            raise ValueError("source run does not exist")
        if self.rounds <= 0 or self.milestone_interval <= 0:
            raise ValueError("rounds and milestone interval must be positive")
        if self.rounds % self.milestone_interval:
            raise ValueError("rounds must end on a retention milestone")


def _preregistered_contract(
    cfg: TraceRankingSweepConfig,
) -> tuple[dict[str, Any], dict[str, float], float]:
    prereg = _load_json(Path(cfg.preregistration))
    if prereg.get("semantics") != PREREGISTRATION_SEMANTICS \
            or prereg.get("status") != "frozen_before_matched_training":
        raise ValueError("trace sweep requires the frozen preregistration")
    expected_fingerprint = prereg.get("fingerprint")
    without_fingerprint = dict(prereg)
    without_fingerprint.pop("fingerprint", None)
    if expected_fingerprint != hash_canonical_value(without_fingerprint):
        raise ValueError("preregistration fingerprint is invalid")
    fixed_inputs = prereg["fixed_inputs"]
    observed = {
        "checkpoint_sha256": sha256_file(cfg.champion_checkpoint),
        "trace_dataset_sha256": sha256_file(cfg.trace_dataset),
        "source_config_sha256": sha256_file(
            Path(cfg.source_run) / "config.json"),
        "replay_snapshot_sha256": sha256_file(cfg.replay_snapshot),
    }
    for field, value in observed.items():
        if fixed_inputs.get(field) != value:
            raise ValueError(f"preregistered input differs: {field}")
    arms_payload = prereg.get("arms", {})
    if tuple(arms_payload) != ARM_ORDER:
        raise ValueError("preregistered arms must be control/light/moderate")
    arms = {
        arm: float(arms_payload[arm]["trace_ranking_weight"])
        for arm in ARM_ORDER
    }
    if arms["control"] != 0.0 \
            or not 0 < arms["light"] < arms["moderate"]:
        raise ValueError("preregistered trace weights are not ordered")
    margin = float(prereg["fixed_training"]["trace_margin"])
    if margin < 0:
        raise ValueError("preregistered trace margin is invalid")
    return prereg, arms, margin


def build_experiment_identity(
    cfg: TraceRankingSweepConfig,
    source_config: dict[str, Any],
    *,
    arms: dict[str, float],
    margin: float,
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
        "arms": arms,
        "trace_loss": {
            "margin": margin,
            "scope": "first_divergent_action_pair_only",
            "target_loss_weight": 0.0,
            "value_loss_weight": 0.0,
            "batching": "one_shuffled_evenly_spread_pass_per_epoch_v1",
        },
        "inputs": {
            "champion_checkpoint_sha256": sha256_file(
                cfg.champion_checkpoint),
            "source_config_sha256": sha256_file(source / "config.json"),
            "replay_snapshot_sha256": sha256_file(cfg.replay_snapshot),
            "trace_dataset_sha256": sha256_file(cfg.trace_dataset),
            "retention_dataset_sha256": sha256_file(
                cfg.retention_dataset),
            "preregistration_sha256": sha256_file(cfg.preregistration),
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


def run_trace_ranking_sweep(
    cfg: TraceRankingSweepConfig,
    *,
    selected_arms: tuple[str, ...] | None = None,
    stop_after_round: int | None = None,
) -> dict[str, Any]:
    cfg.validate()
    prereg, arms, margin = _preregistered_contract(cfg)
    if selected_arms is None:
        selected_arms = ARM_ORDER
    if not selected_arms or any(arm not in ARM_ORDER for arm in selected_arms):
        raise ValueError("selected arms must be a non-empty registered subset")
    if stop_after_round is not None \
            and not 1 <= stop_after_round <= cfg.rounds:
        raise ValueError("stop-after-round is outside the configured sweep")

    source = Path(cfg.source_run)
    source_config = _load_json(source / "config.json")
    _validate_source_config(source_config)
    if int(source_config["learner_milestone_interval"]) \
            != cfg.milestone_interval:
        raise ValueError("milestone interval differs from source run")
    identity = build_experiment_identity(
        cfg, source_config, arms=arms, margin=margin)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiment_path = root / "experiment.json"
    if experiment_path.exists():
        if _load_json(experiment_path) != identity:
            raise RuntimeError("output directory belongs to another experiment")
    else:
        atomic_write_json(experiment_path, identity)
    atomic_write_json(root / "bound_preregistration.json", {
        "path": cfg.preregistration,
        "sha256": identity["inputs"]["preregistration_sha256"],
        "fingerprint": prereg["fingerprint"],
        "arms": arms,
        "margin": margin,
    })

    device = _resolve_device(cfg.device)
    champion = load_checkpoint(cfg.champion_checkpoint, map_location="cpu")
    encoding, model_config, value_norm = configs_from_checkpoint(champion)
    champion_model = model_from_checkpoint(champion, map_location=device)
    champion_sha256 = identity["inputs"]["champion_checkpoint_sha256"]
    replay_records = load_records(cfg.replay_snapshot)
    trace_records = load_records(cfg.trace_dataset)

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
    for arm in selected_arms:
        trace_weight = arms[arm]
        print(
            f"=== trace ranking arm {arm} (weight={trace_weight:g}) ===",
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
                "schema_version": SCHEMA_VERSION,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "trace_ranking_weight": trace_weight,
                "trace_ranking_margin": margin,
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

        final_round = cfg.rounds if stop_after_round is None \
            else min(cfg.rounds, stop_after_round)
        for round_number in range(completed + 1, final_round + 1):
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
                "schema_version": SCHEMA_VERSION,
                "semantics": SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "round": round_number,
                "source_manifest_sha256": sha256_file(manifest_path),
                "source_sample_sha256": manifest["sample"]["sha256"],
                "record_count": len(records),
                "trace_record_count": len(trace_records),
                "trace_dataset_sha256": identity["inputs"][
                    "trace_dataset_sha256"],
                "trace_target_loss_weight": 0.0,
                "trace_value_loss_weight": 0.0,
                "trace_ranking_weight": trace_weight,
                "trace_ranking_margin": margin,
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
                trace_ranking_records=trace_records,
                trace_ranking_weight=trace_weight,
                trace_ranking_margin=margin,
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
                "schema_version": SCHEMA_VERSION,
                "semantics": SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "arm": arm,
                "round": round_number,
                "trace_ranking_weight": trace_weight,
                "trace_ranking_margin": margin,
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

    partial = any(
        not (root / arm / "run_state.json").is_file()
        or int(_load_json(root / arm / "run_state.json")["completed_round"])
        < cfg.rounds
        for arm in ARM_ORDER
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "status": "partial" if partial else "complete",
        "arms": results,
        "sealed_final_test_status": "not_loaded_or_evaluated",
    }
    atomic_write_json(
        root / ("partial_result.json" if partial else "result.json"), result)
    return result


def _parse_args(
    argv: list[str] | None = None,
) -> tuple[TraceRankingSweepConfig, tuple[str, ...] | None, int | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--replay-snapshot", required=True)
    parser.add_argument("--trace-dataset", required=True)
    parser.add_argument("--retention-dataset", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--milestone-interval", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arms", nargs="+", choices=ARM_ORDER)
    parser.add_argument("--stop-after-round", type=int)
    args = vars(parser.parse_args(argv))
    raw_arms = args.pop("arms")
    selected_arms = tuple(raw_arms) if raw_arms else None
    stop_after_round = args.pop("stop_after_round")
    return TraceRankingSweepConfig(**args), selected_arms, stop_after_round


def main(argv: list[str] | None = None) -> int:
    cfg, selected_arms, stop_after_round = _parse_args(argv)
    result = run_trace_ranking_sweep(
        cfg,
        selected_arms=selected_arms,
        stop_after_round=stop_after_round,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
