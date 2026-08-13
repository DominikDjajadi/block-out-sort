"""The alternating protagonist-designer co-training loop.

Each round:

1. Generate levels with the current designer.
2. Estimate a per-level solve rate with the bounded protagonist.
3. Select valid, construction-proven, non-duplicate frontier levels, then run
   bounded exact-first scoring only on that selected subset.
4. Prefer levels whose protagonist solve rate is in the frontier band.
5. Convert accepted levels + reachable states into exact-or-search records.
6. Add them to the protagonist replay buffer.
7. Fine-tune the protagonist (expert-iteration training).
8. Evaluate on frozen validation + benchmark groups. When an external split
   manifest is configured, keep its final-test role sealed for explicit
   post-training evaluation.
9. Promote using validation metrics only.
10. Freeze the promoted protagonist; train the designer against it.

The protagonist and designer are never updated in the same batch. Difficulty
adapts to keep generation near the protagonist's learning frontier. Everything
is disk-checkpointed for resume.
"""

from __future__ import annotations

from collections import Counter
import json
import csv
import io
import random
import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..oracle import Oracle as ExactOracle
from ..schema import Level
from ..serialization import level_to_dict
from ..signature import static_level_signature
from ..dataset.schema import deserialize_state
from ..serialization import level_from_dict

from ..training.checkpoint import (configs_from_checkpoint, load_checkpoint,
                                    model_from_checkpoint, save_checkpoint)
from ..training.transaction import (
    atomic_copy, atomic_write_json, atomic_write_text, refresh_best_checkpoint,
    relative_to_run, resolve_committed_protagonist, resolve_run_path,
    sha256_file)
from ..training.experiment_identity import (
    EVALUATION_SEMANTICS_VERSION, EXPERIMENT_SPEC_FILE,
    PROMOTION_CONTRACT_VERSION, TRANSACTION_SCHEMA_VERSION,
    ExperimentSpecIntegrityError,
    build_experiment_spec, file_identity, hash_canonical_value,
    load_legacy_migration_spec, runtime_device_provenance,
    semantic_dataclass_config, validate_continuation_horizon,
    validate_identified_run_state_presence,
    validate_or_initialize_experiment)
from ..training.dataset import load_records
from ..training.model import PolicyValueNet
from ..training.splits import (SplitRatios, filter_records_for_split, group_key,
                               load_manifest, make_split, save_manifest)
from ..search.graph_search import BlocksortAdapter

from ..expert_iteration.evaluate import evaluate_checkpoint
from ..expert_iteration.promotion import (
    promotion_metric_requires_budget_sweep,
    validate_budget_sweep_promotion_evidence,
    validate_promotion_evidence)
from ..expert_iteration.generate import generate_states
from ..expert_iteration.labeling import label_states
from ..expert_iteration.records import dedup_key, tag_historical
from ..expert_iteration.replay import (
    REPLAY_AGE_BUCKETS,
    ReplayBuffer,
    replay_age_bucket,
)
from ..expert_iteration.train import (
    configure_trainable_part,
    source_weights_for,
    train_expert,
    value_supervision_weights_for,
)
from .frontier_promotion_evaluate import summarize_paired_promotion

from ..designer.actions import DesignerActionSpace
from ..designer.checkpoint import designer_from_checkpoint, load_designer
from ..designer.config import GeneratorConfig, RewardConfig
from ..designer.env import DesignerEnv, FinalizeResult
from ..designer.model import DesignerModelConfig
from ..designer.ppo import PPOConfig, rollout_episode
from ..designer.replay import (LevelReplayBuffer, build_level_record,
                               level_fingerprint)
from ..designer.roles import Oracle as DesignerOracle
from ..designer.roles import Protagonist
from ..designer.score import score_level
from ..model_identity import model_state_sha256
from ..designer.train import TrainConfig, train_designer

from . import benchmark as bench
from .config import CoTrainingConfig, CurriculumConfig, CurriculumState
from .curriculum import adapt_curriculum
from .frontier import (
    estimate_solve_rate, frontier_distance, geometric_budget_sweep, in_frontier,
    select_frontier_backfill)
from .eval_split import (
    EVAL_SPLIT_ALGORITHM,
    evaluation_split_identity,
    load_eval_split_manifest,
)
from .policy_targets import (
    condition_policy_targets,
    incumbent_legal_probabilities,
    recorded_policy_target_summary,
)
from .shadow_learner import (
    apply_learner_transition,
    continuation_decision,
    learner_milestone,
    model_integrity_report,
    policy_drift_report,
)
from .retention import (
    apply_retention_guard,
    evaluate_retention,
    load_retention_pool,
    summarize_retention,
)


_FALLBACK_EVAL_SELECTION_POLICY = "one_initial_state_per_level_v1"


_COTRAIN_INPUT_FIELDS = (
    "protagonist_checkpoint", "designer_checkpoint", "base_dataset",
    "initial_learner_checkpoint",
    "initial_base_split",
    "initial_protagonist_replay", "initial_designer_replay",
    "pretrained_designer_checkpoint", "eval_levels_dataset",
    "eval_split_manifest", "learner_retention_dataset")
_COTRAIN_OPERATIONAL_FIELDS = (
    "output_dir", "generation_checkpoint_interval",
    "generation_progress_interval", "initialize_only",
    "prune_superseded_round_artifacts")
_COTRAIN_HORIZON_FIELDS = ("rounds",)
_COTRAIN_DERIVED_FIELDS = (
    "device", "eval_split_seed", "eval_validation_count",
    # These are fingerprinted through paired_promotion_gate_policy only when
    # the optional gate is enabled, preserving legacy-run compatibility.
    "promotion_paired_gate_enabled",
    "promotion_max_per_budget_regression",
    "promotion_bootstrap_confidence", "promotion_bootstrap_replicates",
    "promotion_bootstrap_seed",
    # Fingerprinted conditionally through learner_retention_policy so runs
    # without the optional pool retain their historical identity.
    "learner_retention_budgets", "learner_retention_per_band",
    "learner_retention_use_full_pool",
    "learner_retention_max_regression", "learner_retention_enforce")
_COTRAIN_SEMANTIC_FIELDS = (
    "levels_per_round", "seed", "solve_rate_trials",
    "stop_after_promotion",
    "frontier_dirichlet_alpha", "frontier_dirichlet_weight",
    "frontier_budget_min_ratio", "frontier_budget_max_ratio",
    "frontier_backfill_target", "min_fresh_levels_to_train",
    "astar_max_nodes", "exploratory_astar_time_limit_seconds",
    "training_astar_time_limit_seconds", "eval_astar_max_nodes",
    "eval_astar_time_limit_seconds", "oracle_simulations", "eval_budgets",
    "eval_limit", "promotion_metric", "promotion_budget",
    "promotion_budgets", "promotion_budget_weights", "promotion_margin",
    "shadow_learner_enabled", "learner_milestone_interval",
    "learner_max_policy_kl", "learner_min_entropy_ratio",
    "states_per_level", "train_sample_size", "epochs", "batch_size",
    "replay_sample_with_replacement", "replay_current_fraction",
    "replay_recent_fraction", "replay_historical_fraction",
    "replay_recent_window",
    "learning_rate", "trainable_part", "weight_decay", "grad_clip",
    "weight_exact_historical", "weight_exact_new", "weight_search",
    "exact_path_policy_confidence",
    "search_value_loss_weight", "policy_anchor_weight",
    "policy_target_profile",
    "max_protagonist_replay", "val_ratio", "test_ratio",
    "designer_episodes", "designer_episodes_per_iter",
    "designer_validation_episodes",
    "designer_frontier_alignment_weight",
    "designer_ppo_epochs", "designer_entropy_coef", "max_designer_replay",
    "benchmark_count", "benchmark_total_limit", "ood_rows", "ood_cols",
    "skip_forgetting_benchmark", "forgetting_only_on_promotion",
    "excluded_signatures", "reward", "curriculum_enabled",
    "use_designer_replay", "seed_historical_replay", "label_mode",
    "train_designer_each_round", "curriculum", "initial_curriculum")


def _designer_input_identity(path: str, kind: str) -> dict[str, Any]:
    checkpoint = load_designer(path, map_location="cpu")
    return file_identity(
        path, kind=kind,
        format_version=int(checkpoint["designer_checkpoint_version"]),
        extra={
            "encoding_config": checkpoint["encoding_config"],
            "model_config": checkpoint["model_config"],
        })


def _protagonist_input_identity(path: str) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    return file_identity(
        path, kind="initial_protagonist_checkpoint",
        format_version=int(checkpoint["checkpoint_version"]),
        extra={
            "encoding_config": checkpoint["encoding_config"],
            "model_config": checkpoint["model_config"],
            "value_norm": checkpoint["value_norm"],
        })


def _learner_input_identity(path: str) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    return file_identity(
        path, kind="initial_shadow_learner_checkpoint",
        format_version=int(checkpoint["checkpoint_version"]),
        extra={
            "encoding_config": checkpoint["encoding_config"],
            "model_config": checkpoint["model_config"],
            "value_norm": checkpoint["value_norm"],
        })


def _validate_learner_checkpoint_compatibility(
    champion: dict[str, Any],
    learner: dict[str, Any],
) -> None:
    """Reject learner ancestry that cannot continue the champion experiment."""
    if champion.get("checkpoint_version") != learner.get("checkpoint_version"):
        raise ValueError(
            "initial learner checkpoint version differs from champion")
    if configs_from_checkpoint(champion) != configs_from_checkpoint(learner):
        raise ValueError(
            "initial learner encoding, model, or value configuration differs "
            "from champion")
    champion_state = champion.get("model_state")
    learner_state = learner.get("model_state")
    if not isinstance(champion_state, dict) or not isinstance(learner_state, dict):
        raise ValueError(
            "champion and initial learner must contain model_state dictionaries")
    if champion_state.keys() != learner_state.keys():
        raise ValueError("initial learner model-state keys differ from champion")
    for name in champion_state:
        champion_tensor = champion_state[name]
        learner_tensor = learner_state[name]
        if (not torch.is_tensor(champion_tensor)
                or not torch.is_tensor(learner_tensor)
                or champion_tensor.shape != learner_tensor.shape
                or champion_tensor.dtype != learner_tensor.dtype):
            raise ValueError(
                "initial learner model-state tensor differs from champion: "
                f"{name}")


def _base_split_for_config(
    cfg: CoTrainingConfig,
    base_records: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = {
        record.get("static_level_signature") or record["level_id"]
        for record in base_records
    }
    if cfg.initial_base_split:
        manifest = load_manifest(cfg.initial_base_split)
        assigned = (
            set(manifest["train_levels"])
            | set(manifest["validation_levels"])
            | set(manifest["test_levels"])
        )
        if assigned != keys:
            raise ExperimentSpecIntegrityError(
                "initial base split membership does not exactly match the "
                "configured base dataset")
        return manifest
    ratios = SplitRatios(
        train=1.0 - cfg.val_ratio - cfg.test_ratio,
        validation=cfg.val_ratio, test=cfg.test_ratio)
    return make_split(sorted(keys), ratios=ratios, seed=cfg.seed)


def _cotraining_experiment_spec(
    cfg: CoTrainingConfig,
    base_records: list[dict[str, Any]],
    *,
    protagonist_identity_path: str | None = None,
    resolved_device: str | torch.device | None = None,
    evaluation_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = sorted({
        record.get("static_level_signature") or record["level_id"]
        for record in base_records})
    expected_split = _base_split_for_config(cfg, base_records)
    inputs: dict[str, Any] = {
        "base_dataset": file_identity(
            cfg.base_dataset, kind="base_dataset", count_lines=True),
        "initial_protagonist": _protagonist_input_identity(
            protagonist_identity_path or cfg.protagonist_checkpoint),
        "initial_designer": _designer_input_identity(
            cfg.designer_checkpoint, "initial_designer_checkpoint"),
        "initial_base_split": (
            file_identity(
                cfg.initial_base_split,
                kind="initial_base_split_manifest",
            )
            if cfg.initial_base_split else None
        ),
        "initial_protagonist_replay": (
            file_identity(
                cfg.initial_protagonist_replay,
                kind="initial_protagonist_replay_snapshot",
                count_lines=True,
            )
            if cfg.initial_protagonist_replay else None
        ),
        "initial_designer_replay": (
            file_identity(
                cfg.initial_designer_replay,
                kind="initial_designer_level_replay_snapshot",
                count_lines=True,
            )
            if cfg.initial_designer_replay else None
        ),
    }
    if cfg.initial_learner_checkpoint:
        inputs["initial_learner"] = _learner_input_identity(
            cfg.initial_learner_checkpoint)
    eval_identity = None
    if cfg.eval_levels_dataset:
        evaluation_manifest = evaluation_manifest or load_eval_split_manifest(
            cfg.eval_split_manifest,
            cfg.eval_levels_dataset,
            expected_split_seed=cfg.eval_split_seed,
            expected_validation_count=cfg.eval_validation_count,
        )
        pool = evaluation_manifest["pool"]
        source = Path(cfg.eval_levels_dataset)
        inputs["evaluation_levels"] = {
            "kind": "evaluation_levels",
            "path_hint": source.as_posix(),
            "sha256": pool["sha256"],
            "line_count": pool["record_count"],
        }
        inputs["evaluation_split_manifest"] = file_identity(
            cfg.eval_split_manifest,
            kind="evaluation_split_manifest",
            extra={
                "evaluation_split_fingerprint":
                    evaluation_manifest["evaluation_split_fingerprint"],
            },
        )
        eval_identity = evaluation_split_identity(
            evaluation_manifest, eval_limit=cfg.eval_limit)
    else:
        inputs["evaluation_levels"] = None
    if cfg.pretrained_designer_checkpoint:
        inputs["pretrained_designer"] = _designer_input_identity(
            cfg.pretrained_designer_checkpoint,
            "pretrained_designer_checkpoint")
    else:
        inputs["pretrained_designer"] = None
    if cfg.learner_retention_dataset:
        inputs["learner_retention_levels"] = file_identity(
            cfg.learner_retention_dataset,
            kind="learner_retention_development_pool",
            count_lines=True,
        )
    semantic = semantic_dataclass_config(
        cfg, semantic_fields=_COTRAIN_SEMANTIC_FIELDS,
        operational_fields=_COTRAIN_OPERATIONAL_FIELDS,
        input_fields=_COTRAIN_INPUT_FIELDS,
        continuation_horizon_fields=_COTRAIN_HORIZON_FIELDS,
        derived_fields=_COTRAIN_DERIVED_FIELDS,
        unordered_fields=("eval_budgets", "excluded_signatures"))
    derived = {
        "split_manifest_sha256": hash_canonical_value(expected_split),
        "split_level_count": len(keys),
    }
    if eval_identity is not None:
        derived["evaluation_split"] = eval_identity
    software_semantics = {
        "evaluation_semantics_version":
            EVALUATION_SEMANTICS_VERSION,
        "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
        "curriculum_adaptation_policy":
            "frontier_yield_aware_v1",
        "frontier_selection_policy":
            "construction_proof_staged_exact_selected_backfill_v2",
        "frontier_estimation_policy":
            "geometric_search_budget_sweep_v1",
        "fresh_data_training_guard":
            "minimum_new_levels_before_finetune_v1",
        "generation_resume_policy":
            "per_level_rng_atomic_partial_checkpoint_v1",
        "validation_cache_policy":
            "content_addressed_checkpoint_split_budget_v1",
        "training_state_sampling_policy":
            "bounded_optimal_and_random_no_near_optimal_v1",
        "oracle_analysis_short_circuit_policy":
            "stop_when_exact_policy_target_impossible_v1",
        "replay_source_weighting_policy":
            "weighted_sampling_source_weighted_policy_loss_v2",
        "replay_age_composition_policy":
            "fresh_recent_historical_quota_v1",
        "search_value_supervision_policy":
            "bounded_estimate_policy_only_by_default_v1",
        "search_record_contract":
            "explicit_approximate_completeness_v1",
        "encoding_block_limit_policy":
            "generator_and_state_must_not_exceed_checkpoint_v1",
        "generated_level_acceptance_policy":
            "construction_proof_with_oracle_contradiction_rejection_v1",
        "frontier_statistics_population":
            "unique_nonduplicate_valid_candidates_v1",
        "loss_aggregation_policy":
            "global_supervision_mass_weighted_v1",
        "policy_target_conditioning_policy":
            "complete_exact_optimal_support_incumbent_mix_v2",
        "designer_frontier_reward_policy":
            "budget_sweep_centered_plateau_alignment_v3",
        "designer_validation_policy":
            "fixed_multilevel_post_update_selection_v1",
        "designer_checkpoint_selection_policy":
            "frontier_in_band_alignment_reward_lexicographic_v1",
        "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
        "split_algorithm_version": expected_split["version"],
        "experiment_identity_version": 1,
        "runtime": runtime_device_provenance(
            requested_device=cfg.device,
            resolved_device=resolved_device or _resolve_device(cfg.device)),
    }
    # Legacy ``hybrid`` intentionally adds no new software key so an old run
    # can resume when the user explicitly selects its unchanged semantics.
    # The corrected search-only behavior and the new path-retaining behavior
    # receive explicit identities and cannot silently resume older runs.
    if cfg.label_mode == "hybrid_path":
        software_semantics["teacher_labeling_policy"] = (
            "full_exact_cached_path_then_neural_search_v1")
    elif cfg.label_mode == "search_only":
        software_semantics["teacher_labeling_policy"] = (
            "neural_search_only_no_astar_v2")
    if eval_identity is not None:
        software_semantics["evaluation_split_algorithm"] = EVAL_SPLIT_ALGORITHM
        software_semantics["external_final_test_policy"] = (
            "sealed_until_explicit_evaluation_v1")
    else:
        software_semantics["fallback_evaluation_selection_policy"] = (
            _FALLBACK_EVAL_SELECTION_POLICY)
    if cfg.promotion_metric == "weighted_budget_sweep_solve_rate":
        software_semantics["full_level_solve_promotion_policy"] = (
            "completed_trajectory_over_total_evaluated_v1")
    if cfg.promotion_paired_gate_enabled:
        software_semantics["paired_promotion_gate_policy"] = {
            "policy": (
                "inclusive_margin_per_budget_guard_"
                "paired_bootstrap_by_level_v1"),
            "maximum_per_budget_regression":
                cfg.promotion_max_per_budget_regression,
            "bootstrap_confidence": cfg.promotion_bootstrap_confidence,
            "bootstrap_replicates": cfg.promotion_bootstrap_replicates,
            "bootstrap_seed": cfg.promotion_bootstrap_seed,
        }
    if cfg.stop_after_promotion:
        software_semantics["promotion_terminal_policy"] = (
            "stop_after_first_durably_committed_promotion_v1")
    if cfg.shadow_learner_enabled:
        software_semantics["shadow_learner_policy"] = (
            "champion_generated_cumulative_learner_milestone_gate_v1")
        software_semantics["learner_continuation_safety_policy"] = (
            "mechanical_each_round_anchor_policy_drift_at_milestones_v1")
        if cfg.initial_learner_checkpoint:
            software_semantics["shadow_learner_initialization_policy"] = (
                "external_checkpoint_import_champion_isolated_v1")
    if cfg.learner_retention_dataset:
        software_semantics["learner_retention_policy"] = {
            "policy": "baseline_difficulty_band_solve_retention_v1",
            "budgets": list(cfg.learner_retention_budgets),
            "selection": (
                "all_source_levels_v1"
                if cfg.learner_retention_use_full_pool
                else "stable_signature_sorted_per_band_slice_v1"),
            "levels_per_band": (
                None if cfg.learner_retention_use_full_pool
                else cfg.learner_retention_per_band),
            "maximum_regression": cfg.learner_retention_max_regression,
            "enforce_continuation": cfg.learner_retention_enforce,
        }
    return build_experiment_spec(
        pipeline="cotraining", semantic_config=semantic, inputs=inputs,
        software_semantics=software_semantics,
        derived=derived)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _gen_cfg(
    state: CurriculumState,
    *,
    max_blocks: int,
) -> GeneratorConfig:
    return GeneratorConfig(rows=state.rows, cols=state.cols,
                           color_count=state.color_count, density=state.density,
                           max_blocks=max_blocks)


def drop_frozen_candidates(candidates, frozen_sigs):
    """Remove any candidate whose level signature is a frozen val/test signature.

    Guards against benchmark/validation leakage into protagonist training.
    """
    return [(s, p) for (s, p) in candidates
            if static_level_signature(s.level) not in frozen_sigs]


def _level_balanced_eval_records(
    env: Environment,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select at most one deterministic, initial-preferred state per level."""
    rows = filter_records_for_split(records, manifest, split)
    by_level: dict[str, list[dict[str, Any]]] = {}
    for record in rows:
        by_level.setdefault(group_key(record), []).append(record)

    level_keys = list(manifest[f"{split}_levels"])
    if limit is not None:
        level_keys = level_keys[:limit]

    selected: list[dict[str, Any]] = []
    for level_key in level_keys:
        candidates = by_level.get(level_key, [])
        if not candidates:
            raise ExperimentSpecIntegrityError(
                f"frozen {split} level {level_key!r} has no base-dataset "
                "records")

        level = level_from_dict(candidates[0]["level"])
        initial_key = env.canonical_key(env.initial_state(level))

        def state_key(record: dict[str, Any]) -> str:
            stored = record.get("state_key")
            if stored is not None:
                return str(stored)
            record_level = level_from_dict(record["level"])
            state = deserialize_state(record_level, record["state"])
            return env.canonical_key(state)

        initial = next(
            (record for record in candidates
             if state_key(record) == initial_key),
            None,
        )
        # A legacy dataset may omit the initial state. Keep the fallback stable
        # under JSONL reordering by choosing the smallest canonical state key.
        selected.append(initial or min(candidates, key=state_key))
    return selected


def _quantile(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _adapt_curriculum_from_generation(
    curriculum: CurriculumState,
    generation: dict[str, Any],
    cfg: CurriculumConfig,
) -> tuple[CurriculumState, dict[str, Any]]:
    """Adapt only when generation produced an independent frontier signal."""
    unique_valid = int(generation["unique_valid_count"])
    if unique_valid == 0:
        state = curriculum.to_dict()
        return curriculum, {
            "direction": "hold",
            "reason": (
                "no unique nonduplicate valid candidates; curriculum unchanged"
            ),
            "mean_solve_rate": None,
            "frontier_acceptance_rate": None,
            "below_frontier_rate": None,
            "above_frontier_rate": None,
            "changed": False,
            "changes": {},
            "before": state,
            "after": state,
        }
    return adapt_curriculum(
        curriculum,
        generation["mean_solve_rate"],
        cfg,
        frontier_acceptance_rate=generation["frontier_acceptance_rate"],
        below_frontier_rate=(
            generation["below_frontier_count"] / unique_valid
        ),
        above_frontier_rate=(
            generation["above_frontier_count"] / unique_valid
        ),
    )


_GENERATION_PARTIAL_SCHEMA_VERSION = 2
_VALIDATION_CACHE_SCHEMA_VERSION = 2


def _generation_partial_identity(
    *,
    experiment_fingerprint: str,
    round_number: int,
    curriculum: CurriculumState,
    run_state: dict[str, Any],
    generator_model_state_sha256: str,
    cfg: CoTrainingConfig,
) -> dict[str, Any]:
    """Identity for resumable cheap-stage generation."""
    return {
        "experiment_fingerprint": experiment_fingerprint,
        "round": round_number,
        "curriculum": curriculum.to_dict(),
        "active_protagonist_sha256":
            run_state["active_protagonist_sha256"],
        "designer_checkpoint_sha256":
            run_state["designer_checkpoint_sha256"],
        "generator_model_state_sha256": generator_model_state_sha256,
        "levels_per_round": cfg.levels_per_round,
        "solve_rate_trials": cfg.solve_rate_trials,
        "frontier_dirichlet_alpha": cfg.frontier_dirichlet_alpha,
        "frontier_dirichlet_weight": cfg.frontier_dirichlet_weight,
        "frontier_budget_min_ratio": cfg.frontier_budget_min_ratio,
        "frontier_budget_max_ratio": cfg.frontier_budget_max_ratio,
        "frontier_simulation_budgets": list(geometric_budget_sweep(
            center=curriculum.protagonist_simulations,
            trials=cfg.solve_rate_trials,
            minimum_ratio=cfg.frontier_budget_min_ratio,
            maximum_ratio=cfg.frontier_budget_max_ratio,
            minimum_simulations=cfg.curriculum.min_protagonist_simulations,
            maximum_simulations=cfg.curriculum.max_protagonist_simulations,
        )),
    }


def _load_generation_partial(
    path: Path,
    *,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"generation partial checkpoint is unreadable: {path}") from exc
    if payload.get("schema_version") != _GENERATION_PARTIAL_SCHEMA_VERSION:
        raise RuntimeError(
            "generation partial checkpoint has an unsupported schema version")
    if payload.get("identity") != identity:
        raise RuntimeError(
            "generation partial checkpoint belongs to different run semantics")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeError(
            "generation partial checkpoint attempts must be a list")
    if len(attempts) > int(identity["levels_per_round"]):
        raise RuntimeError(
            "generation partial checkpoint exceeds configured level count")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or attempt.get("index") != index:
            raise RuntimeError(
                "generation partial checkpoint attempt order is invalid")
        if attempt.get("status") not in {"invalid", "valid"}:
            raise RuntimeError(
                "generation partial checkpoint attempt status is invalid")
        if attempt["status"] == "valid":
            required = {
                "fingerprint", "level", "trajectory", "solve_rate",
                "solve_rate_trials", "solve_rate_solved",
                "solve_rate_budgets", "duplicate",
                "num_mutations",
            }
            if not required.issubset(attempt):
                raise RuntimeError(
                    "generation partial checkpoint valid attempt is incomplete")
            if not isinstance(attempt["duplicate"], bool):
                raise RuntimeError(
                    "generation partial checkpoint duplicate flag is invalid")
            if (not isinstance(attempt["fingerprint"], str)
                    or not attempt["fingerprint"]):
                raise RuntimeError(
                    "generation partial checkpoint fingerprint is invalid")
            if (not isinstance(attempt["trajectory"], list)
                    or any(isinstance(action, bool)
                           or not isinstance(action, int)
                           for action in attempt["trajectory"])):
                raise RuntimeError(
                    "generation partial checkpoint trajectory is invalid")
            rate = attempt["solve_rate"]
            if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                    or not 0.0 <= float(rate) <= 1.0):
                raise RuntimeError(
                    "generation partial checkpoint solve rate is invalid")
            trials = attempt["solve_rate_trials"]
            solved = attempt["solve_rate_solved"]
            if (isinstance(trials, bool) or not isinstance(trials, int)
                    or trials != identity["solve_rate_trials"]):
                raise RuntimeError(
                    "generation partial checkpoint solve-rate trials are invalid")
            if (isinstance(solved, bool) or not isinstance(solved, int)
                    or not 0 <= solved <= trials
                    or abs(float(rate) - solved / trials) > 1e-12):
                raise RuntimeError(
                    "generation partial checkpoint solved count is invalid")
            budgets = attempt["solve_rate_budgets"]
            if (not isinstance(budgets, list)
                    or budgets != identity["frontier_simulation_budgets"]):
                raise RuntimeError(
                    "generation partial checkpoint solve-rate budgets are "
                    "invalid")
            mutations = attempt["num_mutations"]
            if (isinstance(mutations, bool) or not isinstance(mutations, int)
                    or mutations < 0):
                raise RuntimeError(
                    "generation partial checkpoint mutation count is invalid")
            # Parsing validates the serialized level schema eagerly.
            level_from_dict(attempt["level"])
        elif (not isinstance(attempt.get("errors"), list)
              or any(not isinstance(error, str)
                     for error in attempt["errors"])):
            raise RuntimeError(
                "generation partial checkpoint invalid errors are malformed")
    return attempts


def _write_generation_partial(
    path: Path,
    *,
    identity: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> None:
    atomic_write_json(path, {
        "schema_version": _GENERATION_PARTIAL_SCHEMA_VERSION,
        "identity": identity,
        "attempts": attempts,
    })


def _write_or_verify_text(path: Path, text: str, *, label: str) -> None:
    """Persist an immutable round input, or verify an interrupted retry."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                f"persisted {label} differs from deterministic reconstruction: "
                f"{path}")
        return
    atomic_write_text(path, text)


def _write_or_verify_json(path: Path, value: Any, *, label: str) -> None:
    """Persist immutable JSON input without silently replacing prior data."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(
                f"persisted {label} differs from deterministic reconstruction: "
                f"{path}")
        return
    atomic_write_json(path, value)


def _validation_cache_key(metadata: dict[str, Any]) -> str:
    return hash_canonical_value(metadata)


class CoTraining:
    def __init__(self, cfg: CoTrainingConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.output_dir)
        self.device = _resolve_device(cfg.device)
        self.env = Environment()
        self.reward_cfg = cfg.reward or RewardConfig()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _run_state(self) -> dict[str, Any]:
        path = self.root / "run_state.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "completed_rounds": [],
            "best_protagonist": None,
            "designer_checkpoint": self.cfg.designer_checkpoint,
            "designer_checkpoint_sha256": None,
            "curriculum": self.cfg.initial_curriculum.to_dict(),
            "accepted_fingerprints": [],
            "history": [],
        }

    def _save_run_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.root / "run_state.json", state)

    def _resolve_shadow_checkpoint(
        self, state: dict[str, Any], role: str
    ) -> Path:
        path_key = f"active_{role}_checkpoint"
        sha_key = f"active_{role}_sha256"
        relative = state.get(path_key)
        expected = state.get(sha_key)
        if not relative or not expected:
            raise ExperimentSpecIntegrityError(
                f"shadow learner state has no committed {role} identity")
        path = resolve_run_path(self.root, relative)
        if not path.is_file():
            raise ExperimentSpecIntegrityError(
                f"committed {role} checkpoint is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ExperimentSpecIntegrityError(
                f"committed {role} checkpoint integrity failure: "
                f"expected={expected}, observed={observed}")
        load_checkpoint(path, map_location="cpu")
        return path

    def _prepare_shadow_learner_state(self, state: dict[str, Any]) -> None:
        if not self.cfg.shadow_learner_enabled:
            return
        if state.get("active_learner_checkpoint"):
            self._resolve_shadow_checkpoint(state, "learner")
            self._resolve_shadow_checkpoint(state, "learner_anchor")
            return
        champion = resolve_committed_protagonist(self.root, state)
        learner = champion
        initialization = "champion"
        input_sha256 = None
        if self.cfg.initial_learner_checkpoint:
            source = Path(self.cfg.initial_learner_checkpoint)
            if not source.is_file():
                raise FileNotFoundError(
                    f"initial learner checkpoint is missing: {source}")
            champion_payload = load_checkpoint(champion, map_location="cpu")
            learner_payload = load_checkpoint(source, map_location="cpu")
            _validate_learner_checkpoint_compatibility(
                champion_payload, learner_payload)
            input_sha256 = sha256_file(source)
            learner = self.root / "learner" / "initial.pt"
            learner.parent.mkdir(parents=True, exist_ok=True)
            if learner.exists():
                observed = sha256_file(learner)
                if observed != input_sha256:
                    raise ExperimentSpecIntegrityError(
                        "imported initial learner integrity failure: "
                        f"expected={input_sha256}, observed={observed}")
            else:
                atomic_copy(source, learner)
            initialization = "external_checkpoint"
        learner_relative = relative_to_run(learner, self.root)
        learner_sha256 = sha256_file(learner)
        state.update({
            "shadow_learner_schema_version": 1,
            "shadow_learner_initialization": initialization,
            "initial_learner_input_sha256": input_sha256,
            "active_learner_checkpoint": learner_relative,
            "active_learner_sha256": learner_sha256,
            "active_learner_source_round": 0,
            "active_learner_anchor_checkpoint": learner_relative,
            "active_learner_anchor_sha256": learner_sha256,
            "active_learner_anchor_source_round": 0,
            "learner_milestones": [],
            "learner_rollbacks": [],
        })

    def _crash_point(self, stage: str) -> None:
        """Test seam for deterministic crash injection."""

    def _prune_superseded_round_artifacts(
        self,
        state: dict[str, Any],
        *,
        current_round: int,
    ) -> dict[str, Any]:
        """Delete only committed artifacts that no resume path references."""
        root = self.root.resolve()
        promoted_rounds = {
            int(item["round"]) for item in state.get("commits", [])
            if item.get("promoted")
        }
        deleted = []
        for round_number in sorted(
                int(value) for value in state.get("completed_rounds", [])
                if int(value) < current_round):
            milestone = (
                self.cfg.shadow_learner_enabled
                and learner_milestone(
                    round_number, self.cfg.learner_milestone_interval)
            )
            names = ["replay.jsonl", "level_replay.jsonl"]
            if not milestone and round_number not in promoted_rounds:
                names.extend([
                    "training_sample.jsonl",
                    "training_sample_source.jsonl",
                ])
            round_dir = root / f"round_{round_number:03d}"
            for name in names:
                path = (round_dir / name).resolve()
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"storage-pruning target escaped run root: {path}"
                    ) from exc
                if not path.is_file():
                    continue
                size = path.stat().st_size
                path.unlink()
                deleted.append({
                    "round": round_number,
                    "path": path.relative_to(root).as_posix(),
                    "bytes": size,
                })
        event = {
            "schema_version": 1,
            "policy": "superseded_replay_nonmilestone_samples_v1",
            "after_committed_round": current_round,
            "deleted_file_count": len(deleted),
            "deleted_bytes": sum(item["bytes"] for item in deleted),
            "deleted": deleted,
        }
        log_path = root / "storage_pruning.json"
        history = []
        if log_path.exists():
            try:
                history = json.loads(log_path.read_text(encoding="utf-8"))[
                    "events"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                raise RuntimeError("storage-pruning log is corrupt")
        atomic_write_json(log_path, {
            "schema_version": 1,
            "events": [*history, event],
        })
        if deleted:
            print(
                "storage pruning: removed "
                f"{len(deleted)} superseded files "
                f"({event['deleted_bytes'] / (1024 ** 2):.1f} MiB)",
                flush=True,
            )
        return event

    def _evaluate_retention_cached(
        self,
        model,
        encoding_config,
        value_norm,
        *,
        checkpoint_content_sha256: str,
        role: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Content-address a milestone solve-retention evaluation."""
        metadata = {
            "schema_version": 1,
            "semantics": "baseline_difficulty_band_solve_retention_v1",
            "checkpoint_content_sha256": checkpoint_content_sha256,
            "selected": [{
                "static_level_signature": row["static_level_signature"],
                "difficulty_stratum": row["difficulty_stratum"],
            } for row in self.retention_records],
            "budgets": list(self.cfg.learner_retention_budgets),
            "seed": self.cfg.seed,
            "c_puct": 1.5,
        }
        cache_key = hash_canonical_value(metadata)
        path = self.root / "retention_cache" / f"{cache_key}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload["rows"]
                if (payload.get("metadata") == metadata
                        and payload.get("rows_sha256")
                        == hash_canonical_value(rows)):
                    print(
                        f"learner retention {role}: loaded cache "
                        f"{cache_key[:12]}", flush=True)
                    return rows, {"role": role, "hit": True,
                                  "key": cache_key}
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        print(
            f"learner retention {role}: evaluating "
            f"{len(self.retention_records)} levels at "
            f"budgets={list(self.cfg.learner_retention_budgets)}",
            flush=True,
        )
        rows = evaluate_retention(
            self.env, model, encoding_config, value_norm,
            self.retention_records,
            budgets=self.cfg.learner_retention_budgets,
            device=self.device, seed=self.cfg.seed,
        )
        atomic_write_json(path, {
            "schema_version": 1,
            "metadata": metadata,
            "rows": rows,
            "rows_sha256": hash_canonical_value(rows),
        })
        return rows, {"role": role, "hit": False, "key": cache_key}

    def _prepare_checkpoint_state(self, state: dict[str, Any]) -> None:
        designer = state.get("designer_checkpoint")
        if not designer and "schema_version" not in state:
            # Legacy states did not persist the active designer path.  The
            # experiment-identity migration has already validated the original
            # configured input by the time normal resume reaches this method.
            designer = self.cfg.designer_checkpoint
            state["designer_checkpoint"] = designer
        if not designer:
            raise ExperimentSpecIntegrityError(
                "co-training state has no active designer checkpoint")
        load_designer(designer, map_location="cpu")
        observed_designer = sha256_file(designer)
        expected_designer = state.get("designer_checkpoint_sha256")
        if (expected_designer is not None
                and expected_designer != observed_designer):
            raise ExperimentSpecIntegrityError(
                "committed designer checkpoint integrity failure: "
                f"expected={expected_designer}, observed={observed_designer}")
        state["designer_checkpoint_sha256"] = observed_designer

        if state.get("active_protagonist_checkpoint"):
            resolve_committed_protagonist(self.root, state)
            self._prepare_shadow_learner_state(state)
            return
        completed = set(state.get("completed_rounds", []))
        incomplete = [
            path for path in self.root.glob("round_*")
            if int(path.name.split("_")[-1]) not in completed
            and ((path / "candidate.pt").exists()
                 or (path / "report.json").exists()
                 or (path / "report.prepared.json").exists())
        ]
        if state.get("best_protagonist") and incomplete:
            raise RuntimeError(
                "legacy co-training run is ambiguous: uncommitted round artifacts "
                f"exist in {incomplete[0]}; restore the pre-round best.pt or remove "
                "the incomplete directory after inspection")
        source = Path(state.get("best_protagonist") or
                      self.cfg.protagonist_checkpoint)
        if (state.get("best_protagonist") and not source.is_file()
                and (self.root / "best.pt").is_file()):
            source = self.root / "best.pt"
        if not source.is_file():
            raise FileNotFoundError(f"legacy protagonist checkpoint is missing: {source}")
        load_checkpoint(source, map_location="cpu")
        destination = self.root / "protagonist" / (
            "legacy_import.pt" if state.get("best_protagonist") else "initial.pt")
        if not destination.exists():
            atomic_copy(source, destination)
        state["schema_version"] = 2
        state["active_protagonist_checkpoint"] = relative_to_run(
            destination, self.root)
        state["active_protagonist_sha256"] = sha256_file(destination)
        state["active_protagonist_source_round"] = max(completed) if completed else 0
        state["best_protagonist"] = str(self.root / "best.pt")
        self._prepare_shadow_learner_state(state)
        if completed:
            print("upgraded legacy co-training run state with a committed "
                  "checkpoint identity", flush=True)

    def _repair_committed_artifacts(self, state: dict[str, Any]) -> None:
        for commit in state.get("commits", []):
            prepared = resolve_run_path(self.root, commit["prepared_report"])
            if sha256_file(prepared) != commit["prepared_report_sha256"]:
                raise RuntimeError(
                    f"committed prepared report integrity failure: {prepared}")
            report = json.loads(prepared.read_text(encoding="utf-8"))
            report["commit_status"] = "committed"
            report_path = self.root / f"round_{commit['round']:03d}" / "report.json"
            rewrite = True
            if report_path.exists():
                try:
                    rewrite = json.loads(report_path.read_text(encoding="utf-8")) != report
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
            if rewrite:
                atomic_write_json(report_path, report)
            prior = commit.get("prior_levels")
            if prior:
                path = resolve_run_path(self.root, prior)
                if sha256_file(path) != commit["prior_levels_sha256"]:
                    raise RuntimeError(
                        f"committed prior-level artifact integrity failure: {path}")
                levels = [level_from_dict(item) for item in
                          json.loads(path.read_text(encoding="utf-8"))]
                if levels:
                    bench.append_prior_round(self.root, levels)

    def _validate_existing_run(self, state: dict[str, Any]) -> None:
        """Read-only validation required before any resume-side write."""
        completed = set(state.get("completed_rounds", []))
        if state.get("active_protagonist_checkpoint"):
            active = resolve_committed_protagonist(self.root, state)
            load_checkpoint(active, map_location="cpu")
        elif completed:
            raise ExperimentSpecIntegrityError(
                "identified co-training run has committed rounds but no "
                "active protagonist checkpoint")
        if self.cfg.shadow_learner_enabled:
            self._resolve_shadow_checkpoint(state, "learner")
            self._resolve_shadow_checkpoint(state, "learner_anchor")
        designer = state.get("designer_checkpoint")
        if not designer and "schema_version" not in state:
            # Read-only legacy validation: defer persisting this migration to
            # _prepare_checkpoint_state after all existing state is validated.
            designer = self.cfg.designer_checkpoint
        if not designer:
            raise ExperimentSpecIntegrityError(
                "co-training state has no active designer checkpoint")
        load_designer(designer, map_location="cpu")
        expected = state.get("designer_checkpoint_sha256")
        if expected is not None:
            observed = sha256_file(designer)
            if observed != expected:
                raise ExperimentSpecIntegrityError(
                    "committed designer checkpoint integrity failure: "
                    f"expected={expected}, observed={observed}")

        replay_relative = state.get("active_replay_snapshot")
        if replay_relative:
            replay_path = resolve_run_path(self.root, replay_relative)
            if not replay_path.is_file():
                raise ExperimentSpecIntegrityError(
                    f"committed replay snapshot is missing: {replay_path}")
            observed = sha256_file(replay_path)
            if observed != state.get("active_replay_sha256"):
                raise ExperimentSpecIntegrityError(
                    "committed replay snapshot integrity failure: "
                    f"expected {state.get('active_replay_sha256')}, "
                    f"observed {observed}")
            ReplayBuffer(
                self.root / ".identity_validation_replay",
                max_examples=self.cfg.max_protagonist_replay,
                seed=self.cfg.seed).load_snapshot(replay_path)
        elif completed:
            raise ExperimentSpecIntegrityError(
                "identified co-training run has committed rounds but no "
                "active replay snapshot")

        level_relative = state.get("active_level_replay_snapshot")
        if level_relative:
            level_path = resolve_run_path(self.root, level_relative)
            if not level_path.is_file():
                raise ExperimentSpecIntegrityError(
                    f"committed level-replay snapshot is missing: {level_path}")
            observed = sha256_file(level_path)
            if observed != state.get("active_level_replay_sha256"):
                raise ExperimentSpecIntegrityError(
                    "committed level-replay snapshot integrity failure: "
                    f"expected {state.get('active_level_replay_sha256')}, "
                    f"observed {observed}")
            LevelReplayBuffer(
                self.root / ".identity_validation_level_replay",
                max_levels=self.cfg.max_designer_replay,
                seed=self.cfg.seed).load_snapshot(level_path)
        elif completed:
            raise ExperimentSpecIntegrityError(
                "identified co-training run has committed rounds but no "
                "active level-replay snapshot")

        commits = state.get("commits", [])
        committed_numbers = {int(item["round"]) for item in commits}
        if not completed.issubset(committed_numbers):
            raise ExperimentSpecIntegrityError(
                "committed co-training history is missing report evidence for "
                f"rounds {sorted(completed - committed_numbers)}")
        for commit in commits:
            prepared = resolve_run_path(
                self.root, commit["prepared_report"])
            if not prepared.is_file():
                raise ExperimentSpecIntegrityError(
                    f"committed prepared report is missing: {prepared}")
            if sha256_file(prepared) != commit["prepared_report_sha256"]:
                raise ExperimentSpecIntegrityError(
                    f"committed prepared report integrity failure: {prepared}")
            json.loads(prepared.read_text(encoding="utf-8"))
            prior = commit.get("prior_levels")
            if prior:
                prior_path = resolve_run_path(self.root, prior)
                if not prior_path.is_file():
                    raise ExperimentSpecIntegrityError(
                        f"committed prior-level artifact is missing: {prior_path}")
                if sha256_file(prior_path) != commit["prior_levels_sha256"]:
                    raise ExperimentSpecIntegrityError(
                        f"committed prior-level integrity failure: {prior_path}")
                json.loads(prior_path.read_text(encoding="utf-8"))

    def _frozen_split(self, base_records):
        path = self.root / "splits.json"
        if path.exists():
            return load_manifest(path)
        manifest = _base_split_for_config(self.cfg, base_records)
        save_manifest(manifest, path)
        return manifest

    def _states_for_split(self, base_records, manifest, split, limit):
        rows = _level_balanced_eval_records(
            self.env, base_records, manifest, split, limit)
        return [deserialize_state(level_from_dict(r["level"]), r["state"])
                for r in rows]

    def _external_validation_states(self, path, manifest):
        """Load only promotion-validation states in persisted manifest order.

        Records may carry a ``state`` (used as-is) or just a ``level`` (the
        initial state is used). All manifest signatures are returned so both
        roles can be excluded from training/replay without constructing
        final-test states during co-training.
        """
        records = load_records(path)
        by_signature = {}
        for r in records:
            level = level_from_dict(r["level"])
            sig = r.get("static_level_signature") or static_level_signature(level)
            by_signature[sig] = (r, level)
        validation_signatures = [
            item["signature"] for item in manifest["promotion_validation"]]
        test_signatures = [item["signature"] for item in manifest["final_test"]]
        val_states = []
        for sig in validation_signatures:
            record, level = by_signature[sig]
            val_states.append(
                deserialize_state(level, record["state"])
                if record.get("state") else self.env.initial_state(level))
        sigs = set(validation_signatures) | set(test_signatures)
        return val_states, sigs

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        config_path = self.root / "config.json"
        spec_path = self.root / EXPERIMENT_SPEC_FILE
        setup_only_recovery = validate_identified_run_state_presence(
            self.root, pipeline_label="co-training",
            allowed_setup_files=(
                "config.json", "splits.json", "protagonist/initial.pt"))
        legacy_cfg = None
        if config_path.exists() and not spec_path.exists():
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            legacy_cfg = persisted
            persisted_metric = persisted.get("promotion_metric")
            if (isinstance(persisted_metric, str)
                    and persisted_metric != cfg.promotion_metric):
                print("preserving persisted promotion metric on resume: "
                      f"{persisted_metric}", flush=True)
                cfg = replace(cfg, promotion_metric=persisted_metric)
                self.cfg = cfg

        base_records = load_records(cfg.base_dataset)
        evaluation_manifest = None
        requested_eval_identity = None
        if cfg.eval_levels_dataset:
            evaluation_manifest = load_eval_split_manifest(
                cfg.eval_split_manifest,
                cfg.eval_levels_dataset,
                expected_split_seed=cfg.eval_split_seed,
                expected_validation_count=cfg.eval_validation_count,
            )
            requested_eval_identity = evaluation_split_identity(
                evaluation_manifest, eval_limit=cfg.eval_limit)
        run_state_path = self.root / "run_state.json"
        run_state = self._run_state()
        if run_state_path.exists():
            persisted_eval_identity = run_state.get("evaluation_split")
            if requested_eval_identity is not None \
                    and persisted_eval_identity is None:
                raise ExperimentSpecIntegrityError(
                    "legacy co-training run has external held-out evaluation "
                    "levels but no persisted fixed evaluation split; start a "
                    "new output directory with --eval-split-manifest")
            if persisted_eval_identity != requested_eval_identity:
                raise ExperimentSpecIntegrityError(
                    "persisted evaluation split identity differs from the "
                    "requested manifest")
        committed_round = max(
            [int(value) for value in run_state.get("completed_rounds", [])]
            + [int(run_state.get("active_protagonist_source_round", 0))])
        validate_continuation_horizon(
            name="rounds", requested=cfg.rounds, committed=committed_round,
            run_dir=self.root)
        requested_spec = _cotraining_experiment_spec(
            cfg, base_records, resolved_device=self.device,
            evaluation_manifest=evaluation_manifest)
        legacy_spec = None
        if legacy_cfg is not None:
            legacy_spec = load_legacy_migration_spec(
                self.root, pipeline="cotraining",
                unavailable_fields=(
                    "base_dataset.sha256",
                    "initial_base_split.sha256",
                    "initial_designer.sha256",
                    "initial_protagonist_replay.sha256",
                    "initial_designer_replay.sha256",
                    "evaluation_levels.sha256",
                    "evaluation_split_manifest.sha256",
                    "pretrained_designer.sha256"))
        split_path = self.root / "splits.json"
        expected_split_hash = (
            legacy_spec if legacy_spec is not None else requested_spec
        )["derived"]["split_manifest_sha256"]
        validated_split = None
        if split_path.exists():
            actual_split = load_manifest(split_path)
            actual_split_hash = hash_canonical_value(actual_split)
            if actual_split_hash != expected_split_hash:
                raise ExperimentSpecIntegrityError(
                    "persisted split manifest does not match the experiment "
                    f"identity: expected={expected_split_hash}, "
                    f"observed={actual_split_hash}")
            validated_split = actual_split
        fingerprint, migrated = validate_or_initialize_experiment(
            self.root, requested_spec,
            run_state=(run_state if run_state_path.exists() else None),
            legacy_spec=legacy_spec)
        self.experiment_fingerprint = fingerprint
        run_state["experiment_fingerprint"] = fingerprint
        if requested_eval_identity is not None:
            run_state["evaluation_split"] = requested_eval_identity
        if migrated and legacy_spec is not None:
            print("migrated legacy co-training experiment identity", flush=True)

        self.eval_val_states = None
        self.evaluation_split_identity = requested_eval_identity
        eval_signatures: set[str] = set()
        if cfg.eval_levels_dataset:
            self.eval_val_states, eval_signatures = (
                self._external_validation_states(
                    cfg.eval_levels_dataset, evaluation_manifest))
            info = requested_eval_identity
            print(
                "held-out evaluation split:\n"
                f"  manifest: {cfg.eval_split_manifest}\n"
                f"  fingerprint: {info['evaluation_split_fingerprint']}\n"
                f"  pool levels: "
                f"{info['promotion_validation_count'] + info['final_test_count']}\n"
                f"  promotion validation: {info['promotion_validation_count']}\n"
                f"  final test: {info['final_test_count']}\n"
                f"  split seed: {info['split_seed']}",
                flush=True,
            )

        if run_state_path.exists():
            self._validate_existing_run(run_state)
        elif setup_only_recovery:
            initial = self.root / "protagonist" / "initial.pt"
            if initial.exists():
                load_checkpoint(initial, map_location="cpu")
                observed = sha256_file(initial)
                expected = requested_spec["inputs"][
                    "initial_protagonist"]["sha256"]
                if observed != expected:
                    raise ExperimentSpecIntegrityError(
                        "setup-only initial protagonist integrity failure: "
                        f"expected={expected}, observed={observed}")

        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, cfg.to_dict())
        split = validated_split or self._frozen_split(base_records)
        self.frozen_sigs = (set(split["validation_levels"])
                            | set(split["test_levels"])
                            | set(cfg.excluded_signatures))
        self.retention_records: list[dict[str, Any]] = []
        if cfg.learner_retention_dataset:
            (self.retention_records, retention_signatures,
             retention_manifest) = load_retention_pool(
                cfg.learner_retention_dataset,
                per_band=(
                    None if cfg.learner_retention_use_full_pool
                    else cfg.learner_retention_per_band),
            )
            retention_manifest = {
                **retention_manifest,
                "source_identity": file_identity(
                    cfg.learner_retention_dataset,
                    kind="learner_retention_development_pool",
                    count_lines=True,
                ),
                "budgets": list(cfg.learner_retention_budgets),
                "maximum_allowed_regression":
                    cfg.learner_retention_max_regression,
                "enforce_continuation": cfg.learner_retention_enforce,
            }
            _write_or_verify_json(
                self.root / "learner_retention_manifest.json",
                retention_manifest,
                label="learner retention manifest",
            )
            # The entire source pool, not only the selected monitoring slice,
            # remains evaluation-only and cannot enter replay or generation.
            self.frozen_sigs |= retention_signatures
            print(
                "learner retention enabled: "
                f"levels={len(self.retention_records)}; "
                f"budgets={list(cfg.learner_retention_budgets)}; "
                f"maximum regression={cfg.learner_retention_max_regression}; "
                f"enforced={cfg.learner_retention_enforce}",
                flush=True,
            )
        self.learner_safety_states = None
        if cfg.shadow_learner_enabled:
            self.learner_safety_states = self._states_for_split(
                base_records, split, "validation", cfg.eval_limit)
            if not self.learner_safety_states:
                raise ValueError(
                    "shadow learner requires a non-empty base validation "
                    "split for policy-drift safety checks")
            print(
                "shadow learner enabled: champion controls generation; "
                f"milestone interval={cfg.learner_milestone_interval}; "
                f"safety states={len(self.learner_safety_states)}",
                flush=True,
            )

        # Optional harder held-out evaluation levels for promotion (decoupled
        # from a saturated base-dataset split). Their signatures are excluded
        # from training to prevent leakage.
        self.frozen_sigs |= eval_signatures

        self._prepare_checkpoint_state(run_state)

        # Canonical encoding/value-norm from the protagonist checkpoint.
        active = resolve_committed_protagonist(self.root, run_state)
        pck = load_checkpoint(active, map_location="cpu")
        enc, model_cfg, value_norm = configs_from_checkpoint(pck)

        # Protagonist state-replay (seed with base train split historical exact).
        replay = ReplayBuffer(self.root / "replay",
                              max_examples=cfg.max_protagonist_replay,
                              seed=cfg.seed)
        replay_relative = run_state.get("active_replay_snapshot")
        if replay_relative:
            replay_path = resolve_run_path(self.root, replay_relative)
            observed = sha256_file(replay_path)
            if observed != run_state["active_replay_sha256"]:
                raise RuntimeError(
                    "committed replay snapshot integrity failure: "
                    f"expected {run_state['active_replay_sha256']}, "
                    f"observed {observed}")
            replay.load_snapshot(replay_path)
        elif cfg.initial_protagonist_replay:
            replay.load_snapshot(
                cfg.initial_protagonist_replay,
                rebase_as_historical=True,
            )
        else:
            replay.load()
        imported_replay_overlap = {
            record["static_level_signature"]
            for record in replay.records()
        } & self.frozen_sigs
        if imported_replay_overlap:
            raise ExperimentSpecIntegrityError(
                "initial protagonist replay overlaps frozen validation/test "
                f"levels ({len(imported_replay_overlap)} signatures)")
        if len(replay) == 0 and cfg.seed_historical_replay:
            base_train = filter_records_for_split(base_records, split, "train")
            replay.add([tag_historical(r) for r in base_train], iteration=0)
            replay.persist([0])

        # Designer level replay (frontier / difficult levels across rounds).
        level_replay = LevelReplayBuffer(
            self.root / "level_replay", max_levels=cfg.max_designer_replay,
            seed=cfg.seed)
        level_relative = run_state.get("active_level_replay_snapshot")
        if level_relative:
            level_path = resolve_run_path(self.root, level_relative)
            observed = sha256_file(level_path)
            if observed != run_state["active_level_replay_sha256"]:
                raise RuntimeError(
                    "committed level-replay snapshot integrity failure: "
                    f"expected {run_state['active_level_replay_sha256']}, "
                    f"observed {observed}")
            level_replay.load_snapshot(level_path)
        elif cfg.initial_designer_replay:
            level_replay.load_snapshot(cfg.initial_designer_replay)
        else:
            level_replay.load()
        imported_level_overlap = {
            record["static_level_signature"]
            for record in level_replay.records()
        } & self.frozen_sigs
        if imported_level_overlap:
            raise ExperimentSpecIntegrityError(
                "initial designer replay overlaps frozen validation/test "
                f"levels ({len(imported_level_overlap)} signatures)")
        if cfg.initial_designer_replay and not level_relative:
            run_state["accepted_fingerprints"] = sorted(
                set(run_state.get("accepted_fingerprints", []))
                | level_replay.fingerprints()
            )

        if not replay_relative:
            snapshot = self.root / "replay" / "committed_initial.jsonl"
            replay.write_snapshot(snapshot)
            run_state["active_replay_snapshot"] = relative_to_run(snapshot, self.root)
            run_state["active_replay_sha256"] = sha256_file(snapshot)
        if not level_relative:
            snapshot = self.root / "level_replay" / "committed_initial.jsonl"
            level_replay.write_snapshot(snapshot)
            run_state["active_level_replay_snapshot"] = relative_to_run(
                snapshot, self.root)
            run_state["active_level_replay_sha256"] = sha256_file(snapshot)
        self._save_run_state(run_state)
        active = resolve_committed_protagonist(self.root, run_state)
        if refresh_best_checkpoint(
                active, self.root / "best.pt",
                run_state["active_protagonist_sha256"]):
            print("repairing stale best.pt mirror from committed checkpoint", flush=True)
        self._repair_committed_artifacts(run_state)
        replay.persist(run_state["completed_rounds"] + [0])
        level_replay.persist()
        self._write_frontier_diagnostics_summary(run_state)
        print("resuming from committed protagonist: "
              f"round={run_state['active_protagonist_source_round']} "
              f"checkpoint={run_state['active_protagonist_checkpoint']} "
              f"sha256={run_state['active_protagonist_sha256']}", flush=True)
        if cfg.shadow_learner_enabled:
            print(
                "resuming from committed learner: "
                f"round={run_state['active_learner_source_round']} "
                f"checkpoint={run_state['active_learner_checkpoint']} "
                f"sha256={run_state['active_learner_sha256']} "
                f"anchor_round={run_state['active_learner_anchor_source_round']}",
                flush=True,
            )
        if cfg.initialize_only:
            print(
                "initialization-only handoff complete; no benchmark or "
                "co-training round was run",
                flush=True,
            )
            return {
                "run_state": run_state,
                "rounds": len(run_state["completed_rounds"]),
                "initialization_only": True,
            }

        # Frozen benchmark groups (built once).
        init_designer_model = None
        generator_block_limits = [enc.max_blocks]
        ds = run_state.get("designer_checkpoint")
        if ds:
            init_designer_model, denc, _dmc = designer_from_checkpoint(
                load_designer(ds, map_location=self.device), map_location=self.device)
            generator_block_limits.append(denc.max_blocks)
        pretrained_designer_model = None
        if cfg.pretrained_designer_checkpoint:
            pretrained_designer_model, pretrained_denc, _ = designer_from_checkpoint(
                load_designer(cfg.pretrained_designer_checkpoint,
                              map_location=self.device),
                map_location=self.device)
            generator_block_limits.append(pretrained_denc.max_blocks)
        generator_max_blocks = min(generator_block_limits)
        init_cur = CurriculumState(**run_state["curriculum"])
        all_groups = bench.build_benchmark(
            self.root, self.env, base_records,
            adversarial_designer_model=init_designer_model,
            pretrained_designer_model=pretrained_designer_model,
            enc=enc, gen_cfg=_gen_cfg(
                init_cur, max_blocks=generator_max_blocks),
            ood_gen_cfg=GeneratorConfig(rows=cfg.ood_rows, cols=cfg.ood_cols,
                                        color_count=init_cur.color_count,
                                        density=init_cur.density,
                                        max_blocks=generator_max_blocks),
            mutation_budget=init_cur.mutation_budget, count=cfg.benchmark_count,
            device=self.device, seed=cfg.seed)
        if not cfg.skip_forgetting_benchmark:
            label_oracle = ExactOracle(
                self.env,
                max_nodes=cfg.eval_astar_max_nodes,
                time_limit_seconds=cfg.eval_astar_time_limit_seconds,
            )
            bench.ensure_benchmark_labels(self.root, self.env, all_groups,
                                          label_oracle)

        terminal_stop = run_state.get("terminal_stop")
        if (cfg.stop_after_promotion and isinstance(terminal_stop, dict)
                and terminal_stop.get("reason") == "promotion"):
            print(
                "co-training already stopped after committed promotion at "
                f"round {terminal_stop.get('round')}", flush=True)
            return {
                "run_state": run_state,
                "rounds": len(run_state["completed_rounds"]),
                "stopped_after_promotion": True,
            }

        for rnd in range(1, cfg.rounds + 1):
            if rnd in run_state["completed_rounds"]:
                print(f"round {rnd}: already complete, skipping", flush=True)
                continue
            pending_state = copy.deepcopy(run_state)
            report = self._run_round(
                rnd, base_records, split, enc, model_cfg, value_norm,
                replay, level_replay, pending_state)
            self._crash_point("after_artifacts_prepared")
            round_dir = self.root / f"round_{rnd:03d}"
            replay_snapshot = round_dir / "replay.jsonl"
            level_snapshot = round_dir / "level_replay.jsonl"
            replay.write_snapshot(replay_snapshot)
            level_replay.write_snapshot(level_snapshot)
            load_checkpoint(
                resolve_run_path(
                    self.root,
                    report["protagonist"]["candidate_checkpoint"]),
                map_location="cpu")
            json.loads((round_dir / "report.prepared.json").read_text(
                encoding="utf-8"))
            ReplayBuffer(
                self.root / ".validation_replay",
                max_examples=cfg.max_protagonist_replay,
                seed=cfg.seed).load_snapshot(replay_snapshot)
            LevelReplayBuffer(
                self.root / ".validation_level_replay",
                max_levels=cfg.max_designer_replay,
                seed=cfg.seed).load_snapshot(level_snapshot)
            pending_state["completed_rounds"].append(rnd)
            pending_state["history"].append(report["summary"])
            pending_state["active_replay_snapshot"] = relative_to_run(
                replay_snapshot, self.root)
            pending_state["active_replay_sha256"] = sha256_file(replay_snapshot)
            pending_state["active_level_replay_snapshot"] = relative_to_run(
                level_snapshot, self.root)
            pending_state["active_level_replay_sha256"] = sha256_file(level_snapshot)
            stop_after_this_round = bool(
                cfg.stop_after_promotion
                and report["summary"]["promoted"])
            if stop_after_this_round:
                pending_state["terminal_stop"] = {
                    "reason": "promotion",
                    "round": rnd,
                    "candidate_checkpoint": report["protagonist"][
                        "candidate_checkpoint"],
                    "candidate_checkpoint_sha256": report["protagonist"].get(
                        "candidate_checkpoint_sha256"),
                }
            prepared = round_dir / "report.prepared.json"
            prior = round_dir / "prior_levels.json"
            commit = {
                "round": rnd,
                "prepared_report": relative_to_run(prepared, self.root),
                "prepared_report_sha256": sha256_file(prepared),
                "promoted": report["summary"]["promoted"],
            }
            if prior.exists():
                commit["prior_levels"] = relative_to_run(prior, self.root)
                commit["prior_levels_sha256"] = sha256_file(prior)
            pending_state.setdefault("commits", []).append(commit)
            self._save_run_state(pending_state)  # the durable commit point
            run_state = pending_state
            (round_dir / "generation.partial.json").unlink(missing_ok=True)
            self._crash_point("after_state_commit")
            active = resolve_committed_protagonist(self.root, run_state)
            refresh_best_checkpoint(
                active, self.root / "best.pt",
                run_state["active_protagonist_sha256"])
            self._crash_point("after_best_refresh")
            self._repair_committed_artifacts(run_state)
            replay.persist(run_state["completed_rounds"] + [0])
            level_replay.persist()
            self._write_frontier_diagnostics_summary(run_state)
            if cfg.prune_superseded_round_artifacts:
                self._prune_superseded_round_artifacts(
                    run_state, current_round=rnd)
            if stop_after_this_round:
                print(
                    f"stopping after committed promotion at round {rnd}",
                    flush=True)
                break

        return {
            "run_state": run_state,
            "rounds": len(run_state["completed_rounds"]),
            "stopped_after_promotion": bool(
                run_state.get("terminal_stop", {}).get("reason")
                == "promotion"),
        }

    def _write_frontier_diagnostics_summary(self, run_state) -> None:
        rows = []
        for item in run_state.get("history", []):
            rows.append({
                "round": item.get("round"),
                "generated_count": item.get("generated_count",
                                            item.get("generated")),
                "valid_count": item.get("valid_count", item.get("valid")),
                "unique_valid_count": item.get("unique_valid_count"),
                "oracle_solvable_count": item.get(
                    "oracle_solvable_count", item.get("oracle_solvable")),
                "duplicate_count": item.get("duplicate_count",
                                            item.get("duplicates")),
                "duplicate_rate": item.get("duplicate_rate"),
                "duplicate_rate_among_valid":
                    item.get("duplicate_rate_among_valid"),
                "below_frontier_count": item.get("below_frontier_count"),
                "above_frontier_count": item.get("above_frontier_count"),
                "accepted_count": item.get("accepted_count",
                                           item.get("accepted")),
                "strict_frontier_accepted_count":
                    item.get("strict_frontier_accepted_count"),
                "frontier_backfilled_count":
                    item.get("frontier_backfilled_count"),
                "frontier_acceptance_rate":
                    item.get("frontier_acceptance_rate"),
                "training_acceptance_rate":
                    item.get("training_acceptance_rate"),
                "mean_protagonist_solve_rate":
                    item.get("mean_protagonist_solve_rate",
                             item.get("mean_solve_rate")),
                "curriculum_direction":
                    (item.get("curriculum_adjustment") or {}).get("direction"),
            })
        atomic_write_json(self.root / "frontier_diagnostics.json", rows)
        if not rows:
            atomic_write_text(self.root / "frontier_diagnostics.csv", "")
            return
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(self.root / "frontier_diagnostics.csv",
                          handle.getvalue())

    # ------------------------------------------------------------------
    # one round
    # ------------------------------------------------------------------

    def _run_round(self, rnd, base_records, split, enc, model_cfg, value_norm,
                   replay, level_replay, run_state) -> dict[str, Any]:
        cfg = self.cfg
        round_dir = self.root / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== co-training round {rnd} ===", flush=True)

        curriculum = CurriculumState(**run_state["curriculum"])
        prev_ckpt_path = str(resolve_committed_protagonist(self.root, run_state))
        prev_ckpt = load_checkpoint(prev_ckpt_path, map_location="cpu")
        prev_model = model_from_checkpoint(prev_ckpt, map_location=self.device)
        if cfg.shadow_learner_enabled:
            learner_ckpt_path = str(
                self._resolve_shadow_checkpoint(run_state, "learner"))
            learner_ckpt = load_checkpoint(learner_ckpt_path, map_location="cpu")
            learner_model = model_from_checkpoint(
                learner_ckpt, map_location=self.device)
        else:
            learner_ckpt_path = prev_ckpt_path
            learner_ckpt = prev_ckpt
            learner_model = prev_model

        # ---- 1-4. Generate + evaluate + accept (frontier) ----
        gen = self._generate_and_accept(rnd, curriculum, enc, value_norm,
                                        prev_model, run_state)
        accepted_levels = gen["accepted_levels"]    # list[(level_id, Level)]

        # Designer level replay (historical frontier / difficult levels).
        if cfg.use_designer_replay and gen["accepted_records"]:
            level_replay.add(gen["accepted_records"])

        # ---- Curriculum adaptation (recorded; skipped if disabled) ----
        if cfg.curriculum_enabled:
            new_curriculum, cur_record = _adapt_curriculum_from_generation(
                curriculum, gen, cfg.curriculum)
        else:
            new_curriculum = curriculum
            cur_record = {"direction": "disabled",
                          "reason": "curriculum disabled (ablation)",
                          "mean_solve_rate": (
                              round(gen["mean_solve_rate"], 4)
                              if gen["mean_solve_rate"] is not None else None),
                          "changed": False, "changes": {},
                          "before": curriculum.to_dict(),
                          "after": curriculum.to_dict()}
        run_state["curriculum"] = new_curriculum.to_dict()
        gen["frontier_diagnostics"]["curriculum_after"] = new_curriculum.to_dict()
        gen["frontier_diagnostics"]["curriculum_adjustment"] = cur_record

        # ---- 5-9. Fine-tune + promote the protagonist on accepted levels ----
        prot = self._finetune_protagonist(
            rnd, round_dir, accepted_levels, base_records, split, enc, model_cfg,
            value_norm, prev_model, prev_ckpt, prev_ckpt_path, replay, run_state,
            learner_model=learner_model, learner_ckpt=learner_ckpt,
            learner_ckpt_path=learner_ckpt_path)

        # ---- 10. Freeze promoted protagonist, train designer against it ----
        if cfg.train_designer_each_round:
            designer_summary = self._train_designer(
                rnd, round_dir, new_curriculum, run_state, enc)
        else:
            designer_summary = {"skipped": True,
                                "designer_checkpoint":
                                    run_state["designer_checkpoint"],
                                "designer_checkpoint_sha256":
                                    run_state["designer_checkpoint_sha256"]}

        # Stage prior-round benchmark additions.  The shared benchmark is a
        # derived mirror refreshed only after run-state commit.
        if accepted_levels:
            atomic_write_json(
                round_dir / "prior_levels.json",
                [level_to_dict(lvl) for _identifier, lvl in accepted_levels])

        summary = {
            "round": rnd,
            "generated": gen["generated"],
            "generated_count": gen["generated_count"],
            "valid": gen["valid"],
            "valid_count": gen["valid_count"],
            "unique_valid_count": gen["unique_valid_count"],
            "oracle_solvable": gen["oracle_solvable"],
            "oracle_solvable_count": gen["oracle_solvable_count"],
            "duplicates": gen["duplicates"],
            "duplicate_count": gen["duplicate_count"],
            "duplicate_rate": gen["duplicate_rate"],
            "duplicate_rate_among_valid":
                gen["duplicate_rate_among_valid"],
            "below_frontier_count": gen["below_frontier_count"],
            "above_frontier_count": gen["above_frontier_count"],
            "accepted": len(accepted_levels),
            "accepted_count": len(accepted_levels),
            "strict_frontier_accepted_count":
                gen["strict_frontier_accepted_count"],
            "frontier_backfilled_count": gen["frontier_backfilled_count"],
            "frontier_acceptance_rate": gen["frontier_acceptance_rate"],
            "training_acceptance_rate": gen["training_acceptance_rate"],
            "mean_solve_rate": gen["mean_solve_rate"],
            "mean_protagonist_solve_rate": gen["mean_solve_rate"],
            "solve_rate_quantiles": gen["solve_rate_quantiles"],
            "mean_designer_reward": gen["mean_designer_reward"],
            "oracle_solve_rate": gen["oracle_solve_rate"],
            "label_exact": prot["label_exact"],
            "label_exact_path": prot["label_exact_path"],
            "label_search": prot["label_search"],
            "promoted": prot["promoted"],
            **{
                key: value for key, value in prot.items()
                if key.startswith("promotion_")
            },
            "curriculum_adjustment": cur_record,
            "designer": designer_summary,
            "forgetting": prot["forgetting"],
            "rejections": gen["rejections"],
            "rejection_percentages": gen["rejection_percentages"],
        }
        gen_report = {k: v for k, v in gen.items()
                      if k not in ("accepted_levels", "accepted_records")}
        report = {"commit_status": "prepared",
                  "experiment_fingerprint": self.experiment_fingerprint,
                  "summary": summary,
                  "generation": gen_report,
                  "protagonist": prot,
                  "designer": designer_summary,
                  "curriculum": cur_record}
        atomic_write_json(round_dir / "frontier_diagnostics.json",
                          gen["frontier_diagnostics"])
        atomic_write_json(round_dir / "report.prepared.json", report)
        print(json.dumps(summary, indent=2), flush=True)
        return report

    # ------------------------------------------------------------------
    # generation + frontier acceptance
    # ------------------------------------------------------------------

    def _generate_and_accept(self, rnd, curriculum, enc, value_norm, prev_model,
                             run_state) -> dict[str, Any]:
        cfg = self.cfg
        designer_model, denc, _dmc = designer_from_checkpoint(
            load_designer(run_state["designer_checkpoint"], map_location=self.device),
            map_location=self.device)
        generator_max_blocks = min(enc.max_blocks, denc.max_blocks)
        gen_cfg = _gen_cfg(
            curriculum, max_blocks=generator_max_blocks)
        denv = DesignerEnv(gen_cfg, mutation_budget=curriculum.mutation_budget,
                           encoding=denc)
        action_space = DesignerActionSpace(denc)
        generator_model_state_sha256 = model_state_sha256(designer_model)

        protagonist = Protagonist(
            self.env, prev_model, enc, value_norm, self.device,
            simulations=curriculum.protagonist_simulations,
            dirichlet_alpha=cfg.frontier_dirichlet_alpha,
            dirichlet_weight=cfg.frontier_dirichlet_weight)
        oracle = DesignerOracle(self.env, prev_model, enc, value_norm, self.device,
                                 astar_max_nodes=cfg.astar_max_nodes,
                                 astar_time_limit_seconds=(
                                     cfg.exploratory_astar_time_limit_seconds),
                                 search_simulations=cfg.oracle_simulations,
                                 fallback_on_astar_exhaustion=False)

        seen = set(run_state.get("accepted_fingerprints", []))
        partial_path = (
            self.root / f"round_{rnd:03d}" / "generation.partial.json")
        partial_identity = _generation_partial_identity(
            experiment_fingerprint=self.experiment_fingerprint,
            round_number=rnd, curriculum=curriculum, run_state=run_state,
            generator_model_state_sha256=generator_model_state_sha256,
            cfg=cfg)
        attempts = _load_generation_partial(
            partial_path, identity=partial_identity)

        # Reconstruct duplicate state and validate it against the historical
        # accepted set before continuing.
        round_seen = set(seen)
        for attempt in attempts:
            if attempt["status"] != "valid":
                continue
            fp = attempt["fingerprint"]
            expected_duplicate = fp in round_seen
            if attempt["duplicate"] != expected_duplicate:
                raise RuntimeError(
                    "generation partial checkpoint duplicate history is invalid")
            round_seen.add(fp)
        if attempts:
            print(
                f"round {rnd} generation: resumed {len(attempts)}/"
                f"{cfg.levels_per_round} candidates from atomic checkpoint",
                flush=True)

        started = time.perf_counter()
        initial_attempt_count = len(attempts)
        for i in range(initial_attempt_count, cfg.levels_per_round):
            # Per-level RNG independence makes partial resume bit-for-bit
            # deterministic without serializing Python's RNG internals.
            rollout_rng = random.Random(
                (cfg.seed * 1_000_003 + rnd * 104_729 + i * 65_537)
                & 0xFFFFFFFFFFFFFFFF)
            ep = rollout_episode(denv, designer_model, action_space, denc,
                                 seed=cfg.seed * 7919 + rnd * 1009 + i,
                                 device=self.device, rng=rollout_rng,
                                 verify_finalize=False)
            fin = ep.finalize
            if not fin.valid:
                attempts.append({
                    "index": i,
                    "status": "invalid",
                    "errors": list(fin.errors),
                })
            else:
                fp = level_fingerprint(self.env, fin.level)
                is_dup = fp in round_seen
                round_seen.add(fp)
                est = estimate_solve_rate(
                    protagonist, fin.level, trials=cfg.solve_rate_trials,
                    base_seed=cfg.seed,
                    simulation_budgets=(
                        partial_identity["frontier_simulation_budgets"]))
                attempts.append({
                    "index": i,
                    "status": "valid",
                    "fingerprint": fp,
                    "level": level_to_dict(fin.level),
                    "trajectory": list(ep.trajectory),
                    "solve_rate": est.solve_rate,
                    "solve_rate_trials": est.trials,
                    "solve_rate_solved": est.solved,
                    "solve_rate_budgets": list(est.trial_budgets),
                    "duplicate": is_dup,
                    "num_mutations": fin.num_mutations,
                })

            completed = i + 1
            should_checkpoint = (
                completed % cfg.generation_checkpoint_interval == 0
                or completed == cfg.levels_per_round)
            if should_checkpoint:
                _write_generation_partial(
                    partial_path, identity=partial_identity,
                    attempts=attempts)
            if (completed % cfg.generation_progress_interval == 0
                    or completed == cfg.levels_per_round):
                elapsed = time.perf_counter() - started
                completed_here = completed - initial_attempt_count
                per_level = elapsed / max(1, completed_here)
                remaining = cfg.levels_per_round - completed
                valid_so_far = sum(
                    1 for attempt in attempts
                    if attempt["status"] == "valid")
                print(
                    f"round {rnd} generation: {completed}/"
                    f"{cfg.levels_per_round} candidates "
                    f"(valid={valid_so_far}, elapsed={elapsed:.1f}s, "
                    f"eta={remaining * per_level:.1f}s)"
                    + (" [checkpointed]" if should_checkpoint else ""),
                    flush=True)

        # Cheap stage is complete. Every valid DesignerEnv level is solvable by
        # construction, so exact scoring is deferred until after strict
        # frontier selection and ranked backfill.
        valid_attempts = [
            attempt for attempt in attempts
            if attempt["status"] == "valid"]
        valid = len(valid_attempts)
        duplicates = sum(
            1 for attempt in valid_attempts if attempt["duplicate"])
        unique_valid_attempts = [
            attempt for attempt in valid_attempts
            if not attempt["duplicate"]]
        unique_valid = len(unique_valid_attempts)
        solve_rates = [
            float(attempt["solve_rate"]) for attempt in unique_valid_attempts]
        deferred: list[dict[str, Any]] = []
        raw_below = raw_above = strict_frontier_hits = 0
        for attempt in unique_valid_attempts:
            item = {
                "fingerprint": attempt["fingerprint"],
                "level": level_from_dict(attempt["level"]),
                "trajectory": list(attempt["trajectory"]),
                "solve_rate": float(attempt["solve_rate"]),
                "solve_rate_budgets": list(attempt["solve_rate_budgets"]),
                "num_mutations": int(attempt["num_mutations"]),
            }
            if item["solve_rate"] < cfg.curriculum.frontier_min_solve_rate:
                raw_below += 1
            elif item["solve_rate"] > cfg.curriculum.frontier_max_solve_rate:
                raw_above += 1
            else:
                strict_frontier_hits += 1
                item["selection_mode"] = "strict_frontier"
            deferred.append(item)

        strict_items = [
            item for item in deferred
            if in_frontier(item["solve_rate"], cfg.curriculum)]
        outside_items = [
            item for item in deferred
            if not in_frontier(item["solve_rate"], cfg.curriculum)]
        needed = max(
            0, cfg.frontier_backfill_target - len(strict_items))
        selected_backfill = set(select_frontier_backfill(
            (
                (item["fingerprint"], item["solve_rate"])
                for item in outside_items
            ),
            limit=needed,
            cfg=cfg.curriculum,
        ))
        selected_items = list(strict_items)
        backfilled = backfilled_below = backfilled_above = 0
        for item in outside_items:
            if item["fingerprint"] not in selected_backfill:
                continue
            item["selection_mode"] = "ranked_backfill"
            selected_items.append(item)
            backfilled += 1
            if item["solve_rate"] < cfg.curriculum.frontier_min_solve_rate:
                backfilled_below += 1
            else:
                backfilled_above += 1

        accepted_levels: list[tuple[str, Level]] = []
        accepted_records: list[dict[str, Any]] = []
        designer_rewards: list[float] = []
        oracle_rejections: list[dict[str, Any]] = []
        oracle_unsolved = oracle_exact_contradictions = 0
        accepted_strict = accepted_backfilled = 0
        accepted_backfilled_below = accepted_backfilled_above = 0
        protagonist_checkpoint = str(
            resolve_committed_protagonist(self.root, run_state))
        print(
            f"round {rnd} exact scoring: {len(selected_items)} selected of "
            f"{unique_valid} unique valid construction-proven levels "
            f"({valid} total valid)", flush=True)
        for selected_index, item in enumerate(selected_items, start=1):
            fp = item["fingerprint"]
            fin = FinalizeResult(
                level=item["level"], valid=True, errors=(),
                solvable=True, move_count=None,
                num_blocks=item["level"].total_blocks,
                num_mutations=item["num_mutations"])
            scored = score_level(
                self.env, fin, protagonist=protagonist, oracle=oracle,
                reward_cfg=self.reward_cfg, novelty=1.0, seed=cfg.seed,
                astar_max_nodes=cfg.astar_max_nodes,
                construction_solvable=True)
            designer_rewards.append(scored.reward.total)
            oracle_result = scored.oracle_result()
            if oracle_result.get("oracle_solved") is False:
                oracle_unsolved += 1
                oracle_exact = bool(oracle_result.get("oracle_exact"))
                if oracle_exact:
                    oracle_exact_contradictions += 1
                rejection = {
                    "fingerprint": fp,
                    "selection_mode": item["selection_mode"],
                    "solve_rate": item["solve_rate"],
                    "oracle_exact": oracle_exact,
                    "oracle_method": scored.solver_metrics().get(
                        "oracle_method"),
                    "oracle_cost": oracle_result.get("oracle_cost"),
                }
                oracle_rejections.append(rejection)
                print(
                    f"round {rnd} exact scoring: rejected oracle-unsolved "
                    f"level {fp} (exact={oracle_exact})",
                    flush=True,
                )
                continue
            seen.add(fp)
            accepted_levels.append((fp, item["level"]))
            if item["selection_mode"] == "strict_frontier":
                accepted_strict += 1
            else:
                accepted_backfilled += 1
                if (item["solve_rate"]
                        < cfg.curriculum.frontier_min_solve_rate):
                    accepted_backfilled_below += 1
                else:
                    accepted_backfilled_above += 1
            record = build_level_record(
                self.env, item["level"], trajectory=item["trajectory"],
                designer_checkpoint=run_state["designer_checkpoint"],
                generator_model_state_sha256=generator_model_state_sha256,
                protagonist_checkpoint=protagonist_checkpoint,
                oracle_result=oracle_result,
                reward_components=scored.reward.components,
                structural_metrics=scored.structural.to_dict(),
                solver_metrics=scored.solver_metrics(),
                generation_iteration=rnd, reward_total=scored.reward.total)
            record["frontier_selection"] = {
                "mode": item["selection_mode"],
                "solve_rate": item["solve_rate"],
                "simulation_budgets": item["solve_rate_budgets"],
                "distance_to_frontier": frontier_distance(
                    item["solve_rate"], cfg.curriculum),
            }
            record["solvability_evidence"] = "reverse_construction"
            accepted_records.append(record)
            print(
                f"round {rnd} exact scoring: {selected_index}/"
                f"{len(selected_items)}", flush=True)

        run_state["accepted_fingerprints"] = sorted(seen)
        mean_sr = sum(solve_rates) / len(solve_rates) if solve_rates else None
        generated = cfg.levels_per_round
        duplicate_rate = duplicates / generated if generated else 0.0
        duplicate_rate_among_valid = duplicates / valid if valid else 0.0
        rejections: dict[str, int] = {
            "invalid": generated - valid,
            "duplicate": duplicates,
            "oracle_unsolved": oracle_unsolved,
            "below_frontier": raw_below - backfilled_below,
            "above_frontier": raw_above - backfilled_above,
            "accepted": len(accepted_levels),
        }
        rejection_percentages = {
            key: (value / generated if generated else 0.0)
            for key, value in rejections.items()
        }
        frontier_diagnostics = {
            "schema_version": 4,
            "round": rnd,
            "generated_count": generated,
            "valid_count": valid,
            "unique_valid_count": unique_valid,
            "frontier_candidate_count": unique_valid,
            "solve_rate_sample_count": unique_valid,
            "invalid_count": rejections["invalid"],
            "oracle_solvable_count": valid - oracle_unsolved,
            "construction_proven_solvable_count": valid,
            "exact_scored_selected_count": len(selected_items),
            "designer_reward_evaluated_count": len(designer_rewards),
            "duplicate_count": duplicates,
            "duplicate_rate": duplicate_rate,
            "duplicate_rate_among_valid": duplicate_rate_among_valid,
            "below_frontier_count": raw_below,
            "above_frontier_count": raw_above,
            "oracle_unsolved_count": rejections["oracle_unsolved"],
            "oracle_exact_contradiction_count":
                oracle_exact_contradictions,
            "oracle_rejections": oracle_rejections,
            "accepted_count": len(accepted_levels),
            "strict_frontier_selected_count": strict_frontier_hits,
            "frontier_backfill_selected_count": backfilled,
            "strict_frontier_accepted_count": accepted_strict,
            "frontier_backfilled_count": accepted_backfilled,
            "frontier_backfilled_below_count": accepted_backfilled_below,
            "frontier_backfilled_above_count": accepted_backfilled_above,
            "frontier_backfill_target": cfg.frontier_backfill_target,
            "frontier_estimation_policy":
                "geometric_search_budget_sweep_v1",
            "frontier_simulation_budgets":
                partial_identity["frontier_simulation_budgets"],
            "generator_max_blocks": gen_cfg.max_blocks,
            "protagonist_encoding_max_blocks": enc.max_blocks,
            "designer_encoding_max_blocks": denc.max_blocks,
            "frontier_acceptance_rate": (
                accepted_strict / unique_valid if unique_valid else 0.0),
            "training_acceptance_rate": (
                len(accepted_levels) / unique_valid
                if unique_valid else 0.0),
            "mean_protagonist_solve_rate": mean_sr,
            "solve_rate_quantiles": {
                "p10": _quantile(solve_rates, 0.1),
                "p50": _quantile(solve_rates, 0.5),
                "p90": _quantile(solve_rates, 0.9),
            },
            "rejection_counts": dict(rejections),
            "rejection_percentages": rejection_percentages,
            "bucket_distribution": {"designer": generated},
            "curriculum_before": curriculum.to_dict(),
        }
        return {
            "accepted_levels": accepted_levels,
            "accepted_records": accepted_records,
            "generated": generated,
            "generated_count": generated,
            "valid": valid,
            "valid_count": valid,
            "unique_valid_count": unique_valid,
            "frontier_candidate_count": unique_valid,
            "solve_rate_sample_count": unique_valid,
            "oracle_solvable": valid - oracle_unsolved,
            "oracle_solvable_count": valid - oracle_unsolved,
            "oracle_solve_rate": (
                (valid - oracle_unsolved) / valid if valid else 0.0),
            "construction_proven_solvable_count": valid,
            "exact_scored_selected_count": len(selected_items),
            "designer_reward_evaluated_count": len(designer_rewards),
            "duplicates": duplicates,
            "duplicate_count": duplicates,
            "duplicate_rate": duplicate_rate,
            "duplicate_rate_among_valid": duplicate_rate_among_valid,
            "below_frontier_count": raw_below,
            "above_frontier_count": raw_above,
            "oracle_exact_contradiction_count":
                oracle_exact_contradictions,
            "strict_frontier_selected_count": strict_frontier_hits,
            "frontier_backfill_selected_count": backfilled,
            "strict_frontier_accepted_count": accepted_strict,
            "frontier_backfilled_count": accepted_backfilled,
            "frontier_backfilled_below_count": accepted_backfilled_below,
            "frontier_backfilled_above_count": accepted_backfilled_above,
            "frontier_backfill_target": cfg.frontier_backfill_target,
            "frontier_acceptance_rate": (
                accepted_strict / unique_valid if unique_valid else 0.0),
            "training_acceptance_rate": (
                len(accepted_levels) / unique_valid
                if unique_valid else 0.0),
            "mean_solve_rate": mean_sr,
            "solve_rate_quantiles":
                frontier_diagnostics["solve_rate_quantiles"],
            "mean_designer_reward": (sum(designer_rewards) / len(designer_rewards)
                                     if designer_rewards else 0.0),
            "rejections": rejections,
            "rejection_percentages": rejection_percentages,
            "bucket_distribution": frontier_diagnostics["bucket_distribution"],
            "frontier_diagnostics": frontier_diagnostics,
        }

    # ------------------------------------------------------------------
    # protagonist fine-tune + promotion + benchmark/forgetting
    # ------------------------------------------------------------------

    def _finetune_protagonist(self, rnd, round_dir, accepted_levels, base_records,
                              split, enc, model_cfg, value_norm, prev_model,
                              prev_ckpt, prev_ckpt_path, replay, run_state, *,
                              learner_model, learner_ckpt, learner_ckpt_path
                              ) -> dict[str, Any]:
        cfg = self.cfg
        exact_oracle = ExactOracle(
            self.env, max_nodes=cfg.astar_max_nodes,
            time_limit_seconds=cfg.training_astar_time_limit_seconds)

        # 5. Accepted levels + reachable states -> candidates.
        candidates: list = []
        new_records: list = []
        label_exact = label_exact_path = label_search = 0
        if accepted_levels:
            print(
                f"round {rnd} training-state sampling: "
                f"{len(accepted_levels)} accepted levels", flush=True)
            candidates, _sigs = generate_states(
                self.env, exact_oracle, accepted_levels,
                states_per_level=cfg.states_per_level,
                seed=cfg.seed * 104729 + rnd,
                astar_max_nodes=cfg.astar_max_nodes,
                astar_time_limit_seconds=cfg.training_astar_time_limit_seconds,
                optimal_path_state_limit=cfg.states_per_level,
                near_optimal_walks_per_level=0)
            # No leakage: drop any validation/test signatures.
            candidates = drop_frozen_candidates(candidates, self.frozen_sigs)
            print(
                f"round {rnd} training-state sampling: "
                f"{len(candidates)} bounded candidates after dedup/leakage guard",
                flush=True)

        if candidates:
            print(
                f"round {rnd} exact-first labeling: "
                f"{len(candidates)} candidates", flush=True)
            adapter = BlocksortAdapter(self.env, prev_model, enc, value_norm,
                                       self.device)
            new_records, label_stats = label_states(
                self.env, exact_oracle, candidates, iteration=rnd,
                astar_max_nodes=cfg.astar_max_nodes,
                teacher_checkpoint=prev_ckpt_path, search_adapter=adapter,
                search_simulations=cfg.oracle_simulations, search_c_puct=1.5,
                label_policy_temperature=1.0,
                value_norm_constant=value_norm.constant, seed=cfg.seed + rnd,
                label_mode=cfg.label_mode)
            label_exact = label_stats.exact
            label_exact_path = label_stats.exact_path
            label_search = label_stats.search
            replay.add(new_records, rnd)
            print(
                f"round {rnd} exact-first labeling: complete "
                f"(full_exact={label_exact}, exact_path={label_exact_path}, "
                f"search={label_search}, "
                f"records={len(new_records)})", flush=True)

        # 7. Fine-tune from the committed learner. In legacy mode the learner
        # is the champion, preserving the original reset-after-rejection loop.
        candidate_model = PolicyValueNet(enc, model_cfg).to(self.device)
        candidate_model.load_state_dict(learner_ckpt["model_state"])
        fresh_level_count = len(accepted_levels)
        fresh_record_count = len(new_records)
        enough_fresh_levels = (
            fresh_level_count >= cfg.min_fresh_levels_to_train)
        training_eligible = enough_fresh_levels and (
            fresh_record_count > 0 or cfg.min_fresh_levels_to_train == 0)
        if not enough_fresh_levels:
            training_skip_reason = (
                f"fresh level count {fresh_level_count} is below required "
                f"minimum {cfg.min_fresh_levels_to_train}")
        elif fresh_record_count == 0 and cfg.min_fresh_levels_to_train > 0:
            training_skip_reason = (
                "accepted levels produced no non-frozen labeled records")
        else:
            training_skip_reason = None
        replay_age_available = Counter(
            replay_age_bucket(
                record,
                current_iteration=rnd,
                recent_window=cfg.replay_recent_window,
            )
            for record in replay.records()
        )
        replay_age_composition = {
            "policy": "fresh_recent_historical_quota_v1",
            "recent_window": cfg.replay_recent_window,
            "configured_fractions": {
                REPLAY_AGE_BUCKETS[0]: cfg.replay_current_fraction,
                REPLAY_AGE_BUCKETS[1]: cfg.replay_recent_fraction,
                REPLAY_AGE_BUCKETS[2]: cfg.replay_historical_fraction,
            },
            "available_records": {
                name: replay_age_available[name]
                for name in REPLAY_AGE_BUCKETS
            },
            "target_counts": {name: 0 for name in REPLAY_AGE_BUCKETS},
            "realized_counts": {name: 0 for name in REPLAY_AGE_BUCKETS},
            "unique_counts": {name: 0 for name in REPLAY_AGE_BUCKETS},
            "realized_fractions": {name: 0.0 for name in REPLAY_AGE_BUCKETS},
        }
        gradient_weight_mass_by_age = {
            name: 0.0 for name in REPLAY_AGE_BUCKETS}
        train_info = {
            "examples": 0,
            "trainable_part": cfg.trainable_part,
            "policy_target_profile": cfg.policy_target_profile,
            "fresh_levels": fresh_level_count,
            "fresh_records": fresh_record_count,
            "minimum_fresh_levels": cfg.min_fresh_levels_to_train,
            "replay_age_composition": replay_age_composition,
            "gradient_weight_mass_by_age": gradient_weight_mass_by_age,
            "gradient_weight_fraction_by_age": {
                name: 0.0 for name in REPLAY_AGE_BUCKETS},
            "skipped": not training_eligible,
            "reason": training_skip_reason,
        }
        sampled = []
        sampling_seed = cfg.seed * 13 + rnd
        if training_eligible:
            sampled, replay_age_composition = (
                replay.sample_training_set_with_age_quotas(
                    cfg.train_sample_size,
                    current_iteration=rnd,
                    current_fraction=cfg.replay_current_fraction,
                    recent_fraction=cfg.replay_recent_fraction,
                    historical_fraction=cfg.replay_historical_fraction,
                    recent_window=cfg.replay_recent_window,
                    weight_exact_historical=cfg.weight_exact_historical,
                    weight_exact_new=cfg.weight_exact_new,
                    weight_search=cfg.weight_search,
                    seed=sampling_seed,
                    with_replacement=cfg.replay_sample_with_replacement,
                )
            )
            train_info["replay_age_composition"] = replay_age_composition
        if training_eligible and sampled:
            source_sampled = sampled
            incumbent_checkpoint_sha256 = sha256_file(prev_ckpt_path)
            if cfg.policy_target_profile == "recorded":
                policy_target_summary = recorded_policy_target_summary(
                    source_sampled)
            else:
                incumbent_probabilities = incumbent_legal_probabilities(
                    source_sampled,
                    model=prev_model,
                    encoding_config=enc,
                    value_norm=value_norm,
                    device=self.device,
                    batch_size=cfg.batch_size,
                )
                sampled, policy_target_summary = condition_policy_targets(
                    source_sampled,
                    incumbent_probabilities,
                    profile=cfg.policy_target_profile,
                    incumbent_checkpoint_sha256=(
                        incumbent_checkpoint_sha256),
                )
            print(
                f"round {rnd} protagonist fine-tuning: "
                f"{len(sampled)} replay examples, {cfg.epochs} epoch(s), "
                f"policy targets={cfg.policy_target_profile}",
                flush=True)
            weights = source_weights_for(
                sampled,
                rnd,
                weight_exact_historical=cfg.weight_exact_historical,
                weight_exact_new=cfg.weight_exact_new,
                weight_search=cfg.weight_search,
                exact_path_policy_confidence=(
                    cfg.exact_path_policy_confidence),
            )
            base_value_weights = list(weights)
            effective_value_weights = value_supervision_weights_for(
                sampled,
                base_value_weights,
                search_value_loss_weight=cfg.search_value_loss_weight,
            )
            source_sample_path = round_dir / "training_sample_source.jsonl"
            sample_path = round_dir / "training_sample.jsonl"
            policy_weights_path = round_dir / "training_policy_weights.json"
            value_weights_path = round_dir / "training_value_weights.json"
            effective_value_weights_path = (
                round_dir / "training_effective_value_weights.json")
            policy_target_summary_path = (
                round_dir / "training_policy_target_summary.json")
            sample_manifest_path = round_dir / "training_sample_manifest.json"
            source_sample_text = "".join(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in source_sampled
            )
            sample_text = "".join(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in sampled
            )
            _write_or_verify_text(
                source_sample_path,
                source_sample_text,
                label="protagonist source training sample",
            )
            _write_or_verify_text(
                sample_path,
                sample_text,
                label="protagonist training sample",
            )
            _write_or_verify_json(
                policy_weights_path, weights, label="policy weights")
            _write_or_verify_json(
                value_weights_path,
                base_value_weights,
                label="base value weights",
            )
            _write_or_verify_json(
                effective_value_weights_path,
                effective_value_weights,
                label="effective value weights",
            )
            policy_target_summary = {
                **policy_target_summary,
                "source_sample": source_sample_path.name,
                "source_sample_sha256": sha256_file(source_sample_path),
                "sample": sample_path.name,
                "sample_sha256": sha256_file(sample_path),
            }
            _write_or_verify_json(
                policy_target_summary_path,
                policy_target_summary,
                label="policy target summary",
            )
            sample_manifest = {
                "schema_version": 2,
                "semantics": "cotraining_protagonist_training_sample_v2",
                "round": rnd,
                "record_count": len(sampled),
                "source_sample": {
                    "path": source_sample_path.name,
                    "sha256": sha256_file(source_sample_path),
                },
                "sample": {
                    "path": sample_path.name,
                    "sha256": sha256_file(sample_path),
                },
                "policy_weights": {
                    "path": policy_weights_path.name,
                    "sha256": sha256_file(policy_weights_path),
                },
                "value_weights": {
                    "path": value_weights_path.name,
                    "sha256": sha256_file(value_weights_path),
                    "kind": "base_before_exactness_confidence",
                },
                "effective_value_weights": {
                    "path": effective_value_weights_path.name,
                    "sha256": sha256_file(effective_value_weights_path),
                },
                "policy_targets": {
                    "profile": cfg.policy_target_profile,
                    "policy_target_sha256":
                        policy_target_summary["policy_target_sha256"],
                    "summary": {
                        "path": policy_target_summary_path.name,
                        "sha256": sha256_file(policy_target_summary_path),
                    },
                    "incumbent_checkpoint_sha256":
                        incumbent_checkpoint_sha256,
                },
                "sampling": {
                    "policy": "fresh_recent_historical_quota_v1",
                    "sample_size": cfg.train_sample_size,
                    "seed": sampling_seed,
                    "current_iteration": rnd,
                    "recent_window": cfg.replay_recent_window,
                    "current_fraction": cfg.replay_current_fraction,
                    "recent_fraction": cfg.replay_recent_fraction,
                    "historical_fraction": cfg.replay_historical_fraction,
                    "with_replacement": cfg.replay_sample_with_replacement,
                    "realized": replay_age_composition,
                },
                "weighting": {
                    "weight_exact_historical": cfg.weight_exact_historical,
                    "weight_exact_new": cfg.weight_exact_new,
                    "weight_search": cfg.weight_search,
                    "exact_path_policy_confidence":
                        cfg.exact_path_policy_confidence,
                    "search_value_loss_weight":
                        cfg.search_value_loss_weight,
                },
                "training_seed": cfg.seed + rnd,
            }
            _write_or_verify_json(
                sample_manifest_path,
                sample_manifest,
                label="training sample manifest",
            )
            for record, weight in zip(sampled, weights):
                bucket = replay_age_bucket(
                    record,
                    current_iteration=rnd,
                    recent_window=cfg.replay_recent_window,
                )
                gradient_weight_mass_by_age[bucket] += float(weight)
            total_gradient_weight = sum(gradient_weight_mass_by_age.values())
            gradient_weight_fraction_by_age = {
                name: (
                    gradient_weight_mass_by_age[name] / total_gradient_weight
                    if total_gradient_weight else 0.0)
                for name in REPLAY_AGE_BUCKETS
            }
            trainable_info = configure_trainable_part(
                candidate_model, cfg.trainable_part)
            trained = train_expert(
                candidate_model, sampled, weights, encoding_config=enc,
                value_norm=value_norm, epochs=cfg.epochs, batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
                grad_clip=cfg.grad_clip, device=self.device, seed=cfg.seed + rnd,
                value_weights=base_value_weights,
                search_value_loss_weight=cfg.search_value_loss_weight,
                policy_anchor_model=(
                    learner_model if cfg.policy_anchor_weight > 0 else None),
                policy_anchor_weight=cfg.policy_anchor_weight,
                policy_anchor_before_iteration=rnd)
            train_info = {
                **trained,
                **trainable_info,
                "policy_target_profile": cfg.policy_target_profile,
                "policy_target_sha256":
                    policy_target_summary["policy_target_sha256"],
                "sample_unique_examples":
                    len({dedup_key(record) for record in sampled}),
                "sample_unique_levels": len({
                    record["static_level_signature"]
                    for record in sampled
                }),
                "sample_by_iteration": dict(sorted(Counter(
                    str(record.get("generation_iteration", 0))
                    for record in sampled
                ).items())),
                "sample_by_source": dict(sorted(Counter(
                    record.get("target_source", "exact_oracle")
                    for record in sampled
                ).items())),
                "training_sample": {
                    **sample_manifest,
                    "manifest": sample_manifest_path.name,
                    "manifest_sha256": sha256_file(sample_manifest_path),
                },
                "replay_age_composition": replay_age_composition,
                "gradient_weight_mass_by_age": gradient_weight_mass_by_age,
                "gradient_weight_fraction_by_age":
                    gradient_weight_fraction_by_age,
                "sample_with_replacement":
                    cfg.replay_sample_with_replacement,
                "fresh_levels": fresh_level_count,
                "fresh_records": fresh_record_count,
                "minimum_fresh_levels": cfg.min_fresh_levels_to_train,
                "skipped": False,
                "reason": None,
            }
            print(
                f"round {rnd} protagonist fine-tuning: complete", flush=True)
        elif training_eligible:
            training_eligible = False
            training_skip_reason = "protagonist replay produced no training sample"
            train_info["skipped"] = True
            train_info["reason"] = training_skip_reason
        training_performed = (
            training_eligible and int(train_info.get("examples", 0)) > 0)

        candidate_path = round_dir / "candidate.pt"
        save_checkpoint(candidate_path, model=candidate_model, optimizer=None,
                        scheduler=None, epoch=rnd, best_val_metric=None,
                        encoding_config=enc, model_config=model_cfg,
                        value_norm=value_norm, seed=cfg.seed,
                        dataset_version=learner_ckpt.get("dataset_version", 1),
                        split_identity={"frozen_split": "splits.json", "round": rnd},
                        metrics={"train": train_info},
                        experiment_fingerprint=self.experiment_fingerprint)
        candidate_checkpoint_sha256 = bench.checkpoint_content_hash(candidate_path)
        learner_parent_model_sha256 = model_state_sha256(learner_ckpt["model_state"])
        candidate_model_sha256 = model_state_sha256(candidate_model)
        integrity = model_integrity_report(
            candidate_model,
            parent_model_state_sha256=learner_parent_model_sha256,
            candidate_model_state_sha256=candidate_model_sha256,
            training_performed=training_performed,
        )
        shadow_milestone = (
            cfg.shadow_learner_enabled
            and learner_milestone(rnd, cfg.learner_milestone_interval)
        )
        learner_drift = None
        if (cfg.shadow_learner_enabled and shadow_milestone
                and training_performed and integrity["passed"]):
            anchor_path = self._resolve_shadow_checkpoint(
                run_state, "learner_anchor")
            anchor_model = model_from_checkpoint(
                load_checkpoint(anchor_path, map_location="cpu"),
                map_location=self.device,
            )
            learner_drift = {
                "anchor_to_candidate": policy_drift_report(
                    self.env,
                    self.learner_safety_states,
                    reference_model=anchor_model,
                    candidate_model=candidate_model,
                    encoding_config=enc,
                    device=self.device,
                ),
                "champion_to_candidate": policy_drift_report(
                    self.env,
                    self.learner_safety_states,
                    reference_model=prev_model,
                    candidate_model=candidate_model,
                    encoding_config=enc,
                    device=self.device,
                ),
            }
        learner_continuation = (
            continuation_decision(
                training_performed=training_performed,
                milestone=shadow_milestone,
                integrity=integrity,
                drift=(
                    learner_drift["champion_to_candidate"]
                    if learner_drift is not None else None),
                max_policy_kl=cfg.learner_max_policy_kl,
                min_entropy_ratio=cfg.learner_min_entropy_ratio,
            )
            if cfg.shadow_learner_enabled else {
                "decision": "disabled",
                "accepted": False,
                "milestone": False,
                "reasons": [],
            }
        )
        learner_retention = None
        retention_cache_events: list[dict[str, Any]] = []
        if (cfg.learner_retention_dataset and shadow_milestone
                and training_performed and integrity["passed"]):
            reference_rows, cache_event = self._evaluate_retention_cached(
                prev_model, enc, value_norm,
                checkpoint_content_sha256=bench.checkpoint_content_hash(
                    prev_ckpt_path),
                role="champion",
            )
            retention_cache_events.append(cache_event)
            candidate_rows, cache_event = self._evaluate_retention_cached(
                candidate_model, enc, value_norm,
                checkpoint_content_sha256=candidate_checkpoint_sha256,
                role="candidate",
            )
            retention_cache_events.append(cache_event)
            learner_retention = summarize_retention(
                reference_rows, candidate_rows,
                budgets=cfg.learner_retention_budgets,
                max_regression=cfg.learner_retention_max_regression,
            )
            learner_continuation = apply_retention_guard(
                learner_continuation, learner_retention,
                enforce=cfg.learner_retention_enforce,
            )
            print(
                "learner retention: "
                f"passed={learner_retention['passed']}; "
                f"enforced={cfg.learner_retention_enforce}; "
                f"failures={len(learner_retention['failures'])}",
                flush=True,
            )

        # 8. Evaluate prev + candidate on identical frozen states. Prefer the
        # external harder held-out set (non-saturated) when provided.
        candidate_model.eval()
        eval_split_report = None
        if self.eval_val_states is not None:
            lim = cfg.eval_limit
            val_states = (self.eval_val_states[:lim]
                          if lim is not None else self.eval_val_states)
            test_states = None
            eval_split_report = {
                **self.evaluation_split_identity,
                "used_validation_count": len(val_states),
                "used_validation_level_count": len({
                    static_level_signature(state.level) for state in val_states
                }),
                "used_test_count": 0,
                "used_test_level_count": 0,
                "final_test_status": "sealed",
            }
        else:
            val_states = self._states_for_split(base_records, split, "validation",
                                                cfg.eval_limit)
            test_states = self._states_for_split(base_records, split, "test",
                                                 cfg.eval_limit)
            eval_split_report = {
                "source": "base_dataset_frozen_split",
                "selection_policy": _FALLBACK_EVAL_SELECTION_POLICY,
                "eval_limit": cfg.eval_limit,
                "used_validation_count": len(val_states),
                "used_validation_level_count": len({
                    static_level_signature(state.level) for state in val_states
                }),
                "used_test_count": len(test_states),
                "used_test_level_count": len({
                    static_level_signature(state.level) for state in test_states
                }),
                "final_test_status": "evaluated",
            }
        eval_oracle = ExactOracle(
            self.env,
            max_nodes=cfg.eval_astar_max_nodes,
            time_limit_seconds=cfg.eval_astar_time_limit_seconds,
        )
        budgets = list(cfg.eval_budgets)
        validation_cache_events: list[dict[str, Any]] = []

        def _eval(model, states, checkpoint_sha256, role):
            state_identity = [
                {
                    "static_level_signature":
                        static_level_signature(state.level),
                    "canonical_state_key": self.env.canonical_key(state),
                }
                for state in states
            ]
            metadata = {
                "schema_version": _VALIDATION_CACHE_SCHEMA_VERSION,
                "evaluation_semantics_version":
                    EVALUATION_SEMANTICS_VERSION,
                "checkpoint_content_sha256": checkpoint_sha256,
                "states": state_identity,
                "budgets": budgets,
                "eval_astar_max_nodes": cfg.eval_astar_max_nodes,
                "eval_astar_time_limit_seconds":
                    cfg.eval_astar_time_limit_seconds,
                "c_puct": 1.5,
                "seed": cfg.seed,
            }
            cache_key = _validation_cache_key(metadata)
            cache_path = self.root / "validation_cache" / f"{cache_key}.json"
            if cache_path.exists():
                try:
                    payload = json.loads(
                        cache_path.read_text(encoding="utf-8"))
                    report = payload["report"]
                    valid_cache = (
                        payload.get("schema_version")
                        == _VALIDATION_CACHE_SCHEMA_VERSION
                        and payload.get("metadata") == metadata
                        and payload.get("report_sha256")
                        == hash_canonical_value(report))
                    if valid_cache:
                        print(
                            f"validation {role}: loaded cache {cache_key[:12]}",
                            flush=True)
                        validation_cache_events.append({
                            "role": role, "hit": True, "key": cache_key,
                        })
                        return report
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    pass
                print(
                    f"validation {role}: stale/corrupt cache; recomputing",
                    flush=True)
            print(
                f"validation {role}: evaluating {len(states)} states at "
                f"budgets={budgets}", flush=True)
            report = evaluate_checkpoint(
                self.env, model, enc, value_norm, states,
                budgets=budgets, oracle=eval_oracle,
                device=self.device, c_puct=1.5, seed=cfg.seed)
            atomic_write_json(cache_path, {
                "schema_version": _VALIDATION_CACHE_SCHEMA_VERSION,
                "metadata": metadata,
                "report": report,
                "report_sha256": hash_canonical_value(report),
            })
            validation_cache_events.append({
                "role": role, "hit": False, "key": cache_key,
            })
            return report

        incumbent_checkpoint_sha256 = bench.checkpoint_content_hash(
            prev_ckpt_path)
        promotion_evaluated = (
            not cfg.shadow_learner_enabled
            or (shadow_milestone and learner_continuation["accepted"])
        )
        if promotion_evaluated:
            prev_val = _eval(
                prev_model, val_states, incumbent_checkpoint_sha256,
                ("champion_validation" if cfg.shadow_learner_enabled
                 else "incumbent_validation"))
            cand_val = _eval(
                candidate_model, val_states, candidate_checkpoint_sha256,
                "candidate_validation")
            if test_states is None:
                frozen_test_report = {
                    "status": "sealed",
                    "reason": (
                        "external final-test levels are evaluated only by the "
                        "explicit one-shot final-test command"),
                    "previous": None,
                    "candidate": None,
                }
            else:
                frozen_test_report = {
                    "status": "evaluated",
                    "previous": _eval(
                        prev_model, test_states, incumbent_checkpoint_sha256,
                        "incumbent_test"),
                    "candidate": _eval(
                        candidate_model, test_states, candidate_checkpoint_sha256,
                        "candidate_test"),
                }
        else:
            skip_reason = (
                "not_a_preregistered_learner_milestone"
                if not shadow_milestone
                else "learner_failed_continuation_safety")
            prev_val = cand_val = {
                "status": "skipped", "reason": skip_reason,
            }
            frozen_test_report = {
                "status": "sealed" if test_states is None else "skipped",
                "reason": skip_reason,
                "previous": None,
                "candidate": None,
            }
        self._crash_point("after_candidate_evaluation")

        # 9. Promote on validation only.
        self._crash_point("before_promotion_decision")
        evidence = None
        paired_promotion = None
        prev_score = cand_score = None
        counts = ""
        if promotion_evaluated:
            if promotion_metric_requires_budget_sweep(cfg.promotion_metric):
                evidence = validate_budget_sweep_promotion_evidence(
                    prev_val, cand_val, budgets=cfg.promotion_budgets,
                    weights=cfg.promotion_budget_weights,
                    metric=cfg.promotion_metric)
            else:
                evidence = validate_promotion_evidence(
                    prev_val, cand_val, metric=cfg.promotion_metric,
                    budget=cfg.promotion_budget)
            prev_score = evidence.incumbent_score
            cand_score = evidence.candidate_score
            if cfg.promotion_paired_gate_enabled:
                paired_promotion = summarize_paired_promotion(
                    prev_val["paired_level_solve_outcomes"],
                    cand_val["paired_level_solve_outcomes"],
                    budgets=list(cfg.promotion_budgets),
                    weights=list(cfg.promotion_budget_weights),
                    minimum_delta=cfg.promotion_margin,
                    maximum_per_budget_regression=(
                        cfg.promotion_max_per_budget_regression),
                    bootstrap_confidence=cfg.promotion_bootstrap_confidence,
                    bootstrap_replicates=cfg.promotion_bootstrap_replicates,
                    bootstrap_seed=cfg.promotion_bootstrap_seed,
                )
            if evidence.evidence_kind == "solved":
                counts = (
                    f" incumbent={prev_score:.4f}"
                    f" ({evidence.incumbent_confirmed_count}/"
                    f"{evidence.comparison_count} solved)"
                    f" candidate={cand_score:.4f}"
                    f" ({evidence.candidate_confirmed_count}/"
                    f"{evidence.comparison_count} solved)")
            else:
                counts = (
                    f" incumbent={prev_score:.4f}"
                    f" ({evidence.incumbent_confirmed_count}/"
                    f"{evidence.comparison_count} confirmed, "
                    f"{evidence.incumbent_known_count}/"
                    f"{evidence.comparison_count} classified)"
                    f" candidate={cand_score:.4f}"
                    f" ({evidence.candidate_confirmed_count}/"
                    f"{evidence.comparison_count} confirmed, "
                    f"{evidence.candidate_known_count}/"
                    f"{evidence.comparison_count} classified)")
        scalar_gate_passed = bool(
            cand_score is not None and prev_score is not None
            and cand_score > prev_score + cfg.promotion_margin)
        promotion_gate_passed = (
            bool(paired_promotion["promoted"])
            if paired_promotion is not None else scalar_gate_passed)
        promoted = bool(
            promotion_evaluated and training_performed and val_states
            and promotion_gate_passed)
        self._crash_point("after_promotion_decision")
        if cfg.shadow_learner_enabled:
            apply_learner_transition(
                run_state,
                round_number=rnd,
                candidate_checkpoint=relative_to_run(candidate_path, self.root),
                candidate_checkpoint_sha256=candidate_checkpoint_sha256,
                continuation=learner_continuation,
            )
        if not training_performed:
            print(
                f"skipped protagonist fine-tuning: {training_skip_reason}",
                flush=True)
        elif promoted:
            run_state["active_protagonist_checkpoint"] = relative_to_run(
                candidate_path, self.root)
            run_state["active_protagonist_sha256"] = candidate_checkpoint_sha256
            run_state["active_protagonist_source_round"] = rnd
            if cfg.shadow_learner_enabled:
                candidate_relative = relative_to_run(candidate_path, self.root)
                run_state["active_learner_checkpoint"] = candidate_relative
                run_state["active_learner_sha256"] = candidate_checkpoint_sha256
                run_state["active_learner_source_round"] = rnd
                run_state["active_learner_anchor_checkpoint"] = candidate_relative
                run_state["active_learner_anchor_sha256"] = (
                    candidate_checkpoint_sha256)
                run_state["active_learner_anchor_source_round"] = rnd
            print(
                f"PROMOTED protagonist metric={cfg.promotion_metric} "
                f"budget={cfg.promotion_budget} "
                f"budgets={list(cfg.promotion_budgets)}{counts}", flush=True)
        elif cfg.shadow_learner_enabled and not promotion_evaluated:
            print(
                "shadow learner: "
                f"{learner_continuation['decision']}; promotion not evaluated",
                flush=True,
            )
        else:
            print(
                f"rejected protagonist metric={cfg.promotion_metric} "
                f"budget={cfg.promotion_budget} "
                f"budgets={list(cfg.promotion_budgets)}{counts}", flush=True)
        if cfg.shadow_learner_enabled and shadow_milestone:
            run_state.setdefault("learner_milestones", []).append({
                "round": rnd,
                "learner_parent_checkpoint": relative_to_run(
                    learner_ckpt_path, self.root),
                "learner_parent_checkpoint_sha256": sha256_file(
                    learner_ckpt_path),
                "candidate_checkpoint": relative_to_run(
                    candidate_path, self.root),
                "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
                "integrity": integrity,
                "policy_drift": learner_drift,
                "retention": learner_retention,
                "retention_cache": retention_cache_events,
                "continuation": learner_continuation,
                "promotion_evaluated": promotion_evaluated,
                "promoted": promoted,
                "promotion_score_champion": prev_score,
                "promotion_score_candidate": cand_score,
                "paired_promotion": paired_promotion,
            })

        # Benchmark groups + forgetting (candidate vs frozen round-0 baseline).
        forgetting: dict[str, Any] = {"skipped": True, "reason": "not evaluated"}
        cand_bench: dict[str, Any] = {}
        if not cfg.skip_forgetting_benchmark and (
                promoted or not cfg.forgetting_only_on_promotion):
            all_groups = bench.build_benchmark(
                self.root, self.env, base_records, enc=enc,
                gen_cfg=GeneratorConfig(max_blocks=enc.max_blocks),
                ood_gen_cfg=GeneratorConfig(max_blocks=enc.max_blocks),
                mutation_budget=4, count=cfg.benchmark_count, device=self.device,
                seed=cfg.seed)
            sampled = bench.sample_benchmark_groups(
                all_groups, per_group_limit=cfg.eval_limit,
                total_limit=cfg.benchmark_total_limit, seed=cfg.seed)
            labels_path = self.root / bench.BENCHMARK_LABELS_FILE
            precomputed = (json.loads(labels_path.read_text(encoding="utf-8"))
                           if labels_path.exists() else None)
            progress_dir = self.root / bench.BENCHMARK_EVAL_DIR
            forget_metric_budget = (
                cfg.promotion_budgets[-1]
                if promotion_metric_requires_budget_sweep(cfg.promotion_metric)
                else cfg.promotion_budget)
            forget_budgets = [forget_metric_budget]
            print("forgetting benchmark: evaluating candidate", flush=True)
            cand_bench = bench.evaluate_groups(
                self.env, candidate_model, enc, value_norm, sampled,
                exact_oracle=eval_oracle, budgets=forget_budgets,
                device=self.device, c_puct=1.5, seed=cfg.seed,
                precomputed_labels=precomputed, progress_dir=progress_dir,
                tag="candidate",
                checkpoint_sha256=candidate_checkpoint_sha256)
            # Forgetting is always measured against the original protagonist,
            # not whichever incumbent happens to precede this round.
            baseline_path = Path(cfg.protagonist_checkpoint)
            baseline_ckpt = load_checkpoint(baseline_path, map_location="cpu")
            baseline_model = model_from_checkpoint(
                baseline_ckpt, map_location=self.device)
            print("forgetting benchmark: evaluating baseline", flush=True)
            baseline = bench.evaluate_groups(
                self.env, baseline_model, enc, value_norm, sampled,
                exact_oracle=eval_oracle, budgets=forget_budgets,
                device=self.device, c_puct=1.5, seed=cfg.seed,
                precomputed_labels=precomputed, progress_dir=progress_dir,
                tag="baseline",
                checkpoint_sha256=bench.checkpoint_content_hash(baseline_path))
            forgetting = bench.forgetting_report(
                baseline, cand_bench, metric_budget=forget_metric_budget)
        elif cfg.skip_forgetting_benchmark:
            forgetting = {"skipped": True, "reason": "disabled"}
        elif not promoted:
            forgetting = {"skipped": True, "reason": "not promoted"}

        promotion_fields = (
            evidence.report_fields() if evidence is not None else {
                "promotion_evidence_kind": None,
                "promotion_score_prev": None,
                "promotion_score_candidate": None,
                "promotion_total_count": 0,
                "promotion_comparison_count": 0,
            }
        )
        report = {
            "incumbent_checkpoint": relative_to_run(prev_ckpt_path, self.root),
            "incumbent_checkpoint_sha256": incumbent_checkpoint_sha256,
            "learner_parent_checkpoint": relative_to_run(
                learner_ckpt_path, self.root),
            "learner_parent_checkpoint_sha256": sha256_file(learner_ckpt_path),
            "learner_parent_model_state_sha256":
                learner_parent_model_sha256,
            "candidate_checkpoint": relative_to_run(candidate_path, self.root),
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
            "candidate_model_state_sha256": candidate_model_sha256,
            "resulting_active_checkpoint":
                run_state["active_protagonist_checkpoint"],
            "resulting_active_checkpoint_sha256":
                run_state["active_protagonist_sha256"],
            "resulting_learner_checkpoint": run_state.get(
                "active_learner_checkpoint"),
            "resulting_learner_checkpoint_sha256": run_state.get(
                "active_learner_sha256"),
            "resulting_learner_anchor_checkpoint": run_state.get(
                "active_learner_anchor_checkpoint"),
            "resulting_learner_anchor_checkpoint_sha256": run_state.get(
                "active_learner_anchor_sha256"),
            "label_exact": label_exact,
            "label_exact_path": label_exact_path,
            "label_search": label_search,
            "train": train_info,
            "training_eligible": training_eligible,
            "training_performed": training_performed,
            "training_skip_reason": training_skip_reason,
            "replay_size": len(replay),
            "replay_composition": replay.counts_by_source(),
            "promotion_metric": cfg.promotion_metric,
            "promotion_budget": cfg.promotion_budget,
            "promotion_budgets": list(cfg.promotion_budgets),
            "promotion_budget_weights": list(cfg.promotion_budget_weights),
            **promotion_fields,
            "promotion_margin": cfg.promotion_margin,
            "promotion_paired_gate": paired_promotion,
            "promotion_decision": (
                "skipped" if not training_performed
                else "not_evaluated" if not promotion_evaluated
                else ("promoted" if promoted
                      else ("tie" if cand_score == prev_score else "rejected"))),
            "promotion_evaluated": promotion_evaluated,
            "promoted": promoted,
            "shadow_learner": {
                "enabled": cfg.shadow_learner_enabled,
                "milestone": shadow_milestone,
                "milestone_interval": cfg.learner_milestone_interval,
                "integrity": integrity,
                "policy_drift": learner_drift,
                "retention": learner_retention,
                "retention_cache": retention_cache_events,
                "continuation": learner_continuation,
            },
            "validation": {"previous": prev_val, "candidate": cand_val},
            "validation_cache": validation_cache_events,
            "frozen_test": frozen_test_report,
            "benchmark_candidate": cand_bench,
            "forgetting": forgetting,
        }
        if eval_split_report is not None:
            report["evaluation_split"] = eval_split_report
        return report

    # ------------------------------------------------------------------
    # designer training against the frozen promoted protagonist
    # ------------------------------------------------------------------

    def _train_designer(
        self, rnd, round_dir, curriculum, run_state, enc=None
    ) -> dict[str, Any]:
        cfg = self.cfg
        attempts_root = round_dir / "designer_attempts"
        existing_attempts = sorted(attempts_root.glob("attempt_*"))
        attempt_number = (
            max(int(path.name.split("_")[-1]) for path in existing_attempts)
            + 1 if existing_attempts else 1)
        designer_out = attempts_root / f"attempt_{attempt_number:03d}"
        protagonist_checkpoint = str(
            resolve_committed_protagonist(self.root, run_state))
        if enc is None:
            protagonist_state = load_checkpoint(
                protagonist_checkpoint, map_location="cpu")
            enc, _model_cfg, _value_norm = configs_from_checkpoint(
                protagonist_state)
        # Warm-start with the input designer's own architecture.
        dck = load_designer(run_state["designer_checkpoint"], map_location="cpu")
        dmc = DesignerModelConfig.from_dict(dck["model_config"])
        designer_max_blocks = int(dck["encoding_config"]["max_blocks"])
        tc = TrainConfig(
            protagonist_checkpoint=protagonist_checkpoint,
            output_dir=str(designer_out),
            init_designer=run_state["designer_checkpoint"],
            episodes=cfg.designer_episodes,
            episodes_per_iter=cfg.designer_episodes_per_iter,
            validation_episodes=cfg.designer_validation_episodes,
            mutation_budget=curriculum.mutation_budget,
            protagonist_simulations=curriculum.protagonist_simulations,
            oracle_simulations=cfg.oracle_simulations,
            astar_max_nodes=cfg.astar_max_nodes,
            astar_time_limit_seconds=(
                cfg.exploratory_astar_time_limit_seconds),
            frontier_solve_rate_trials=cfg.solve_rate_trials,
            frontier_min_solve_rate=(
                cfg.curriculum.frontier_min_solve_rate),
            frontier_max_solve_rate=(
                cfg.curriculum.frontier_max_solve_rate),
            frontier_alignment_weight=(
                cfg.designer_frontier_alignment_weight),
            frontier_dirichlet_alpha=cfg.frontier_dirichlet_alpha,
            frontier_dirichlet_weight=cfg.frontier_dirichlet_weight,
            frontier_budget_min_ratio=cfg.frontier_budget_min_ratio,
            frontier_budget_max_ratio=cfg.frontier_budget_max_ratio,
            frontier_min_simulations=(
                cfg.curriculum.min_protagonist_simulations),
            frontier_max_simulations=(
                cfg.curriculum.max_protagonist_simulations),
            seed=cfg.seed + rnd, device=cfg.device,
            max_replay=cfg.max_designer_replay,
            generator=_gen_cfg(
                curriculum,
                max_blocks=min(enc.max_blocks, designer_max_blocks)),
            model=dmc,
            ppo=PPOConfig(epochs=cfg.designer_ppo_epochs,
                          entropy_coef=cfg.designer_entropy_coef))
        summary = train_designer(tc)
        new_designer = Path(summary.get("best_checkpoint",
                                        designer_out / "best.pt"))
        if new_designer.exists():
            run_state["designer_checkpoint"] = str(new_designer)
            run_state["designer_checkpoint_sha256"] = sha256_file(new_designer)
        return {
            "designer_checkpoint": run_state["designer_checkpoint"],
            "designer_checkpoint_sha256":
                run_state["designer_checkpoint_sha256"],
            "iterations": summary.get("iterations"),
            "best_mean_reward": summary.get("best_mean_reward"),
            "best_validation_mean_reward":
                summary.get("best_validation_mean_reward"),
            "best_validation_metrics":
                summary.get("best_validation_metrics", {}),
            "best_selection_metric":
                summary.get("best_selection_metric", {}),
            "protagonist_checkpoint": protagonist_checkpoint,
        }


def run_cotraining(cfg: CoTrainingConfig) -> dict[str, Any]:
    return CoTraining(cfg).run()
