"""The expert-iteration pipeline (generate -> label -> mix -> train -> evaluate
-> promote), with disk checkpointing for resume.

Layout under ``output_dir``::

    config.json
    splits.json            # frozen grouped split (created once)
    run_state.json         # atomic commit record + immutable active checkpoint
    replay/                # committed snapshots + compatibility shards/manifest
    best.pt                # derived compatibility mirror
    iter_001/candidate.pt
    iter_001/report.json
    ...

Run state is the sole commit point.  It references immutable checkpoint, replay,
and prepared-report artifacts by relative path and content hash.  Compatibility
mirrors are refreshed only after commit and repaired idempotently on resume.
"""

from __future__ import annotations

import json
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..oracle import Oracle
from ..dataset.schema import deserialize_state
from ..serialization import level_from_dict
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from ..training.transaction import (
    atomic_copy,
    atomic_write_json,
    refresh_best_checkpoint,
    relative_to_run,
    resolve_committed_protagonist,
    resolve_run_path,
    sha256_file,
)
from ..training.experiment_identity import (
    EVALUATION_SEMANTICS_VERSION, EXPERIMENT_SPEC_FILE,
    PROMOTION_CONTRACT_VERSION, TRANSACTION_SCHEMA_VERSION,
    ExperimentSpecIntegrityError, build_experiment_spec, file_identity,
    hash_canonical_value, load_legacy_migration_spec,
    runtime_device_provenance, semantic_dataclass_config,
    validate_continuation_horizon,
    validate_identified_run_state_presence,
    validate_or_initialize_experiment)
from ..training.config import ModelConfig
from ..training.dataset import load_records
from ..training.splits import (
    SplitRatios,
    filter_records_for_split,
    load_manifest,
    make_split,
    save_manifest,
)
from ..training.model import PolicyValueNet
from .config import ExpertIterationConfig
from .evaluate import evaluate_checkpoint
from .promotion import validate_promotion_evidence
from .generate import base_train_levels, generate_states, load_level_pool, sample_levels
from .labeling import label_states
from .records import tag_historical
from .replay import ReplayBuffer
from ..search.graph_search import BlocksortAdapter
from .train import source_weights_for, train_expert


_EXPERT_INPUT_FIELDS = ("initial_checkpoint", "base_dataset", "extra_levels")
_EXPERT_OPERATIONAL_FIELDS = ("output_dir",)
_EXPERT_HORIZON_FIELDS = ("iterations",)
_EXPERT_DERIVED_FIELDS = ("device",)
_EXPERT_SEMANTIC_FIELDS = (
    "levels_per_iteration", "states_per_level", "seed",
    "astar_max_nodes", "search_simulations", "search_c_puct",
    "label_policy_temperature", "label_mode", "val_ratio", "test_ratio",
    "max_replay_examples", "epochs", "batch_size", "learning_rate",
    "weight_decay", "grad_clip", "train_sample_size", "policy_loss_weight",
    "value_loss_weight", "search_value_loss_weight", "policy_anchor_weight",
    "weight_exact_historical", "weight_exact_new", "weight_search",
    "exact_path_policy_confidence", "eval_budgets", "eval_limit", "promotion_metric",
    "promotion_budget", "promotion_margin")


def _expert_checkpoint_identity(path: str) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    return file_identity(
        path, kind="initial_protagonist_checkpoint",
        format_version=int(checkpoint["checkpoint_version"]),
        extra={
            "encoding_config": checkpoint["encoding_config"],
            "model_config": checkpoint["model_config"],
            "value_norm": checkpoint["value_norm"],
        })


def _expert_iteration_experiment_spec(
    cfg: ExpertIterationConfig,
    base_records: list[dict[str, Any]],
    *,
    initial_identity_path: str | None = None,
    resolved_device: str | torch.device | None = None,
) -> dict[str, Any]:
    keys = sorted({
        record.get("static_level_signature") or record["level_id"]
        for record in base_records})
    ratios = SplitRatios(
        train=1.0 - cfg.val_ratio - cfg.test_ratio,
        validation=cfg.val_ratio, test=cfg.test_ratio)
    expected_split = make_split(keys, ratios=ratios, seed=cfg.seed)
    inputs = {
        "base_dataset": file_identity(
            cfg.base_dataset, kind="base_dataset", count_lines=True),
        "initial_protagonist": _expert_checkpoint_identity(
            initial_identity_path or cfg.initial_checkpoint),
        "extra_levels": (
            file_identity(cfg.extra_levels, kind="extra_level_pool")
            if cfg.extra_levels else None),
    }
    semantic = semantic_dataclass_config(
        cfg, semantic_fields=_EXPERT_SEMANTIC_FIELDS,
        operational_fields=_EXPERT_OPERATIONAL_FIELDS,
        input_fields=_EXPERT_INPUT_FIELDS,
        continuation_horizon_fields=_EXPERT_HORIZON_FIELDS,
        derived_fields=_EXPERT_DERIVED_FIELDS,
        unordered_fields=("eval_budgets",))
    return build_experiment_spec(
        pipeline="expert_iteration", semantic_config=semantic, inputs=inputs,
        software_semantics={
            "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
            "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
            "split_algorithm_version": expected_split["version"],
            "experiment_identity_version": 1,
            "replay_source_weighting_policy":
                "weighted_sampling_source_weighted_policy_loss_v2",
            "search_value_supervision_policy":
                "bounded_estimate_policy_only_by_default_v1",
            "search_record_contract":
                "explicit_approximate_completeness_v1",
            "encoding_block_limit_policy":
                "state_must_not_exceed_checkpoint_v1",
            "loss_aggregation_policy":
                "global_supervision_mass_weighted_v1",
            "runtime": runtime_device_provenance(
                requested_device=cfg.device,
                resolved_device=resolved_device or _resolve_device(cfg.device)),
        },
        derived={
            "split_manifest_sha256": hash_canonical_value(expected_split),
            "split_level_count": len(keys),
        })


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _states_for_split(records: list[dict[str, Any]], manifest, split: str,
                      limit: Optional[int]):
    rows = filter_records_for_split(records, manifest, split)
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        level = level_from_dict(r["level"])
        out.append(deserialize_state(level, r["state"]))
    return out


class ExpertIteration:
    def __init__(self, config: ExpertIterationConfig) -> None:
        self.cfg = config
        self.root = Path(config.output_dir)
        self.device = _resolve_device(config.device)
        self.env = Environment()
        self.log_lines: list[str] = []

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)
        print(msg, flush=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _frozen_split(self, base_records):
        path = self.root / "splits.json"
        if path.exists():
            return load_manifest(path)
        keys = sorted({r.get("static_level_signature") or r["level_id"]
                       for r in base_records})
        ratios = SplitRatios(
            train=1.0 - self.cfg.val_ratio - self.cfg.test_ratio,
            validation=self.cfg.val_ratio, test=self.cfg.test_ratio)
        manifest = make_split(keys, ratios=ratios, seed=self.cfg.seed)
        save_manifest(manifest, path)
        return manifest

    def _run_state(self) -> dict[str, Any]:
        path = self.root / "run_state.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"completed_iterations": [], "best_checkpoint": None, "history": []}

    def _save_run_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.root / "run_state.json", state)

    def _crash_point(self, stage: str) -> None:
        """Test seam for deterministic crash injection."""

    def _prepare_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("active_protagonist_checkpoint"):
            resolve_committed_protagonist(self.root, state)
            return
        completed = set(state.get("completed_iterations", []))
        incomplete = [
            path for path in self.root.glob("iter_*")
            if int(path.name.split("_")[-1]) not in completed
            and ((path / "candidate.pt").exists()
                 or (path / "report.json").exists()
                 or (path / "report.prepared.json").exists())
        ]
        if state.get("best_checkpoint") and incomplete:
            raise RuntimeError(
                "legacy expert-iteration run is ambiguous: uncommitted iteration "
                f"artifacts exist in {incomplete[0]}; restore the pre-iteration "
                "best.pt or remove the incomplete directory after inspection")
        source = Path(state.get("best_checkpoint") or self.cfg.initial_checkpoint)
        if (state.get("best_checkpoint") and not source.is_file()
                and (self.root / "best.pt").is_file()):
            source = self.root / "best.pt"
        if not source.is_file():
            raise FileNotFoundError(f"legacy protagonist checkpoint is missing: {source}")
        # Validate before publishing the migration.
        load_checkpoint(source, map_location="cpu")
        destination = self.root / "protagonist" / (
            "legacy_import.pt" if state.get("best_checkpoint") else "initial.pt")
        if not destination.exists():
            atomic_copy(source, destination)
        state["schema_version"] = 2
        state["active_protagonist_checkpoint"] = relative_to_run(
            destination, self.root)
        state["active_protagonist_sha256"] = sha256_file(destination)
        state["active_protagonist_source_iteration"] = (
            max(completed) if completed else 0)
        state["best_checkpoint"] = str(self.root / "best.pt")
        if completed:
            self.log("upgraded legacy expert-iteration run state with a committed "
                     "checkpoint identity")

    def _repair_committed_reports(self, state: dict[str, Any]) -> None:
        for commit in state.get("commits", []):
            prepared = resolve_run_path(self.root, commit["prepared_report"])
            expected = commit["prepared_report_sha256"]
            if sha256_file(prepared) != expected:
                raise RuntimeError(
                    f"committed prepared report integrity failure: {prepared}")
            report_path = self.root / f"iter_{commit['iteration']:03d}" / "report.json"
            report = json.loads(prepared.read_text(encoding="utf-8"))
            report["commit_status"] = "committed"
            rewrite = True
            if report_path.exists():
                try:
                    rewrite = json.loads(
                        report_path.read_text(encoding="utf-8")) != report
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
            if rewrite:
                atomic_write_json(report_path, report)

    def _validate_existing_run(self, state: dict[str, Any]) -> None:
        """Read-only validation required before any resume-side write."""
        completed = set(state.get("completed_iterations", []))
        if state.get("active_protagonist_checkpoint"):
            active = resolve_committed_protagonist(self.root, state)
            load_checkpoint(active, map_location="cpu")
        elif completed:
            raise ExperimentSpecIntegrityError(
                "identified expert-iteration run has committed iterations but "
                "no active protagonist checkpoint")

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
                max_examples=self.cfg.max_replay_examples,
                seed=self.cfg.seed).load_snapshot(replay_path)
        elif completed:
            raise ExperimentSpecIntegrityError(
                "identified expert-iteration run has committed iterations but "
                "no active replay snapshot")

        commits = state.get("commits", [])
        committed_numbers = {int(item["iteration"]) for item in commits}
        if not completed.issubset(committed_numbers):
            raise ExperimentSpecIntegrityError(
                "committed expert-iteration history is missing report evidence "
                f"for iterations {sorted(completed - committed_numbers)}")
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        config_path = self.root / "config.json"
        spec_path = self.root / EXPERIMENT_SPEC_FILE
        setup_only_recovery = validate_identified_run_state_presence(
            self.root, pipeline_label="expert-iteration",
            allowed_setup_files=(
                "config.json", "splits.json", "protagonist/initial.pt"))
        legacy_cfg = None
        if config_path.exists() and not spec_path.exists():
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            legacy_cfg = persisted
            persisted_metric = persisted.get("promotion_metric")
            if (isinstance(persisted_metric, str)
                    and persisted_metric != self.cfg.promotion_metric):
                self.log(
                    "preserving persisted promotion metric on resume: "
                    f"{persisted_metric}")
                self.cfg = replace(
                    self.cfg, promotion_metric=persisted_metric)

        base_records = load_records(self.cfg.base_dataset)
        run_state_path = self.root / "run_state.json"
        run_state = self._run_state()
        committed_iteration = max(
            [int(value) for value in run_state.get("completed_iterations", [])]
            + [int(run_state.get(
                "active_protagonist_source_iteration", 0))])
        validate_continuation_horizon(
            name="iterations", requested=self.cfg.iterations,
            committed=committed_iteration, run_dir=self.root)
        requested_spec = _expert_iteration_experiment_spec(
            self.cfg, base_records, resolved_device=self.device)
        legacy_spec = None
        if legacy_cfg is not None:
            legacy_spec = load_legacy_migration_spec(
                self.root, pipeline="expert_iteration",
                unavailable_fields=(
                    "base_dataset.sha256",
                    "initial_protagonist.sha256",
                    "extra_levels.sha256"))
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
        if migrated and legacy_spec is not None:
            self.log("migrated legacy expert-iteration experiment identity")

        extra_level_pool = (
            load_level_pool(self.cfg.extra_levels)
            if self.cfg.extra_levels else [])
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
        atomic_write_json(config_path, self.cfg.to_dict())
        split = validated_split or self._frozen_split(base_records)
        train_sigs = set(split["train_levels"])

        self._prepare_checkpoint_state(run_state)
        replay = ReplayBuffer(
            self.root / "replay", max_examples=self.cfg.max_replay_examples,
            seed=self.cfg.seed)
        replay_relative = run_state.get("active_replay_snapshot")
        if replay_relative:
            replay_path = resolve_run_path(self.root, replay_relative)
            observed = sha256_file(replay_path)
            expected = run_state["active_replay_sha256"]
            if observed != expected:
                raise RuntimeError(
                    "committed replay snapshot integrity failure: "
                    f"expected {expected}, observed {observed}")
            replay.load_snapshot(replay_path)
        else:
            replay.load()

        # Seed replay with historical exact base data (train split only).
        if len(replay) == 0:
            base_train = filter_records_for_split(base_records, split, "train")
            replay.add([tag_historical(r) for r in base_train], iteration=0)
            replay.persist([0])
            self.log(f"seeded replay with {len(replay)} historical exact examples")

        if not replay_relative:
            snapshot = self.root / "replay" / "committed_initial.jsonl"
            replay.write_snapshot(snapshot)
            run_state["active_replay_snapshot"] = relative_to_run(snapshot, self.root)
            run_state["active_replay_sha256"] = sha256_file(snapshot)
            self._save_run_state(run_state)
        active = resolve_committed_protagonist(self.root, run_state)
        if refresh_best_checkpoint(
                active, self.root / "best.pt",
                run_state["active_protagonist_sha256"]):
            self.log("repairing stale best.pt mirror from committed checkpoint")
        self._repair_committed_reports(run_state)
        replay.persist(run_state["completed_iterations"] + [0])
        self.log(
            "resuming from committed protagonist: "
            f"iteration={run_state['active_protagonist_source_iteration']} "
            f"checkpoint={run_state['active_protagonist_checkpoint']} "
            f"sha256={run_state['active_protagonist_sha256']}")

        level_pool = base_train_levels(base_records, train_sigs)
        level_pool = level_pool + extra_level_pool

        for iteration in range(1, self.cfg.iterations + 1):
            if iteration in run_state["completed_iterations"]:
                self.log(f"iteration {iteration}: already complete, skipping")
                continue
            pending_state = copy.deepcopy(run_state)
            report = self._run_iteration(
                iteration, base_records, split, level_pool, replay, run_state)
            self._crash_point("after_artifacts_prepared")
            snapshot = self.root / f"iter_{iteration:03d}" / "replay.jsonl"
            replay.write_snapshot(snapshot)
            load_checkpoint(
                resolve_run_path(self.root, report["candidate_checkpoint"]),
                map_location="cpu")
            json.loads((self.root / f"iter_{iteration:03d}" /
                        "report.prepared.json").read_text(encoding="utf-8"))
            ReplayBuffer(
                self.root / ".validation_replay",
                max_examples=self.cfg.max_replay_examples,
                seed=self.cfg.seed).load_snapshot(snapshot)
            pending_state["completed_iterations"].append(iteration)
            pending_state["history"].append({
                "iteration": iteration,
                "promoted": report["promoted"],
                "promotion_score_prev": report["promotion_score_prev"],
                "promotion_score_candidate": report["promotion_score_candidate"],
            })
            if report["promoted"]:
                pending_state["active_protagonist_checkpoint"] = \
                    report["candidate_checkpoint"]
                pending_state["active_protagonist_sha256"] = \
                    report["candidate_checkpoint_sha256"]
                pending_state["active_protagonist_source_iteration"] = iteration
            pending_state["active_replay_snapshot"] = relative_to_run(
                snapshot, self.root)
            pending_state["active_replay_sha256"] = sha256_file(snapshot)
            prepared = self.root / f"iter_{iteration:03d}" / "report.prepared.json"
            pending_state.setdefault("commits", []).append({
                "iteration": iteration,
                "prepared_report": relative_to_run(prepared, self.root),
                "prepared_report_sha256": sha256_file(prepared),
                "promoted": report["promoted"],
            })
            self._save_run_state(pending_state)  # the durable commit point
            run_state = pending_state
            self._crash_point("after_state_commit")
            active = resolve_committed_protagonist(self.root, run_state)
            refresh_best_checkpoint(
                active, self.root / "best.pt",
                run_state["active_protagonist_sha256"])
            self._crash_point("after_best_refresh")
            self._repair_committed_reports(run_state)
            replay.persist(run_state["completed_iterations"] + [0])

        return {"run_state": run_state, "log": self.log_lines}

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    def _run_iteration(self, iteration, base_records, split, level_pool, replay,
                        run_state) -> dict[str, Any]:
        cfg = self.cfg
        iter_dir = self.root / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"=== iteration {iteration} ===")

        prev_ckpt_path = str(resolve_committed_protagonist(self.root, run_state))
        incumbent_sha256 = run_state["active_protagonist_sha256"]
        prev_ckpt = load_checkpoint(prev_ckpt_path, map_location="cpu")
        enc, model_cfg, value_norm = configs_from_checkpoint(prev_ckpt)
        prev_model = model_from_checkpoint(prev_ckpt, map_location=self.device)

        # 1-3. Sample levels and generate deduplicated candidate states.
        levels = sample_levels(level_pool, cfg.levels_per_iteration,
                               cfg.seed * 7919 + iteration)
        oracle = Oracle(self.env, max_nodes=cfg.astar_max_nodes)
        candidates, gen_sigs = generate_states(
            self.env, oracle, levels, states_per_level=cfg.states_per_level,
            seed=cfg.seed * 104729 + iteration, astar_max_nodes=cfg.astar_max_nodes)

        # No leakage: never train on validation/test level signatures.
        frozen = set(split["validation_levels"]) | set(split["test_levels"])
        candidates = [(s, p) for (s, p) in candidates
                      if self.env_signature(s) not in frozen]
        self.log(f"generated {len(candidates)} unique non-frozen candidate states")

        # 4-6. Label: exact A* first, search only on exhaustion.
        adapter = BlocksortAdapter(self.env, prev_model, enc, value_norm, self.device)
        new_records, label_stats = label_states(
            self.env, oracle, candidates, iteration=iteration,
            astar_max_nodes=cfg.astar_max_nodes, teacher_checkpoint=prev_ckpt_path,
            search_adapter=adapter, search_simulations=cfg.search_simulations,
            search_c_puct=cfg.search_c_puct,
            label_policy_temperature=cfg.label_policy_temperature,
            value_norm_constant=value_norm.constant, seed=cfg.seed + iteration,
            label_mode=cfg.label_mode)
        self.log(f"labeled: {label_stats.as_dict()}")

        # 7. Mix with historical replay (dedup, exact priority).
        add_stats = replay.add(new_records, iteration)
        self.log(f"replay add: {add_stats}; replay size={len(replay)} "
                 f"composition={replay.counts_by_source()}")

        # 8. Train a new model from the previous checkpoint on a weighted sample.
        sampled = replay.sample_training_set(
            cfg.train_sample_size, current_iteration=iteration,
            weight_exact_historical=cfg.weight_exact_historical,
            weight_exact_new=cfg.weight_exact_new,
            weight_search=cfg.weight_search,
            seed=cfg.seed * 13 + iteration)
        weights = source_weights_for(
            sampled,
            iteration,
            weight_exact_historical=cfg.weight_exact_historical,
            weight_exact_new=cfg.weight_exact_new,
            weight_search=cfg.weight_search,
            exact_path_policy_confidence=cfg.exact_path_policy_confidence,
        )

        candidate_model = PolicyValueNet(enc, model_cfg).to(self.device)
        candidate_model.load_state_dict(prev_ckpt["model_state"])
        train_info = train_expert(
            candidate_model, sampled, weights, encoding_config=enc,
            value_norm=value_norm, epochs=cfg.epochs, batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip, device=self.device,
            seed=cfg.seed + iteration, policy_loss_weight=cfg.policy_loss_weight,
            value_loss_weight=cfg.value_loss_weight,
            search_value_loss_weight=cfg.search_value_loss_weight,
            policy_anchor_model=(
                prev_model if cfg.policy_anchor_weight > 0 else None),
            policy_anchor_weight=cfg.policy_anchor_weight,
            policy_anchor_before_iteration=iteration)
        self.log(f"trained on {train_info['examples']} examples; "
                 f"final loss={train_info['history'][-1]['loss']:.4f}")

        candidate_path = iter_dir / "candidate.pt"
        save_checkpoint(
            candidate_path, model=candidate_model, optimizer=None, scheduler=None,
            epoch=iteration, best_val_metric=None, encoding_config=enc,
            model_config=model_cfg, value_norm=value_norm, seed=cfg.seed,
            dataset_version=prev_ckpt.get("dataset_version", 1),
            split_identity={"frozen_split": "splits.json", "iteration": iteration},
            metrics={"train": train_info},
            experiment_fingerprint=self.experiment_fingerprint)

        # 9. Evaluate previous + candidate on identical frozen states.
        candidate_model.eval()
        val_states = _states_for_split(base_records, split, "validation",
                                       cfg.eval_limit)
        test_states = _states_for_split(base_records, split, "test", cfg.eval_limit)
        eval_oracle = Oracle(self.env, max_nodes=cfg.astar_max_nodes)
        budgets = list(cfg.eval_budgets)

        def _eval(model, states):
            return evaluate_checkpoint(
                self.env, model, enc, value_norm, states, budgets=budgets,
                oracle=eval_oracle, device=self.device, c_puct=cfg.search_c_puct,
                seed=cfg.seed)

        prev_val = _eval(prev_model, val_states)
        cand_val = _eval(candidate_model, val_states)
        prev_test = _eval(prev_model, test_states)
        cand_test = _eval(candidate_model, test_states)

        # Old vs new levels: a few new-level initial states this iteration.
        new_states = self._new_level_states(levels, limit=cfg.eval_limit)
        prev_new = _eval(prev_model, new_states) if new_states else None
        cand_new = _eval(candidate_model, new_states) if new_states else None
        self._crash_point("after_candidate_evaluation")

        # 10. Promote on validation only.
        self._crash_point("before_promotion_decision")
        evidence = validate_promotion_evidence(
            prev_val, cand_val, metric=cfg.promotion_metric,
            budget=cfg.promotion_budget)
        prev_score = evidence.incumbent_score
        cand_score = evidence.candidate_score
        promoted = cand_score > prev_score + cfg.promotion_margin
        self._crash_point("after_promotion_decision")
        counts = (
            f" incumbent={prev_score:.4f}"
            f" ({evidence.incumbent_confirmed_count}/{evidence.total_count}"
            f" confirmed, {evidence.incumbent_known_count}/"
            f"{evidence.total_count} classified)"
            f" candidate={cand_score:.4f}"
            f" ({evidence.candidate_confirmed_count}/{evidence.total_count}"
            f" confirmed, {evidence.candidate_known_count}/"
            f"{evidence.total_count} classified)")
        if promoted:
            self.log(
                f"PROMOTED candidate metric={cfg.promotion_metric} "
                f"budget={cfg.promotion_budget}{counts}")
        else:
            self.log(
                f"rejected candidate metric={cfg.promotion_metric} "
                f"budget={cfg.promotion_budget}{counts}")

        report = {
            "iteration": iteration,
            "experiment_fingerprint": self.experiment_fingerprint,
            "commit_status": "prepared",
            "teacher_checkpoint": prev_ckpt_path,
            "incumbent_checkpoint": run_state["active_protagonist_checkpoint"],
            "incumbent_checkpoint_sha256": incumbent_sha256,
            "candidate_checkpoint": relative_to_run(candidate_path, self.root),
            "candidate_checkpoint_sha256": sha256_file(candidate_path),
            "states_generated": len(candidates),
            "label_stats": label_stats.as_dict(),
            "replay_add": add_stats,
            "replay_composition": replay.counts_by_source(),
            "replay_size": len(replay),
            "train": train_info,
            "promotion_metric": cfg.promotion_metric,
            "promotion_budget": cfg.promotion_budget,
            **evidence.report_fields(),
            "promoted": promoted,
            "resulting_active_checkpoint": (
                relative_to_run(candidate_path, self.root) if promoted
                else run_state["active_protagonist_checkpoint"]),
            "resulting_active_checkpoint_sha256": (
                sha256_file(candidate_path) if promoted else incumbent_sha256),
            "validation": {"previous": prev_val, "candidate": cand_val},
            "frozen_test": {"previous": prev_test, "candidate": cand_test},
            "new_levels": {"previous": prev_new, "candidate": cand_new},
        }
        atomic_write_json(iter_dir / "report.prepared.json", report)
        return report

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def env_signature(self, state) -> str:
        from ..signature import static_level_signature
        return static_level_signature(state.level)

    def _new_level_states(self, levels, limit):
        states = []
        for _lid, level in levels:
            states.append(self.env.initial_state(level))
            if limit is not None and len(states) >= max(1, limit // 4):
                break
        return states


def run_expert_iteration(config: ExpertIterationConfig) -> dict[str, Any]:
    return ExpertIteration(config).run()
