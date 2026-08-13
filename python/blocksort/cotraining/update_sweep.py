"""Train small, fixed-replay protagonist update ablations.

The sweep deliberately performs no generation, labeling, promotion, or
designer training. Every candidate starts from one incumbent checkpoint and
uses the same persisted replay sample so learning-rate/update-size comparisons
are controlled and restart-safe. Prefer ``--sample-jsonl`` with its aligned
policy/value weight files from a co-training round. When those artifacts are
unavailable, reconstruction uses the same age quotas and source-confidence
weighting as the live co-training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from ..expert_iteration.records import dedup_key
from ..expert_iteration.replay import ReplayBuffer
from ..expert_iteration.train import (
    TRAINABLE_PARTS,
    configure_trainable_part,
    source_weights_for,
    train_expert,
)
from ..training.dataset import load_records
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from ..training.model import PolicyValueNet
from ..training.transaction import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from .policy_targets import (
    POLICY_TARGET_PROFILES,
    condition_policy_targets as _condition_policy_targets,
    incumbent_legal_probabilities as _incumbent_legal_probabilities,
    recorded_policy_target_summary as _recorded_policy_target_summary,
)


SCHEMA_VERSION = 12
SEMANTICS = "fixed_replay_protagonist_update_sweep_v12"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
VALUE_WEIGHT_PROFILES = frozenset(("standard", "hard_tail"))
VALUE_SAMPLE_PROFILES = frozenset(("shared", "depth_stratified"))
VALUE_DEPTH_FRACTIONS = (0.35, 0.30, 0.25, 0.10)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    learning_rate: float
    epochs: int = 1
    update_fraction: float = 1.0
    trainable_part: str = "all"
    value_weight_profile: str = "standard"
    value_anchor_weight: float = 0.0
    value_sample_profile: str = "shared"
    policy_anchor_weight: float = 0.0
    policy_target_profile: str = "recorded"
    policy_ranking_weight: float = 0.0

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(
                "candidate names must contain only letters, numbers, "
                "underscores, and hyphens")
        if (not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0):
            raise ValueError("candidate learning rates must be positive")
        if (isinstance(self.epochs, bool)
                or not isinstance(self.epochs, int)
                or self.epochs <= 0):
            raise ValueError("candidate epochs must be positive integers")
        if (not math.isfinite(self.update_fraction)
                or not 0 < self.update_fraction <= 1):
            raise ValueError("candidate update fractions must be in (0, 1]")
        if self.trainable_part not in TRAINABLE_PARTS:
            choices = ", ".join(sorted(TRAINABLE_PARTS))
            raise ValueError(f"trainable part must be one of: {choices}")
        if self.value_weight_profile not in VALUE_WEIGHT_PROFILES:
            choices = ", ".join(sorted(VALUE_WEIGHT_PROFILES))
            raise ValueError(
                f"value weight profile must be one of: {choices}")
        if (not math.isfinite(self.value_anchor_weight)
                or self.value_anchor_weight < 0):
            raise ValueError(
                "value anchor weight must be finite and non-negative")
        if (self.value_weight_profile != "standard"
                and self.trainable_part != "value_head"):
            raise ValueError(
                "value weighting is restricted to value-head-only candidates")
        if (self.value_anchor_weight > 0
                and self.trainable_part not in
                ("all", "policy_trunk", "value_head")):
            raise ValueError(
                "value anchoring requires a candidate whose trainable "
                "parameters can affect value predictions")
        if self.value_sample_profile not in VALUE_SAMPLE_PROFILES:
            choices = ", ".join(sorted(VALUE_SAMPLE_PROFILES))
            raise ValueError(
                f"value sample profile must be one of: {choices}")
        if (self.value_sample_profile != "shared"
                and self.trainable_part != "value_head"):
            raise ValueError(
                "value-specific sampling is restricted to "
                "value-head-only candidates")
        if (not math.isfinite(self.policy_anchor_weight)
                or self.policy_anchor_weight < 0):
            raise ValueError(
                "policy anchor weight must be finite and non-negative")
        if (self.policy_anchor_weight > 0
                and self.trainable_part not in
                ("all", "policy_adapter", "policy_head", "policy_trunk")):
            raise ValueError(
                "policy anchoring requires an all-parameter or "
                "policy-head-only candidate")
        if self.policy_target_profile not in POLICY_TARGET_PROFILES:
            choices = ", ".join(sorted(POLICY_TARGET_PROFILES))
            raise ValueError(
                f"policy target profile must be one of: {choices}")
        if (self.policy_target_profile != "recorded"
                and self.trainable_part not in
                ("all", "policy_adapter", "policy_head", "policy_trunk")):
            raise ValueError(
                "incumbent-guided policy targets require an all-parameter "
                "or policy-head-only candidate")
        if (self.policy_target_profile != "recorded"
                and self.value_sample_profile != "shared"):
            raise ValueError(
                "incumbent-guided policy targets require the shared sample")
        if (not math.isfinite(self.policy_ranking_weight)
                or self.policy_ranking_weight < 0):
            raise ValueError(
                "policy ranking weight must be finite and non-negative")
        if (self.policy_ranking_weight > 0
                and self.trainable_part not in
                ("all", "policy_adapter", "policy_head", "policy_trunk")):
            raise ValueError(
                "policy ranking requires an all-parameter or "
                "policy-head-only candidate")


DEFAULT_CANDIDATES = (
    CandidateSpec("gentle", 1e-4, 1, 1.0),
    CandidateSpec("very_gentle", 3e-5, 1, 1.0),
    CandidateSpec("anchored_half", 1e-4, 1, 0.5),
)


@dataclass(frozen=True)
class UpdateSweepConfig:
    incumbent_checkpoint: str
    replay_snapshot: str
    output_dir: str
    candidates: tuple[CandidateSpec, ...] = DEFAULT_CANDIDATES
    sample_size: int = 2_000
    current_iteration: int = 2
    sample_jsonl: str | None = None
    policy_weights_json: str | None = None
    value_weights_json: str | None = None
    replay_current_fraction: float = 0.35
    replay_recent_fraction: float = 0.25
    replay_historical_fraction: float = 0.40
    replay_recent_window: int = 2
    replay_sample_with_replacement: bool = False
    weight_exact_historical: float = 1.0
    weight_exact_new: float = 1.5
    weight_search: float = 0.5
    exact_path_policy_confidence: float = 0.5
    search_value_loss_weight: float = 0.0
    batch_size: int = 128
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    sample_seed: int = 26_509
    train_seed: int = 2_041
    device: str = "cpu"
    policy_ranking_margin: float = 0.25

    def validate(self) -> None:
        for label, path in (
                ("incumbent checkpoint", self.incumbent_checkpoint),
                ("replay snapshot", self.replay_snapshot)):
            if not Path(path).is_file():
                raise ValueError(f"{label} does not exist: {path}")
        supplied_sample_artifacts = (
            self.sample_jsonl,
            self.policy_weights_json,
            self.value_weights_json,
        )
        if any(path is not None for path in supplied_sample_artifacts):
            if not all(path is not None for path in supplied_sample_artifacts):
                raise ValueError(
                    "--sample-jsonl, --policy-weights-json, and "
                    "--value-weights-json must be supplied together")
            for label, path in zip(
                    ("sample JSONL", "policy weights", "value weights"),
                    supplied_sample_artifacts):
                if not Path(str(path)).is_file():
                    raise ValueError(f"{label} does not exist: {path}")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        names: set[str] = set()
        for candidate in self.candidates:
            candidate.validate()
            if candidate.name in names:
                raise ValueError(f"duplicate candidate name: {candidate.name}")
            names.add(candidate.name)
        for label, value in (
                ("sample_size", self.sample_size),
                ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (isinstance(self.current_iteration, bool)
                or not isinstance(self.current_iteration, int)
                or self.current_iteration < 0):
            raise ValueError("current_iteration must be a non-negative integer")
        if (isinstance(self.replay_recent_window, bool)
                or not isinstance(self.replay_recent_window, int)
                or self.replay_recent_window < 0):
            raise ValueError("replay_recent_window must be non-negative")
        replay_fractions = (
            self.replay_current_fraction,
            self.replay_recent_fraction,
            self.replay_historical_fraction,
        )
        if any(not math.isfinite(value) or value < 0
               for value in replay_fractions):
            raise ValueError("replay age fractions must be finite and non-negative")
        if not math.isclose(sum(replay_fractions), 1.0, abs_tol=1e-9):
            raise ValueError("replay age fractions must sum to 1")
        for label, value in (
                ("weight_exact_historical", self.weight_exact_historical),
                ("weight_exact_new", self.weight_exact_new),
                ("weight_search", self.weight_search),
                ("exact_path_policy_confidence",
                 self.exact_path_policy_confidence),
                ("search_value_loss_weight", self.search_value_loss_weight),
                ("policy_ranking_margin", self.policy_ranking_margin),
                ("weight_decay", self.weight_decay),
                ("grad_clip", self.grad_clip)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.exact_path_policy_confidence > 1:
            raise ValueError(
                "exact_path_policy_confidence must be at most 1")
        for label, value in (
                ("sample_seed", self.sample_seed),
                ("train_seed", self.train_seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")


def _parse_candidate(value: str) -> CandidateSpec:
    name, separator, raw = value.partition("=")
    fields = raw.split(",") if separator else []
    if not name or len(fields) not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        raise argparse.ArgumentTypeError(
            "candidates must use "
            "NAME=LR[,EPOCHS[,UPDATE_FRACTION[,TRAINABLE_PART"
            "[,VALUE_PROFILE[,VALUE_ANCHOR[,VALUE_SAMPLE"
            "[,POLICY_ANCHOR[,POLICY_TARGET[,POLICY_RANKING]]]]]]]]]")
    try:
        spec = CandidateSpec(
            name=name,
            learning_rate=float(fields[0]),
            epochs=int(fields[1]) if len(fields) >= 2 else 1,
            update_fraction=float(fields[2]) if len(fields) >= 3 else 1.0,
            trainable_part=fields[3] if len(fields) >= 4 else "all",
            value_weight_profile=(
                fields[4] if len(fields) >= 5 else "standard"),
            value_anchor_weight=(
                float(fields[5]) if len(fields) >= 6 else 0.0),
            value_sample_profile=(
                fields[6] if len(fields) >= 7 else "shared"),
            policy_anchor_weight=(
                float(fields[7]) if len(fields) >= 8 else 0.0),
            policy_target_profile=(
                fields[8] if len(fields) >= 9 else "recorded"),
            policy_ranking_weight=(
                float(fields[9]) if len(fields) >= 10 else 0.0),
        )
        spec.validate()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return spec


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_text(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def _sample_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(records),
        "unique_examples": len({dedup_key(record) for record in records}),
        "unique_levels": len({
            record["static_level_signature"] for record in records
        }),
        "by_iteration": dict(sorted(Counter(
            str(record.get("generation_iteration", 0))
            for record in records
        ).items())),
        "by_source": dict(sorted(Counter(
            record.get("target_source", "exact_oracle")
            for record in records
        ).items())),
    }


def _persist_policy_target_profile(
    root: Path,
    *,
    profile: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    profile_dir = root / "policy_targets"
    profile_dir.mkdir(parents=True, exist_ok=True)
    text = _sample_text(records)
    sample_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sample_path = profile_dir / f"{profile}.jsonl"
    if sample_path.exists():
        if sample_path.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                "persisted policy-target profile differs from deterministic "
                f"reconstruction: {profile}")
    else:
        atomic_write_text(sample_path, text)
    persisted_summary = {
        **summary,
        "sample": str(sample_path),
        "sample_sha256": sample_sha256,
    }
    atomic_write_json(
        profile_dir / f"{profile}_summary.json", persisted_summary)
    return sample_sha256, persisted_summary


def _depth_bucket(record: dict[str, Any]) -> str:
    raw_moves = float(record["value_target"]["raw_optimal_moves"])
    if raw_moves <= 3:
        return "1_to_3"
    if raw_moves <= 6:
        return "4_to_6"
    if raw_moves < 10:
        return "7_to_9"
    return "10_plus"


def _value_sample_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_sample_summary(records),
        "by_depth": dict(sorted(Counter(
            _depth_bucket(record) for record in records
        ).items())),
        "depth_fractions_requested": {
            key: fraction
            for key, fraction in zip(
                ("1_to_3", "4_to_6", "7_to_9", "10_plus"),
                VALUE_DEPTH_FRACTIONS,
            )
        },
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _configure_trainable_part(model, trainable_part: str) -> dict[str, int | str]:
    """Backward-compatible wrapper for diagnostic callers and tests."""
    return configure_trainable_part(model, trainable_part)


def _candidate_model(
    checkpoint: dict[str, Any],
    *,
    encoding,
    incumbent_model_config,
    trainable_part: str,
    device: torch.device,
) -> tuple[PolicyValueNet, Any]:
    """Construct a candidate, expanding legacy models only for adapters."""
    candidate_model_config = incumbent_model_config
    if (trainable_part == "policy_adapter"
            and incumbent_model_config.policy_adapter_blocks == 0):
        candidate_model_config = replace(
            incumbent_model_config, policy_adapter_blocks=1)
    model = PolicyValueNet(encoding, candidate_model_config)
    incompatible = model.load_state_dict(
        checkpoint["model_state"],
        strict=(candidate_model_config == incumbent_model_config),
    )
    if candidate_model_config != incumbent_model_config:
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        expected_missing = {
            name for name in model.state_dict()
            if name.startswith("policy_adapter.")
        }
        if missing != expected_missing or unexpected:
            raise RuntimeError(
                "legacy checkpoint could not be expanded with a policy "
                f"adapter: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}")
    return model.to(device), candidate_model_config


def _value_depth_multiplier(raw_moves: float, profile: str) -> float:
    if profile == "standard":
        return 1.0
    if profile != "hard_tail":
        choices = ", ".join(sorted(VALUE_WEIGHT_PROFILES))
        raise ValueError(f"value weight profile must be one of: {choices}")
    if raw_moves <= 6:
        return 1.0
    if raw_moves < 10:
        return 2.0
    return 3.0


def _value_weights_for(
    records: list[dict[str, Any]],
    source_weights: list[float],
    profile: str,
) -> tuple[list[float], dict[str, Any]]:
    if len(records) != len(source_weights):
        raise ValueError("one source weight is required per training record")
    weights = []
    bins = {
        "1_to_3": {"examples": 0, "weight_mass": 0.0},
        "4_to_6": {"examples": 0, "weight_mass": 0.0},
        "7_to_9": {"examples": 0, "weight_mass": 0.0},
        "10_plus": {"examples": 0, "weight_mass": 0.0},
    }
    for record, source_weight in zip(records, source_weights):
        raw_moves = float(
            record["value_target"]["raw_optimal_moves"])
        if not math.isfinite(raw_moves) or raw_moves < 0:
            raise ValueError(
                f"invalid raw value target for depth weighting: {raw_moves}")
        multiplier = _value_depth_multiplier(raw_moves, profile)
        weight = source_weight * multiplier
        weights.append(weight)
        if raw_moves <= 3:
            key = "1_to_3"
        elif raw_moves <= 6:
            key = "4_to_6"
        elif raw_moves < 10:
            key = "7_to_9"
        else:
            key = "10_plus"
        bins[key]["examples"] += 1
        bins[key]["weight_mass"] += weight
    total = sum(weights)
    for item in bins.values():
        item["weight_share"] = item["weight_mass"] / total if total else 0.0
    return weights, {
        "profile": profile,
        "total_weight_mass": total,
        "by_depth": bins,
    }


def _interpolate_toward_incumbent(
    model,
    incumbent_state: dict[str, torch.Tensor],
    fraction: float,
) -> None:
    if fraction == 1.0:
        return
    trained_state = model.state_dict()
    interpolated = {}
    for name, trained in trained_state.items():
        incumbent = incumbent_state[name].to(
            device=trained.device, dtype=trained.dtype)
        if torch.is_floating_point(trained) or torch.is_complex(trained):
            interpolated[name] = incumbent + fraction * (trained - incumbent)
        else:
            interpolated[name] = trained
    model.load_state_dict(interpolated)


def _identity(
    cfg: UpdateSweepConfig,
    *,
    sample_sha256: str,
    value_sample_sha256: dict[str, str],
    policy_weight_sha256: dict[str, str],
    value_weight_sha256: dict[str, str],
    policy_target_sha256: dict[str, str],
) -> dict[str, Any]:
    semantic_config = json.loads(json.dumps(asdict(cfg)))
    semantic_config.pop("output_dir")
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": semantic_config,
        "inputs": {
            "incumbent_checkpoint_sha256":
                sha256_file(cfg.incumbent_checkpoint),
            "replay_snapshot_sha256": sha256_file(cfg.replay_snapshot),
            "sample_sha256": sample_sha256,
            "value_sample_sha256": value_sample_sha256,
            "policy_weight_sha256": policy_weight_sha256,
            "value_weight_sha256": value_weight_sha256,
            "policy_target_sha256": policy_target_sha256,
            "provided_sample_sha256": (
                sha256_file(cfg.sample_jsonl)
                if cfg.sample_jsonl is not None else None),
            "provided_policy_weights_sha256": (
                sha256_file(cfg.policy_weights_json)
                if cfg.policy_weights_json is not None else None),
            "provided_value_weights_sha256": (
                sha256_file(cfg.value_weights_json)
                if cfg.value_weights_json is not None else None),
        },
    }
    result["fingerprint"] = _canonical_sha256(result)
    return result


def _load_weight_vector(
    path: str,
    *,
    label: str,
    expected_count: int,
) -> list[float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise ValueError(
            f"{label} must contain one entry per sample record; "
            f"expected {expected_count}, observed "
            f"{len(raw) if isinstance(raw, list) else 'non-list'}")
    weights = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}[{index}] must be numeric")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{label}[{index}] must be finite and non-negative")
        weights.append(weight)
    return weights


def _load_or_create_sample(
    cfg: UpdateSweepConfig,
    root: Path,
) -> tuple[
        list[dict[str, Any]], str, dict[str, Any], list[float], list[float]]:
    if cfg.sample_jsonl is not None:
        records = load_records(cfg.sample_jsonl)
        policy_weights = _load_weight_vector(
            str(cfg.policy_weights_json),
            label="policy weights",
            expected_count=len(records),
        )
        value_weights = _load_weight_vector(
            str(cfg.value_weights_json),
            label="value weights",
            expected_count=len(records),
        )
        sampling_summary = {
            "mode": "persisted_exact_sample",
            "provided_sample": cfg.sample_jsonl,
            "provided_sample_sha256": sha256_file(cfg.sample_jsonl),
            "provided_policy_weights": cfg.policy_weights_json,
            "provided_policy_weights_sha256":
                sha256_file(str(cfg.policy_weights_json)),
            "provided_value_weights": cfg.value_weights_json,
            "provided_value_weights_sha256":
                sha256_file(str(cfg.value_weights_json)),
        }
    else:
        replay = ReplayBuffer(
            root / ".replay_loader",
            max_examples=max(cfg.sample_size, 1),
            seed=cfg.sample_seed,
        ).load_snapshot(cfg.replay_snapshot)
        records, age_composition = replay.sample_training_set_with_age_quotas(
            cfg.sample_size,
            current_iteration=cfg.current_iteration,
            current_fraction=cfg.replay_current_fraction,
            recent_fraction=cfg.replay_recent_fraction,
            historical_fraction=cfg.replay_historical_fraction,
            recent_window=cfg.replay_recent_window,
            weight_exact_historical=cfg.weight_exact_historical,
            weight_exact_new=cfg.weight_exact_new,
            weight_search=cfg.weight_search,
            seed=cfg.sample_seed,
            with_replacement=cfg.replay_sample_with_replacement,
        )
        policy_weights = source_weights_for(
            records,
            cfg.current_iteration,
            weight_exact_historical=cfg.weight_exact_historical,
            weight_exact_new=cfg.weight_exact_new,
            weight_search=cfg.weight_search,
            exact_path_policy_confidence=cfg.exact_path_policy_confidence,
        )
        value_weights = list(policy_weights)
        sampling_summary = {
            "mode": "fresh_recent_historical_quota_v1",
            "sample_size": cfg.sample_size,
            "seed": cfg.sample_seed,
            "current_iteration": cfg.current_iteration,
            "current_fraction": cfg.replay_current_fraction,
            "recent_fraction": cfg.replay_recent_fraction,
            "historical_fraction": cfg.replay_historical_fraction,
            "recent_window": cfg.replay_recent_window,
            "with_replacement": cfg.replay_sample_with_replacement,
            "realized": age_composition,
        }
    if not records:
        raise RuntimeError("fixed replay produced no training sample")
    if sum(policy_weights) <= 0:
        raise ValueError("policy weights must have positive total mass")
    text = _sample_text(records)
    sample_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sample_path = root / "sample.jsonl"
    if sample_path.exists():
        existing = sample_path.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(
                "persisted update-sweep sample differs from deterministic "
                "reconstruction")
    else:
        atomic_write_text(sample_path, text)
    summary = _sample_summary(records)
    atomic_write_json(root / "sample_summary.json", {
        **summary,
        "sample_sha256": sample_sha256,
        "policy_weight_sha256": _canonical_sha256(policy_weights),
        "value_weight_sha256": _canonical_sha256(value_weights),
        "sampling": sampling_summary,
        "source_replay": cfg.replay_snapshot,
        "source_replay_sha256": sha256_file(cfg.replay_snapshot),
    })
    summary = {**summary, "sampling": sampling_summary}
    return (
        records, sample_sha256, summary, policy_weights, value_weights)


def _load_or_create_depth_value_sample(
    cfg: UpdateSweepConfig,
    root: Path,
) -> tuple[
        list[dict[str, Any]], str, dict[str, Any], list[float], list[float]]:
    replay = ReplayBuffer(
        root / ".value_replay_loader",
        max_examples=max(cfg.sample_size, 1),
        seed=cfg.sample_seed,
    ).load_snapshot(cfg.replay_snapshot)
    records = replay.sample_value_training_set(
        cfg.sample_size,
        current_iteration=cfg.current_iteration,
        weight_exact_historical=cfg.weight_exact_historical,
        weight_exact_new=cfg.weight_exact_new,
        weight_search=cfg.weight_search,
        seed=cfg.sample_seed,
        depth_fractions=VALUE_DEPTH_FRACTIONS,
    )
    if not records:
        raise RuntimeError(
            "depth-stratified replay produced no value-training sample")
    policy_weights = source_weights_for(
        records,
        cfg.current_iteration,
        weight_exact_historical=cfg.weight_exact_historical,
        weight_exact_new=cfg.weight_exact_new,
        weight_search=cfg.weight_search,
        exact_path_policy_confidence=cfg.exact_path_policy_confidence,
    )
    value_weights = list(policy_weights)
    text = _sample_text(records)
    sample_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sample_dir = root / "value_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "depth_stratified.jsonl"
    if sample_path.exists():
        existing = sample_path.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(
                "persisted depth-stratified value sample differs from "
                "deterministic reconstruction")
    else:
        atomic_write_text(sample_path, text)
    summary = _value_sample_summary(records)
    atomic_write_json(sample_dir / "depth_stratified_summary.json", {
        **summary,
        "sample_sha256": sample_sha256,
        "source_replay": cfg.replay_snapshot,
        "source_replay_sha256": sha256_file(cfg.replay_snapshot),
    })
    return records, sample_sha256, summary, policy_weights, value_weights


def _candidate_summary_is_valid(
    path: Path,
    *,
    expected_spec: CandidateSpec,
    experiment_fingerprint: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(summary.get("checkpoint", ""))
    expected = {
        "spec": asdict(expected_spec),
        "experiment_fingerprint": experiment_fingerprint,
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected or not checkpoint.is_file():
        raise RuntimeError(f"incompatible candidate summary: {path}")
    if sha256_file(checkpoint) != summary.get("checkpoint_sha256"):
        raise RuntimeError(f"candidate checkpoint integrity failure: {checkpoint}")
    return summary


def run_update_sweep(cfg: UpdateSweepConfig) -> dict[str, Any]:
    cfg.validate()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (records, sample_sha256, sample_summary, sample_policy_weights,
     sample_value_weights) = _load_or_create_sample(cfg, root)
    sample_profiles = {
        "shared": (
            records,
            sample_sha256,
            sample_summary,
            sample_policy_weights,
            sample_value_weights,
        ),
    }
    if any(
            candidate.value_sample_profile == "depth_stratified"
            for candidate in cfg.candidates):
        sample_profiles["depth_stratified"] = (
            _load_or_create_depth_value_sample(cfg, root))
    checkpoint = load_checkpoint(cfg.incumbent_checkpoint, map_location="cpu")
    encoding, model_config, value_norm = configs_from_checkpoint(checkpoint)
    device = _resolve_device(cfg.device)
    incumbent_checkpoint_sha256 = sha256_file(cfg.incumbent_checkpoint)
    recorded_target_summary = _recorded_policy_target_summary(
        records,
        sample_sha256=sample_sha256,
        sample_path=str(root / "sample.jsonl"),
    )
    policy_target_profiles = {
        "recorded": (records, sample_sha256, recorded_target_summary),
    }
    requested_conditioned_profiles = sorted({
        candidate.policy_target_profile
        for candidate in cfg.candidates
        if candidate.policy_target_profile != "recorded"
    })
    if requested_conditioned_profiles:
        incumbent_model = model_from_checkpoint(
            checkpoint, map_location=device)
        incumbent_probabilities = _incumbent_legal_probabilities(
            records,
            model=incumbent_model,
            encoding_config=encoding,
            value_norm=value_norm,
            device=device,
            batch_size=cfg.batch_size,
        )
        for profile in requested_conditioned_profiles:
            transformed, target_summary = _condition_policy_targets(
                records,
                incumbent_probabilities,
                profile=profile,
                incumbent_checkpoint_sha256=incumbent_checkpoint_sha256,
            )
            transformed_sha256, target_summary = (
                _persist_policy_target_profile(
                    root,
                    profile=profile,
                    records=transformed,
                    summary=target_summary,
                ))
            policy_target_profiles[profile] = (
                transformed, transformed_sha256, target_summary)
        del incumbent_model
    identity = _identity(
        cfg,
        sample_sha256=sample_sha256,
        value_sample_sha256={
            name: profile[1]
            for name, profile in sample_profiles.items()
        },
        policy_weight_sha256={
            name: _canonical_sha256(profile[3])
            for name, profile in sample_profiles.items()
        },
        value_weight_sha256={
            name: _canonical_sha256(profile[4])
            for name, profile in sample_profiles.items()
        },
        policy_target_sha256={
            name: profile[2]["policy_target_sha256"]
            for name, profile in policy_target_profiles.items()
        },
    )
    identity_path = root / "experiment.json"
    if identity_path.exists():
        persisted = json.loads(identity_path.read_text(encoding="utf-8"))
        if persisted != identity:
            raise RuntimeError(
                "update-sweep output directory belongs to different inputs or "
                "settings")
    else:
        atomic_write_json(identity_path, identity)

    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in cfg.candidates:
        candidate_path = candidate_dir / f"{spec.name}.pt"
        summary_path = candidate_dir / f"{spec.name}.json"
        cached = _candidate_summary_is_valid(
            summary_path,
            expected_spec=spec,
            experiment_fingerprint=identity["fingerprint"],
        )
        if cached is not None:
            print(f"reusing candidate {spec.name}", flush=True)
            summaries.append(cached)
            continue
        if candidate_path.exists():
            partial = load_checkpoint(candidate_path, map_location="cpu")
            if partial.get("experiment_fingerprint") != identity["fingerprint"]:
                raise RuntimeError(
                    f"orphan candidate belongs to another experiment: "
                    f"{candidate_path}")
            train_info = partial.get("metrics", {}).get("train")
            if not isinstance(train_info, dict):
                raise RuntimeError(
                    f"orphan candidate lacks training evidence: {candidate_path}")
            summary = {
                "spec": asdict(spec),
                "experiment_fingerprint": identity["fingerprint"],
                "checkpoint": str(candidate_path),
                "checkpoint_sha256": sha256_file(candidate_path),
                "train": train_info,
            }
            atomic_write_json(summary_path, summary)
            summaries.append(summary)
            continue

        print(
            f"training {spec.name}: lr={spec.learning_rate:g}, "
            f"epochs={spec.epochs}, update_fraction={spec.update_fraction:g}, "
            f"trainable_part={spec.trainable_part}, "
            f"value_profile={spec.value_weight_profile}, "
            f"value_anchor={spec.value_anchor_weight:g}, "
            f"value_sample={spec.value_sample_profile}, "
            f"policy_anchor={spec.policy_anchor_weight:g}, "
            f"policy_target={spec.policy_target_profile}, "
            f"policy_ranking={spec.policy_ranking_weight:g}",
            flush=True,
        )
        (base_candidate_records, _base_candidate_sample_sha256,
         base_candidate_sample_summary,
         candidate_policy_weights, candidate_base_value_weights) = (
            sample_profiles[spec.value_sample_profile])
        if spec.policy_target_profile == "recorded":
            candidate_records = base_candidate_records
            candidate_sample_sha256 = _base_candidate_sample_sha256
            policy_target_summary = _recorded_policy_target_summary(
                candidate_records,
                sample_sha256=candidate_sample_sha256,
                sample_path=(
                    recorded_target_summary["sample"]
                    if spec.value_sample_profile == "shared"
                    else str(
                        root / "value_samples"
                        / f"{spec.value_sample_profile}.jsonl")),
            )
        else:
            (candidate_records, candidate_sample_sha256,
             policy_target_summary) = policy_target_profiles[
                spec.policy_target_profile]
        candidate_sample_summary = {
            **base_candidate_sample_summary,
            "policy_target_profile": policy_target_summary,
        }
        model, candidate_model_config = _candidate_model(
            checkpoint,
            encoding=encoding,
            incumbent_model_config=model_config,
            trainable_part=spec.trainable_part,
            device=device,
        )
        initial_model_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        trainable_info = _configure_trainable_part(
            model, spec.trainable_part)
        value_weights, value_weight_summary = _value_weights_for(
            candidate_records,
            candidate_base_value_weights,
            spec.value_weight_profile,
        )
        value_anchor_model = None
        if spec.value_anchor_weight > 0:
            value_anchor_model = model_from_checkpoint(
                checkpoint, map_location=device)
        policy_anchor_model = None
        if spec.policy_anchor_weight > 0:
            policy_anchor_model = model_from_checkpoint(
                checkpoint, map_location=device)
        train_info = train_expert(
            model,
            candidate_records,
            candidate_policy_weights,
            encoding_config=encoding,
            value_norm=value_norm,
            epochs=spec.epochs,
            batch_size=cfg.batch_size,
            learning_rate=spec.learning_rate,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
            device=device,
            seed=cfg.train_seed,
            value_weights=value_weights,
            value_anchor_model=value_anchor_model,
            value_anchor_weight=spec.value_anchor_weight,
            search_value_loss_weight=cfg.search_value_loss_weight,
            policy_anchor_model=policy_anchor_model,
            policy_anchor_weight=spec.policy_anchor_weight,
            policy_anchor_before_iteration=cfg.current_iteration,
            policy_ranking_weight=spec.policy_ranking_weight,
            policy_ranking_margin=cfg.policy_ranking_margin,
        )
        _interpolate_toward_incumbent(
            model, initial_model_state, spec.update_fraction)
        train_info = {
            **train_info,
            "sample": candidate_sample_summary,
            "sample_sha256": candidate_sample_sha256,
            "sample_profile": spec.value_sample_profile,
            "policy_target_profile": spec.policy_target_profile,
            "policy_target_sha256":
                policy_target_summary["policy_target_sha256"],
            "update_fraction": spec.update_fraction,
            "value_weighting": value_weight_summary,
            "value_anchor_weight": spec.value_anchor_weight,
            "policy_anchor_weight": spec.policy_anchor_weight,
            "policy_ranking_weight": spec.policy_ranking_weight,
            "policy_ranking_margin": cfg.policy_ranking_margin,
            **trainable_info,
        }
        save_checkpoint(
            candidate_path,
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=spec.epochs,
            best_val_metric=None,
            encoding_config=encoding,
            model_config=candidate_model_config,
            value_norm=value_norm,
            seed=cfg.train_seed,
            dataset_version=checkpoint.get("dataset_version", 1),
            split_identity={
                "kind": "fixed_replay_update_sweep",
                "source_replay_sha256": sha256_file(cfg.replay_snapshot),
                "sample_sha256": candidate_sample_sha256,
                "sample_profile": spec.value_sample_profile,
                "policy_target_profile": spec.policy_target_profile,
                "policy_target_sha256":
                    policy_target_summary["policy_target_sha256"],
                "candidate": asdict(spec),
            },
            metrics={"train": train_info, "candidate_spec": asdict(spec)},
            experiment_fingerprint=identity["fingerprint"],
        )
        summary = {
            "spec": asdict(spec),
            "experiment_fingerprint": identity["fingerprint"],
            "checkpoint": str(candidate_path),
            "checkpoint_sha256": sha256_file(candidate_path),
            "train": train_info,
        }
        atomic_write_json(summary_path, summary)
        summaries.append(summary)
        print(f"completed candidate {spec.name}", flush=True)

    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "incumbent_checkpoint": cfg.incumbent_checkpoint,
        "incumbent_checkpoint_sha256":
            identity["inputs"]["incumbent_checkpoint_sha256"],
        "replay_snapshot": cfg.replay_snapshot,
        "replay_snapshot_sha256":
            identity["inputs"]["replay_snapshot_sha256"],
        "sample": sample_summary,
        "sample_sha256": sample_sha256,
        "value_samples": {
            name: {
                "sample_sha256": profile[1],
                "summary": profile[2],
                "policy_weight_sha256": _canonical_sha256(profile[3]),
                "value_weight_sha256": _canonical_sha256(profile[4]),
            }
            for name, profile in sample_profiles.items()
        },
        "policy_targets": {
            name: profile[2]
            for name, profile in policy_target_profiles.items()
        },
        "candidates": summaries,
    }
    atomic_write_json(root / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train controlled protagonist updates from one frozen replay.")
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--replay-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=None,
        metavar=(
            "NAME=LR[,EPOCHS[,UPDATE_FRACTION[,TRAINABLE_PART"
            "[,VALUE_PROFILE[,VALUE_ANCHOR[,VALUE_SAMPLE"
            "[,POLICY_ANCHOR[,POLICY_TARGET[,POLICY_RANKING]]]]]]]]]"),
        help=(
            "candidate update; TRAINABLE_PART may be all, policy_head, "
            "policy_adapter, policy_trunk, or value_head; VALUE_PROFILE may "
            "be standard or "
            "hard_tail; "
            "VALUE_SAMPLE may be shared or depth_stratified; POLICY_ANCHOR "
            "is an incumbent-policy KL weight on historical records; "
            "POLICY_TARGET may be recorded, incumbent_optimal, "
            "incumbent_optimal_blend50, or incumbent_optimal_sharp; "
            "POLICY_RANKING is the optional search-utility ranking weight"),
    )
    parser.add_argument("--sample-size", type=int, default=2_000)
    parser.add_argument("--current-iteration", type=int, default=2)
    parser.add_argument(
        "--sample-jsonl",
        default=None,
        help="exact persisted co-training sample; requires both weight files",
    )
    parser.add_argument(
        "--policy-weights-json",
        default=None,
        help="policy weights aligned one-for-one with --sample-jsonl",
    )
    parser.add_argument(
        "--value-weights-json",
        default=None,
        help=(
            "base value weights aligned with --sample-jsonl, before the "
            "search-value confidence multiplier"),
    )
    parser.add_argument("--replay-current-fraction", type=float, default=0.35)
    parser.add_argument("--replay-recent-fraction", type=float, default=0.25)
    parser.add_argument(
        "--replay-historical-fraction", type=float, default=0.40)
    parser.add_argument("--replay-recent-window", type=int, default=2)
    parser.add_argument(
        "--replay-sample-with-replacement", action="store_true")
    parser.add_argument(
        "--weight-exact-historical", type=float, default=1.0,
        help="replay-sampling preference for historical exact records")
    parser.add_argument(
        "--weight-exact-new", type=float, default=1.5,
        help="replay-sampling preference for current-iteration exact records")
    parser.add_argument(
        "--weight-search", type=float, default=0.5,
        help="replay-sampling preference for search-derived records")
    parser.add_argument(
        "--exact-path-policy-confidence", type=float, default=0.5)
    parser.add_argument("--search-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--policy-ranking-margin", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=26_509)
    parser.add_argument("--train-seed", type=int, default=2_041)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = UpdateSweepConfig(
        incumbent_checkpoint=args.incumbent_checkpoint,
        replay_snapshot=args.replay_snapshot,
        output_dir=args.output_dir,
        candidates=tuple(args.candidate or DEFAULT_CANDIDATES),
        sample_size=args.sample_size,
        current_iteration=args.current_iteration,
        sample_jsonl=args.sample_jsonl,
        policy_weights_json=args.policy_weights_json,
        value_weights_json=args.value_weights_json,
        replay_current_fraction=args.replay_current_fraction,
        replay_recent_fraction=args.replay_recent_fraction,
        replay_historical_fraction=args.replay_historical_fraction,
        replay_recent_window=args.replay_recent_window,
        replay_sample_with_replacement=args.replay_sample_with_replacement,
        weight_exact_historical=args.weight_exact_historical,
        weight_exact_new=args.weight_exact_new,
        weight_search=args.weight_search,
        exact_path_policy_confidence=args.exact_path_policy_confidence,
        search_value_loss_weight=args.search_value_loss_weight,
        policy_ranking_margin=args.policy_ranking_margin,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        sample_seed=args.sample_seed,
        train_seed=args.train_seed,
        device=args.device,
    )
    result = run_update_sweep(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
