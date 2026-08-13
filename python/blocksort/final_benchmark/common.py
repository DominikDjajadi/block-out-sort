"""Shared helpers for the final-benchmark experiment drivers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..schema import Level
from ..training.checkpoint import (configs_from_checkpoint, load_checkpoint,
                                   model_from_checkpoint)
from ..training.experiment_identity import (
    EXPERIMENT_SPEC_FILE, ExperimentSpecIntegrityError,
    LegacyRunMigrationError, hash_canonical_value, hash_file_streaming,
    load_persisted_experiment_spec)
from ..training.transaction import resolve_run_path
from ..designer.actions import DesignerActionSpace
from ..designer.checkpoint import designer_from_checkpoint, load_designer
from ..designer.config import GeneratorConfig
from ..designer.encoding import DESIGNER_EXTRA_GLOBALS
from ..designer.env import DesignerEnv
from ..designer.model import DesignerModelConfig, DesignerNet
from ..designer.ppo import rollout_episode


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass(frozen=True)
class ValidatedCheckpointSource:
    checkpoint_path: Path
    checkpoint_sha256: str
    bytes: int
    source_kind: str
    pipeline: str | None
    experiment_fingerprint: str | None
    committed_role: str | None
    committed_progress: int | None
    encoding_fingerprint: str | None = None
    authority_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.checkpoint_path),
            "checkpoint_path": str(self.checkpoint_path),
            "exists": True,
            "bytes": self.bytes,
            "sha256": self.checkpoint_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_kind": self.source_kind,
            "pipeline": self.pipeline,
            "source_pipeline": self.pipeline,
            "source_experiment_fingerprint":
                self.experiment_fingerprint,
            "experiment_fingerprint": self.experiment_fingerprint,
            "committed_role": self.committed_role,
            "committed_progress": self.committed_progress,
            "encoding_fingerprint": self.encoding_fingerprint,
            "authority_sha256": self.authority_sha256,
        }


def _source_run_root(path: Path) -> Path | None:
    if (path.parent / EXPERIMENT_SPEC_FILE).is_file():
        return path.parent
    if (path.parent.name in {"checkpoints", "protagonist"}
            and (path.parent.parent / EXPERIMENT_SPEC_FILE).is_file()):
        return path.parent.parent
    return None


def _load_source_state(root: Path, fingerprint: str) -> dict[str, Any]:
    state_path = root / "run_state.json"
    if not state_path.is_file():
        raise ExperimentSpecIntegrityError(
            f"identified source run is missing authoritative state: {state_path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecIntegrityError(
            f"cannot validate source run state: {state_path}") from exc
    if state.get("experiment_fingerprint") != fingerprint:
        raise ExperimentSpecIntegrityError(
            "source run-state experiment fingerprint does not match "
            f"{root / EXPERIMENT_SPEC_FILE}")
    return state


def _validate_state_artifact(
    root: Path, relative: str, expected: str, label: str
) -> tuple[Path, str]:
    artifact = resolve_run_path(root, relative)
    if not artifact.is_file():
        raise ExperimentSpecIntegrityError(
            f"identified source {label} is missing: {artifact}")
    observed = hash_file_streaming(artifact)
    if observed != expected:
        raise ExperimentSpecIntegrityError(
            f"identified source {label} integrity failure: "
            f"expected={expected}, observed={observed}")
    return artifact, observed


def validate_checkpoint_source(path: str | Path) -> ValidatedCheckpointSource:
    """Validate checkpoint content and its authoritative role, when identified."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint source is missing: {p}")
    digest = hash_file_streaming(p)
    size = p.stat().st_size
    root = _source_run_root(p)
    if root is None:
        run_markers = (
            "run_state.json", "config.json", "history.json",
            "summary.json", "replay", "splits.json")
        if any((p.parent / marker).exists() for marker in run_markers):
            raise LegacyRunMigrationError(
                "final-benchmark source appears to be a legacy run but "
                f"has no {EXPERIMENT_SPEC_FILE}: {p.parent}. Migrate the "
                "source run or use a standalone content-addressed checkpoint.")
        return ValidatedCheckpointSource(
            checkpoint_path=p, checkpoint_sha256=digest, bytes=size,
            source_kind="standalone_checkpoint", pipeline=None,
            experiment_fingerprint=None, committed_role=None,
            committed_progress=None, authority_sha256=None)

    spec, fingerprint = load_persisted_experiment_spec(root)
    pipeline = spec["pipeline"]
    role = None
    progress = None
    encoding_fingerprint = None
    authority_sha256 = None
    if pipeline in {"cotraining", "expert_iteration"}:
        state = _load_source_state(root, fingerprint)
        authority_sha256 = hash_file_streaming(root / "run_state.json")
        active, active_digest = _validate_state_artifact(
            root, state["active_protagonist_checkpoint"],
            state["active_protagonist_sha256"], "active checkpoint")
        if digest != active_digest:
            raise ExperimentSpecIntegrityError(
                f"checkpoint {p} is not the authoritative active checkpoint "
                f"for identified {pipeline} run {root}")
        role = "active"
        progress = int(state.get(
            "active_protagonist_source_round"
            if pipeline == "cotraining"
            else "active_protagonist_source_iteration", 0))
        load_checkpoint(active, map_location="cpu")
    elif pipeline == "supervised_protagonist":
        state = _load_source_state(root, fingerprint)
        authority_sha256 = hash_file_streaming(root / "run_state.json")
        active, active_digest = _validate_state_artifact(
            root, state["active_checkpoint"],
            state["active_checkpoint_sha256"], "active checkpoint")
        best_digest = None
        if state.get("best_checkpoint"):
            _best, best_digest = _validate_state_artifact(
                root, state["best_checkpoint"],
                state["best_checkpoint_sha256"], "best checkpoint")
        if p.name == "best.pt":
            if best_digest is None or digest != best_digest:
                raise ExperimentSpecIntegrityError(
                    "supervised best.pt does not match committed best checkpoint")
            role = "best"
        elif p.name == "last.pt":
            if digest != active_digest:
                raise ExperimentSpecIntegrityError(
                    "supervised last.pt does not match committed active checkpoint")
            role = "active"
        elif p.resolve() == active.resolve() and digest == active_digest:
            role = "active"
        elif (state.get("best_checkpoint")
              and p.resolve() == resolve_run_path(
                  root, state["best_checkpoint"]).resolve()
              and digest == best_digest):
            role = "best"
        else:
            raise ExperimentSpecIntegrityError(
                f"supervised checkpoint {p} is not referenced by committed state")
        progress = int(state["completed_epochs"])
        load_checkpoint(p, map_location="cpu")
    elif pipeline in {"designer_training", "designer_pretraining"}:
        summary_path = root / (
            "summary.json" if pipeline == "designer_training"
            else "pretrain_summary.json")
        if not summary_path.is_file():
            raise ExperimentSpecIntegrityError(
                f"identified designer source is missing provenance summary: "
                f"{summary_path}")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExperimentSpecIntegrityError(
                f"cannot validate designer provenance summary: "
                f"{summary_path}") from exc
        authority_sha256 = hash_file_streaming(summary_path)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        embedded = (
            payload.get("experiment_fingerprint")
            or payload.get("metadata", {}).get("experiment_fingerprint"))
        if embedded != fingerprint:
            raise ExperimentSpecIntegrityError(
                "identified designer checkpoint lacks matching committed "
                "experiment provenance")
        if pipeline == "designer_training":
            roles = {
                "best": (
                    summary.get("best_checkpoint"),
                    summary.get("best_checkpoint_sha256")),
                "last": (
                    summary.get("last_checkpoint"),
                    summary.get("last_checkpoint_sha256")),
            }
        else:
            roles = {
                "best": (
                    summary.get("checkpoint"),
                    summary.get("checkpoint_sha256")),
            }
        for candidate_role, (candidate_path, expected) in roles.items():
            if (candidate_path and Path(candidate_path).resolve() == p.resolve()
                    and expected == digest):
                role = candidate_role
                break
        if role is None:
            raise ExperimentSpecIntegrityError(
                f"designer checkpoint {p} is not a hash-matching committed "
                "output in its provenance summary")
        encoding_fingerprint = hash_canonical_value({
            "encoding_config": payload.get("encoding_config"),
            "model_config": payload.get("model_config"),
        })
        if summary.get("encoding_fingerprint") != encoding_fingerprint:
            raise ExperimentSpecIntegrityError(
                "designer checkpoint encoding fingerprint does not match "
                "persisted provenance")
    else:
        raise ExperimentSpecIntegrityError(
            f"unsupported identified checkpoint source pipeline: {pipeline!r}")

    return ValidatedCheckpointSource(
        checkpoint_path=p, checkpoint_sha256=digest, bytes=size,
        source_kind="identified_run", pipeline=pipeline,
        experiment_fingerprint=fingerprint, committed_role=role,
        committed_progress=progress,
        encoding_fingerprint=encoding_fingerprint,
        authority_sha256=authority_sha256)


def checkpoint_identity(path: str) -> dict:
    """Compatibility dictionary for validated checkpoint source provenance."""
    p = Path(path)
    if not p.is_file():
        return {
            "path": str(path), "exists": False,
            "source_kind": "missing_checkpoint",
            "source_experiment_fingerprint": None,
        }
    return validate_checkpoint_source(p).to_dict()


class Protagonist:
    """Convenience wrapper bundling a protagonist model + its encoding."""

    def __init__(
        self,
        checkpoint: str | Path | ValidatedCheckpointSource,
        device: torch.device,
    ):
        self.source = (
            checkpoint if isinstance(checkpoint, ValidatedCheckpointSource)
            else validate_checkpoint_source(checkpoint))
        self.checkpoint = str(self.source.checkpoint_path)
        ckpt = load_checkpoint(self.checkpoint, map_location="cpu")
        self.enc, self.model_cfg, self.value_norm = configs_from_checkpoint(ckpt)
        self.model = model_from_checkpoint(ckpt, map_location=device)
        self.model.eval()
        self.device = device


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def designer_encoding_fingerprint(enc) -> str:
    return _canonical_hash({
        "encoding_config": enc.to_dict(),
        "designer_feature_schema": "base_encoding_plus_designer_globals_v1",
        "designer_extra_globals": DESIGNER_EXTRA_GLOBALS,
        "designer_action_schema": "designer_action_space_v1",
    })


@dataclass(frozen=True)
class LoadedDesigner:
    checkpoint_path: Path
    checkpoint_sha256: str
    model: DesignerNet
    encoding: Any
    encoding_fingerprint: str
    model_config: DesignerModelConfig
    checkpoint_version: int
    source: ValidatedCheckpointSource

    def provenance(self) -> dict[str, Any]:
        return {
            **self.source.to_dict(),
            "source_encoding_fingerprint":
                self.source.encoding_fingerprint,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "encoding_fingerprint": self.encoding_fingerprint,
            "encoding_config": self.encoding.to_dict(),
            "model_config": self.model_config.to_dict(),
            "checkpoint_version": self.checkpoint_version,
        }


def load_designer_bundle(
    checkpoint: str | Path | ValidatedCheckpointSource,
    device: torch.device,
) -> LoadedDesigner:
    source = (
        checkpoint if isinstance(checkpoint, ValidatedCheckpointSource)
        else validate_checkpoint_source(checkpoint))
    path = source.checkpoint_path
    ckpt = load_designer(path, map_location=device)
    if not isinstance(ckpt.get("encoding_config"), dict):
        raise ValueError(
            f"designer checkpoint {path} is missing encoding metadata")
    model, enc, model_cfg = designer_from_checkpoint(
        ckpt, map_location=device)
    model.eval()
    return LoadedDesigner(
        checkpoint_path=path,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model=model,
        encoding=enc,
        encoding_fingerprint=designer_encoding_fingerprint(enc),
        model_config=model_cfg,
        checkpoint_version=int(ckpt["designer_checkpoint_version"]),
        source=source,
    )


def designer_generation_identity(
    bundle: LoadedDesigner,
    gen_cfg: GeneratorConfig,
    *,
    mutation_budget: int,
    count: int,
    seed: int,
) -> str:
    return _canonical_hash({
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "encoding_fingerprint": bundle.encoding_fingerprint,
        "generator_config": asdict(gen_cfg),
        "mutation_budget": mutation_budget,
        "count": count,
        "seed": seed,
    })


def validate_designer_generation_config(
    bundle: LoadedDesigner, gen_cfg: GeneratorConfig
) -> None:
    enc = bundle.encoding
    if gen_cfg.rows > enc.max_rows or gen_cfg.cols > enc.max_cols:
        raise ValueError(
            f"designer checkpoint {bundle.checkpoint_path} encoding supports at "
            f"most {enc.max_rows}x{enc.max_cols}, requested "
            f"{gen_cfg.rows}x{gen_cfg.cols}")
    if gen_cfg.color_count > len(enc.colors):
        raise ValueError(
            f"designer checkpoint {bundle.checkpoint_path} encoding supports "
            f"{len(enc.colors)} colors, requested {gen_cfg.color_count}")


def designer_levels(env: Environment, bundle: LoadedDesigner,
                    gen_cfg: GeneratorConfig, *,
                    mutation_budget: int, count: int, device: torch.device,
                    seed: int) -> list[Level]:
    """Roll out a designer policy ``count`` times, keeping valid levels."""
    validate_designer_generation_config(bundle, gen_cfg)
    denv = DesignerEnv(
        gen_cfg, mutation_budget=mutation_budget, encoding=bundle.encoding)
    action_space = DesignerActionSpace(bundle.encoding)
    rng = random.Random(seed)
    out: list[Level] = []
    for i in range(count):
        ep = rollout_episode(denv, bundle.model, action_space, bundle.encoding,
                             seed=seed * 100003 + i, device=device, rng=rng,
                             verify_finalize=False)
        if ep.finalize.valid:
            out.append(ep.finalize.level)
    return out
