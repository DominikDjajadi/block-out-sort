"""Canonical, content-based experiment identity for resumable run directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .transaction import atomic_write_json

EXPERIMENT_SPEC_SCHEMA_VERSION = 1
EXPERIMENT_SPEC_FILE = "experiment_spec.json"
LEGACY_MIGRATION_MANIFEST_FILE = "legacy_migration_manifest.json"
EVALUATION_SEMANTICS_VERSION = 9
PROMOTION_CONTRACT_VERSION = 2
TRANSACTION_SCHEMA_VERSION = 2
_DIAGNOSTIC_KEYS = {"path_hint", "requested_device"}


class ExperimentIdentityError(RuntimeError):
    """Requested invocation does not match an initialized experiment."""


class ExperimentSpecIntegrityError(ExperimentIdentityError):
    """Persisted experiment specification or fingerprint is corrupt."""


class LegacyRunMigrationError(ExperimentIdentityError):
    """A legacy run cannot be assigned an identity without guessing."""


class UnsupportedResumeError(ExperimentIdentityError):
    """A pipeline was asked to reuse output without exact resume support."""


class ContinuationHorizonError(ExperimentIdentityError):
    """A requested total horizon is behind already committed progress."""


class MissingRunStateError(ExperimentSpecIntegrityError):
    """An identified run has artifacts but no authoritative transaction state."""


@dataclass(frozen=True)
class ExperimentDifference:
    field: str
    persisted: Any
    requested: Any
    category: str


def hash_file_streaming(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any, *, fingerprint: bool) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, fingerprint=fingerprint)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (fingerprint and str(key) in _DIAGNOSTIC_KEYS)
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item, fingerprint=fingerprint) for item in value]
    if isinstance(value, set):
        normalized = [_normalize(item, fingerprint=fingerprint) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("experiment specifications cannot contain non-finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported experiment specification value: {type(value)!r}")


def canonical_experiment_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize(spec, fingerprint=False)
    allowed = {
        "schema_version", "pipeline", "semantic_config", "inputs",
        "software_semantics", "derived"}
    unknown = set(normalized) - allowed
    missing = allowed - set(normalized)
    if unknown or missing:
        raise ValueError(
            "invalid experiment specification fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}")
    if normalized.get("schema_version") != EXPERIMENT_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported experiment specification schema version")
    pipeline = normalized.get("pipeline")
    if not isinstance(pipeline, str) or not pipeline:
        raise ValueError("experiment specification requires a pipeline name")
    for required in ("semantic_config", "inputs", "software_semantics"):
        if not isinstance(normalized.get(required), dict):
            raise ValueError(
                f"experiment specification requires mapping {required!r}")

    def validate_values(value: Any, field: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{field}.{key}" if field else str(key)
                if key in {"sha256", "manifest_sha256"}:
                    _validate_fingerprint(item, child)
                if key.endswith("_fingerprint") and item is not None:
                    _validate_fingerprint(item, child)
                if (key.endswith("_count") or key == "bytes") and (
                        not isinstance(item, int) or isinstance(item, bool)
                        or item < 0):
                    raise ValueError(
                        f"{child} must be a non-negative integer")
                validate_values(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                validate_values(item, f"{field}[{index}]")
    validate_values(normalized)
    return normalized


def fingerprint_experiment_spec(spec: Mapping[str, Any]) -> str:
    canonical = canonical_experiment_spec(spec)
    identity = _normalize(canonical, fingerprint=True)
    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_canonical_value(value: Any) -> str:
    normalized = _normalize(value, fingerprint=False)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_identity(
    path: str | Path,
    *,
    kind: str,
    count_lines: bool = False,
    format_version: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{kind} input is missing: {source}")
    identity: dict[str, Any] = {
        "kind": kind,
        "path_hint": source.as_posix(),
        "sha256": hash_file_streaming(source),
        "bytes": source.stat().st_size,
    }
    if count_lines:
        with source.open("rb") as handle:
            identity["record_count"] = sum(1 for line in handle if line.strip())
    if format_version is not None:
        identity["format_version"] = format_version
    if extra:
        identity.update(extra)
    return identity


def semantic_dataclass_config(
    config: Any,
    *,
    semantic_fields: Iterable[str],
    operational_fields: Iterable[str],
    input_fields: Iterable[str],
    continuation_horizon_fields: Iterable[str] = (),
    derived_fields: Iterable[str] = (),
    unsupported_resume_fields: Iterable[str] = (),
    unordered_fields: Iterable[str] = (),
) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("semantic config classification requires a dataclass")
    semantic = tuple(semantic_fields)
    categories = {
        "semantic": set(semantic),
        "operational": set(operational_fields),
        "input": set(input_fields),
        "continuation_horizon": set(continuation_horizon_fields),
        "derived": set(derived_fields),
        "unsupported_on_resume": set(unsupported_resume_fields),
    }
    actual = {field.name for field in fields(config)}
    validate_field_classification(actual, categories)
    unordered = set(unordered_fields)
    result = {}
    for name in semantic:
        value = getattr(config, name)
        if name in unordered:
            value = sorted(set(value))
        result[name] = _normalize(value, fingerprint=False)
    return result


def validate_field_classification(
    actual_fields: Iterable[str],
    categories: Mapping[str, Iterable[str]],
) -> None:
    actual = set(actual_fields)
    memberships: dict[str, list[str]] = {}
    for category, names in categories.items():
        for name in names:
            memberships.setdefault(name, []).append(category)
    unclassified = sorted(actual - set(memberships))
    unknown = sorted(set(memberships) - actual)
    duplicated = {
        name: values for name, values in sorted(memberships.items())
        if len(values) != 1}
    if unclassified or unknown or duplicated:
        raise ValueError(
            "incomplete config field classification: "
            f"unclassified={unclassified}, unknown={unknown}, "
            f"multiple_categories={duplicated}")


def runtime_device_provenance(
    *, requested_device: str, resolved_device: str | torch.device
) -> dict[str, Any]:
    """Return strict runtime identity used to accept or reject resume.

    Resolved device class, PyTorch/CUDA/cuDNN versions, and deterministic
    algorithm mode are resume-sensitive. CUDA fields normalize to ``None`` for
    CPU execution regardless of whether the installed PyTorch build supports
    CUDA.
    """
    resolved = torch.device(resolved_device).type
    return {
        "requested_device": requested_device,
        "resolved_device": resolved,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if resolved == "cuda" else None,
        "cudnn_version": (
            torch.backends.cudnn.version() if resolved == "cuda" else None),
        "deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
    }


def validate_continuation_horizon(
    *, name: str, requested: int, committed: int, run_dir: str | Path
) -> None:
    if requested < committed:
        raise ContinuationHorizonError(
            f"Cannot resume run {str(Path(run_dir))!r}: requested {name} "
            f"{requested} is below committed progress {committed}. "
            "Use a target at least as large as committed progress; committed "
            "artifacts cannot be rewound.")


def ensure_fresh_output_directory(
    run_dir: str | Path, *, pipeline_label: str
) -> None:
    root = Path(run_dir)
    if root.exists() and any(root.iterdir()):
        spec_exists = (root / EXPERIMENT_SPEC_FILE).is_file()
        completed = any(
            (root / marker).exists()
            for marker in ("summary.json", "pretrain_summary.json",
                           "best.pt", "last.pt"))
        if spec_exists and completed:
            state = "a completed identified run"
        elif spec_exists:
            state = "a partially initialized identified run"
        else:
            state = "an arbitrary nonempty directory"
        raise UnsupportedResumeError(
            f"{pipeline_label} output directory contains {state}: {root}. "
            f"{pipeline_label} does not currently support exact resume or "
            "partial-initialization recovery. Reusing this directory would "
            "combine a fresh model with existing state. Use a new output "
            "directory.")


def validate_identified_run_state_presence(
    run_dir: str | Path,
    *,
    pipeline_label: str,
    allowed_setup_files: Iterable[str],
) -> bool:
    """Validate missing-state recovery; return whether setup-only recovery applies."""
    root = Path(run_dir)
    if not (root / EXPERIMENT_SPEC_FILE).is_file():
        return False
    if (root / "run_state.json").is_file():
        return False
    allowed = {
        EXPERIMENT_SPEC_FILE,
        LEGACY_MIGRATION_MANIFEST_FILE,
        *tuple(Path(value).as_posix() for value in allowed_setup_files),
    }
    detected: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in allowed:
            detected.append(relative)
        elif path.is_dir():
            if not any(
                    item == relative or item.startswith(f"{relative}/")
                    for item in allowed):
                detected.append(f"{relative}/")
    if detected:
        details = "\n".join(f"  {value}" for value in detected[:20])
        raise MissingRunStateError(
            f"Cannot resume identified {pipeline_label} run: "
            "run_state.json is missing, but progress or unknown artifacts "
            f"exist.\n\nDetected:\n{details}\n\nThe authoritative transaction "
            "state is unavailable. Restore run_state.json from backup or use "
            "a new output directory.")
    return True


def build_experiment_spec(
    *,
    pipeline: str,
    semantic_config: Mapping[str, Any],
    inputs: Mapping[str, Any],
    software_semantics: Mapping[str, Any],
    derived: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return canonical_experiment_spec({
        "schema_version": EXPERIMENT_SPEC_SCHEMA_VERSION,
        "pipeline": pipeline,
        "semantic_config": semantic_config,
        "inputs": inputs,
        "software_semantics": software_semantics,
        "derived": dict(derived or {}),
    })


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        out = {}
        for key in sorted(value):
            if key in _DIAGNOSTIC_KEYS:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], path))
        return out
    return {prefix: value}


def compare_experiment_specs(
    persisted: Mapping[str, Any], requested: Mapping[str, Any]
) -> list[ExperimentDifference]:
    expected = _flatten(canonical_experiment_spec(persisted))
    actual = _flatten(canonical_experiment_spec(requested))
    differences = []
    for field in sorted(set(expected) | set(actual)):
        if expected.get(field) == actual.get(field):
            continue
        if field.startswith("inputs."):
            category = "input content"
        elif field.startswith("software_semantics."):
            category = "software semantics"
        else:
            category = "semantic config"
        differences.append(ExperimentDifference(
            field=field, persisted=expected.get(field),
            requested=actual.get(field), category=category))
    return differences


def _validate_fingerprint(value: Any, context: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ExperimentSpecIntegrityError(
            f"{context} has an invalid SHA-256 fingerprint")
    return value


def _load_experiment_spec_document(path: Path) -> tuple[dict, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecIntegrityError(
            f"cannot read experiment specification: {path}") from exc
    expected_document_fields = {
        "schema_version", "experiment_fingerprint",
        "spec_integrity_sha256", "spec"}
    if (not isinstance(document, dict)
            or set(document) != expected_document_fields
            or not isinstance(document.get("spec"), dict)):
        raise ExperimentSpecIntegrityError(
            f"malformed experiment specification document: {path}")
    if document.get("schema_version") != EXPERIMENT_SPEC_SCHEMA_VERSION:
        raise ExperimentSpecIntegrityError(
            f"unsupported experiment specification document schema: {path}")
    stored = _validate_fingerprint(
        document.get("experiment_fingerprint"), str(path))
    stored_integrity = _validate_fingerprint(
        document.get("spec_integrity_sha256"), str(path))
    try:
        observed_integrity = hash_canonical_value(document["spec"])
    except (TypeError, ValueError) as exc:
        raise ExperimentSpecIntegrityError(
            f"invalid persisted experiment specification: {path}") from exc
    if stored_integrity != observed_integrity:
        raise ExperimentSpecIntegrityError(
            "experiment specification document integrity failure: "
            f"stored={stored_integrity}, observed={observed_integrity}")
    try:
        observed = fingerprint_experiment_spec(document["spec"])
    except (TypeError, ValueError, ExperimentSpecIntegrityError) as exc:
        raise ExperimentSpecIntegrityError(
            f"invalid persisted experiment specification: {path}") from exc
    if stored != observed:
        raise ExperimentSpecIntegrityError(
            "experiment specification integrity failure: "
            f"stored={stored}, observed={observed}")
    return canonical_experiment_spec(document["spec"]), stored


def load_persisted_experiment_spec(run_dir: str | Path) -> tuple[dict, str]:
    return _load_experiment_spec_document(
        Path(run_dir) / EXPERIMENT_SPEC_FILE)


def load_legacy_migration_spec(
    run_dir: str | Path,
    *,
    pipeline: str,
    unavailable_fields: Iterable[str],
) -> dict[str, Any]:
    path = Path(run_dir) / LEGACY_MIGRATION_MANIFEST_FILE
    if not path.is_file():
        fields_text = "\n".join(
            f"  {field}" for field in unavailable_fields)
        raise LegacyRunMigrationError(
            "Legacy run cannot be safely migrated.\n\n"
            "Historical identity is unavailable for:\n"
            f"{fields_text}\n\n"
            "The files currently present at external paths cannot prove what "
            "the run originally used.\n\nUse a new output directory, or add "
            f"an independently verified {LEGACY_MIGRATION_MANIFEST_FILE}.")
    spec, _fingerprint = _load_experiment_spec_document(path)
    if spec["pipeline"] != pipeline:
        raise LegacyRunMigrationError(
            f"legacy migration manifest pipeline is {spec['pipeline']!r}, "
            f"expected {pipeline!r}")
    return spec


def persist_experiment_spec(run_dir: str | Path, spec: Mapping[str, Any]) -> str:
    canonical = canonical_experiment_spec(spec)
    fingerprint = fingerprint_experiment_spec(canonical)
    atomic_write_json(Path(run_dir) / EXPERIMENT_SPEC_FILE, {
        "schema_version": EXPERIMENT_SPEC_SCHEMA_VERSION,
        "experiment_fingerprint": fingerprint,
        "spec_integrity_sha256": hash_canonical_value(canonical),
        "spec": canonical,
    })
    return fingerprint


def persist_legacy_migration_manifest(
    run_dir: str | Path, spec: Mapping[str, Any]
) -> str:
    canonical = canonical_experiment_spec(spec)
    fingerprint = fingerprint_experiment_spec(canonical)
    atomic_write_json(
        Path(run_dir) / LEGACY_MIGRATION_MANIFEST_FILE, {
            "schema_version": EXPERIMENT_SPEC_SCHEMA_VERSION,
            "experiment_fingerprint": fingerprint,
            "spec_integrity_sha256": hash_canonical_value(canonical),
            "spec": canonical,
        })
    return fingerprint


def format_identity_mismatch(
    run_dir: str | Path, differences: Iterable[ExperimentDifference]
) -> str:
    lines = [
        f"Cannot resume run {str(Path(run_dir))!r}: experiment identity differs.",
        "",
        "Changed fields:",
    ]
    for difference in differences:
        lines.extend([
            f"  {difference.field} ({difference.category}):",
            f"    persisted: {difference.persisted!r}",
            f"    requested: {difference.requested!r}",
        ])
    lines.extend(["", "Use a new --output-dir for a different experiment."])
    return "\n".join(lines)


def validate_or_initialize_experiment(
    run_dir: str | Path,
    requested_spec: Mapping[str, Any],
    *,
    run_state: Mapping[str, Any] | None,
    legacy_spec: Mapping[str, Any] | None = None,
    extra_legacy_markers: Iterable[str] = (),
) -> tuple[str, bool]:
    """Validate before writes; return ``(fingerprint, initialized_or_migrated)``."""
    root = Path(run_dir)
    spec_path = root / EXPERIMENT_SPEC_FILE
    if spec_path.exists():
        persisted, fingerprint = load_persisted_experiment_spec(root)
        if persisted.get("pipeline") != requested_spec.get("pipeline"):
            raise ExperimentIdentityError(
                f"run pipeline is {persisted.get('pipeline')!r}, requested "
                f"{requested_spec.get('pipeline')!r}")
        state_fingerprint = (run_state or {}).get("experiment_fingerprint")
        if run_state is not None and state_fingerprint != fingerprint:
            raise ExperimentSpecIntegrityError(
                "committed run-state experiment fingerprint does not match "
                f"{EXPERIMENT_SPEC_FILE}")
        differences = compare_experiment_specs(persisted, requested_spec)
        if differences:
            raise ExperimentIdentityError(
                format_identity_mismatch(root, differences))
        return fingerprint, False

    legacy_markers = (
        "run_state.json", "config.json", "best.pt", "last.pt",
        "history.json", "summary.json", "replay", "splits.json",
        *tuple(extra_legacy_markers))
    is_legacy = root.exists() and any(
        (root / marker).exists() for marker in legacy_markers)
    if is_legacy:
        if legacy_spec is None:
            raise LegacyRunMigrationError(
                f"Legacy run cannot be safely migrated: {EXPERIMENT_SPEC_FILE} "
                "is missing and original identity cannot be reconstructed. "
                "Use a new --output-dir or provide an explicit migration manifest.")
        differences = compare_experiment_specs(legacy_spec, requested_spec)
        if differences:
            raise ExperimentIdentityError(
                format_identity_mismatch(root, differences))
        fingerprint = fingerprint_experiment_spec(legacy_spec)
        # Crash-recoverable staged migration (not one all-or-nothing write):
        # publish the state reference first, then the spec. The helper never
        # returns between stages, so no intermediate state can enter training.
        # A retry deterministically reconstructs the same fingerprint.
        if run_state is not None:
            migrated_state = dict(run_state)
            migrated_state["experiment_fingerprint"] = fingerprint
            atomic_write_json(root / "run_state.json", migrated_state)
        persisted_fingerprint = persist_experiment_spec(root, legacy_spec)
        if persisted_fingerprint != fingerprint:
            raise ExperimentSpecIntegrityError(
                "legacy migration produced an inconsistent fingerprint")
        return fingerprint, True

    fingerprint = persist_experiment_spec(root, requested_spec)
    return fingerprint, True
