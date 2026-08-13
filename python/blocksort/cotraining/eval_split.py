"""Persisted, training-seed-independent held-out evaluation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..training.transaction import atomic_write_json

EVAL_SPLIT_SCHEMA_VERSION = 1
EVAL_SPLIT_ALGORITHM = "stable_signature_sha256_rank_v1"
DEFAULT_EVAL_SPLIT_SEED = 1729
HARD_EVAL_POOL_SCHEMA_VERSION = 1


class EvaluationSplitError(ValueError):
    """Held-out pool or split manifest is missing, inconsistent, or corrupted."""


def _hash_canonical_value(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_eval_pool_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash canonical pool records independent of their JSONL line order."""
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: record["static_level_signature"])
    return _hash_canonical_value(ordered)


def _read_pool(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"held-out evaluation pool is missing: {source}")
    by_signature: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationSplitError(
                    f"{source}: line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise EvaluationSplitError(
                    f"{source}: line {line_no}: record must be an object")
            level_data = record.get("level")
            if not isinstance(level_data, dict):
                raise EvaluationSplitError(
                    f"{source}: line {line_no}: missing or invalid level")
            try:
                level = level_from_dict(level_data)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise EvaluationSplitError(
                    f"{source}: line {line_no}: invalid level: {exc}") from exc
            signature = static_level_signature(level)
            declared = record.get("static_level_signature")
            if declared is not None and declared != signature:
                raise EvaluationSplitError(
                    f"{source}: line {line_no}: static_level_signature does "
                    "not match the level")
            schema = record.get("hard_eval_pool_schema_version")
            if schema is not None:
                _validate_hard_pool_record(
                    record, source=source, line_no=line_no,
                    expected_level_hash=_hash_canonical_value(
                        level_to_dict(level)))
            if signature in by_signature:
                raise EvaluationSplitError(
                    f"{source}: duplicate held-out signature {signature!r} "
                    f"on line {line_no}")
            canonical = dict(record)
            canonical["static_level_signature"] = signature
            canonical.setdefault("level_id", signature)
            by_signature[signature] = canonical

    records = [by_signature[sig] for sig in sorted(by_signature)]
    if len(records) < 2:
        raise EvaluationSplitError(
            "held-out evaluation pool must contain at least two unique levels")
    pool_hash = canonical_eval_pool_sha256(records)
    return records, pool_hash


def _validate_hard_pool_record(
    record: Mapping[str, Any], *, source: Path, line_no: int,
    expected_level_hash: str,
) -> None:
    if record.get("hard_eval_pool_schema_version") != \
            HARD_EVAL_POOL_SCHEMA_VERSION:
        raise EvaluationSplitError(
            f"{source}: line {line_no}: unsupported hard eval pool schema")
    for field in (
            "generation_bucket", "generation_seed", "generation_parameters",
            "oracle_validation", "canonical_level_sha256"):
        if field not in record:
            raise EvaluationSplitError(
                f"{source}: line {line_no}: hard eval record missing {field}")
    if not isinstance(record["generation_bucket"], str) \
            or not record["generation_bucket"]:
        raise EvaluationSplitError(
            f"{source}: line {line_no}: generation_bucket must be a string")
    if isinstance(record["generation_seed"], bool) \
            or not isinstance(record["generation_seed"], int):
        raise EvaluationSplitError(
            f"{source}: line {line_no}: generation_seed must be an integer")
    if not isinstance(record["generation_parameters"], dict):
        raise EvaluationSplitError(
            f"{source}: line {line_no}: generation_parameters must be an object")
    if record["generation_bucket"] == "adversarial_designer_hard":
        parameters = record["generation_parameters"]
        if parameters.get("generation_method") != "trained_designer":
            raise EvaluationSplitError(
                f"{source}: line {line_no}: adversarial designer records must "
                "declare generation_method=trained_designer")
        checkpoint_digest = parameters.get("designer_checkpoint_sha256")
        if (not isinstance(checkpoint_digest, str)
                or len(checkpoint_digest) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in checkpoint_digest)):
            raise EvaluationSplitError(
                f"{source}: line {line_no}: adversarial designer records must "
                "include a valid designer checkpoint SHA-256")
    oracle = record["oracle_validation"]
    if not isinstance(oracle, dict):
        raise EvaluationSplitError(
            f"{source}: line {line_no}: oracle_validation must be an object")
    if oracle.get("exact") is not True or oracle.get("solvable") is not True:
        raise EvaluationSplitError(
            f"{source}: line {line_no}: hard eval records must be "
            "oracle-solvable with exact validation")
    digest = record["canonical_level_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)):
        raise EvaluationSplitError(
            f"{source}: line {line_no}: canonical_level_sha256 is invalid")
    if digest != expected_level_hash:
        raise EvaluationSplitError(
            f"{source}: line {line_no}: canonical_level_sha256 does not "
            "match the level")
    protagonist = record.get("protagonist_filter")
    if protagonist is not None and not isinstance(protagonist, dict):
        raise EvaluationSplitError(
            f"{source}: line {line_no}: protagonist_filter must be an object")


def load_eval_pool_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a held-out pool with the same strict validation used by manifests."""
    records, _pool_hash = _read_pool(path)
    return records


def _rank(split_seed: int, signature: str) -> str:
    payload = f"{split_seed}\0{signature}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _membership_fingerprint(
    *,
    pool_sha256: str,
    split_seed: int,
    validation_signatures: Iterable[str],
    test_signatures: Iterable[str],
) -> str:
    return _hash_canonical_value({
        "schema_version": EVAL_SPLIT_SCHEMA_VERSION,
        "split_algorithm": EVAL_SPLIT_ALGORITHM,
        "pool_sha256": pool_sha256,
        "split_seed": split_seed,
        "promotion_validation": sorted(validation_signatures),
        "final_test": sorted(test_signatures),
    })


def _signature_hash(signatures: Iterable[str]) -> str:
    return _hash_canonical_value(sorted(signatures))


def _without_integrity_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: value for key, value in manifest.items()
        if key != "manifest_sha256"}
    payload = json.loads(json.dumps(payload))
    pool = payload.get("pool")
    if isinstance(pool, dict):
        # Informational relocation hint, deliberately outside identity.
        pool.pop("source_path_hint", None)
    return payload


def create_eval_split_manifest(
    pool_path: str | Path,
    output_path: str | Path,
    *,
    validation_count: int,
    split_seed: int = DEFAULT_EVAL_SPLIT_SEED,
) -> dict[str, Any]:
    """Create one immutable split manifest; never overwrite an existing file."""
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise EvaluationSplitError("eval split seed must be an integer")
    records, pool_hash = _read_pool(pool_path)
    total = len(records)
    if (isinstance(validation_count, bool)
            or not isinstance(validation_count, int)
            or not 1 <= validation_count < total):
        raise EvaluationSplitError(
            "validation_count must leave both promotion-validation and "
            f"final-test nonempty; got {validation_count!r} for {total} levels")

    ranked = sorted(
        records,
        key=lambda record: (
            _rank(split_seed, record["static_level_signature"]),
            record["static_level_signature"],
        ),
    )
    validation = ranked[:validation_count]
    test = ranked[validation_count:]

    def entry(record: Mapping[str, Any]) -> dict[str, str]:
        return {
            "level_id": str(record.get("level_id")
                            or record["static_level_signature"]),
            "signature": str(record["static_level_signature"]),
        }

    validation_entries = [entry(record) for record in validation]
    test_entries = [entry(record) for record in test]
    validation_signatures = [item["signature"] for item in validation_entries]
    test_signatures = [item["signature"] for item in test_entries]
    fingerprint = _membership_fingerprint(
        pool_sha256=pool_hash,
        split_seed=split_seed,
        validation_signatures=validation_signatures,
        test_signatures=test_signatures,
    )
    manifest: dict[str, Any] = {
        "schema_version": EVAL_SPLIT_SCHEMA_VERSION,
        "split_algorithm": EVAL_SPLIT_ALGORITHM,
        "pool": {
            "source_path_hint": Path(pool_path).as_posix(),
            "sha256": pool_hash,
            "record_count": total,
            "unique_signature_count": total,
        },
        "split_config": {
            "split_seed": split_seed,
            "validation_count": len(validation_entries),
            "test_count": len(test_entries),
        },
        "promotion_validation": validation_entries,
        "final_test": test_entries,
        "validation_signature_hash": _signature_hash(validation_signatures),
        "test_signature_hash": _signature_hash(test_signatures),
        "evaluation_split_fingerprint": fingerprint,
    }
    manifest["manifest_sha256"] = _hash_canonical_value(
        _without_integrity_hash(manifest))
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"evaluation split manifest already exists: {destination}")
    atomic_write_json(destination, manifest)
    return manifest


def load_eval_split_manifest(
    manifest_path: str | Path,
    pool_path: str | Path,
    *,
    expected_split_seed: int | None = None,
    expected_validation_count: int | None = None,
) -> dict[str, Any]:
    """Load and fully verify integrity, pool identity, and exact membership."""
    source = Path(manifest_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"held-out evaluation split manifest is missing: {source}. "
            "Create it explicitly; it will not be regenerated.")
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationSplitError(
            f"invalid evaluation split manifest JSON: {source}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EvaluationSplitError("evaluation split manifest must be an object")
    if manifest.get("schema_version") != EVAL_SPLIT_SCHEMA_VERSION:
        raise EvaluationSplitError(
            f"unsupported evaluation split schema "
            f"{manifest.get('schema_version')!r}")
    if manifest.get("split_algorithm") != EVAL_SPLIT_ALGORITHM:
        raise EvaluationSplitError(
            f"unsupported evaluation split algorithm "
            f"{manifest.get('split_algorithm')!r}")
    expected_integrity = _hash_canonical_value(_without_integrity_hash(manifest))
    if manifest.get("manifest_sha256") != expected_integrity:
        raise EvaluationSplitError(
            "evaluation split manifest integrity hash mismatch")

    records, pool_hash = _read_pool(pool_path)
    pool = manifest.get("pool")
    config = manifest.get("split_config")
    validation = manifest.get("promotion_validation")
    test = manifest.get("final_test")
    if not isinstance(pool, dict) or not isinstance(config, dict):
        raise EvaluationSplitError("evaluation split manifest metadata is malformed")
    if not isinstance(validation, list) or not isinstance(test, list):
        raise EvaluationSplitError("evaluation split memberships must be lists")
    if pool.get("sha256") != pool_hash:
        raise EvaluationSplitError(
            "held-out pool content hash differs from the split manifest")
    if (pool.get("record_count") != len(records)
            or pool.get("unique_signature_count") != len(records)):
        raise EvaluationSplitError(
            "held-out pool count differs from the split manifest")

    split_seed = config.get("split_seed")
    validation_count = config.get("validation_count")
    test_count = config.get("test_count")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise EvaluationSplitError(
            "evaluation split manifest seed must be an integer")
    if (isinstance(validation_count, bool)
            or not isinstance(validation_count, int)
            or isinstance(test_count, bool)
            or not isinstance(test_count, int)
            or validation_count < 1
            or test_count < 1
            or validation_count + test_count != len(records)):
        raise EvaluationSplitError(
            "evaluation split manifest counts must be positive integers "
            "covering the complete held-out pool")
    if expected_split_seed is not None and split_seed != expected_split_seed:
        raise EvaluationSplitError(
            f"evaluation split seed mismatch: manifest={split_seed!r}, "
            f"configured={expected_split_seed!r}")
    if (expected_validation_count is not None
            and validation_count != expected_validation_count):
        raise EvaluationSplitError(
            "evaluation validation-count mismatch: "
            f"manifest={validation_count!r}, "
            f"configured={expected_validation_count!r}")
    if validation_count != len(validation) or test_count != len(test):
        raise EvaluationSplitError(
            "evaluation split configured counts do not match membership")

    def signatures(entries: list[Any], role: str) -> list[str]:
        result: list[str] = []
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get("signature"), str):
                raise EvaluationSplitError(
                    f"malformed {role} membership entry")
            result.append(item["signature"])
        if len(result) != len(set(result)):
            raise EvaluationSplitError(
                f"duplicate signature within {role} membership")
        return result

    validation_signatures = signatures(validation, "promotion_validation")
    test_signatures = signatures(test, "final_test")
    validation_set = set(validation_signatures)
    test_set = set(test_signatures)
    pool_set = {record["static_level_signature"] for record in records}
    if validation_set & test_set:
        raise EvaluationSplitError(
            "promotion-validation and final-test memberships overlap")
    if validation_set | test_set != pool_set:
        raise EvaluationSplitError(
            "evaluation split membership does not completely cover the pool")

    ranked = sorted(
        pool_set, key=lambda sig: (_rank(split_seed, sig), sig))
    if validation_signatures != ranked[:validation_count] \
            or test_signatures != ranked[validation_count:]:
        raise EvaluationSplitError(
            "evaluation split membership does not match the stable algorithm")
    if manifest.get("validation_signature_hash") != \
            _signature_hash(validation_signatures):
        raise EvaluationSplitError("validation signature hash mismatch")
    if manifest.get("test_signature_hash") != _signature_hash(test_signatures):
        raise EvaluationSplitError("test signature hash mismatch")
    fingerprint = _membership_fingerprint(
        pool_sha256=pool_hash,
        split_seed=split_seed,
        validation_signatures=validation_signatures,
        test_signatures=test_signatures,
    )
    if manifest.get("evaluation_split_fingerprint") != fingerprint:
        raise EvaluationSplitError("evaluation split fingerprint mismatch")
    return manifest


def evaluation_split_identity(
    manifest: Mapping[str, Any],
    *,
    eval_limit: int | None,
    evaluation_semantics_version: int | None = None,
) -> dict[str, Any]:
    """Reusable identity stored in run specs, state, and reports."""
    if evaluation_semantics_version is None:
        from ..training.experiment_identity import EVALUATION_SEMANTICS_VERSION
        evaluation_semantics_version = EVALUATION_SEMANTICS_VERSION
    return {
        "evaluation_split_fingerprint":
            manifest["evaluation_split_fingerprint"],
        "evaluation_split_manifest_sha256": manifest["manifest_sha256"],
        "evaluation_pool_sha256": manifest["pool"]["sha256"],
        "split_algorithm": manifest["split_algorithm"],
        "split_seed": manifest["split_config"]["split_seed"],
        "promotion_validation_count":
            manifest["split_config"]["validation_count"],
        "final_test_count": manifest["split_config"]["test_count"],
        "validation_signature_hash": manifest["validation_signature_hash"],
        "test_signature_hash": manifest["test_signature_hash"],
        "eval_limit": eval_limit,
        "evaluation_semantics_version": evaluation_semantics_version,
    }


def validate_common_evaluation_split(
    run_identities: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject aggregation of runs that used incomparable evaluation roles."""
    identities = list(run_identities)
    if not identities:
        raise EvaluationSplitError(
            "cannot validate an empty set of run evaluation identities")
    first = dict(identities[0])
    fields = (
        "evaluation_split_fingerprint",
        "evaluation_split_manifest_sha256",
        "evaluation_pool_sha256",
        "split_algorithm",
        "split_seed",
        "promotion_validation_count",
        "final_test_count",
        "validation_signature_hash",
        "test_signature_hash",
        "eval_limit",
        "evaluation_semantics_version",
    )
    missing = [field for field in fields if field not in first]
    if missing:
        raise EvaluationSplitError(
            f"run evaluation identity is missing fields: {missing}")
    fingerprint = first["evaluation_split_fingerprint"]
    if not isinstance(fingerprint, str):
        raise EvaluationSplitError(
            "run evaluation identity has invalid evaluation_split_fingerprint")
    for index, identity in enumerate(identities[1:], start=1):
        missing = [field for field in fields if field not in identity]
        if missing:
            raise EvaluationSplitError(
                f"run[{index}] evaluation identity is missing fields: {missing}")
        differences = [
            field for field in fields
            if identity.get(field) != first.get(field)
        ]
        if differences:
            raise EvaluationSplitError(
                "Cannot aggregate runs: evaluation split identities differ. "
                f"run[0]={fingerprint}, "
                f"run[{index}]={identity.get('evaluation_split_fingerprint')}; "
                f"different fields={differences}. All compared seeds must use "
                "the same held-out split manifest and evaluation limits.")
    return first


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one immutable held-out evaluation split manifest.")
    parser.add_argument("--eval-levels-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-count", required=True, type=int)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_EVAL_SPLIT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = create_eval_split_manifest(
        args.eval_levels_dataset,
        args.output,
        validation_count=args.validation_count,
        split_seed=args.split_seed,
    )
    print(json.dumps({
        "manifest": str(args.output),
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_split_fingerprint":
            manifest["evaluation_split_fingerprint"],
        "pool_levels": manifest["pool"]["unique_signature_count"],
        "promotion_validation": manifest["split_config"]["validation_count"],
        "final_test": manifest["split_config"]["test_count"],
        "split_seed": manifest["split_config"]["split_seed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
