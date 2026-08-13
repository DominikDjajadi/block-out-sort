"""Safely combine policy-value JSONL datasets.

Records are deduplicated by static-level signature plus canonical state key.
A full-successor exact record supersedes an exact-path-policy record for the
same state.  Conflicting exact values or same-strength labels are rejected.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from ..training.transaction import atomic_write_json, atomic_write_text, sha256_file
from .schema import LABEL_EXACT_PATH_POLICY, LABEL_FULL_EXACT


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path} line {line_number} must contain a JSON object")
            records.append(record)
    return records


def _identity(record: dict[str, Any]) -> tuple[str, str]:
    signature = record.get("static_level_signature")
    state_key = record.get("state_key")
    if not isinstance(signature, str) or not isinstance(state_key, str):
        raise ValueError("record is missing a string signature/state identity")
    return signature, state_key


def _label_kind(record: dict[str, Any]) -> str:
    kind = record.get("label_kind", LABEL_FULL_EXACT)
    if kind not in (LABEL_FULL_EXACT, LABEL_EXACT_PATH_POLICY):
        raise ValueError(f"unsupported exact label kind: {kind!r}")
    return kind


def _strength(record: dict[str, Any]) -> int:
    return 2 if _label_kind(record) == LABEL_FULL_EXACT else 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _merge_provenance(target: dict[str, Any], other: dict[str, Any]) -> None:
    merged: list[Any] = []
    seen: set[str] = set()
    for entry in list(target.get("provenance") or []) + list(
            other.get("provenance") or []):
        key = _canonical(entry)
        if key not in seen:
            seen.add(key)
            merged.append(copy.deepcopy(entry))
    target["provenance"] = merged


def _assert_compatible(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> None:
    if existing.get("optimal_remaining_moves") != incoming.get(
            "optimal_remaining_moves"):
        raise ValueError(
            f"conflicting exact root values for state {_identity(existing)}")
    if _strength(existing) != _strength(incoming):
        return
    fields = (
        "legal_actions", "optimal_actions", "action_costs", "action_regrets",
        "policy", "policy_target", "value_target",
    )
    disagreements = [
        field for field in fields
        if _canonical(existing.get(field)) != _canonical(incoming.get(field))
    ]
    if disagreements:
        raise ValueError(
            f"conflicting {_label_kind(existing)} labels for state "
            f"{_identity(existing)}: {', '.join(disagreements)}")


def combine_records(
    inputs: list[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministically deduplicated records and merge statistics."""
    if not inputs:
        raise ValueError("at least one input dataset is required")
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    input_rows = 0
    duplicates = 0
    stronger_replacements = 0
    source_rows: dict[str, int] = {}
    for raw_path in inputs:
        path = Path(raw_path)
        records = _read_jsonl(path)
        source_rows[str(path)] = len(records)
        input_rows += len(records)
        for incoming in records:
            identity = _identity(incoming)
            _label_kind(incoming)
            existing = by_identity.get(identity)
            if existing is None:
                by_identity[identity] = copy.deepcopy(incoming)
                continue
            duplicates += 1
            _assert_compatible(existing, incoming)
            if _strength(incoming) > _strength(existing):
                replacement = copy.deepcopy(incoming)
                _merge_provenance(replacement, existing)
                by_identity[identity] = replacement
                stronger_replacements += 1
            else:
                _merge_provenance(existing, incoming)

    records = list(by_identity.values())
    label_counts = Counter(_label_kind(record) for record in records)
    summary = {
        "input_rows": input_rows,
        "output_records": len(records),
        "duplicates": duplicates,
        "stronger_replacements": stronger_replacements,
        "source_rows": source_rows,
        "label_kinds": dict(sorted(label_counts.items())),
    }
    return records, summary


def write_combined_dataset(
    inputs: list[str | Path],
    output: str | Path,
    *,
    report: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    report_path = (
        Path(report) if report is not None
        else destination.with_name(f"{destination.stem}_report.json"))
    if destination.resolve() in {Path(path).resolve() for path in inputs}:
        raise ValueError("combined output cannot overwrite an input dataset")
    if not overwrite and (destination.exists() or report_path.exists()):
        raise FileExistsError(
            "combined output/report already exists; pass --overwrite to replace")

    records, summary = combine_records(inputs)
    text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records)
    atomic_write_text(destination, text)
    source_descriptors = [
        {
            "path": str(Path(path)),
            "sha256": sha256_file(path),
            "records": summary["source_rows"][str(Path(path))],
        }
        for path in inputs
    ]
    report_document = {
        "schema_version": 1,
        "sources": source_descriptors,
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "records": len(records),
        },
        **summary,
    }
    atomic_write_json(report_path, report_document)
    return {**report_document, "report": str(report_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely combine and deduplicate exact policy-value datasets")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_combined_dataset(
        args.inputs, args.output, report=args.report, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
