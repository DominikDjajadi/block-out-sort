"""Build reproducible, mixed-size level pools for exact supervised training.

The output is a plain JSON list accepted directly by
``blocksort.dataset.generate``.  A sibling report records generation parameters,
frozen-pool exclusions, per-stratum coverage, rejection counts, and identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..designer.config import GeneratorConfig
from ..environment import Environment
from ..level_generation import random_level
from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..training.transaction import atomic_write_json, sha256_file
from ..validation import validate_level


TRAINING_POOL_SCHEMA_VERSION = 1
DEFAULT_STRATA = ((5, 5, 2), (5, 5, 3), (6, 6, 2), (6, 6, 3))


@dataclass(frozen=True)
class TrainingPoolConfig:
    output: str
    exclude_level_files: tuple[str, ...]
    per_stratum_count: int = 25
    seed: int = 2051
    density_min: float = 0.35
    density_max: float = 0.55
    mutation_budget_min: int = 4
    mutation_budget_max: int = 12
    max_attempts_per_level: int = 200
    report: str | None = None
    overwrite: bool = False


def _stratum_key(rows: int, cols: int, color_count: int) -> str:
    return f"{rows}x{cols}_c{color_count}"


def _attempt_seed(global_seed: int, stratum: str, attempt: int) -> int:
    payload = f"{global_seed}:{stratum}:{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _decode_json_or_jsonl(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read excluded level source: {path}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"excluded level source is not valid JSON or JSONL: {path}"
            ) from exc


def _looks_like_level(value: dict[str, Any]) -> bool:
    return all(key in value for key in ("rows", "cols", "blocks", "exits"))


def _extract_signatures(value: Any) -> set[str]:
    """Extract level identities from level files, wrapped records, or manifests."""
    signatures: set[str] = set()
    if isinstance(value, list):
        for item in value:
            signatures.update(_extract_signatures(item))
        return signatures
    if not isinstance(value, dict):
        return signatures

    if _looks_like_level(value):
        try:
            signatures.add(static_level_signature(level_from_dict(value)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("excluded source contains an invalid level") from exc
        return signatures

    for field in ("static_level_signature", "signature"):
        signature = value.get(field)
        if isinstance(signature, str) and signature:
            signatures.add(signature)

    # Recurse so JSONL wrappers (``level``), {"levels": [...]}, exact dataset
    # records, and frozen split manifests are all accepted.
    for child in value.values():
        if isinstance(child, (dict, list)):
            signatures.update(_extract_signatures(child))
    return signatures


def load_excluded_signatures(
    paths: Iterable[str],
) -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    sources: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        decoded = _decode_json_or_jsonl(path)
        signatures = _extract_signatures(decoded)
        if not signatures:
            raise ValueError(
                f"excluded level source contains no level identities: {path}")
        excluded.update(signatures)
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "signature_count": len(signatures),
        })
    return excluded, sources


def _validate_config(cfg: TrainingPoolConfig) -> None:
    for name, value in (
        ("per_stratum_count", cfg.per_stratum_count),
        ("max_attempts_per_level", cfg.max_attempts_per_level),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int):
        raise ValueError("seed must be an integer")
    for name, value in (
        ("density_min", cfg.density_min),
        ("density_max", cfg.density_max),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ValueError(f"{name} must be finite")
    if not 0 < cfg.density_min <= cfg.density_max <= 1:
        raise ValueError("density range must be ordered within (0, 1]")
    if (isinstance(cfg.mutation_budget_min, bool)
            or isinstance(cfg.mutation_budget_max, bool)
            or not isinstance(cfg.mutation_budget_min, int)
            or not isinstance(cfg.mutation_budget_max, int)
            or not 0 <= cfg.mutation_budget_min <= cfg.mutation_budget_max):
        raise ValueError(
            "mutation budget range must contain ordered non-negative integers")
    if not cfg.exclude_level_files:
        raise ValueError(
            "at least one frozen level source is required; pass --exclude-levels")


def _report_path(cfg: TrainingPoolConfig) -> Path:
    output = Path(cfg.output)
    if cfg.report:
        return Path(cfg.report)
    return output.with_name(f"{output.stem}_report.json")


def _prepare_destinations(cfg: TrainingPoolConfig) -> tuple[Path, Path]:
    output, report = Path(cfg.output), _report_path(cfg)
    if output.resolve() == report.resolve():
        raise ValueError("output and report paths must be different")
    if not cfg.overwrite:
        existing = [str(path) for path in (output, report) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing training-pool artifacts: "
                + ", ".join(existing))
    return output, report


def _verify_generated_pool(
    levels: list[dict[str, Any]],
    excluded: set[str],
    per_stratum_count: int,
) -> dict[str, Any]:
    signatures: list[str] = []
    counts = {_stratum_key(*spec): 0 for spec in DEFAULT_STRATA}
    for raw in levels:
        level = level_from_dict(raw)
        errors = validate_level(level)
        if errors:
            raise RuntimeError(f"generated training level is invalid: {errors}")
        signature = static_level_signature(level)
        signatures.append(signature)
        colors = len({block.color for block in level.blocks})
        key = _stratum_key(level.rows, level.cols, colors)
        if key not in counts:
            raise RuntimeError(f"generated level falls outside requested strata: {key}")
        counts[key] += 1
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("generated training pool contains duplicate signatures")
    overlap = sorted(set(signatures) & excluded)
    if overlap:
        raise RuntimeError(
            f"generated training pool overlaps frozen identities: {overlap[:3]}")
    expected = {key: per_stratum_count for key in counts}
    if counts != expected:
        raise RuntimeError(
            f"generated training pool has wrong stratum distribution: {counts}")
    return {
        "valid_levels": True,
        "unique_static_signatures": True,
        "frozen_signature_overlap_count": 0,
        "stratum_counts_match_request": True,
        "observed_stratum_counts": counts,
    }


def generate_training_pool(cfg: TrainingPoolConfig) -> dict[str, Any]:
    """Generate the pool and report, returning the report dictionary."""
    _validate_config(cfg)
    output_path, report_path = _prepare_destinations(cfg)
    excluded, exclusion_sources = load_excluded_signatures(
        cfg.exclude_level_files)
    env = Environment()
    accepted_signatures: set[str] = set()
    output_levels: list[dict[str, Any]] = []
    level_reports: list[dict[str, Any]] = []
    rejection_counts = {
        "generation_failed": 0,
        "wrong_actual_color_count": 0,
        "invalid": 0,
        "frozen_signature": 0,
        "duplicate_signature": 0,
    }
    attempts_by_stratum: dict[str, int] = {}

    for rows, cols, color_count in DEFAULT_STRATA:
        stratum = _stratum_key(rows, cols, color_count)
        accepted = 0
        attempt = 0
        limit = cfg.per_stratum_count * cfg.max_attempts_per_level
        while accepted < cfg.per_stratum_count and attempt < limit:
            seed = _attempt_seed(cfg.seed, stratum, attempt)
            attempt += 1
            rng = random.Random(seed)
            density = rng.uniform(cfg.density_min, cfg.density_max)
            mutation_budget = rng.randint(
                cfg.mutation_budget_min, cfg.mutation_budget_max)
            gen_cfg = GeneratorConfig(
                rows=rows,
                cols=cols,
                color_count=color_count,
                density=density,
            )
            level = random_level(
                env, gen_cfg, rng, reverse_depth=mutation_budget)
            if level is None:
                rejection_counts["generation_failed"] += 1
                continue
            actual_colors = len({block.color for block in level.blocks})
            if actual_colors != color_count:
                rejection_counts["wrong_actual_color_count"] += 1
                continue
            errors = validate_level(level)
            if errors:
                rejection_counts["invalid"] += 1
                continue
            signature = static_level_signature(level)
            if signature in excluded:
                rejection_counts["frozen_signature"] += 1
                continue
            if signature in accepted_signatures:
                rejection_counts["duplicate_signature"] += 1
                continue

            name = f"mixed_exact_{stratum}_{accepted:04d}_{signature[:10]}"
            raw = level_to_dict(level)
            raw["name"] = name
            output_levels.append(raw)
            accepted_signatures.add(signature)
            occupied_cells = sum(len(block.cells) for block in level.blocks)
            level_reports.append({
                "name": name,
                "static_level_signature": signature,
                "stratum": stratum,
                "generation_seed": seed,
                "requested_density": density,
                "actual_cell_density": occupied_cells / (rows * cols),
                "mutation_budget": mutation_budget,
                "block_count": len(level.blocks),
                "actual_color_count": actual_colors,
            })
            accepted += 1
        attempts_by_stratum[stratum] = attempt
        if accepted < cfg.per_stratum_count:
            raise RuntimeError(
                f"could not generate {cfg.per_stratum_count} unique levels for "
                f"{stratum} after {attempt} attempts")

    verification = _verify_generated_pool(
        output_levels, excluded, cfg.per_stratum_count)
    atomic_write_json(output_path, output_levels)
    report = {
        "training_pool_schema_version": TRAINING_POOL_SCHEMA_VERSION,
        "generation_method": "random_reverse_construction",
        "solvability_guarantee": "construction_proven",
        "config": {
            **asdict(cfg),
            "exclude_level_files": list(cfg.exclude_level_files),
            "strata": [
                {"rows": rows, "cols": cols, "color_count": colors}
                for rows, cols, colors in DEFAULT_STRATA
            ],
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "level_count": len(output_levels),
            "unique_static_signature_count": len(accepted_signatures),
        },
        "exclusions": {
            "mandatory": True,
            "sources": exclusion_sources,
            "unique_signature_count": len(excluded),
        },
        "attempts_by_stratum": attempts_by_stratum,
        "rejections": rejection_counts,
        "verification": verification,
        "levels": level_reports,
    }
    atomic_write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced 5x5/6x6, two-/three-color level pool for "
            "exact supervised training."))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-levels",
        action="append",
        required=True,
        dest="exclude_level_files",
        help=(
            "frozen level JSON/JSONL or split manifest to exclude; repeat for "
            "every validation/final-test source"),
    )
    parser.add_argument("--per-stratum-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2051)
    parser.add_argument("--density-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"), default=(0.35, 0.55))
    parser.add_argument("--mutation-budget-range", type=int, nargs=2,
                        metavar=("MIN", "MAX"), default=(4, 12))
    parser.add_argument("--max-attempts-per-level", type=int, default=200)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainingPoolConfig(
        output=args.output,
        exclude_level_files=tuple(args.exclude_level_files),
        per_stratum_count=args.per_stratum_count,
        seed=args.seed,
        density_min=args.density_range[0],
        density_max=args.density_range[1],
        mutation_budget_min=args.mutation_budget_range[0],
        mutation_budget_max=args.mutation_budget_range[1],
        max_attempts_per_level=args.max_attempts_per_level,
        report=args.report,
        overwrite=args.overwrite,
    )
    report = generate_training_pool(cfg)
    print(json.dumps({
        "output": report["output"],
        "report": str(_report_path(cfg)),
        "verification": report["verification"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
