"""Candidate-blind, frontier-stratified promotion-pool generation.

The official pool is generated from a frozen JSON manifest.  Only the champion
is loaded for difficulty stratification; a candidate checkpoint is neither a
configuration field nor a runtime input.  Generation is resumable and the
completed JSONL remains compatible with the existing immutable split contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..environment import Environment
from ..oracle import Oracle
from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..training.transaction import (
    atomic_write_json, atomic_write_text, sha256_file)
from .eval_split import (
    HARD_EVAL_POOL_SCHEMA_VERSION, canonical_eval_pool_sha256)
from .hard_eval_pool import (
    _canonical_sha256, _designer_level, _level_items_from_file,
    _load_designer_generator)


EXCLUSION_MANIFEST_SCHEMA_VERSION = 1
GENERATION_MANIFEST_SCHEMA_VERSION = 1
PARTIAL_SCHEMA_VERSION = 1
GENERATION_SEMANTICS = "candidate_blind_frontier_stratified_pool_v1"

STRATA = (
    "solved_by_20",
    "first_solved_by_34",
    "first_solved_by_57",
    "first_solved_by_95",
    "first_solved_by_160_or_unsolved_through_160",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_frontier_stratum(
    budgets: Iterable[int], solved: Iterable[bool]
) -> str:
    """Classify by the smallest solved budget, with the tail merged."""
    budget_list = tuple(int(value) for value in budgets)
    solved_list = tuple(bool(value) for value in solved)
    if budget_list != (20, 34, 57, 95, 160):
        raise ValueError("frontier strata require budgets [20, 34, 57, 95, 160]")
    if len(solved_list) != len(budget_list):
        raise ValueError("one solved flag is required per budget")
    if solved_list[0]:
        return STRATA[0]
    if solved_list[1]:
        return STRATA[1]
    if solved_list[2]:
        return STRATA[2]
    if solved_list[3]:
        return STRATA[3]
    return STRATA[4]


def _resolve_globs(patterns: Iterable[str]) -> list[Path]:
    import glob

    paths: dict[str, Path] = {}
    for pattern in patterns:
        matches = [Path(value).resolve() for value in glob.glob(
            pattern, recursive=True)]
        if not matches:
            raise ValueError(f"exclusion source glob matched no files: {pattern}")
        for path in matches:
            if path.is_file():
                paths[str(path).casefold()] = path
    if not paths:
        raise ValueError("no exclusion source files were resolved")
    return sorted(paths.values(), key=lambda value: value.as_posix().casefold())


def build_exclusion_manifest(
    *, source_globs: Iterable[str], output_path: str | Path
) -> dict[str, Any]:
    """Freeze exact source hashes and their aggregate static signatures."""
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"exclusion manifest already exists: {destination}")
    sources = []
    signatures: set[str] = set()
    for path in _resolve_globs(source_globs):
        if path.stat().st_size == 0:
            items = []
        else:
            items = _level_items_from_file(path)
        source_signatures = {
            static_level_signature(level_from_dict(item)) for item in items}
        signatures.update(source_signatures)
        sources.append({
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "record_count": len(items),
            "unique_static_signature_count": len(source_signatures),
        })
    ordered = sorted(signatures)
    manifest: dict[str, Any] = {
        "schema_version": EXCLUSION_MANIFEST_SCHEMA_VERSION,
        "semantics": "frozen_level_signature_exclusions_v1",
        "source_count": len(sources),
        "sources": sources,
        "unique_static_signature_count": len(ordered),
        "static_signatures": ordered,
        "static_signature_set_sha256": _canonical_hash(ordered),
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    atomic_write_json(destination, manifest)
    return manifest


def load_exclusion_manifest(path: str | Path) -> tuple[dict[str, Any], set[str]]:
    source = Path(path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXCLUSION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported exclusion manifest schema")
    stored_hash = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items()
               if key != "manifest_sha256"}
    if stored_hash != _canonical_hash(payload):
        raise ValueError("exclusion manifest integrity hash mismatch")
    signatures = manifest.get("static_signatures")
    if (not isinstance(signatures, list)
            or any(not isinstance(value, str) for value in signatures)
            or signatures != sorted(set(signatures))):
        raise ValueError("exclusion manifest signatures are invalid")
    if manifest.get("unique_static_signature_count") != len(signatures):
        raise ValueError("exclusion manifest signature count mismatch")
    if manifest.get("static_signature_set_sha256") != _canonical_hash(signatures):
        raise ValueError("exclusion manifest signature-set hash mismatch")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("exclusion manifest sources are invalid")
    for item in sources:
        if not isinstance(item, dict) or not Path(item.get("path", "")).is_file():
            raise ValueError("an exclusion source is missing")
        if sha256_file(item["path"]) != item.get("sha256"):
            raise ValueError(f"exclusion source hash mismatch: {item['path']}")
    return manifest, set(signatures)


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_generation_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frontier pool generation manifest schema")
    if manifest.get("semantics") != GENERATION_SEMANTICS:
        raise ValueError("unsupported frontier pool generation semantics")
    stored = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items()
               if key != "manifest_sha256"}
    if stored != _canonical_hash(payload):
        raise ValueError("generation manifest integrity hash mismatch")
    forbidden_candidate_inputs = {
        "candidate", "candidate_checkpoint", "learner",
        "learner_checkpoint"}
    if forbidden_candidate_inputs.intersection(
            key.casefold() for key in manifest):
        raise ValueError("candidate inputs are forbidden in pool generation")
    budgets = manifest.get("budgets")
    if budgets != [20, 34, 57, 95, 160]:
        raise ValueError("generation manifest has the wrong frontier budgets")
    targets = manifest.get("target_counts")
    if (not isinstance(targets, dict) or set(targets) != set(STRATA)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in targets.values())):
        raise ValueError("generation manifest target counts are invalid")
    for field in ("preregistration", "champion_checkpoint",
                  "designer_checkpoint", "exclusion_manifest",
                  "generation_implementation"):
        item = manifest.get(field)
        if not isinstance(item, dict):
            raise ValueError(f"generation manifest is missing {field}")
        resolved = _resolve_manifest_path(source, item.get("path", ""))
        if not resolved.is_file() or sha256_file(resolved) != item.get("sha256"):
            raise ValueError(f"generation manifest input mismatch: {field}")
        item["resolved_path"] = str(resolved)
    output = manifest.get("output")
    if not isinstance(output, str) or not output:
        raise ValueError("generation manifest output is invalid")
    manifest["resolved_output"] = str(_resolve_manifest_path(source, output))
    manifest["source_path"] = str(source)
    return manifest


def _generation_identity(manifest: Mapping[str, Any]) -> str:
    semantic = json.loads(json.dumps({
        key: value for key, value in manifest.items()
        if key not in ("manifest_sha256", "resolved_output", "source_path")}
    ))
    for item in semantic.values():
        if isinstance(item, dict):
            item.pop("resolved_path", None)
    return _canonical_hash(semantic)


def _load_champion(manifest: Mapping[str, Any]):
    import torch

    from ..designer.roles import Protagonist
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    device = torch.device(manifest.get("device", "cuda"))
    checkpoint_path = manifest["champion_checkpoint"]["resolved_path"]
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    encoding, _model_cfg, value_norm = configs_from_checkpoint(checkpoint)
    model = model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    return Protagonist(
        Environment(), model, encoding, value_norm, device,
        simulations=max(manifest["budgets"]), c_puct=manifest["c_puct"],
        temperature=0.0), model


def _partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial.json")


def _validate_partial(
    partial: Mapping[str, Any], *, identity: str,
    targets: Mapping[str, int], excluded: set[str]
) -> tuple[list[dict[str, Any]], int, Counter]:
    if partial.get("schema_version") != PARTIAL_SCHEMA_VERSION:
        raise ValueError("unsupported frontier pool partial schema")
    if partial.get("generation_identity_sha256") != identity:
        raise ValueError("frontier pool partial belongs to different settings")
    records = partial.get("records")
    if not isinstance(records, list):
        raise ValueError("frontier pool partial records are invalid")
    seen: set[str] = set()
    counts: Counter = Counter()
    for record in records:
        level = level_from_dict(record["level"])
        signature = static_level_signature(level)
        if signature != record.get("static_level_signature"):
            raise ValueError("frontier pool partial signature mismatch")
        if signature in seen or signature in excluded:
            raise ValueError("frontier pool partial contains forbidden overlap")
        seen.add(signature)
        stratum = record.get("protagonist_filter", {}).get("difficulty_stratum")
        if stratum not in targets:
            raise ValueError("frontier pool partial contains an invalid stratum")
        counts[stratum] += 1
        if counts[stratum] > targets[stratum]:
            raise ValueError("frontier pool partial exceeds a stratum target")
    attempts = partial.get("attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        raise ValueError("frontier pool partial attempts are invalid")
    return list(records), attempts, counts


def generate_frontier_promotion_pool(manifest_path: str | Path) -> dict[str, Any]:
    """Generate the frozen pool without ever accepting a candidate input."""
    manifest = load_generation_manifest(manifest_path)
    output = Path(manifest["resolved_output"])
    if output.exists():
        raise FileExistsError(f"frontier promotion pool already exists: {output}")
    partial_path = _partial_path(output)
    exclusion_manifest, excluded = load_exclusion_manifest(
        manifest["exclusion_manifest"]["resolved_path"])
    targets = {key: int(value)
               for key, value in manifest["target_counts"].items()}
    identity = _generation_identity(manifest)
    env = Environment()
    oracle_cfg = manifest["oracle"]
    oracle = Oracle(
        env, max_nodes=int(oracle_cfg["max_nodes"]),
        time_limit_seconds=oracle_cfg["time_limit_seconds"])
    protagonist, champion_model = _load_champion(manifest)
    designer_cfg = type("DesignerConfig", (), {
        "designer_checkpoint":
            manifest["designer_checkpoint"]["resolved_path"],
        "device": manifest.get("device", "cuda"),
    })()
    designer = _load_designer_generator(designer_cfg, required=True)

    rng = random.Random(int(manifest["generation_seed"]))
    records: list[dict[str, Any]] = []
    attempts = 0
    counts: Counter = Counter()
    counters: Counter = Counter()
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        records, attempts, counts = _validate_partial(
            partial, identity=identity, targets=targets, excluded=excluded)
        counters.update(partial.get("counters", {}))
        for _ in range(attempts):
            rng.randrange(2**63)
        print(
            f"resuming frontier pool at {len(records)}/{sum(targets.values())} "
            f"after {attempts} attempts", flush=True)

    seen = {record["static_level_signature"] for record in records}

    def save_partial() -> None:
        atomic_write_json(partial_path, {
            "schema_version": PARTIAL_SCHEMA_VERSION,
            "generation_identity_sha256": identity,
            "attempts": attempts,
            "counts": dict(counts),
            "counters": dict(counters),
            "records": records,
        })

    generation_cfg = manifest["generator"]
    from ..designer.config import GeneratorConfig
    from ..search.seeding import derive_trial_seed, level_search_identity

    gen_cfg = GeneratorConfig(
        rows=int(generation_cfg["rows"]),
        cols=int(generation_cfg["cols"]),
        color_count=int(generation_cfg["color_count"]),
        density=float(generation_cfg["density"]))
    target_total = sum(targets.values())
    max_attempts = int(manifest["max_attempts"])
    checkpoint_every = int(manifest["checkpoint_every_attempts"])
    try:
        while len(records) < target_total and attempts < max_attempts:
            attempts += 1
            generation_seed = rng.randrange(2**63)
            local_rng = random.Random(generation_seed)
            level = _designer_level(
                designer, gen_cfg,
                mutation_budget=int(generation_cfg["mutation_budget"]),
                generation_seed=generation_seed, rng=local_rng)
            if level is None:
                counters["invalid"] += 1
                if attempts % checkpoint_every == 0:
                    save_partial()
                continue
            signature = static_level_signature(level)
            if signature in excluded:
                counters["excluded"] += 1
                if attempts % checkpoint_every == 0:
                    save_partial()
                continue
            if signature in seen:
                counters["duplicate"] += 1
                if attempts % checkpoint_every == 0:
                    save_partial()
                continue
            identity_value = level_search_identity(env, level)
            solved = []
            costs = []
            for index, budget in enumerate(manifest["budgets"]):
                seed = derive_trial_seed(
                    int(manifest["screening_seed"]), trial_index=index,
                    level_identity=identity_value,
                    evaluation_context=(
                        f"promotion_pool_screen_budget={budget}"))
                outcome = protagonist.solve(
                    level, seed=seed, simulations=int(budget))
                solved.append(bool(outcome.solved))
                costs.append(outcome.cost)
            stratum = classify_frontier_stratum(manifest["budgets"], solved)
            counters[f"screened_{stratum}"] += 1
            if counts[stratum] >= targets[stratum]:
                counters["stratum_full"] += 1
                if attempts % checkpoint_every == 0:
                    save_partial()
                continue
            value = oracle.value(env.initial_state(level))
            if not (value.exact and value.solvable):
                counters["oracle_rejected"] += 1
                if attempts % checkpoint_every == 0:
                    save_partial()
                continue
            level_data = level_to_dict(level)
            record = {
                "hard_eval_pool_schema_version":
                    HARD_EVAL_POOL_SCHEMA_VERSION,
                "level_id": signature,
                "static_level_signature": signature,
                "canonical_level_sha256": _canonical_sha256(level_data),
                "generation_bucket": "adversarial_designer_hard",
                "generation_seed": generation_seed,
                "generation_parameters": {
                    "generation_method": "trained_designer",
                    "global_seed": int(manifest["generation_seed"]),
                    "rows": gen_cfg.rows,
                    "cols": gen_cfg.cols,
                    "color_count": gen_cfg.color_count,
                    "density": gen_cfg.density,
                    "mutation_budget":
                        int(generation_cfg["mutation_budget"]),
                    "designer_checkpoint":
                        manifest["designer_checkpoint"]["path"],
                    "designer_checkpoint_sha256":
                        manifest["designer_checkpoint"]["sha256"],
                    "generation_manifest_sha256":
                        manifest["manifest_sha256"],
                },
                "oracle_validation": {
                    "max_nodes": int(oracle_cfg["max_nodes"]),
                    "time_limit_seconds": oracle_cfg["time_limit_seconds"],
                    "exact": value.exact,
                    "solvable": value.solvable,
                    "optimal_remaining_moves": value.value,
                },
                "protagonist_filter": {
                    "enabled": True,
                    "checkpoint": manifest["champion_checkpoint"]["path"],
                    "checkpoint_sha256":
                        manifest["champion_checkpoint"]["sha256"],
                    "budgets": list(manifest["budgets"]),
                    "solved": solved,
                    "costs": costs,
                    "difficulty_stratum": stratum,
                    "screening_seed": int(manifest["screening_seed"]),
                    "retained": True,
                },
                "level": level_data,
            }
            records.append(record)
            seen.add(signature)
            counts[stratum] += 1
            save_partial()
            print(
                f"frontier pool: {len(records)}/{target_total} accepted; "
                f"{stratum}={counts[stratum]}/{targets[stratum]}; "
                f"attempts={attempts}", flush=True)
        if len(records) != target_total:
            save_partial()
            raise RuntimeError(
                f"frontier promotion pool infeasible under frozen recipe: "
                f"accepted {len(records)}/{target_total} after {attempts}/"
                f"{max_attempts} attempts; counts={dict(counts)}")
    except BaseException:
        save_partial()
        raise

    records.sort(key=lambda item: (
        STRATA.index(item["protagonist_filter"]["difficulty_stratum"]),
        item["static_level_signature"]))
    text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records)
    atomic_write_text(output, text)
    if partial_path.exists():
        partial_path.unlink()
    del champion_model
    return {
        "output": str(output),
        "record_count": len(records),
        "counts": dict(counts),
        "attempts": attempts,
        "counters": dict(counters),
        "canonical_pool_sha256": canonical_eval_pool_sha256(records),
        "file_sha256": sha256_file(output),
        "generation_manifest_sha256": manifest["manifest_sha256"],
        "exclusion_manifest_sha256":
            exclusion_manifest["manifest_sha256"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    exclusions = subparsers.add_parser("build-exclusions")
    exclusions.add_argument("--source-glob", action="append", required=True)
    exclusions.add_argument("--output", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-exclusions":
        result = build_exclusion_manifest(
            source_globs=args.source_glob, output_path=args.output)
    else:
        result = generate_frontier_promotion_pool(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
