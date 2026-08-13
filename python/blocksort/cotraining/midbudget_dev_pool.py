"""Candidate-blind generation for a fresh exploratory mid-budget dev pool."""

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
from ..training.transaction import atomic_write_json, atomic_write_text, sha256_file
from .eval_split import HARD_EVAL_POOL_SCHEMA_VERSION, canonical_eval_pool_sha256
from .hard_eval_pool import (
    _canonical_sha256,
    _designer_level,
    _level_items_from_file,
    _load_designer_generator,
)


SCHEMA_VERSION = 1
SEMANTICS = "candidate_blind_midbudget_dev_pool_v1"
EXCLUSION_SEMANTICS = "expanded_midbudget_dev_exclusions_v1"
SCREENING_BUDGETS = (64, 72, 80, 88, 95, 104, 112, 120, 128)
STRATA = (
    "solved_by_64",
    "first_solved_72_through_88",
    "first_solved_95_through_112",
    "first_solved_120_or_later_or_unsolved",
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def classify_stratum(solved: Iterable[bool]) -> str:
    values = tuple(bool(value) for value in solved)
    if len(values) != len(SCREENING_BUDGETS):
        raise ValueError("one solve flag is required per screening budget")
    if values[0]:
        return STRATA[0]
    if any(values[1:4]):
        return STRATA[1]
    if any(values[4:7]):
        return STRATA[2]
    return STRATA[3]


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _validated_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    stored = document.get("manifest_sha256")
    payload = {key: value for key, value in document.items()
               if key != "manifest_sha256"}
    if stored != _canonical_hash(payload):
        raise ValueError(f"manifest integrity mismatch: {path}")
    return document


def build_expanded_exclusions(
    *, recipe_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    recipe_source = Path(recipe_path).resolve()
    recipe = json.loads(recipe_source.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != SCHEMA_VERSION \
            or recipe.get("semantics") != EXCLUSION_SEMANTICS \
            or recipe.get("status") != "frozen":
        raise ValueError("unsupported exclusion recipe")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"expanded exclusion manifest exists: {output}")
    base_item = recipe["base_exclusion_manifest"]
    base_path = _resolve(recipe_source, base_item["path"])
    if sha256_file(base_path) != base_item["sha256"]:
        raise ValueError("base exclusion manifest file mismatch")
    base = _validated_document(base_path)
    signatures = set(base["static_signatures"])
    sources = [{
        "kind": "imported_signature_manifest",
        "path": str(base_path),
        "sha256": base_item["sha256"],
        "unique_static_signature_count": len(signatures),
    }]
    for item in recipe["additional_sources"]:
        path = _resolve(recipe_source, item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"additional exclusion source mismatch: {path}")
        levels = _level_items_from_file(path)
        observed = {
            static_level_signature(level_from_dict(level)) for level in levels}
        signatures.update(observed)
        sources.append({
            "kind": "level_source",
            "path": str(path),
            "sha256": item["sha256"],
            "record_count": len(levels),
            "unique_static_signature_count": len(observed),
        })
    ordered = sorted(signatures)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "semantics": EXCLUSION_SEMANTICS,
        "recipe_path": str(recipe_source),
        "recipe_sha256": sha256_file(recipe_source),
        "sources": sources,
        "source_count": len(sources),
        "unique_static_signature_count": len(ordered),
        "static_signatures": ordered,
        "static_signature_set_sha256": _canonical_hash(ordered),
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    atomic_write_json(output, manifest)
    return manifest


def _contains_candidate_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if "candidate" in lowered or "learner" in lowered:
                return True
            if _contains_candidate_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_candidate_key(item) for item in value)
    return False


def _load_contract(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    contract = json.loads(source.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA_VERSION \
            or contract.get("contract_id") != "midbudget_dev_pool_v1" \
            or contract.get("status") != "frozen_before_pool_generation":
        raise ValueError("unsupported mid-budget pool contract")
    if _contains_candidate_key(contract):
        raise ValueError("candidate and learner inputs are forbidden")
    if contract["screening"]["budgets"] != list(SCREENING_BUDGETS):
        raise ValueError("screening budgets differ from implementation")
    targets = contract["target_counts"]
    if set(targets) != set(STRATA) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in targets.values()):
        raise ValueError("invalid target counts")
    for name in ("baseline_checkpoint", "designer_checkpoint",
                 "exclusion_manifest", "implementation"):
        item = contract[name]
        resolved = _resolve(source, item["path"])
        if not resolved.is_file() or sha256_file(resolved) != item["sha256"]:
            raise ValueError(f"pool contract input mismatch: {name}")
        item["resolved_path"] = str(resolved)
    if Path(contract["implementation"]["resolved_path"]) != \
            Path(__file__).resolve():
        raise ValueError("pool implementation path mismatch")
    contract["source_path"] = str(source)
    contract["resolved_output"] = str(_resolve(source, contract["output"]))
    return source, contract


def _load_exclusions(path: str | Path) -> tuple[dict[str, Any], set[str]]:
    manifest = _validated_document(Path(path))
    if manifest.get("semantics") != EXCLUSION_SEMANTICS:
        raise ValueError("wrong exclusion semantics")
    values = manifest["static_signatures"]
    if values != sorted(set(values)) \
            or manifest["static_signature_set_sha256"] != _canonical_hash(values):
        raise ValueError("invalid exclusion signature set")
    return manifest, set(values)


def _load_baseline(contract: Mapping[str, Any]):
    import torch

    from ..designer.roles import Protagonist
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    device = torch.device(contract.get("device", "cuda"))
    checkpoint = load_checkpoint(
        contract["baseline_checkpoint"]["resolved_path"], map_location="cpu")
    encoding, _model_config, value_norm = configs_from_checkpoint(checkpoint)
    model = model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    return Protagonist(
        Environment(), model, encoding, value_norm, device,
        simulations=max(SCREENING_BUDGETS),
        c_puct=contract["screening"]["c_puct"], temperature=0.0), model


def _partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial.json")


def generate_pool(contract_path: str | Path) -> dict[str, Any]:
    contract_source, contract = _load_contract(contract_path)
    output = Path(contract["resolved_output"])
    if output.exists():
        raise FileExistsError(f"development pool already exists: {output}")
    exclusion, excluded = _load_exclusions(
        contract["exclusion_manifest"]["resolved_path"])
    identity = _canonical_hash({
        "contract_sha256": sha256_file(contract_source),
        "implementation_sha256": sha256_file(Path(__file__)),
    })
    targets = {key: int(value) for key, value in contract["target_counts"].items()}
    records: list[dict[str, Any]] = []
    attempts = 0
    counts: Counter = Counter()
    counters: Counter = Counter()
    partial_path = _partial_path(output)
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("generation_identity") != identity:
            raise ValueError("incompatible development-pool partial")
        records = partial["records"]
        attempts = int(partial["attempts"])
        counts.update(partial.get("counts", {}))
        counters.update(partial.get("counters", {}))
    seen = {record["static_level_signature"] for record in records}
    if seen.intersection(excluded) or len(seen) != len(records):
        raise ValueError("development-pool partial has overlap or duplicates")

    def save_partial() -> None:
        atomic_write_json(partial_path, {
            "schema_version": SCHEMA_VERSION,
            "generation_identity": identity,
            "attempts": attempts,
            "counts": dict(counts),
            "counters": dict(counters),
            "records": records,
        })

    env = Environment()
    oracle_config = contract["oracle"]
    oracle = Oracle(
        env, max_nodes=int(oracle_config["max_nodes"]),
        time_limit_seconds=oracle_config["time_limit_seconds"])
    baseline, baseline_model = _load_baseline(contract)
    designer_config = type("DesignerConfig", (), {
        "designer_checkpoint": contract["designer_checkpoint"]["resolved_path"],
        "device": contract.get("device", "cuda"),
    })()
    designer = _load_designer_generator(designer_config, required=True)
    from ..designer.config import GeneratorConfig
    from ..search.seeding import derive_trial_seed, level_search_identity

    generation = contract["generator"]
    generator_config = GeneratorConfig(
        rows=int(generation["rows"]), cols=int(generation["cols"]),
        color_count=int(generation["color_count"]),
        density=float(generation["density"]))
    rng = random.Random(int(contract["generation_seed"]))
    for _ in range(attempts):
        rng.randrange(2**63)
    target_total = sum(targets.values())
    checkpoint_every = int(contract["checkpoint_every_attempts"])
    while len(records) < target_total and attempts < int(contract["max_attempts"]):
        attempts += 1
        generation_seed = rng.randrange(2**63)
        local_rng = random.Random(generation_seed)
        level = _designer_level(
            designer, generator_config,
            mutation_budget=int(generation["mutation_budget"]),
            generation_seed=generation_seed, rng=local_rng)
        if level is None:
            counters["invalid"] += 1
            continue
        signature = static_level_signature(level)
        if signature in excluded:
            counters["excluded"] += 1
            continue
        if signature in seen:
            counters["duplicate"] += 1
            continue
        search_identity = level_search_identity(env, level)
        solved = []
        costs = []
        for index, budget in enumerate(SCREENING_BUDGETS):
            seed = derive_trial_seed(
                int(contract["screening"]["seed"]), trial_index=index,
                level_identity=search_identity,
                evaluation_context="midbudget_dev_pool.screening_v1")
            outcome = baseline.solve(level, seed=seed, simulations=budget)
            solved.append(bool(outcome.solved))
            costs.append(outcome.cost)
        stratum = classify_stratum(solved)
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
        records.append({
            "hard_eval_pool_schema_version": HARD_EVAL_POOL_SCHEMA_VERSION,
            "level_id": signature,
            "static_level_signature": signature,
            "canonical_level_sha256": _canonical_sha256(level_data),
            "generation_bucket": "candidate_blind_midbudget_dev",
            "generation_seed": generation_seed,
            "generation_parameters": {
                "generation_method": "frozen_pre_candidate_designer",
                "global_seed": int(contract["generation_seed"]),
                "rows": generator_config.rows, "cols": generator_config.cols,
                "color_count": generator_config.color_count,
                "density": generator_config.density,
                "mutation_budget": int(generation["mutation_budget"]),
                "contract_sha256": sha256_file(contract_source),
            },
            "oracle_validation": {
                "exact": value.exact, "solvable": value.solvable,
                "optimal_remaining_moves": value.value,
                "max_nodes": int(oracle_config["max_nodes"]),
                "time_limit_seconds": oracle_config["time_limit_seconds"],
            },
            "baseline_filter": {
                "checkpoint_sha256":
                    contract["baseline_checkpoint"]["sha256"],
                "budgets": list(SCREENING_BUDGETS),
                "solved": solved, "costs": costs,
                "difficulty_stratum": stratum,
                "screening_seed": int(contract["screening"]["seed"]),
            },
            "level": level_data,
        })
        seen.add(signature)
        counts[stratum] += 1
        save_partial()
        print(
            f"midbudget pool: {len(records)}/{target_total}; "
            f"{stratum}={counts[stratum]}/{targets[stratum]}; "
            f"attempts={attempts}", flush=True)
    if len(records) != target_total:
        save_partial()
        raise RuntimeError(
            f"pool infeasible: {len(records)}/{target_total} after "
            f"{attempts} attempts; counts={dict(counts)}")
    records.sort(key=lambda item: (
        STRATA.index(item["baseline_filter"]["difficulty_stratum"]),
        item["static_level_signature"]))
    atomic_write_text(output, "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records))
    partial_path.unlink(missing_ok=True)
    overlap = len({item["static_level_signature"] for item in records}
                  .intersection(excluded))
    seal = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "status": "sealed_before_candidate_evaluation",
        "pool": str(output),
        "record_count": len(records),
        "file_sha256": sha256_file(output),
        "canonical_pool_sha256": canonical_eval_pool_sha256(records),
        "static_signature_sha256": _canonical_hash(sorted(seen)),
        "difficulty_stratum_counts": dict(counts),
        "exclusion_manifest_sha256": exclusion["manifest_sha256"],
        "excluded_signature_overlap_count": overlap,
        "attempts": attempts,
        "counters": dict(counters),
        "candidate_checkpoint_loaded": False,
    }
    seal["seal_sha256"] = _canonical_hash(seal)
    seal_path = output.with_name(output.name + ".seal.json")
    atomic_write_json(seal_path, seal)
    del baseline_model
    return {**seal, "seal_path": str(seal_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exclusions = commands.add_parser("build-exclusions")
    exclusions.add_argument("--recipe", required=True)
    exclusions.add_argument("--output", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--contract", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-exclusions":
        result = build_expanded_exclusions(
            recipe_path=args.recipe, output_path=args.output)
    else:
        result = generate_pool(args.contract)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
