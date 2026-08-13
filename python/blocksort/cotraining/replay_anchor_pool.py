"""Build a leakage-free, champion-screened replay anchor pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..environment import Environment
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..training.transaction import atomic_write_json, atomic_write_text, sha256_file
from ..dataset.training_pool import load_excluded_signatures
from ..search.seeding import derive_trial_seed, level_search_identity
from .midbudget_dev_pool import SCREENING_BUDGETS, STRATA, classify_stratum


SCHEMA_VERSION = 1
SEMANTICS = "training_only_baseline_difficulty_replay_anchor_v1"
EXACT_SOURCES = frozenset(("exact_oracle", "exact_astar_path"))


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _rank(seed: int, signature: str) -> str:
    return hashlib.sha256(f"{seed}\0{signature}".encode()).hexdigest()


def _load_replay_by_level(
    path: Path,
    *,
    excluded: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            counters["source_records"] += 1
            level = level_from_dict(record["level"])
            signature = static_level_signature(level)
            if record.get("static_level_signature") != signature:
                raise ValueError(
                    f"replay line {line_number} has inconsistent signature")
            if signature in excluded:
                counters["excluded_records"] += 1
                continue
            if record.get("target_source") not in EXACT_SOURCES:
                counters["nonexact_records"] += 1
                continue
            by_level[signature].append(record)
            counters["eligible_records"] += 1
    counters["eligible_levels"] = len(by_level)
    return dict(by_level), dict(counters)


def _base_holdout_signatures(path: Path) -> set[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    signatures = set(manifest.get("validation_levels", []))
    signatures.update(manifest.get("test_levels", []))
    if not signatures:
        raise ValueError("base split contains no validation/test signatures")
    return signatures


def _load_champion(path: Path, device_name: str):
    import torch

    from ..designer.roles import Protagonist
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    device = torch.device(device_name)
    checkpoint = load_checkpoint(path, map_location="cpu")
    encoding, _model_config, value_norm = configs_from_checkpoint(checkpoint)
    model = model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    return Protagonist(
        Environment(), model, encoding, value_norm, device,
        simulations=max(SCREENING_BUDGETS), c_puct=1.5, temperature=0.0), model


def _screen_level(
    protagonist,
    env: Environment,
    level,
    *,
    screening_seed: int,
) -> tuple[str, list[bool | None], list[float | None]]:
    identity = level_search_identity(env, level)
    solved: list[bool | None] = []
    costs: list[float | None] = []
    # Preserve the exact nine-budget definition. Stop only when a solve makes
    # the final stratum unambiguous; unclassified levels run all budgets.
    for index, budget in enumerate(SCREENING_BUDGETS):
        seed = derive_trial_seed(
            screening_seed, trial_index=index, level_identity=identity,
            evaluation_context="training_replay_anchor.screening_v1")
        outcome = protagonist.solve(level, seed=seed, simulations=budget)
        solved.append(bool(outcome.solved))
        costs.append(outcome.cost)
        if outcome.solved and index == 0:
            solved.extend([None] * (len(SCREENING_BUDGETS) - len(solved)))
            costs.extend([None] * (len(SCREENING_BUDGETS) - len(costs)))
            return STRATA[0], solved, costs
        if outcome.solved and 1 <= index <= 3:
            solved.extend([None] * (len(SCREENING_BUDGETS) - len(solved)))
            costs.extend([None] * (len(SCREENING_BUDGETS) - len(costs)))
            return STRATA[1], solved, costs
        if outcome.solved and 4 <= index <= 6:
            solved.extend([None] * (len(SCREENING_BUDGETS) - len(solved)))
            costs.extend([None] * (len(SCREENING_BUDGETS) - len(costs)))
            return STRATA[2], solved, costs
    return classify_stratum(solved), solved, costs


def _annotated_records(
    by_level: Mapping[str, list[dict[str, Any]]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for signature in sorted(assignments, key=lambda value: (
            STRATA.index(assignments[value]["difficulty_stratum"]), value)):
        metadata = assignments[signature]
        for source in sorted(
                by_level[signature],
                key=lambda record: (
                    record.get("state_key", ""),
                    _canonical_hash(record),
                )):
            record = dict(source)
            record["training_anchor"] = {
                "schema_version": SCHEMA_VERSION,
                "semantics": SEMANTICS,
                "difficulty_stratum": metadata["difficulty_stratum"],
                "champion_checkpoint_sha256":
                    metadata["champion_checkpoint_sha256"],
                "screening_budgets": list(SCREENING_BUDGETS),
                "screening_solved": metadata["solved"],
                "screening_seed": metadata["screening_seed"],
            }
            records.append(record)
    return records


def _verify_pool(
    records: Iterable[Mapping[str, Any]],
    *,
    excluded: set[str],
    levels_per_band: int,
) -> dict[str, Any]:
    level_bands: dict[str, str] = {}
    record_counts: Counter = Counter()
    total = 0
    for record in records:
        total += 1
        signature = record["static_level_signature"]
        if signature in excluded:
            raise RuntimeError("anchor pool overlaps an evaluation identity")
        band = record["training_anchor"]["difficulty_stratum"]
        prior = level_bands.setdefault(signature, band)
        if prior != band:
            raise RuntimeError("one anchor level has multiple bands")
        record_counts[band] += 1
    level_counts = Counter(level_bands.values())
    expected = {band: levels_per_band for band in STRATA}
    if dict(level_counts) != expected:
        raise RuntimeError(
            f"anchor level bands are not balanced: {dict(level_counts)}")
    return {
        "record_count": total,
        "unique_level_count": len(level_bands),
        "level_counts_by_band": dict(level_counts),
        "record_counts_by_band": dict(record_counts),
        "excluded_signature_overlap_count": 0,
    }


def build_anchor_pool(
    *,
    replay_path: str | Path,
    champion_checkpoint: str | Path,
    base_split_manifest: str | Path,
    exclusion_files: Iterable[str | Path],
    output_path: str | Path,
    levels_per_band: int = 50,
    selection_seed: int = 8243,
    screening_seed: int = 8244,
    device: str = "cuda",
) -> dict[str, Any]:
    if levels_per_band <= 0:
        raise ValueError("levels_per_band must be positive")
    replay = Path(replay_path).resolve()
    champion_path = Path(champion_checkpoint).resolve()
    split_path = Path(base_split_manifest).resolve()
    exclusions = tuple(Path(path).resolve() for path in exclusion_files)
    output = Path(output_path).resolve()
    contract_path = output.with_name(output.name + ".contract.json")
    partial_path = output.with_name(output.name + ".partial.json")
    seal_path = output.with_name(output.name + ".seal.json")
    if output.exists() or seal_path.exists():
        raise FileExistsError("sealed anchor-pool output already exists")

    external_excluded, external_sources = load_excluded_signatures(
        str(path) for path in exclusions)
    base_excluded = _base_holdout_signatures(split_path)
    excluded = external_excluded | base_excluded
    contract = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": "training_replay_anchor_pool_v1",
        "status": "frozen_before_champion_screening",
        "semantics": SEMANTICS,
        "replay": {"path": str(replay), "sha256": sha256_file(replay)},
        "champion_checkpoint": {
            "path": str(champion_path), "sha256": sha256_file(champion_path)},
        "base_split_manifest": {
            "path": str(split_path), "sha256": sha256_file(split_path),
            "holdout_signature_count": len(base_excluded),
        },
        "evaluation_exclusion_sources": external_sources,
        "excluded_signature_count": len(excluded),
        "levels_per_band": levels_per_band,
        "selection_seed": selection_seed,
        "screening_seed": screening_seed,
        "screening_budgets": list(SCREENING_BUDGETS),
        "device": device,
        "output": str(output),
    }
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("persisted anchor-pool contract differs")
    else:
        atomic_write_json(contract_path, contract)
    identity = _canonical_hash(contract)
    by_level, replay_summary = _load_replay_by_level(
        replay, excluded=excluded)
    ordered = sorted(
        by_level, key=lambda signature: (_rank(selection_seed, signature), signature))

    cursor = 0
    assignments: dict[str, dict[str, Any]] = {}
    screened_counts: Counter = Counter()
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("contract_sha256") != identity:
            raise ValueError("anchor-pool partial contract differs")
        cursor = int(partial["cursor"])
        assignments = partial["assignments"]
        screened_counts.update(partial.get("screened_counts", {}))
    selected_counts = Counter(
        value["difficulty_stratum"] for value in assignments.values())

    def persist_partial() -> None:
        atomic_write_json(partial_path, {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": identity,
            "cursor": cursor,
            "screened_counts": dict(screened_counts),
            "selected_counts": dict(selected_counts),
            "assignments": assignments,
        })

    protagonist, champion_model = _load_champion(champion_path, device)
    env = Environment()
    target_total = levels_per_band * len(STRATA)
    while len(assignments) < target_total and cursor < len(ordered):
        signature = ordered[cursor]
        cursor += 1
        level = level_from_dict(by_level[signature][0]["level"])
        band, solved, costs = _screen_level(
            protagonist, env, level, screening_seed=screening_seed)
        screened_counts[band] += 1
        if selected_counts[band] < levels_per_band:
            assignments[signature] = {
                "difficulty_stratum": band,
                "solved": solved,
                "costs": costs,
                "screening_seed": screening_seed,
                "champion_checkpoint_sha256": sha256_file(champion_path),
            }
            selected_counts[band] += 1
        if cursor % 5 == 0 or len(assignments) == target_total:
            persist_partial()
            print(
                f"anchor screening: candidates={cursor}; "
                f"selected={len(assignments)}/{target_total}; "
                f"bands={dict(selected_counts)}", flush=True)
    if len(assignments) != target_total:
        persist_partial()
        raise RuntimeError(
            "replay cannot fill balanced anchor pool: "
            f"selected={dict(selected_counts)} screened={dict(screened_counts)}")

    records = _annotated_records(by_level, assignments)
    verification = _verify_pool(
        records, excluded=excluded, levels_per_band=levels_per_band)
    atomic_write_text(output, "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records))
    partial_path.unlink(missing_ok=True)
    assignment_identity = [{
        "static_level_signature": signature,
        **assignments[signature],
    } for signature in sorted(assignments)]
    seal = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "status": "sealed_training_only_anchor_pool",
        "contract_path": str(contract_path),
        "contract_sha256": identity,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "assignment_sha256": _canonical_hash(assignment_identity),
        "verification": verification,
        "replay_summary": replay_summary,
        "screened_candidate_count": cursor,
        "screened_counts_by_band": dict(screened_counts),
        "candidate_or_learner_checkpoint_loaded": False,
    }
    seal["seal_sha256"] = _canonical_hash(seal)
    atomic_write_json(seal_path, seal)
    del champion_model
    return seal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--base-split-manifest", required=True)
    parser.add_argument("--exclude", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--levels-per-band", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=8243)
    parser.add_argument("--screening-seed", type=int, default=8244)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = build_anchor_pool(
        replay_path=args.replay,
        champion_checkpoint=args.champion_checkpoint,
        base_split_manifest=args.base_split_manifest,
        exclusion_files=args.exclude,
        output_path=args.output,
        levels_per_band=args.levels_per_band,
        selection_seed=args.selection_seed,
        screening_seed=args.screening_seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
