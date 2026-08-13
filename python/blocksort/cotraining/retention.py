"""Difficulty-band retention checks for cumulative shadow learners."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..search.seeding import derive_trial_seed, level_search_identity
from ..serialization import level_from_dict
from ..signature import static_level_signature
from .eval_split import load_eval_pool_records


RETENTION_SEMANTICS = "baseline_difficulty_band_solve_retention_v1"


def load_retention_pool(
    path: str | Path,
    *,
    per_band: int | None,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Validate a stratified pool and select a stable slice or the full pool."""
    records = load_eval_pool_records(path)
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_signatures: set[str] = set()
    for record in records:
        level = level_from_dict(record["level"])
        signature = record.get("static_level_signature") \
            or static_level_signature(level)
        if signature in all_signatures:
            raise ValueError(f"duplicate retention level: {signature}")
        all_signatures.add(signature)
        baseline_filter = record.get("baseline_filter")
        band = (
            baseline_filter.get("difficulty_stratum")
            if isinstance(baseline_filter, Mapping) else None
        )
        if not isinstance(band, str) or not band:
            raise ValueError(
                "every retention record needs "
                "baseline_filter.difficulty_stratum")
        by_band[band].append({
            "static_level_signature": signature,
            "difficulty_stratum": band,
            "level": record["level"],
        })
    if len(by_band) < 2:
        raise ValueError("retention pool must contain at least two bands")
    selected: list[dict[str, Any]] = []
    band_counts = {}
    selected_band_counts = {}
    for band in sorted(by_band):
        rows = sorted(
            by_band[band], key=lambda item: item["static_level_signature"])
        requested = len(rows) if per_band is None else per_band
        if len(rows) < requested:
            raise ValueError(
                f"retention band {band!r} has {len(rows)} levels; "
                f"{requested} required")
        selected.extend(rows[:requested])
        band_counts[band] = len(rows)
        selected_band_counts[band] = requested
    manifest = {
        "schema_version": 1,
        "semantics": RETENTION_SEMANTICS,
        "source": str(Path(path).resolve()),
        "source_level_count": len(records),
        "source_band_counts": band_counts,
        "per_band": per_band,
        "selection_policy": (
            "all_source_levels_v1" if per_band is None
            else "stable_signature_sorted_per_band_slice_v1"),
        "selected_band_counts": selected_band_counts,
        "selected_level_count": len(selected),
        "selected": [{
            "static_level_signature": row["static_level_signature"],
            "difficulty_stratum": row["difficulty_stratum"],
        } for row in selected],
    }
    return selected, all_signatures, manifest


@torch.no_grad()
def evaluate_retention(
    env,
    model,
    encoding_config,
    value_norm,
    records: Iterable[Mapping[str, Any]],
    *,
    budgets: Iterable[int],
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate deterministic bounded solves for every retained band level."""
    model.eval()
    budgets = list(budgets)
    adapter = BlocksortAdapter(
        env, model, encoding_config, value_norm, device)
    rows = []
    for record in records:
        level = level_from_dict(record["level"])
        signature = record["static_level_signature"]
        state = env.initial_state(level)
        search_identity = level_search_identity(env, level)
        outcomes = {}
        for budget_index, budget in enumerate(budgets):
            trial_seed = derive_trial_seed(
                seed, trial_index=budget_index,
                level_identity=search_identity,
                evaluation_context="shadow_learner_retention_v1")
            config = SearchConfig(
                simulations=budget, c_puct=1.5, temperature=0.0,
                value_normalization_constant=getattr(
                    value_norm, "constant", 20.0),
                seed=trial_seed)
            result = GraphSearch(adapter, config).run(state)
            outcomes[str(budget)] = {
                "solved": bool(result.solved),
                "solution_length": result.solution_length,
                "termination_reason": result.termination_reason,
            }
        rows.append({
            "static_level_signature": signature,
            "difficulty_stratum": record["difficulty_stratum"],
            "budgets": outcomes,
        })
    return rows


def summarize_retention(
    reference_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    budgets: Iterable[int],
    max_regression: float,
) -> dict[str, Any]:
    """Apply a paired solve-rate guard independently in every band/budget."""
    reference = {
        row["static_level_signature"]: row for row in reference_rows}
    candidate = {
        row["static_level_signature"]: row for row in candidate_rows}
    if not reference or set(reference) != set(candidate):
        raise ValueError("retention evaluations are not paired")
    grouped: dict[str, list[str]] = defaultdict(list)
    for signature, row in reference.items():
        band = row["difficulty_stratum"]
        if candidate[signature]["difficulty_stratum"] != band:
            raise ValueError("retention difficulty bands differ")
        grouped[band].append(signature)
    bands: dict[str, Any] = {}
    failures = []
    for band in sorted(grouped):
        signatures = grouped[band]
        per_budget = {}
        for budget in budgets:
            key = str(budget)
            ref_solved = sum(
                bool(reference[sig]["budgets"][key]["solved"])
                for sig in signatures)
            cand_solved = sum(
                bool(candidate[sig]["budgets"][key]["solved"])
                for sig in signatures)
            delta = (cand_solved - ref_solved) / len(signatures)
            ref_only = sum(
                bool(reference[sig]["budgets"][key]["solved"])
                and not bool(candidate[sig]["budgets"][key]["solved"])
                for sig in signatures)
            cand_only = sum(
                bool(candidate[sig]["budgets"][key]["solved"])
                and not bool(reference[sig]["budgets"][key]["solved"])
                for sig in signatures)
            passed = delta >= -max_regression
            per_budget[key] = {
                "level_count": len(signatures),
                "reference_solved": ref_solved,
                "candidate_solved": cand_solved,
                "solve_rate_delta": delta,
                "reference_only_solves": ref_only,
                "candidate_only_solves": cand_only,
                "passed": passed,
            }
            if not passed:
                failures.append({
                    "difficulty_stratum": band,
                    "budget": budget,
                    "solve_rate_delta": delta,
                })
        bands[band] = {"per_budget": per_budget}
    return {
        "semantics": RETENTION_SEMANTICS,
        "passed": not failures,
        "maximum_allowed_regression": max_regression,
        "bands": bands,
        "failures": failures,
    }


def apply_retention_guard(
    continuation: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    enforce: bool,
) -> dict[str, Any]:
    """Optionally turn retention evidence into a learner rollback decision."""
    decision = dict(continuation)
    decision["retention_enforced"] = enforce
    if report is None:
        decision["retention_passed"] = None
        return decision
    decision["retention_passed"] = bool(report["passed"])
    if enforce and decision.get("accepted") and not report["passed"]:
        decision["accepted"] = False
        decision["decision"] = "rollback_to_anchor"
        decision["reasons"] = [
            *decision.get("reasons", []),
            "difficulty_band_retention_regression",
        ]
    return decision
