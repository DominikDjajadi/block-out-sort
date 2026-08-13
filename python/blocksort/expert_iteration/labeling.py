"""Teacher labeling with explicit exact-path retention.

For each candidate state we try an exact oracle analysis within the A* node
budget. If the state's value and *every* successor value are proven, we emit an
exact-oracle example. If only successor proof exhausts, an already-proven root
path can be retained as an exact-value, partial-policy label. If the root itself
is not proven, neural-guided graph search remains the final fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..environment import Environment
from ..oracle import Oracle
from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..state import State
from .records import (
    build_exact_example,
    build_exact_path_example,
    build_search_example,
)

LABEL_MODE_HYBRID_PATH = "hybrid_path"
LABEL_MODE_HYBRID = "hybrid"
LABEL_MODE_SEARCH_ONLY = "search_only"
LABEL_MODES = (
    LABEL_MODE_HYBRID_PATH,
    LABEL_MODE_HYBRID,
    LABEL_MODE_SEARCH_ONLY,
)


@dataclass
class LabelStats:
    exact: int = 0
    exact_path: int = 0
    search: int = 0
    astar_exhausted: int = 0
    astar_skipped: int = 0
    skipped_terminal: int = 0
    skipped_unsolvable: int = 0
    skipped_other: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "exact_labeled": self.exact,
            "exact_path_labeled": self.exact_path,
            "search_labeled": self.search,
            "astar_exhausted": self.astar_exhausted,
            "astar_skipped": self.astar_skipped,
            "skipped_terminal": self.skipped_terminal,
            "skipped_unsolvable": self.skipped_unsolvable,
            "skipped_other": self.skipped_other,
        }


def label_states(
    env: Environment,
    oracle: Oracle,
    candidates: list[tuple[State, dict[str, Any]]],
    *,
    iteration: int,
    astar_max_nodes: int,
    teacher_checkpoint: Optional[str],
    search_adapter: Optional[BlocksortAdapter],
    search_simulations: int,
    search_c_puct: float,
    label_policy_temperature: float,
    value_norm_constant: float,
    seed: int = 0,
    label_mode: str = LABEL_MODE_HYBRID_PATH,
) -> tuple[list[dict[str, Any]], LabelStats]:
    """Label candidates according to an explicit, reproducible strategy."""
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"label_mode must be one of: {', '.join(LABEL_MODES)}")
    records: list[dict[str, Any]] = []
    stats = LabelStats()

    for i, (state, provenance) in enumerate(candidates):
        if env.is_terminal(state):
            stats.skipped_terminal += 1
            continue

        analysis = None
        if label_mode == LABEL_MODE_SEARCH_ONLY:
            stats.astar_skipped += 1
        else:
            analysis = oracle.analyze(state)

        if (analysis is not None and analysis.exact and analysis.solvable
                and analysis.all_successors_exact):
            rec = build_exact_example(
                analysis, state, level_id=provenance.get("level_id", "?"),
                iteration=iteration, astar_max_nodes=astar_max_nodes,
                teacher_checkpoint=teacher_checkpoint, provenance=provenance,
                value_norm_constant=value_norm_constant)
            if rec is not None:
                records.append(rec)
                stats.exact += 1
                continue
            stats.skipped_other += 1
            continue

        if analysis is not None and analysis.exact and not analysis.solvable:
            # Proven unsolvable: nothing useful to learn, and not "exhausted".
            stats.skipped_unsolvable += 1
            continue

        # A solved root remains a rigorous exact-value and partial-policy
        # teacher even when proving every successor exhausts. Recover the path
        # from the oracle cache; this must not rerun A*.
        if (analysis is not None
                and analysis.exact and analysis.solvable
                and not analysis.all_successors_exact):
            stats.astar_exhausted += 1
            if label_mode == LABEL_MODE_HYBRID_PATH:
                result = oracle.cached_solve_result(state)
                if result is not None:
                    rec = build_exact_path_example(
                        result, state, env,
                        level_id=provenance.get("level_id", "?"),
                        iteration=iteration,
                        astar_max_nodes=astar_max_nodes,
                        teacher_checkpoint=teacher_checkpoint,
                        provenance=provenance,
                        value_norm_constant=value_norm_constant,
                    )
                    if rec is not None:
                        records.append(rec)
                        stats.exact_path += 1
                        continue
        elif analysis is not None:
            # The root itself exhausted.
            stats.astar_exhausted += 1

        if search_adapter is None:
            stats.skipped_other += 1
            continue

        cfg = SearchConfig(
            simulations=search_simulations, c_puct=search_c_puct,
            temperature=label_policy_temperature,
            value_normalization_constant=value_norm_constant,
            seed=(seed + i) & 0xFFFFFFFF)
        result = GraphSearch(search_adapter, cfg).run(state)
        rec = build_search_example(
            result, state, level_id=provenance.get("level_id", "?"),
            static_signature=(
                analysis.static_signature
                if analysis is not None else oracle.signature(state)),
            iteration=iteration, teacher_checkpoint=teacher_checkpoint,
            simulations=search_simulations,
            policy_temperature=label_policy_temperature, provenance=provenance,
            astar_max_nodes=astar_max_nodes,
            astar_reason=(
                "disabled_search_only"
                if analysis is None else "budget_exhausted"),
            value_norm_constant=value_norm_constant)
        if rec is not None:
            records.append(rec)
            stats.search += 1
        else:
            stats.skipped_other += 1

    return records, stats
