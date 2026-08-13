"""Search result and per-node statistics types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SearchStats:
    """Bookkeeping counters for one search run."""

    seed: int = 0
    simulations: int = 0
    nodes_expanded: int = 0
    unique_states: int = 0
    transposition_hits: int = 0
    cycle_rejections: int = 0
    deadlocks: int = 0
    model_evaluations: int = 0
    model_evaluation_batches: int = 0
    model_evaluation_cache_hits: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class SearchResult:
    # Chosen action, as a stable locator (color/cells/dir/distance/exit) plus the
    # concrete Action object resolved in the root state.
    chosen_action: Optional[Any]
    chosen_action_locator: Optional[dict[str, Any]]

    # Root edge statistics, aligned with ``legal_actions``.
    legal_actions: list[Any] = field(default_factory=list)
    legal_action_locators: list[dict[str, Any]] = field(default_factory=list)
    visit_counts: list[int] = field(default_factory=list)
    visit_policy: list[float] = field(default_factory=list)
    action_q_cost: list[float] = field(default_factory=list)
    priors: list[float] = field(default_factory=list)

    # Value estimates (cost space = remaining moves; lower is better).
    root_value_cost_model: float = 0.0       # raw model estimate at the root
    search_value_cost: float = 0.0           # best (min) visited edge Q-cost
    search_value_normalized: float = 0.0     # -search_value_cost / constant

    # Principal variation (greedy max-visit walk) as locators.
    principal_variation: list[dict[str, Any]] = field(default_factory=list)

    # Solution (a terminal-reaching path), verified by replay.
    solved: bool = False
    solution_locators: Optional[list[dict[str, Any]]] = None
    solution_actions: Optional[list[Any]] = None
    solution_length: Optional[int] = None
    solution_verified: bool = False
    # First simulation whose backup discovered any verified terminal path.
    # This permits exact prefix-budget diagnostics without rerunning identical
    # deterministic search prefixes.
    first_solution_simulation: Optional[int] = None
    # Why the search stopped at this root. One of: solved, deadlock,
    # budget_exhausted.
    termination_reason: str = "budget_exhausted"

    stats: SearchStats = field(default_factory=SearchStats)

    def visit_policy_dict(self) -> dict[int, float]:
        return {i: p for i, p in enumerate(self.visit_policy)}
