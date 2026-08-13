"""PUCT Monte Carlo *graph* search with transposition sharing.

Cost convention (documented and tested)
----------------------------------------
Everything is in cost space: a state's value ``V(s)`` is the estimated number of
remaining moves to clear the board, ``V(terminal) = 0`` and lower is better.
Every move costs 1, so the cost-to-go of an edge ``(s, a)`` along a trajectory
that reaches a leaf ``L`` in ``m`` further steps is ``m + V(L)``; equivalently
``Q(s, a) = 1 + V(child)``.

Selection minimizes ``Q(s, a) - U(s, a)`` where
``U = c_puct * P(a) * sqrt(sum_b N_b + 1) / (1 + N(s, a))``.
Subtracting the exploration bonus (rather than adding it) is the cost-space
mirror of AlphaZero's ``argmax(Q + U)``: high-prior, low-visit edges look cheaper
and are explored first.

Backup pushes the leaf cost up the traversed edges, adding 1 per step:
``g = V(leaf); for edge in reversed(path): g += 1; N+=1; W+=g; Q=W/N``.

Graph handling: nodes are keyed by ``(static_signature, canonical_state_key)`` so
equivalent states share one node with possibly many parents. A cycle (revisiting
a state already on the current simulation path) is never followed.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Optional, Protocol

from .config import SearchConfig
from .node import Node, TranspositionTable
from .result import SearchResult, SearchStats


class SearchAdapter(Protocol):
    """Everything the search needs from the environment + model.

    Implemented by :class:`BlocksortAdapter`; tests may supply a mock to exercise
    the selection/backup math on a tiny abstract graph.
    """

    def key(self, state: Any) -> Hashable: ...
    def is_terminal(self, state: Any) -> bool: ...
    def is_deadlock(self, state: Any) -> bool: ...
    def legal_actions(self, state: Any) -> list[Any]: ...
    def apply(self, state: Any, action: Any) -> Any: ...
    def evaluate(self, state: Any) -> tuple[list[float], float]: ...
    def to_locator(self, state: Any, action: Any) -> dict[str, Any]: ...
    def from_locator(self, state: Any, locator: dict[str, Any]) -> Any: ...


class NoLegalActionsError(RuntimeError):
    """A nonterminal state has no actions and was not classified as deadlocked."""


@dataclass
class _PendingSimulation:
    path: list[tuple[Node, int]]
    node: Node
    state: Any
    leaf_reason: str
    leaf_cost: float | None
    needs_evaluation: bool
    legal_actions: list[Any] | None
    virtual_costs: list[float]
    selection_trace: list[dict[str, Any]] = field(default_factory=list)


class GraphSearch:
    def __init__(self, adapter: SearchAdapter, config: SearchConfig) -> None:
        self.adapter = adapter
        self.config = config
        self.table = TranspositionTable()
        self.stats = SearchStats()
        self._rng = random.Random(config.seed)
        # Shortest verified terminal path discovered so far (list of locators).
        self._best_solution: Optional[list[dict[str, Any]]] = None
        self._first_solution_simulation: Optional[int] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, root_state: Any, *, seed: int | None = None) -> SearchResult:
        """Run independently, using ``seed`` or the configured one-off default."""
        self._validate_config()
        self._reset_per_run_state(seed)
        start = time.perf_counter()
        evaluations_before = int(getattr(
            self.adapter, "model_evaluations", 0))
        evaluation_batches_before = int(getattr(
            self.adapter, "model_evaluation_batches", 0))
        cache_hits_before = int(getattr(
            self.adapter, "model_evaluation_cache_hits", 0))
        root_key = self.adapter.key(root_state)
        root, created = self.table.get_or_create(root_key, root_state)
        if created:
            self._expand(root, root_state)
            if (self.config.dirichlet_alpha > 0
                    and self.config.dirichlet_weight > 0
                    and root.num_actions > 1):
                self._apply_root_noise(root)

        batch_evaluator = getattr(
            self.adapter, "evaluate_batch_with_legal_actions", None)
        if (self.config.inference_batch_size > 1
                and callable(batch_evaluator)):
            remaining = self.config.simulations
            while remaining > 0:
                completed = self._simulate_batch(
                    root,
                    root_state,
                    min(self.config.inference_batch_size, remaining),
                    batch_evaluator,
                )
                if completed <= 0:
                    raise RuntimeError(
                        "batched search made no simulation progress")
                remaining -= completed
        else:
            for _ in range(self.config.simulations):
                self.stats.simulations += 1
                self._simulate(root, root_state)

        self.stats.unique_states = len(self.table)
        self.stats.transposition_hits = self.table.hits
        self.stats.model_evaluations = int(getattr(
            self.adapter, "model_evaluations", 0)) - evaluations_before
        self.stats.model_evaluation_batches = int(getattr(
            self.adapter, "model_evaluation_batches", 0)) - \
            evaluation_batches_before
        self.stats.model_evaluation_cache_hits = int(getattr(
            self.adapter, "model_evaluation_cache_hits", 0)) - \
            cache_hits_before
        self.stats.elapsed_seconds = time.perf_counter() - start
        return self._build_result(root, root_state)

    def _validate_config(self) -> None:
        if self.config.simulations <= 0:
            raise ValueError("SearchConfig.simulations must be positive")
        if (isinstance(self.config.inference_batch_size, bool)
                or not isinstance(self.config.inference_batch_size, int)
                or self.config.inference_batch_size <= 0):
            raise ValueError(
                "SearchConfig.inference_batch_size must be a positive integer")
        if (not math.isfinite(self.config.virtual_loss)
                or self.config.virtual_loss < 0):
            raise ValueError(
                "SearchConfig.virtual_loss must be finite and non-negative")
        if self.config.max_depth <= 0:
            raise ValueError("SearchConfig.max_depth must be positive")
        if self.config.c_puct < 0:
            raise ValueError("SearchConfig.c_puct must be non-negative")
        if (not math.isfinite(self.config.temperature)
                or self.config.temperature < 0):
            raise ValueError(
                "SearchConfig.temperature must be finite and non-negative")
        if self.config.max_cost < 0 or not math.isfinite(self.config.max_cost):
            raise ValueError("SearchConfig.max_cost must be finite and non-negative")
        constant = self.config.value_normalization_constant
        if not math.isfinite(constant) or constant <= 0:
            raise ValueError(
                "SearchConfig.value_normalization_constant must be finite "
                f"and greater than 0; got {constant!r}")

    def _reset_per_run_state(self, seed: int | None) -> None:
        """Make every public run an independent, reproducible search."""
        actual_seed = self.config.seed if seed is None else seed
        if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
            raise TypeError("search seed must be an integer")
        self.table = TranspositionTable()
        self.stats = SearchStats(seed=actual_seed)
        self._rng = random.Random(actual_seed)
        self._best_solution = None
        self._first_solution_simulation = None

    # ------------------------------------------------------------------
    # One simulation
    # ------------------------------------------------------------------

    def _simulate_batch(
        self,
        root: Node,
        root_state: Any,
        target_size: int,
        batch_evaluator,
        trace_rows: list[dict[str, Any]] | None = None,
    ) -> int:
        """Collect distinct neural leaves, evaluate once, then back them up."""
        prepared: dict[Hashable, tuple[Any, list[Any]]] = {}
        pending: list[_PendingSimulation] = []
        try:
            for _ in range(target_size):
                simulation = self._collect_pending_simulation(
                    root, root_state, prepared,
                    trace=trace_rows is not None)
                # A second path reached a neural leaf already awaiting
                # evaluation. End this batch instead of spending another
                # simulation on the identical unevaluated node.
                if simulation is None:
                    break
                simulation.virtual_costs = self._reserve_virtual(
                    simulation.path)
                pending.append(simulation)
                self.stats.simulations += 1
        except Exception:
            for simulation in pending:
                self._release_virtual(
                    simulation.path, simulation.virtual_costs)
            raise

        requests: dict[Hashable, tuple[Node, Any, list[Any]]] = {}
        first_simulation_number = self.stats.simulations - len(pending) + 1
        for offset, simulation in enumerate(pending):
            if simulation.needs_evaluation:
                assert simulation.legal_actions is not None
                requests.setdefault(
                    simulation.node.key,
                    (simulation.node, simulation.state,
                     simulation.legal_actions),
                )
        try:
            if requests:
                request_rows = list(requests.values())
                evaluated = batch_evaluator(
                    [row[1] for row in request_rows],
                    [row[2] for row in request_rows],
                )
                if len(evaluated) != len(request_rows):
                    raise ValueError(
                        "batched evaluate result count does not match leaf count")
                for (node, state, actions), (priors, value_cost) in zip(
                        request_rows, evaluated):
                    self._complete_expansion(
                        node, state, actions, priors, value_cost)
        except Exception:
            for simulation in pending:
                self._release_virtual(
                    simulation.path, simulation.virtual_costs)
            raise

        traced_start = len(trace_rows) if trace_rows is not None else 0
        for simulation in pending:
            solution_before = (
                copy.deepcopy(self._best_solution)
                if trace_rows is not None else None)
            self._release_virtual(
                simulation.path, simulation.virtual_costs)
            if simulation.needs_evaluation:
                leaf_cost = simulation.node.value_cost
            else:
                assert simulation.leaf_cost is not None
                leaf_cost = simulation.leaf_cost
            if simulation.leaf_reason == "terminal" and simulation.path:
                self._record_solution(
                    simulation.path,
                    simulation_number=first_simulation_number + offset)
            self._backup(simulation.path, leaf_cost)
            if trace_rows is not None:
                path_locators = copy.deepcopy([
                    node.locators[index]
                    for node, index in simulation.path
                ])
                trace_rows.append({
                    "root_edge_index": (
                        simulation.path[0][1]
                        if simulation.path else None),
                    "path_length": len(simulation.path),
                    "path_locators": path_locators,
                    "leaf_reason": simulation.leaf_reason,
                    "leaf_cost": float(leaf_cost),
                    "solution_changed":
                        solution_before != self._best_solution,
                    "best_solution_length": (
                        len(self._best_solution)
                        if self._best_solution is not None else None),
                    "selection_trace": simulation.selection_trace,
                    "leaf_expansion": {
                        "depth": len(simulation.path),
                        "node_key": copy.deepcopy(simulation.node.key),
                        "terminal": bool(simulation.node.terminal),
                        "deadlock": bool(simulation.node.deadlock),
                        "value_cost": float(simulation.node.value_cost),
                        "priors": list(simulation.node.priors),
                        "locators": copy.deepcopy(simulation.node.locators),
                    },
                })
        if trace_rows is not None:
            # Batched selection happens before any ordinary backup.  Attach
            # the stable batch-end root state to every simulation in this
            # batch rather than exposing temporary virtual reservations.
            visit_counts = list(root.N)
            q_costs = list(root.Q)
            selection_scores = self._root_selection_scores(root)
            principal_variation = copy.deepcopy(
                self._principal_variation(root, root_state))
            for row in trace_rows[traced_start:]:
                row["root_visit_counts"] = visit_counts
                row["root_action_q_costs"] = q_costs
                row["next_root_selection_scores"] = selection_scores
                row["principal_variation"] = principal_variation
        return len(pending)

    def _root_selection_scores(self, node: Node) -> list[float]:
        """Return cost-space PUCT scores without mutating search state."""
        sqrt_total = math.sqrt(node.total_visits + 1)
        scores = []
        for index in range(node.num_actions):
            visits = node.N[index]
            q_cost = node.Q[index] if visits > 0 else node.value_cost
            exploration = (
                self.config.c_puct
                * node.priors[index]
                * sqrt_total
                / (1 + visits)
            )
            scores.append(q_cost - exploration)
        return scores

    def _collect_pending_simulation(
        self,
        root: Node,
        root_state: Any,
        prepared: dict[Hashable, tuple[Any, list[Any]]],
        *,
        trace: bool = False,
    ) -> _PendingSimulation | None:
        """Select one path without evaluating its unexpanded neural leaf."""
        path: list[tuple[Node, int]] = []
        path_keys: set[Hashable] = set()
        node = root
        state = root_state
        depth = 0
        selection_trace: list[dict[str, Any]] = []

        while True:
            path_keys.add(node.key)
            if not node.expanded:
                if node.key in prepared:
                    return None
                actions = self._prepare_expansion(node, state)
                if node.terminal:
                    return _PendingSimulation(
                        path, node, state, "terminal", 0.0, False, None, [],
                        selection_trace)
                if node.deadlock:
                    return _PendingSimulation(
                        path, node, state, "deadlock", self.config.max_cost,
                        False, None, [], selection_trace)
                assert actions is not None
                prepared[node.key] = (state, actions)
                return _PendingSimulation(
                    path, node, state, "model_leaf", None, True, actions, [],
                    selection_trace)

            if node.terminal:
                return _PendingSimulation(
                    path, node, state, "terminal", 0.0, False, None, [],
                    selection_trace)
            if node.deadlock:
                return _PendingSimulation(
                    path, node, state, "deadlock", self.config.max_cost,
                    False, None, [], selection_trace)
            if depth >= self.config.max_depth:
                return _PendingSimulation(
                    path, node, state, "max_depth", node.value_cost,
                    False, None, [], selection_trace)

            edge = self._select_edge(node, state, path_keys)
            if edge is None:
                return _PendingSimulation(
                    path, node, state, "cycle_stop", node.value_cost,
                    False, None, [], selection_trace)
            index, child_state, child_key = edge
            if trace:
                sqrt_total = math.sqrt(node.total_visits + 1)
                q_for_selection = [
                    node.Q[action_index]
                    if node.N[action_index] > 0 else node.value_cost
                    for action_index in range(node.num_actions)
                ]
                exploration = [
                    self.config.c_puct
                    * node.priors[action_index]
                    * sqrt_total
                    / (1 + node.N[action_index])
                    for action_index in range(node.num_actions)
                ]
                selection_trace.append({
                    "depth": depth,
                    "node_key": copy.deepcopy(node.key),
                    "node_value_cost": float(node.value_cost),
                    "total_visits": int(node.total_visits),
                    "priors": list(node.priors),
                    "visit_counts": list(node.N),
                    "q_costs": list(node.Q),
                    "q_costs_for_selection": q_for_selection,
                    "exploration_u": exploration,
                    "selection_scores": [
                        q_cost - u
                        for q_cost, u in zip(q_for_selection, exploration)
                    ],
                    "locators": copy.deepcopy(node.locators),
                    "selected_edge_index": index,
                    "selected_locator": copy.deepcopy(node.locators[index]),
                })
            path.append((node, index))
            child, _created = self.table.get_or_create(
                child_key, child_state)
            node = child
            state = child_state
            depth += 1

    def _reserve_virtual(
        self,
        path: list[tuple[Node, int]],
    ) -> list[float]:
        """Temporarily make a selected path costlier within the current batch."""
        reserved: list[float] = []
        for node, index in path:
            baseline = (
                node.Q[index] if node.N[index] > 0 else node.value_cost)
            virtual_cost = baseline + self.config.virtual_loss
            node.N[index] += 1
            node.W[index] += virtual_cost
            node.Q[index] = node.W[index] / node.N[index]
            node.total_visits += 1
            reserved.append(virtual_cost)
        return reserved

    @staticmethod
    def _release_virtual(
        path: list[tuple[Node, int]],
        virtual_costs: list[float],
    ) -> None:
        if len(path) != len(virtual_costs):
            raise RuntimeError("virtual reservation length mismatch")
        for (node, index), virtual_cost in zip(path, virtual_costs):
            node.N[index] -= 1
            node.W[index] -= virtual_cost
            node.total_visits -= 1
            if node.N[index] > 0:
                node.Q[index] = node.W[index] / node.N[index]
            else:
                node.N[index] = 0
                node.W[index] = 0.0
                node.Q[index] = 0.0

    def _simulate(
        self,
        root: Node,
        root_state: Any,
        *,
        trace: bool = False,
    ) -> dict[str, Any] | None:
        path: list[tuple[Node, int]] = []  # (node, edge_index) actually traversed
        path_keys: set[Hashable] = set()
        node = root
        state = root_state
        depth = 0
        leaf_cost: float
        leaf_reason: str
        solution_before = (
            copy.deepcopy(self._best_solution) if trace else None)
        selection_trace: list[dict[str, Any]] = []
        leaf_expansion: dict[str, Any] | None = None

        while True:
            path_keys.add(node.key)

            if not node.expanded:
                self._expand(node, state)
                if trace:
                    leaf_expansion = {
                        "depth": depth,
                        "node_key": copy.deepcopy(node.key),
                        "terminal": bool(node.terminal),
                        "deadlock": bool(node.deadlock),
                        "value_cost": float(node.value_cost),
                        "priors": list(node.priors),
                        "locators": copy.deepcopy(node.locators),
                    }
                if node.terminal:
                    leaf_cost = 0.0
                    leaf_reason = "terminal"
                    if path:  # terminal reached via >=1 move -> a candidate solution
                        self._record_solution(
                            path, simulation_number=self.stats.simulations)
                elif node.deadlock:
                    leaf_cost = self.config.max_cost
                    leaf_reason = "deadlock"
                else:
                    leaf_cost = node.value_cost
                    leaf_reason = "model_leaf"
                break

            if node.terminal:
                leaf_cost = 0.0
                leaf_reason = "terminal"
                if path:
                    self._record_solution(
                        path, simulation_number=self.stats.simulations)
                break

            if node.deadlock:
                leaf_cost = self.config.max_cost
                leaf_reason = "deadlock"
                break

            if depth >= self.config.max_depth:
                leaf_cost = node.value_cost
                leaf_reason = "max_depth"
                break

            edge = self._select_edge(node, state, path_keys)
            if edge is None:
                # All children are ancestors (cycle) or unavailable; stop here.
                leaf_cost = node.value_cost
                leaf_reason = "cycle_stop"
                break

            i, child_state, child_key = edge
            if trace:
                sqrt_total = math.sqrt(node.total_visits + 1)
                q_for_selection = [
                    node.Q[index]
                    if node.N[index] > 0 else node.value_cost
                    for index in range(node.num_actions)
                ]
                exploration = [
                    self.config.c_puct
                    * node.priors[index]
                    * sqrt_total
                    / (1 + node.N[index])
                    for index in range(node.num_actions)
                ]
                selection_trace.append({
                    "depth": depth,
                    "node_key": copy.deepcopy(node.key),
                    "node_value_cost": float(node.value_cost),
                    "total_visits": int(node.total_visits),
                    "priors": list(node.priors),
                    "visit_counts": list(node.N),
                    "q_costs": list(node.Q),
                    "q_costs_for_selection": q_for_selection,
                    "exploration_u": exploration,
                    "selection_scores": [
                        q_cost - u
                        for q_cost, u in zip(q_for_selection, exploration)
                    ],
                    "locators": copy.deepcopy(node.locators),
                    "selected_edge_index": i,
                    "selected_locator": copy.deepcopy(node.locators[i]),
                })
            path.append((node, i))
            child, created = self.table.get_or_create(child_key, child_state)
            if created:
                pass  # expanded lazily when next reached as a leaf
            node = child
            state = child_state
            depth += 1

        self._backup(path, leaf_cost)
        if not trace:
            return None
        path_locators = copy.deepcopy(
            [path_node.locators[index] for path_node, index in path])
        return {
            "root_edge_index": path[0][1] if path else None,
            "path_length": len(path),
            "path_locators": path_locators,
            "leaf_reason": leaf_reason,
            "leaf_cost": float(leaf_cost),
            "solution_changed": solution_before != self._best_solution,
            "best_solution_length": (
                len(self._best_solution)
                if self._best_solution is not None else None
            ),
            "selection_trace": selection_trace,
            "leaf_expansion": leaf_expansion,
        }

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: Node, state: Any) -> None:
        actions = self._prepare_expansion(node, state)
        if node.expanded:
            return
        assert actions is not None
        evaluate_with_actions = getattr(
            self.adapter, "evaluate_with_legal_actions", None)
        if callable(evaluate_with_actions):
            priors, value_cost = evaluate_with_actions(state, actions)
        else:
            # Backward-compatible path for abstract/test adapters.
            priors, value_cost = self.adapter.evaluate(state)
        self._complete_expansion(
            node, state, actions, priors, value_cost)

    def _prepare_expansion(
        self,
        node: Node,
        state: Any,
    ) -> list[Any] | None:
        """Classify terminal/deadlock nodes or return actions needing a model."""
        if self.adapter.is_terminal(state):
            node.expanded = True
            node.terminal = True
            node.value_cost = 0.0
            self.stats.nodes_expanded += 1
            return None

        actions = self.adapter.legal_actions(state)
        if not isinstance(actions, list):
            raise TypeError("legal_actions() must return a list")
        if any(actions[i] == actions[j]
               for i in range(len(actions)) for j in range(i)):
            raise ValueError("legal_actions() returned duplicate actions")
        if not actions:
            classify = getattr(self.adapter, "is_deadlock", None)
            if callable(classify) and bool(classify(state)):
                node.expanded = True
                node.deadlock = True
                node.value_cost = self.config.max_cost
                self.stats.nodes_expanded += 1
                self.stats.deadlocks += 1
                return None
            level = getattr(state, "level", None)
            level_id = getattr(level, "name", None)
            raise NoLegalActionsError(
                "nonterminal state has no legal actions but adapter did not "
                "classify it as a deadlock "
                f"(context=GraphSearch._expand, key={node.key!r}, "
                f"terminal=False, level_id={level_id!r}, state={state!r}). "
                "Check the state, terminal classifier, and legal-action generator."
            )
        return actions

    def _complete_expansion(
        self,
        node: Node,
        state: Any,
        actions: list[Any],
        priors,
        value_cost,
    ) -> None:
        """Install one neural evaluation onto a prepared graph node."""
        if node.expanded:
            raise RuntimeError(f"node {node.key!r} was expanded twice")
        if len(priors) != len(actions):
            raise ValueError("evaluate() priors length != legal action count")
        priors = [float(p) for p in priors]
        if any(not math.isfinite(p) or p < 0 for p in priors):
            raise ValueError("evaluate() priors must be finite and non-negative")
        prior_total = sum(priors)
        priors = ([p / prior_total for p in priors] if prior_total > 0
                  else [1.0 / len(actions)] * len(actions))
        value_cost = float(value_cost)
        if not math.isfinite(value_cost):
            raise ValueError("evaluate() value_cost must be finite")

        node.actions = list(actions)
        node.locators = [self.adapter.to_locator(state, a) for a in actions]
        node.priors = priors
        node.N = [0] * len(actions)
        node.W = [0.0] * len(actions)
        node.Q = [0.0] * len(actions)
        node.child_key = [None] * len(actions)
        node.value_cost = max(0.0, min(self.config.max_cost, value_cost))
        node.expanded = True

        self.stats.nodes_expanded += 1

    def _apply_root_noise(self, node: Node) -> None:
        alpha = self.config.dirichlet_alpha
        eps = self.config.dirichlet_weight
        noise = [self._rng.gammavariate(alpha, 1.0) for _ in node.priors]
        s = sum(noise) or 1.0
        noise = [x / s for x in noise]
        node.priors = [(1 - eps) * p + eps * n for p, n in zip(node.priors, noise)]

    # ------------------------------------------------------------------
    # Selection (PUCT in cost space; minimize Q - U)
    # ------------------------------------------------------------------

    def _select_edge(
        self, node: Node, state: Any, path_keys: set[Hashable]
    ) -> Optional[tuple[int, Any, Hashable]]:
        """Return ``(edge_index, child_state, child_key)`` or ``None``.

        Candidates are considered best-first; the first one whose child is not an
        ancestor on the current path is taken. Cycle-leading edges are skipped
        (and counted) so an ancestor state is never followed.
        """
        total = node.total_visits
        sqrt_total = math.sqrt(total + 1)
        c = self.config.c_puct

        scored: list[tuple[float, int]] = []
        for i in range(node.num_actions):
            n_i = node.N[i]
            q_i = node.Q[i] if n_i > 0 else node.value_cost  # FPU = node estimate
            u_i = c * node.priors[i] * sqrt_total / (1 + n_i)
            scored.append((q_i - u_i, i))
        scored.sort(key=lambda t: (t[0], t[1]))

        for _, i in scored:
            child_state, child_key = self._child(node, state, i)
            if child_key in path_keys:
                self.stats.cycle_rejections += 1
                continue
            return i, child_state, child_key
        return None

    def _child(self, node: Node, state: Any, i: int) -> tuple[Any, Hashable]:
        """Resolve edge ``i`` to a concrete child state (from ``state``) and key.

        The action is stored against the node's representative; it is translated
        through its stable locator so it applies correctly to ``state`` even when
        ``state`` orders its blocks differently from the representative.
        """
        locator = node.locators[i]
        action = self.adapter.from_locator(state, locator)
        child_state = self.adapter.apply(state, action)
        key = node.child_key[i]
        if key is None:
            key = self.adapter.key(child_state)
            node.child_key[i] = key
        return child_state, key

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def _backup(self, path: list[tuple[Node, int]], leaf_cost: float) -> None:
        g = float(leaf_cost)
        for node, i in reversed(path):
            g += 1.0
            node.N[i] += 1
            node.W[i] += g
            node.Q[i] = node.W[i] / node.N[i]
            node.total_visits += 1

    # ------------------------------------------------------------------
    # Solutions
    # ------------------------------------------------------------------

    def _record_solution(
        self,
        path: list[tuple[Node, int]],
        *,
        simulation_number: int,
    ) -> None:
        if self._first_solution_simulation is None:
            self._first_solution_simulation = simulation_number
        locators = copy.deepcopy([node.locators[i] for node, i in path])
        if self._best_solution is None or len(locators) < len(self._best_solution):
            self._best_solution = locators

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(self, root: Node, root_state: Any) -> SearchResult:
        cfg = self.config
        result = SearchResult(chosen_action=None, chosen_action_locator=None)
        result.stats = replace(self.stats)
        result.first_solution_simulation = self._first_solution_simulation

        result.legal_actions = copy.deepcopy(root.actions)
        result.legal_action_locators = copy.deepcopy(root.locators)
        result.visit_counts = list(root.N)
        result.priors = list(root.priors)
        result.action_q_cost = list(root.Q)
        result.root_value_cost_model = root.value_cost

        visited = [i for i in range(root.num_actions) if root.N[i] > 0]
        if visited:
            best_cost = min(root.Q[i] for i in visited)
        else:
            best_cost = root.value_cost
        result.search_value_cost = best_cost
        const = cfg.value_normalization_constant
        result.search_value_normalized = -best_cost / const

        result.visit_policy = self._visit_policy(root.N, cfg.temperature)
        chosen = self._choose_action(root)
        if chosen is not None:
            result.chosen_action_locator = copy.deepcopy(root.locators[chosen])
            result.chosen_action = copy.deepcopy(
                self.adapter.from_locator(root_state, root.locators[chosen])
            )
        result.principal_variation = copy.deepcopy(
            self._principal_variation(root, root_state)
        )

        if root.terminal:
            result.termination_reason = "solved"
        elif root.deadlock:
            result.termination_reason = "deadlock"

        if self._best_solution is not None:
            actions = self._verify(root_state, self._best_solution)
            if actions is not None:
                result.solved = True
                result.solution_verified = True
                result.solution_locators = copy.deepcopy(self._best_solution)
                result.solution_actions = copy.deepcopy(actions)
                result.solution_length = len(actions)
                result.termination_reason = "solved"
        return result

    def _visit_policy(self, counts: list[int], temperature: float) -> list[float]:
        total = sum(counts)
        if total == 0 or not counts:
            n = len(counts)
            return [1.0 / n] * n if n else []
        if temperature <= 0:
            best = max(range(len(counts)), key=lambda i: (counts[i], -i))
            policy = [0.0] * len(counts)
            policy[best] = 1.0
            return policy
        powered = [c ** (1.0 / temperature) for c in counts]
        s = sum(powered) or 1.0
        return [p / s for p in powered]

    def _choose_action(self, root: Node) -> Optional[int]:
        if root.num_actions == 0:
            return None
        total = sum(root.N)
        if total == 0:
            # No simulation expanded an edge (e.g. simulations=0): fall back to
            # the highest-prior legal action.
            return max(range(root.num_actions), key=lambda i: (root.priors[i], -i))
        if self.config.temperature <= 0:
            return max(range(root.num_actions), key=lambda i: (root.N[i], -i))
        policy = self._visit_policy(root.N, self.config.temperature)
        r = self._rng.random()
        acc = 0.0
        for i, p in enumerate(policy):
            acc += p
            if r <= acc:
                return i
        return len(policy) - 1

    def _principal_variation(self, root: Node, root_state: Any) -> list[dict[str, Any]]:
        pv: list[dict[str, Any]] = []
        seen: set[Hashable] = {root.key}
        node = root
        state = root_state
        for _ in range(self.config.max_depth):
            if not node.expanded or node.terminal or node.num_actions == 0:
                break
            visited = [i for i in range(node.num_actions) if node.N[i] > 0]
            if not visited:
                break
            i = max(visited, key=lambda j: (node.N[j], -j))
            pv.append(node.locators[i])
            child_state, child_key = self._child(node, state, i)
            if child_key in seen:
                break
            seen.add(child_key)
            child = self.table.get(child_key)
            if child is None:
                break
            node, state = child, child_state
        return pv

    def _verify(self, root_state: Any, locators: list[dict[str, Any]]
                ) -> Optional[list[Any]]:
        actions: list[Any] = []
        state = root_state
        try:
            for loc in locators:
                action = self.adapter.from_locator(state, loc)
                actions.append(action)
                state = self.adapter.apply(state, action)
        except Exception:
            return None
        if not self.adapter.is_terminal(state):
            return None
        return actions


# ======================================================================
# Block Out Sort adapter (environment + trained model)
# ======================================================================

class BlocksortAdapter:
    """Couples the game :class:`Environment` and a trained policy-value model."""

    def __init__(
        self,
        env,
        model,
        encoding_config,
        value_norm,
        device,
        *,
        evaluation_cache_size: int = 50_000,
    ) -> None:
        import torch  # local import keeps abstract-graph tests torch-free

        self._torch = torch
        self.env = env
        self.model = model
        self.encoding_config = encoding_config
        self.value_norm = value_norm
        self.device = device
        self._sig_cache: dict[Any, str] = {}
        if (isinstance(evaluation_cache_size, bool)
                or not isinstance(evaluation_cache_size, int)
                or evaluation_cache_size < 0):
            raise ValueError(
                "evaluation_cache_size must be a non-negative integer")
        self.evaluation_cache_size = evaluation_cache_size
        self._evaluation_cache: OrderedDict[
            Hashable, tuple[Any, float]
        ] = OrderedDict()
        self.model_evaluations = 0
        self.model_evaluation_batches = 0
        self.model_evaluation_cache_hits = 0

    # ----- identity -----

    def _signature(self, state) -> str:
        from ..signature import static_level_signature
        sig = self._sig_cache.get(state.level)
        if sig is None:
            sig = static_level_signature(state.level)
            self._sig_cache[state.level] = sig
        return sig

    def key(self, state) -> Hashable:
        return (self._signature(state), self.env.canonical_key(state))

    # ----- environment -----

    def is_terminal(self, state) -> bool:
        return self.env.is_terminal(state)

    def is_deadlock(self, state) -> bool:
        return self.env.is_deadlock(state)

    def legal_actions(self, state) -> list[Any]:
        return self.env.legal_actions(state)

    def apply(self, state, action) -> Any:
        return self.env.apply_action(state, action)

    def to_locator(self, state, action) -> dict[str, Any]:
        from ..solution import serialize_action
        return serialize_action(state, action)

    def from_locator(self, state, locator) -> Any:
        from ..solution import deserialize_action
        return deserialize_action(state, locator)

    # ----- model -----

    def evaluate(self, state) -> tuple[list[float], float]:
        legal = self.env.legal_actions(state)
        return self.evaluate_with_legal_actions(state, legal)

    def evaluate_with_legal_actions(
        self,
        state,
        legal_actions: list[Any],
    ) -> tuple[list[float], float]:
        """Evaluate one state through the same cache-aware batched path."""
        return self.evaluate_batch_with_legal_actions(
            [state], [legal_actions])[0]

    def evaluate_batch_with_legal_actions(
        self,
        states: list[Any],
        legal_action_batches: list[list[Any]],
    ) -> list[tuple[list[float], float]]:
        """Evaluate uncached canonical states in one model forward pass."""
        from ..training.action_encoding import action_index, build_legal_action_mask
        from ..training.encoding import encode_state
        from ..training.losses import masked_policy_probs

        if len(states) != len(legal_action_batches):
            raise ValueError(
                "states and legal-action batches must have equal length")
        if not states:
            return []
        torch = self._torch
        keys = [self.key(state) for state in states]
        outputs: list[tuple[Any, float] | None] = [None] * len(states)
        missing_by_key: dict[Hashable, list[int]] = {}
        for index, key in enumerate(keys):
            cached = self._evaluation_cache.get(key)
            if cached is not None:
                self._evaluation_cache.move_to_end(key)
                outputs[index] = cached
                self.model_evaluation_cache_hits += 1
            else:
                missing_by_key.setdefault(key, []).append(index)

        if missing_by_key:
            representative_indices = [
                indices[0] for indices in missing_by_key.values()]
            encoded_batch = [
                encode_state(self.env, states[index], self.encoding_config)
                for index in representative_indices
            ]
            board = torch.stack(
                [encoded.board for encoded in encoded_batch]).to(self.device)
            glob = torch.stack(
                [encoded.global_features for encoded in encoded_batch]
            ).to(self.device)
            with torch.no_grad():
                logits_batch, value_batch = self.model(board, glob)
            logits_batch = logits_batch.detach().cpu()
            values = value_batch.detach().cpu().reshape(-1).tolist()
            self.model_evaluations += len(representative_indices)
            self.model_evaluation_batches += 1
            for batch_index, (key, indices) in enumerate(
                    missing_by_key.items()):
                representative = indices[0]
                predicted_cost = self.value_norm.denormalize(
                    float(values[batch_index]))
                value_cost = max(
                    float(states[representative].remaining), predicted_cost)
                evaluated = (logits_batch[batch_index:batch_index + 1],
                             value_cost)
                for index in indices:
                    outputs[index] = evaluated
                if self.evaluation_cache_size:
                    self._evaluation_cache[key] = evaluated
                    self._evaluation_cache.move_to_end(key)
                while len(self._evaluation_cache) > self.evaluation_cache_size:
                    self._evaluation_cache.popitem(last=False)

        results: list[tuple[list[float], float]] = []
        for state, legal_actions, evaluated in zip(
                states, legal_action_batches, outputs):
            if evaluated is None:  # defensive; every slot is cache hit or miss
                raise RuntimeError("batched model evaluation left an empty slot")
            logits_cpu, value_cost = evaluated
            mask = build_legal_action_mask(
                self.env,
                state,
                self.encoding_config,
                legal_actions=legal_actions,
            )
            probs = masked_policy_probs(logits_cpu, mask.unsqueeze(0))[0]
            priors = [
                float(probs[action_index(state, action, self.encoding_config)])
                for action in legal_actions
            ]
            total = sum(priors)
            if total > 0:
                priors = [prior / total for prior in priors]
            elif legal_actions:
                priors = [1.0 / len(legal_actions)] * len(legal_actions)
            results.append((priors, value_cost))
        return results

    def clear_evaluation_cache(self) -> None:
        """Discard cached outputs after replacing or mutating model weights."""
        self._evaluation_cache.clear()


def search(env, state, model, *, encoding_config, value_norm, device="cpu",
           simulations: int = 800, config: Optional[SearchConfig] = None
           ) -> SearchResult:
    """Convenience wrapper: build the adapter and run a search from ``state``."""
    if config is None:
        const = getattr(value_norm, "constant", 20.0)
        config = SearchConfig(simulations=simulations,
                              value_normalization_constant=const)
    adapter = BlocksortAdapter(env, model, encoding_config, value_norm, device)
    return GraphSearch(adapter, config).run(state)
