"""Graph nodes and the transposition table.

A node represents an *equivalence class* of states identified by
``(static_level_signature, canonical_state_key)``. Equivalent states therefore
share a single node, which may be reached through many parents. Edge statistics
(visit count ``N``, accumulated cost ``W``, mean cost ``Q``, prior ``P``) are
stored on the node per legal action -- a property of the state-action, not of any
particular parent -- so backups stay correct regardless of the path taken.
"""

from __future__ import annotations

from typing import Any, Hashable, Optional


class Node:
    __slots__ = (
        "key", "state", "terminal", "deadlock", "expanded",
        "actions", "locators", "priors",
        "N", "W", "Q", "child_key",
        "total_visits", "value_cost",
    )

    def __init__(self, key: Hashable, state: Any) -> None:
        self.key: Hashable = key
        # Representative state for this equivalence class (fixes edge ordering).
        self.state: Any = state
        self.terminal: bool = False
        self.deadlock: bool = False
        self.expanded: bool = False
        self.actions: list[Any] = []
        self.locators: list[dict[str, Any]] = []
        self.priors: list[float] = []
        self.N: list[int] = []
        self.W: list[float] = []
        self.Q: list[float] = []
        # Cached child identity per edge (canonical, parent-independent).
        self.child_key: list[Optional[Hashable]] = []
        self.total_visits: int = 0
        self.value_cost: float = 0.0

    @property
    def num_actions(self) -> int:
        return len(self.actions)


class TranspositionTable:
    """Maps a canonical graph key to its unique node."""

    def __init__(self) -> None:
        self._nodes: dict[Hashable, Node] = {}
        self.hits: int = 0
        self.misses: int = 0

    def __contains__(self, key: Hashable) -> bool:
        return key in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def get_or_create(self, key: Hashable, state: Any) -> tuple[Node, bool]:
        """Return ``(node, created)``; reuses an existing node for equal keys."""
        node = self._nodes.get(key)
        if node is not None:
            self.hits += 1
            return node, False
        self.misses += 1
        node = Node(key, state)
        self._nodes[key] = node
        return node, True

    def get(self, key: Hashable) -> Optional[Node]:
        return self._nodes.get(key)
