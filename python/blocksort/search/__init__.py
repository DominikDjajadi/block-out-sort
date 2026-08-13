"""Neural-guided Monte Carlo graph search for Block Out Sort.

A trained policy-value checkpoint guides a PUCT search that shares transpositions
(equivalent states are one node) and rejects cycles within a simulation path.
Everything works in **cost space**: a node's value is its estimated number of
remaining moves to clear the board (>= 0; terminal = 0), and lower is better.

This package implements search and evaluation only -- no expert iteration,
self-play, multiprocessing, or serving (see the milestone scope).
"""

from __future__ import annotations

from .config import SearchConfig
from .result import SearchResult
from .graph_search import BlocksortAdapter, GraphSearch, search

__all__ = [
    "SearchConfig",
    "SearchResult",
    "GraphSearch",
    "BlocksortAdapter",
    "search",
]
