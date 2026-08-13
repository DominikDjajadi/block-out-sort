"""Search configuration.

All quantities are in *cost space* (estimated remaining moves). The value
normalization constant matches the trained model: the network predicts
``normalized = -remaining_moves / constant``, so a node's cost estimate is
``denormalize(normalized) = -normalized * constant`` clamped to the physical
lower bound of one exit move per remaining block.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 800
    # Number of simulations whose unexpanded leaves may be collected before a
    # neural forward pass. Adapters without batch evaluation fall back to the
    # exact sequential path. One explicitly preserves legacy search ordering.
    inference_batch_size: int = 8
    # Temporary extra edge cost used to diversify paths collected in the same
    # batch. Reservations are removed exactly before ordinary backup.
    virtual_loss: float = 1.0
    c_puct: float = 1.5
    # Action-selection temperature for the *returned* visit policy / chosen action.
    # 0 -> deterministic argmax visit count. >0 -> sampled / softened target.
    temperature: float = 0.0
    # Value normalization constant (taken from the checkpoint by default).
    value_normalization_constant: float = 20.0
    # Safety cap on a single simulation's descent depth (the state graph is
    # finite and cycles are rejected, but this bounds pathological cases).
    max_depth: int = 256
    # Optional root exploration noise (off by default for deterministic search).
    dirichlet_alpha: float = 0.0
    dirichlet_weight: float = 0.0
    # First-play-urgency: unvisited edges are scored using the node's own cost
    # estimate (a neutral, parent-derived prior) so priors break first-visit ties.
    seed: int = 0
    # Maximum cost used when the model value is missing/degenerate.
    max_cost: float = 1_000.0
