"""Export search-improved policy targets (for a later expert-iteration step).

Each record captures the root's legal actions, search visit counts, the
normalized visit policy, the search value estimate, and solution info. Records
are intentionally self-describing (they carry the model checkpoint and
simulation budget) so a future training run can trust their provenance.

This module only *produces* records; it does not implement expert iteration.
"""

from __future__ import annotations

from typing import Any, Optional

from .result import SearchResult

SEARCH_POLICY_VERSION = 1


def build_search_policy_record(
    state_key: str,
    result: SearchResult,
    *,
    model_checkpoint: str,
    simulations: int,
    static_signature: Optional[str] = None,
) -> dict[str, Any]:
    """A JSON-serializable search-policy record for ``state_key``."""
    return {
        "version": SEARCH_POLICY_VERSION,
        "state_key": state_key,
        "static_level_signature": static_signature,
        "legal_actions": list(result.legal_action_locators),
        "visit_counts": [int(n) for n in result.visit_counts],
        "visit_policy": [float(p) for p in result.visit_policy],
        "search_value": float(result.search_value_normalized),
        "search_value_cost": float(result.search_value_cost),
        "simulations": int(simulations),
        "model_checkpoint": str(model_checkpoint),
        "solved": bool(result.solved),
        "solution_length": result.solution_length,
        "termination_reason": result.termination_reason,
    }


def write_search_policy_jsonl(records: list[dict[str, Any]], path: str) -> None:
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
