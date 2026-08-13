"""Audit an exact policy-value dataset before training.

    python -m blocksort.training.audit_dataset data/training/pv_examples.jsonl

Reports distributions and whether the configured board/action limits can
represent every record. Exits non-zero if any record cannot be encoded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Optional

from ..environment import Environment
from ..serialization import level_from_dict
from ..dataset.schema import deserialize_state
from .action_encoding import normalized_action_index
from .config import EncodingConfig, EncodingError
from .dataset import load_records
from .encoding import encode_state


def audit(records: list[dict[str, Any]], config: EncodingConfig) -> dict[str, Any]:
    env = Environment()
    boards = Counter()
    remaining = Counter()
    legal_counts = Counter()
    optimal_counts = Counter()
    optimal_moves = Counter()
    regret_hist = Counter()
    policy_types = Counter()
    label_kinds = Counter()
    value_schemes = Counter()
    provenance = Counter()
    states = set()
    levels = set()
    terminal = 0
    max_distance = 0
    max_rows = max_cols = 0
    unencodable: list[dict[str, Any]] = []

    for i, r in enumerate(records):
        sig = r.get("static_level_signature")
        levels.add(sig or r["level_id"])
        states.add((sig, r["state_key"]))
        boards[(r["level"]["rows"], r["level"]["cols"])] += 1
        remaining[r["remaining_blocks"]] += 1
        legal_counts[len(r["legal_actions"])] += 1
        optimal_counts[len(r["optimal_actions"])] += 1
        optimal_moves[r["optimal_remaining_moves"]] += 1
        policy_types[r.get("policy", {}).get("type", "?")] += 1
        label_kinds[r.get("label_kind", "full-exact")] += 1
        value_schemes[r["value_target"]["normalization"]["scheme"]] += 1
        provenance[(r.get("provenance") or [{}])[0].get("sampling", "?")] += 1
        if not r["legal_actions"]:
            terminal += 1
        for reg in r["action_regrets"]:
            regret_hist["inf" if reg is None else reg] += 1
        for a in r["legal_actions"]:
            if not a["exit"]:
                max_distance = max(max_distance, a["distance"])
        max_rows = max(max_rows, r["level"]["rows"])
        max_cols = max(max_cols, r["level"]["cols"])

        # Encodability check.
        try:
            level = level_from_dict(r["level"])
            state = deserialize_state(level, r["state"])
            encode_state(env, state, config)
            for a in r["legal_actions"]:
                normalized_action_index(a, config)
        except EncodingError as exc:
            unencodable.append({"index": i, "level_id": r["level_id"],
                                "state_key": r["state_key"], "error": str(exc)})

    duplicate_states = len(records) - len(states)
    return {
        "records": len(records),
        "unique_levels": len(levels),
        "unique_states": len(states),
        "duplicate_state_records": duplicate_states,
        "terminal_states": terminal,
        "board_size_distribution": _key_sorted(boards),
        "remaining_block_distribution": _key_sorted(remaining),
        "legal_action_count_distribution": _key_sorted(legal_counts),
        "optimal_action_count_distribution": _key_sorted(optimal_counts),
        "optimal_remaining_move_distribution": _key_sorted(optimal_moves),
        "action_regret_distribution": _key_sorted(regret_hist),
        "policy_target_types": dict(policy_types),
        "label_kinds": dict(label_kinds),
        "value_normalization_schemes": dict(value_schemes),
        "provenance_types": dict(provenance),
        "max_required_slide_distance": max_distance,
        "max_required_board": [max_rows, max_cols],
        "config": {"max_rows": config.max_rows, "max_cols": config.max_cols,
                   "max_slide_distance": config.max_slide_distance,
                   "action_space_size": config.action_space_size},
        "fits_config": (max_rows <= config.max_rows and max_cols <= config.max_cols
                        and max_distance <= config.max_slide_distance
                        and not unencodable),
        "unencodable_records": unencodable,
    }


def _key_sorted(counter: Counter) -> dict:
    return {str(k): counter[k] for k in sorted(counter, key=lambda x: str(x))}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit an exact policy-value dataset.")
    p.add_argument("dataset")
    p.add_argument("--max-rows", type=int, default=8)
    p.add_argument("--max-cols", type=int, default=8)
    p.add_argument("--max-slide-distance", type=int, default=8)
    p.add_argument("--max-blocks", type=int, default=16)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = EncodingConfig(max_rows=args.max_rows, max_cols=args.max_cols,
                            max_slide_distance=args.max_slide_distance,
                            max_blocks=args.max_blocks)
    report = audit(load_records(args.dataset), config)
    print(json.dumps(report, indent=2))
    if not report["fits_config"]:
        print("AUDIT FAILED: dataset does not fit the configured limits.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
