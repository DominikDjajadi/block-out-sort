"""CLI: solve a level's initial state with neural-guided graph search.

    python -m blocksort.search.solve \\
        --checkpoint runs/pv/best.pt --level data/example.json --simulations 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..environment import Environment
from ..serialization import level_from_dict
from .config import SearchConfig
from .graph_search import BlocksortAdapter, GraphSearch


def _load_level(path: str, index: int):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        data = data["levels"]
    if isinstance(data, list):
        data = data[index]
    return level_from_dict(data)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural-guided graph search solve.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--level-index", type=int, default=0)
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--virtual-loss", type=float, default=1.0)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--compare-oracle", action="store_true")
    p.add_argument("--max-nodes", type=int, default=250_000)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    from ..training.predict import load_model_bundle
    from ..training.train import resolve_device

    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, encoding_config, value_norm, _ = load_model_bundle(args.checkpoint, device)

    env = Environment()
    level = _load_level(args.level, args.level_index)
    state = env.initial_state(level)

    cfg = SearchConfig(simulations=args.simulations, c_puct=args.c_puct,
                       inference_batch_size=args.inference_batch_size,
                       virtual_loss=args.virtual_loss,
                       temperature=args.temperature, seed=args.seed,
                       value_normalization_constant=getattr(value_norm, "constant",
                                                            20.0))
    adapter = BlocksortAdapter(env, model, encoding_config, value_norm, device)
    result = GraphSearch(adapter, cfg).run(state)

    order = sorted(range(len(result.legal_actions)),
                   key=lambda i: -result.visit_counts[i])
    top = []
    for i in order[:args.top_k]:
        loc = result.legal_action_locators[i]
        top.append({
            "action": _fmt(loc),
            "visits": result.visit_counts[i],
            "visit_prob": round(result.visit_policy[i], 4),
            "prior": round(result.priors[i], 4),
            "q_cost": round(result.action_q_cost[i], 3),
        })

    out = {
        "predicted_remaining_moves_model": round(result.root_value_cost_model, 3),
        "search_estimated_remaining_moves": round(result.search_value_cost, 3),
        "chosen_action": _fmt(result.chosen_action_locator)
        if result.chosen_action_locator else None,
        "solved": result.solved,
        "termination_reason": result.termination_reason,
        "solution_verified": result.solution_verified,
        "solution_length": result.solution_length,
        "top_actions": top,
        "principal_variation": [_fmt(l) for l in result.principal_variation],
        "stats": {
            "simulations": result.stats.simulations,
            "nodes_expanded": result.stats.nodes_expanded,
            "unique_states": result.stats.unique_states,
            "transposition_hits": result.stats.transposition_hits,
            "cycle_rejections": result.stats.cycle_rejections,
            "deadlocks": result.stats.deadlocks,
            "model_evaluations": result.stats.model_evaluations,
            "model_evaluation_batches":
                result.stats.model_evaluation_batches,
            "model_evaluation_cache_hits":
                result.stats.model_evaluation_cache_hits,
            "elapsed_seconds": round(result.stats.elapsed_seconds, 4),
        },
    }

    if args.compare_oracle:
        from ..oracle import Oracle
        oracle = Oracle(env, max_nodes=args.max_nodes)
        v = oracle.optimal_remaining_moves(state)
        out["exact_remaining_moves"] = v
        if result.solution_length is not None and v is not None:
            out["solution_length_gap"] = result.solution_length - v

    print(json.dumps(out, indent=2))
    return 0


def _fmt(loc) -> Optional[str]:
    if not loc:
        return None
    move = "EXIT" if loc.get("exit") else loc.get("distance")
    cells = loc.get("cells")
    anchor = min((r, c) for r, c in cells) if cells else None
    return f"{loc.get('color')} at {anchor} {loc.get('dir')} {move}"


if __name__ == "__main__":
    raise SystemExit(main())
