"""CLI: evaluate search vs the raw model against the exact oracle, over budgets.

    python -m blocksort.search.evaluate \\
        --checkpoint runs/pv/best.pt --dataset data/training/pv_examples.jsonl \\
        --split test --budgets 1 25 100 400 800
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from ..environment import Environment
from ..serialization import level_from_dict
from ..dataset.schema import deserialize_state
from .evaluation import evaluate_states
from .export import build_search_policy_record, write_search_policy_jsonl


def _load_states(dataset: str, split: Optional[str], manifest_path: Optional[str],
                 limit: Optional[int]):
    from ..training.dataset import load_records
    from ..training.splits import filter_records_for_split, load_manifest

    records = load_records(dataset)
    if split and manifest_path:
        records = filter_records_for_split(records, load_manifest(manifest_path), split)
    if limit:
        records = records[:limit]
    states = []
    keys = []
    for r in records:
        level = level_from_dict(r["level"])
        states.append(deserialize_state(level, r["state"]))
        keys.append((r.get("static_level_signature"), r["state_key"]))
    return states, keys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exact-oracle evaluation of search.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default=None, choices=[None, "train", "validation",
                                                     "test"])
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 25, 100, 400, 800])
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate at most N states (keeps A* affordable)")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--virtual-loss", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-nodes", type=int, default=250_000)
    p.add_argument("--export-policy", default=None,
                   help="write search-policy JSONL at the largest budget")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    from ..training.predict import load_model_bundle
    from ..training.train import resolve_device

    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, encoding_config, value_norm, _ = load_model_bundle(args.checkpoint, device)
    env = Environment()
    states, keys = _load_states(args.dataset, args.split, args.split_manifest,
                                args.limit)

    report = evaluate_states(
        env, model, encoding_config, value_norm, states,
        budgets=args.budgets, device=device, c_puct=args.c_puct, seed=args.seed,
        inference_batch_size=args.inference_batch_size,
        virtual_loss=args.virtual_loss,
        oracle_max_nodes=args.max_nodes)

    out = {"checkpoint": args.checkpoint, "split": args.split,
           "evaluated_states": len(states), "budgets": report}

    if args.export_policy:
        from .config import SearchConfig
        from .graph_search import BlocksortAdapter, GraphSearch
        budget = max(args.budgets)
        adapter = BlocksortAdapter(env, model, encoding_config, value_norm, device)
        records = []
        for state, (sig, state_key) in zip(states, keys):
            if env.is_terminal(state):
                continue
            cfg = SearchConfig(simulations=budget, c_puct=args.c_puct,
                               inference_batch_size=args.inference_batch_size,
                               virtual_loss=args.virtual_loss,
                               value_normalization_constant=getattr(value_norm,
                                                                    "constant", 20.0),
                               seed=args.seed)
            res = GraphSearch(adapter, cfg).run(state)
            records.append(build_search_policy_record(
                state_key, res, model_checkpoint=args.checkpoint,
                simulations=budget, static_signature=sig))
        write_search_policy_jsonl(records, args.export_policy)
        out["exported_policy_records"] = len(records)
        out["exported_policy_path"] = args.export_policy

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
