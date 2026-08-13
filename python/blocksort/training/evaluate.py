"""Evaluation CLI: metrics, baselines, grouped breakdowns, and examples.

    python -m blocksort.training.evaluate \\
        --dataset data/training/pv_examples.jsonl \\
        --checkpoint runs/pv_baseline/best.pt --split test \\
        --split-manifest runs/pv_baseline/splits.json --device auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from ..environment import Environment
from ..oracle import Oracle
from ..serialization import level_from_dict
from ..dataset.schema import deserialize_state
from .dataset import PolicyValueDataset, collate_batch, load_records
from .metrics import (
    MetricAccumulator,
    policy_baselines,
    policy_baselines_by_strategy,
    value_baselines,
)
from .predict import load_model_bundle, top_legal_actions
from .splits import filter_records_for_split, load_manifest
from .train import evaluate_loader, resolve_device


def evaluate_split(
    records: list[dict[str, Any]], manifest: dict[str, Any], split: str,
    model, encoding_config, value_norm, device: torch.device,
    *, batch_size: int = 256, policy_weight: float = 1.0,
    value_weight: float = 1.0, value_loss_type: str = "huber",
) -> dict[str, Any]:
    split_records = filter_records_for_split(records, manifest, split)
    if not split_records:
        return {"split": split, "size": 0, "note": "no records in split"}

    dataset = PolicyValueDataset(split_records, encoding_config=encoding_config,
                                 value_norm=value_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_batch)
    metrics, _ = evaluate_loader(
        model, loader, value_norm, device,
        policy_weight=policy_weight, value_weight=value_weight,
        value_loss_type=value_loss_type)

    acc = MetricAccumulator(value_norm)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            logits, value = model(moved["board"], moved["global_features"])
            acc.update(logits, value, moved)

    base = {**policy_baselines(dataset.items, encoding_config),
            **value_baselines(dataset.items, value_norm)}

    return {
        "split": split,
        "size": len(split_records),
        "levels": len({r.get("static_level_signature") or r["level_id"]
                       for r in split_records}),
        "metrics": metrics,
        "baselines": base,
        "baselines_by_policy_strategy": policy_baselines_by_strategy(
            dataset.items, encoding_config),
        "grouped_by_board": _strkeys(acc.grouped("board")),
        "grouped_by_remaining": _strkeys(acc.grouped("remaining")),
        "grouped_by_legal_count": _strkeys(acc.grouped("legal_count")),
        "grouped_by_optimal_moves": _strkeys(acc.grouped("optimal_moves")),
        "grouped_by_provenance": _strkeys(acc.grouped("provenance")),
        "grouped_by_policy_strategy": _strkeys(
            acc.grouped("policy_strategy")),
    }


def _strkeys(d: dict) -> dict:
    return {str(k): v for k, v in d.items()}


def interpretable_examples(
    records: list[dict[str, Any]], manifest: dict[str, Any], split: str,
    model, encoding_config, value_norm, device: torch.device,
    *, n: int = 5, max_nodes: int = 250_000,
) -> list[dict[str, Any]]:
    split_records = filter_records_for_split(records, manifest, split)
    env = Environment()
    oracle = Oracle(env, max_nodes=max_nodes)
    out = []
    step = max(1, len(split_records) // max(n, 1))
    for r in split_records[::step][:n]:
        level = level_from_dict(r["level"])
        state = deserialize_state(level, r["state"])
        pred = top_legal_actions(model, env, state, encoding_config, value_norm,
                                 device, top_k=3, oracle=oracle)
        out.append({
            "level_id": r["level_id"],
            "state_key": r["state_key"],
            "true_optimal_remaining_moves": r["optimal_remaining_moves"],
            "predicted_remaining_moves": pred["predicted_remaining_moves"],
            "top_actions": pred["top_actions"],
            "top_action_is_optimal": pred["top_actions"][0]["optimal"]
            if pred["top_actions"] else None,
        })
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a policy-value checkpoint.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=250_000)
    p.add_argument("--policy-loss-weight", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--value-loss", default="huber", choices=["huber", "mse"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, encoding_config, value_norm, _ = load_model_bundle(args.checkpoint, device)
    records = load_records(args.dataset)
    manifest = load_manifest(args.split_manifest)

    report = evaluate_split(records, manifest, args.split, model, encoding_config,
                            value_norm, device,
                            policy_weight=args.policy_loss_weight,
                            value_weight=args.value_loss_weight,
                            value_loss_type=args.value_loss)
    examples = interpretable_examples(
        records, manifest, args.split, model, encoding_config, value_norm, device,
        n=args.examples, max_nodes=args.max_nodes)
    report["examples"] = examples
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
