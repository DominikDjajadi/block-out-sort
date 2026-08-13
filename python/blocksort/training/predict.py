"""Single-state prediction / debugging CLI and shared inference helpers.

    python -m blocksort.training.predict \\
        --checkpoint runs/pv_baseline/best.pt --level data/example_level.json \\
        --top-k 10 --compare-oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..oracle import Oracle
from ..serialization import level_from_dict
from ..state import State
from .action_encoding import action_index, build_legal_action_mask, decode_action
from .checkpoint import configs_from_checkpoint, load_checkpoint, model_from_checkpoint
from .encoding import encode_state
from .losses import masked_policy_probs


def load_model_bundle(checkpoint_path: str | Path, device: torch.device):
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    encoding_config, model_config, value_norm = configs_from_checkpoint(ckpt)
    model = model_from_checkpoint(ckpt, map_location=device)
    return model, encoding_config, value_norm, ckpt


@torch.no_grad()
def predict_state(model, env: Environment, state: State, encoding_config,
                  device: torch.device):
    encoded = encode_state(env, state, encoding_config)
    board = encoded.board.unsqueeze(0).to(device)
    glob = encoded.global_features.unsqueeze(0).to(device)
    logits, value = model(board, glob)
    return logits[0].cpu(), float(value[0].item())


@torch.no_grad()
def top_legal_actions(
    model, env: Environment, state: State, encoding_config, value_norm,
    device: torch.device, *, top_k: int = 10, oracle: Oracle | None = None,
) -> dict[str, Any]:
    legal_actions = env.legal_actions(state)
    if not legal_actions:
        reason = "solved" if env.is_terminal(state) else "deadlock"
        return {
            "predicted_remaining_moves": None,
            "exact_remaining_moves": 0 if reason == "solved" else None,
            "predicted_normalized_value": None,
            "top_actions": [],
            "termination_reason": reason,
        }
    logits, value = predict_state(model, env, state, encoding_config, device)
    mask = build_legal_action_mask(env, state, encoding_config)
    probs = masked_policy_probs(logits.unsqueeze(0), mask.unsqueeze(0))[0]
    legal_idx = mask.nonzero(as_tuple=True)[0].tolist()
    legal_idx.sort(key=lambda i: -float(probs[i].item()))

    regret_by_index: dict[int, Any] = {}
    exact_value = None
    if oracle is not None:
        analysis = oracle.analyze(state)
        exact_value = analysis.value
        for aa in analysis.actions:
            regret_by_index[action_index(state, aa.action, encoding_config)] = aa.regret

    rows = []
    for idx in legal_idx[:top_k]:
        action = decode_action(env, state, idx, encoding_config)
        block = state.blocks[action.block_index]
        anchor = min((c.r, c.c) for c in block.cells)
        rows.append({
            "action_index": idx,
            "color": block.color,
            "anchor": list(anchor),
            "direction": action.direction.value,
            "move": "EXIT" if action.exit else action.distance,
            "prob": round(float(probs[idx].item()), 4),
            "regret": regret_by_index.get(idx),
            "optimal": (regret_by_index.get(idx) == 0) if oracle else None,
        })

    return {
        "predicted_remaining_moves": round(value_norm.denormalize(value), 3),
        "exact_remaining_moves": exact_value,
        "predicted_normalized_value": round(value, 4),
        "top_actions": rows,
        "termination_reason": "actions_available",
    }


def _load_level(path: str | Path, index: int):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        data = data["levels"]
    if isinstance(data, list):
        data = data[index]
    return level_from_dict(data)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict policy/value for one state.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--level-index", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--compare-oracle", action="store_true")
    p.add_argument("--max-nodes", type=int, default=250_000)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    from .train import resolve_device
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, encoding_config, value_norm, _ = load_model_bundle(args.checkpoint, device)
    env = Environment()
    level = _load_level(args.level, args.level_index)
    state = env.initial_state(level)
    oracle = Oracle(env, max_nodes=args.max_nodes) if args.compare_oracle else None
    result = top_legal_actions(model, env, state, encoding_config, value_norm,
                               device, top_k=args.top_k, oracle=oracle)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
