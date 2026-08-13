"""Behaviour-cloning pretraining for the designer policy.

Imitates trajectories produced by the existing reverse-construction generator:
from a valid solvable base, take random *legal* reverse slides (the generator's
scramble) and then ``STOP``. The designer policy is trained with masked
cross-entropy to reproduce those choices, giving adversarial training a sensible
starting point. Optionally seeds bases from a provided level pool.

    python -m blocksort.designer.pretrain \\
        --levels data/generated/levels.jsonl --output-dir runs/designer_pretrain
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..serialization import level_from_dict
from ..training.config import EncodingConfig
from ..training.experiment_identity import (
    TRANSACTION_SCHEMA_VERSION, build_experiment_spec,
    ensure_fresh_output_directory, file_identity, runtime_device_provenance,
    hash_canonical_value, hash_file_streaming, validate_field_classification,
    validate_or_initialize_experiment)
from .actions import STOP_INDEX, DesignerAction, DesignerActionSpace
from .checkpoint import save_designer
from .config import GeneratorConfig
from .encoding import encode_designer_state
from .env import DesignerEnv, DesignerState
from .model import DesignerModelConfig, DesignerNet
from .ppo import masked_log_probs


_PRETRAIN_INPUT_FIELDS = {"levels"}
_PRETRAIN_OPERATIONAL_FIELDS = {"output_dir"}
_PRETRAIN_DERIVED_FIELDS = {"device"}
_PRETRAIN_SEMANTIC_FIELDS = {
    "trajectories", "epochs", "batch_size", "learning_rate", "generator",
    "model_config", "encoding", "seed"}


def _load_base_levels(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        data = data["levels"]
    if isinstance(data, dict):
        data = [data]
    return [level_from_dict(raw) for raw in data]


def generate_bc_examples(env: DesignerEnv, encoding: EncodingConfig, *,
                         trajectories: int, seed: int,
                         base_levels: Optional[list] = None) -> list[dict[str, Any]]:
    """Generate ``(board, globals, mask, action_index)`` imitation examples."""
    action_space = env.action_space
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []

    for t in range(trajectories):
        if base_levels:
            level = base_levels[t % len(base_levels)]
            state = DesignerState(level=level, history=(), budget_used=0,
                                  max_budget=env.mutation_budget, stopped=False,
                                  seed=seed + t)
        else:
            state = env.reset(seed * 7919 + t)

        scramble = rng.randint(1, env.mutation_budget)
        for _ in range(scramble):
            moves = env.legal_moves(state)
            if not moves:
                break
            move = rng.choice(moves)
            enc = encode_designer_state(env.env, state, encoding)
            examples.append({
                "board": enc.board, "globals": enc.global_features,
                "mask": torch.tensor(env.legal_mask(state), dtype=torch.bool),
                "action_index": action_space.move_index(move)})
            state = env.step(state, DesignerAction(
                kind="reverse", anchor=move.anchor, direction=move.direction,
                distance=move.distance))
        # Teach STOP at the end of each trajectory.
        enc = encode_designer_state(env.env, state, encoding)
        examples.append({
            "board": enc.board, "globals": enc.global_features,
            "mask": torch.tensor(env.legal_mask(state), dtype=torch.bool),
            "action_index": STOP_INDEX})
    return examples


class _BCDataset(Dataset):
    def __init__(self, examples): self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def _collate(batch):
    return {
        "board": torch.stack([b["board"] for b in batch]),
        "globals": torch.stack([b["globals"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "action_index": torch.tensor([b["action_index"] for b in batch],
                                     dtype=torch.long)}


def pretrain_designer(*, output_dir: str, levels: Optional[str] = None,
                      trajectories: int = 200, epochs: int = 5,
                      batch_size: int = 64, learning_rate: float = 1e-3,
                      generator: Optional[GeneratorConfig] = None,
                      model_config: Optional[DesignerModelConfig] = None,
                      encoding: Optional[EncodingConfig] = None,
                      device: str = "cpu", seed: int = 42) -> dict[str, Any]:
    validate_field_classification(
        inspect.signature(pretrain_designer).parameters, {
            "semantic": _PRETRAIN_SEMANTIC_FIELDS,
            "operational": _PRETRAIN_OPERATIONAL_FIELDS,
            "input": _PRETRAIN_INPUT_FIELDS,
            "derived": _PRETRAIN_DERIVED_FIELDS,
        })
    enc = encoding or EncodingConfig()
    gen = generator or GeneratorConfig()
    model_cfg = model_config or DesignerModelConfig()
    root = Path(output_dir)
    ensure_fresh_output_directory(
        root, pipeline_label="Designer pretraining")
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
    spec = build_experiment_spec(
        pipeline="designer_pretraining",
        semantic_config={
            "trajectories": trajectories,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "generator": gen,
            "model_config": model_cfg,
            "encoding": enc,
            "mutation_budget": 12,
            "seed": seed,
        },
        inputs={
            "base_levels": (
                file_identity(levels, kind="designer_base_levels")
                if levels else None),
        },
        software_semantics={
            "designer_action_schema_version": 1,
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
            "experiment_identity_version": 1,
            "runtime": runtime_device_provenance(
                requested_device=device, resolved_device=dev),
        })
    experiment_fingerprint, _ = validate_or_initialize_experiment(
        root, spec, run_state=None,
        extra_legacy_markers=("pretrain_summary.json",))
    env = DesignerEnv(gen, mutation_budget=12, encoding=enc)
    base_levels = _load_base_levels(levels) if levels else None
    examples = generate_bc_examples(env, enc, trajectories=trajectories, seed=seed,
                                    base_levels=base_levels)

    torch.manual_seed(seed)
    random.seed(seed)
    model = DesignerNet(enc, model_cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(_BCDataset(examples), batch_size=batch_size, shuffle=True,
                        collate_fn=_collate,
                        generator=torch.Generator().manual_seed(seed))

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total, nb = 0.0, 0
        for batch in loader:
            board = batch["board"].to(dev)
            glob = batch["globals"].to(dev)
            mask = batch["mask"].to(dev)
            target = batch["action_index"].to(dev)
            logits, _ = model(board, glob)
            logp = masked_log_probs(logits, mask)
            loss = F.nll_loss(logp, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            nb += 1
        history.append({"epoch": epoch, "loss": total / max(1, nb)})
        print(json.dumps(history[-1]), flush=True)

    root.mkdir(parents=True, exist_ok=True)
    save_designer(root / "best.pt", model=model, encoding_config=enc,
                  model_config=model_cfg, seed=seed,
                  metadata={"pretrain": True, "examples": len(examples),
                            "experiment_fingerprint": experiment_fingerprint,
                            "history": history})
    summary = {"examples": len(examples), "history": history,
               "experiment_fingerprint": experiment_fingerprint,
               "checkpoint": str(root / "best.pt"),
               "checkpoint_sha256": hash_file_streaming(root / "best.pt"),
               "encoding_fingerprint": hash_canonical_value({
                   "encoding_config": enc.to_dict(),
                   "model_config": model_cfg.to_dict(),
               })}
    (root / "pretrain_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Designer behaviour-cloning pretrain")
    p.add_argument("--levels", default=None,
                   help="optional JSON pool of base levels to scramble from")
    p.add_argument("--output-dir", default="runs/designer_pretrain")
    p.add_argument("--trajectories", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--color-count", type=int, default=3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--residual-blocks", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    gen = GeneratorConfig(rows=args.rows, cols=args.cols,
                          color_count=args.color_count)
    model_cfg = DesignerModelConfig(channels=args.channels,
                                    residual_blocks=args.residual_blocks,
                                    hidden_size=args.hidden_size)
    summary = pretrain_designer(
        output_dir=args.output_dir, levels=args.levels,
        trajectories=args.trajectories, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        generator=gen, model_config=model_cfg, device=args.device, seed=args.seed)
    print(f"\npretrained on {summary['examples']} examples; "
          f"checkpoint={summary['checkpoint']}")


if __name__ == "__main__":
    main()
