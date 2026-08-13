"""PyTorch dataset over the exact policy-value JSONL records.

Each record is reconstructed into an environment state, encoded once into board /
global / mask tensors, and paired with the dense policy and value targets. The
dataset is held in memory (the oracle datasets are small); large files are read
once and parsed once.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..dataset.schema import (
    LABEL_FULL_EXACT,
    LABEL_SEARCH_VISIT_POLICY,
    deserialize_state,
)
from ..environment import Environment
from ..serialization import level_from_dict
from .action_encoding import (
    encode_sparse_policy_target,
    legal_mask_from_record,
    normalized_action_index,
)
from .config import EncodingConfig, EncodingError, ValueNormConfig
from .encoding import encode_state

# Sentinel regret for legal actions that lead to an unsolvable (infinite) state.
_INF_REGRET = 1.0e9


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL dataset into a list of record dicts."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"invalid dataset record on line {line_no} of {path}: "
                    "JSON row must be an object")
            records.append(record)
    return records


def provenance_type(record: dict[str, Any]) -> str:
    prov = record.get("provenance") or []
    if prov and isinstance(prov, list):
        return prov[0].get("sampling", "unknown")
    return "unknown"


class PolicyValueDataset(Dataset):
    """Encoded (state, policy, value) training items from oracle records."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        encoding_config: EncodingConfig,
        value_norm: ValueNormConfig,
        env: Environment | None = None,
    ) -> None:
        self.encoding_config = encoding_config
        self.value_norm = value_norm
        self.env = env or Environment()
        self.items: list[dict[str, Any]] = []
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"failed to encode record {i}: JSON row must be an object")
            try:
                self.items.append(self._encode_record(record))
            except (KeyError, TypeError, ValueError, IndexError, EncodingError) as exc:
                rid = f"{record.get('level_id', '?')}/{record.get('state_key', '?')}"
                raise ValueError(
                    f"failed to encode record {i} ({rid}): {exc}"
                ) from exc

    def _encode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("version") != 1:
            raise EncodingError(f"unsupported record version {record.get('version')}")
        if (record.get("label_kind") == LABEL_SEARCH_VISIT_POLICY
                or record.get("target_source") == "graph_search"
                or record.get("value_exact") is False
                or record.get("policy_exact") is False):
            raise EncodingError(
                "approximate search records are not accepted by the exact "
                "PolicyValueDataset; use ExpertDataset so policy and value "
                "confidence remain explicit")
        level = level_from_dict(record["level"])
        state = deserialize_state(level, record["state"])
        encoded = encode_state(self.env, state, self.encoding_config)

        legal_mask = legal_mask_from_record(record, self.encoding_config)
        policy_target = encode_sparse_policy_target(record, self.encoding_config)

        # Policy probabilities must sum to ~1 and place no mass on illegal cells.
        psum = float(policy_target.sum().item())
        if not abs(psum - 1.0) < 1e-4:
            raise EncodingError(f"policy target sums to {psum}, expected ~1")
        if float((policy_target * (1.0 - legal_mask)).sum().item()) > 1e-6:
            raise EncodingError("policy target places mass on an illegal action")

        # Dense regret tensor aligned with legal action indices.
        regret = torch.zeros(self.encoding_config.action_space_size, dtype=torch.float32)
        regret_known = torch.zeros(
            self.encoding_config.action_space_size, dtype=torch.float32)
        for norm, reg in zip(record["legal_actions"], record["action_regrets"]):
            idx = normalized_action_index(norm, self.encoding_config)
            regret[idx] = _INF_REGRET if reg is None else float(reg)
            if reg is not None:
                regret_known[idx] = 1.0

        raw_value = record["value_target"]["raw_optimal_moves"]
        if (isinstance(raw_value, bool) or not isinstance(raw_value, Real)
                or not math.isfinite(float(raw_value))
                or not float(raw_value).is_integer()):
            raise EncodingError(
                "exact raw_optimal_moves must be a finite integer; "
                "fractional bounded estimates belong in ExpertDataset")
        raw_moves = int(raw_value)
        value = self.value_norm.normalize(raw_moves)
        exit_slots = torch.arange(
            self.encoding_config.action_space_size
        ) % self.encoding_config.move_code_count == \
            self.encoding_config.exit_move_code
        legal_exit_available = bool(
            ((legal_mask > 0) & exit_slots).any())
        optimal_exit_available = bool(
            ((regret == 0) & (regret_known > 0) &
             (legal_mask > 0) & exit_slots).any())
        if not torch.isfinite(policy_target).all():
            raise EncodingError("policy target contains nonfinite values")
        if not torch.isfinite(regret).all():
            raise EncodingError("action regret target contains nonfinite values")
        if not torch.isfinite(encoded.board).all() \
                or not torch.isfinite(encoded.global_features).all():
            raise EncodingError("encoded model inputs contain nonfinite values")
        if not torch.isfinite(torch.tensor(value, dtype=torch.float32)):
            raise EncodingError("value target is nonfinite")

        return {
            "board": encoded.board,
            "global_features": encoded.global_features,
            "valid_cell_mask": encoded.valid_cell_mask,
            "legal_action_mask": legal_mask,
            "policy_target": policy_target,
            "value_target": torch.tensor(value, dtype=torch.float32),
            "raw_optimal_moves": int(raw_moves),
            "action_regret": regret,
            "action_regret_known_mask": regret_known,
            "optimal_actions_complete": torch.tensor(bool(record.get(
                "optimal_actions_complete",
                record.get("label_kind", LABEL_FULL_EXACT) == LABEL_FULL_EXACT,
            ))),
            "legal_exit_available": torch.tensor(legal_exit_available),
            "optimal_exit_available": torch.tensor(optimal_exit_available),
            "remaining_blocks": int(record["remaining_blocks"]),
            "legal_action_count": len(record["legal_actions"]),
            "rows": level.rows,
            "cols": level.cols,
            "provenance_type": provenance_type(record),
            "level_id": record["level_id"],
            "state_key": record["state_key"],
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


_TENSOR_KEYS = (
    "board", "global_features", "valid_cell_mask", "legal_action_mask",
    "policy_target", "value_target", "action_regret",
    "action_regret_known_mask", "optimal_actions_complete",
    "legal_exit_available", "optimal_exit_available",
)
_SCALAR_KEYS = ("raw_optimal_moves", "remaining_blocks", "legal_action_count",
                "rows", "cols")
_META_KEYS = ("provenance_type", "level_id", "state_key")


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep scalars as tensors and metadata as lists."""
    out: dict[str, Any] = {}
    for key in _TENSOR_KEYS:
        out[key] = torch.stack([item[key] for item in batch])
    for key in _SCALAR_KEYS:
        out[key] = torch.tensor([item[key] for item in batch], dtype=torch.long
                                if key != "raw_optimal_moves" else torch.float32)
    for key in _META_KEYS:
        out[key] = [item[key] for item in batch]
    return out
