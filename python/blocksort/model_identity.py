"""Stable content identities for persisted model provenance and caching."""

from __future__ import annotations

import ctypes
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def model_state_sha256(model_or_state: Any) -> str:
    """Hash names, dtypes, shapes, and bytes in a model state deterministically."""
    state = (model_or_state.state_dict()
             if hasattr(model_or_state, "state_dict") else model_or_state)
    if not isinstance(state, Mapping):
        raise TypeError("expected a model or model-state mapping")

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_canonical_json(list(tensor.shape)).encode("ascii"))
            # Hash the contiguous CPU memory directly. NumPy is optional in
            # PyTorch and is intentionally not a blocksort dependency.
            if tensor.numel():
                raw = (ctypes.c_ubyte * tensor.nbytes).from_address(
                    tensor.data_ptr())
                digest.update(raw)
        else:
            digest.update(_canonical_json(value).encode("utf-8"))
    return digest.hexdigest()
