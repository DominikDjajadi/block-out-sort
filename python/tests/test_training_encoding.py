"""Tests for the neural state and action encodings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from blocksort import Environment, level_from_dict
from blocksort.state import State
from blocksort.training import action_encoding as ae
from blocksort.training.config import EncodingConfig, EncodingError
from blocksort.training.encoding import _ChannelMap, channel_names, encode_state

ENV = Environment()
CFG = EncodingConfig()
REPO_ROOT = Path(__file__).resolve().parents[2]


def lvl(**kw):
    kw.setdefault("name", "t")
    return level_from_dict(kw)


def handcrafted(i):
    return level_from_dict(json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())[i])


# --------------------------------------------------------------------------
# State encoding
# --------------------------------------------------------------------------

def test_encode_shapes_and_channel_names():
    state = ENV.initial_state(handcrafted(0))
    enc = encode_state(ENV, state, CFG)
    assert enc.board.shape == (CFG.num_board_channels, CFG.max_rows, CFG.max_cols)
    assert enc.global_features.shape == (CFG.num_global_features,)
    assert enc.valid_cell_mask.shape == (CFG.max_rows, CFG.max_cols)
    assert len(channel_names(CFG)) == CFG.num_board_channels


def test_encoding_is_deterministic():
    state = ENV.initial_state(handcrafted(1))
    a = encode_state(ENV, state, CFG).board
    b = encode_state(ENV, state, CFG).board
    assert torch.equal(a, b)


def test_padding_and_valid_mask():
    level = lvl(cols=4, rows=3, blocks=[{"color": "red", "cells": [[0, 0]]}],
               exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}])
    enc = encode_state(ENV, ENV.initial_state(level), CFG)
    ch = _ChannelMap(CFG)
    # valid mask is 1 exactly on the real 3x4 board.
    assert enc.valid_cell_mask[:3, :4].sum() == 12
    assert enc.valid_cell_mask.sum() == 12
    # board_extent matches the mask; padded cells carry no occupancy.
    assert torch.equal(enc.board[ch.board_extent], enc.valid_cell_mask)
    assert enc.board[ch.occupancy0:ch.occupancy0 + CFG.num_colors, 3:, :].sum() == 0


def test_oversized_board_raises():
    level = lvl(cols=20, rows=3, blocks=[], exits=[])
    with pytest.raises(EncodingError):
        encode_state(ENV, ENV.initial_state(level), CFG)


def test_state_exceeding_checkpoint_block_limit_raises():
    level = lvl(
        cols=3,
        rows=3,
        blocks=[{"color": "red", "cells": [[1, 1]]}],
        exits=[{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    )
    oversized = State(
        level, tuple(level.blocks), total_blocks=CFG.max_blocks + 1)

    with pytest.raises(EncodingError, match="total block count"):
        encode_state(ENV, oversized, CFG)


def test_block_order_invariance():
    level = lvl(cols=5, rows=5,
                blocks=[{"color": "red", "cells": [[0, 0]]},
                        {"color": "blue", "cells": [[4, 4]]}],
                exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}])
    state = ENV.initial_state(level)
    reordered = State(level, tuple(reversed(state.blocks)), state.total_blocks)
    assert torch.equal(encode_state(ENV, state, CFG).board,
                       encode_state(ENV, reordered, CFG).board)


def test_touching_same_color_blocks_distinct_from_combined():
    ch = _ChannelMap(CFG)
    two = lvl(cols=4, rows=3,
              blocks=[{"color": "red", "cells": [[0, 0]]},
                      {"color": "red", "cells": [[0, 1]]}],
              exits=[{"edge": "top", "start": 0, "length": 4, "color": "red"}])
    combined = lvl(cols=4, rows=3,
                   blocks=[{"color": "red", "cells": [[0, 0], [0, 1]]}],
                   exits=[{"edge": "top", "start": 0, "length": 4, "color": "red"}])
    two_b = encode_state(ENV, ENV.initial_state(two), CFG).board
    comb_b = encode_state(ENV, ENV.initial_state(combined), CFG).board
    # Same color occupancy ...
    assert torch.equal(two_b[ch.occupancy0], comb_b[ch.occupancy0])
    # ... but different anchors and connectivity make them distinguishable.
    assert two_b[ch.anchor].sum() == 2
    assert comb_b[ch.anchor].sum() == 1
    assert comb_b[ch.connect_right][0, 0] == 1
    assert two_b[ch.connect_right][0, 0] == 0
    assert not torch.equal(two_b, comb_b)


def test_frozen_block_encoding():
    ch = _ChannelMap(CFG)
    level = lvl(cols=4, rows=4,
                blocks=[{"color": "red", "cells": [[1, 1]], "unlockAt": 2}],
                exits=[{"edge": "top", "start": 1, "length": 1, "color": "red"}])
    # cleared = total - remaining; total=1 -> cleared 0 -> frozen.
    frozen_state = State(level, (level.blocks[0],), total_blocks=1)
    enc = encode_state(ENV, frozen_state, CFG)
    assert enc.board[ch.frozen_now][1, 1] == 1
    assert enc.board[ch.frozen_remaining][1, 1] == pytest.approx(2 / CFG.max_blocks)
    # total=3 -> cleared 2 -> thawed.
    thawed = State(level, (level.blocks[0],), total_blocks=3)
    enc2 = encode_state(ENV, thawed, CFG)
    assert enc2.board[ch.frozen_now][1, 1] == 0


def test_locked_region_encoding_and_unlock_change():
    ch = _ChannelMap(CFG)
    level = lvl(cols=4, rows=4, blocks=[{"color": "red", "cells": [[0, 0]]}],
                exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}],
                lockedRegions=[{"cells": [[2, 2]], "unlockAt": 1}])
    locked = State(level, (level.blocks[0],), total_blocks=1)  # cleared 0
    enc = encode_state(ENV, locked, CFG)
    assert enc.board[ch.locked_now][2, 2] == 1
    assert enc.board[ch.locked_remaining][2, 2] == pytest.approx(1 / CFG.max_blocks)
    unlocked = State(level, (level.blocks[0],), total_blocks=2)  # cleared 1
    enc2 = encode_state(ENV, unlocked, CFG)
    assert enc2.board[ch.locked_now][2, 2] == 0


def test_gate_color_direction_active_locked():
    ch = _ChannelMap(CFG)
    level = lvl(cols=4, rows=4, blocks=[{"color": "red", "cells": [[0, 0]]}],
                exits=[{"edge": "right", "start": 0, "length": 1, "color": "blue",
                        "unlockAt": 1}])
    state = State(level, (level.blocks[0],), total_blocks=1)  # cleared 0 -> locked
    enc = encode_state(ENV, state, CFG)
    blue = CFG.colors.index("blue")
    right = 3  # right == dir index 3
    # right gate at (0, cols-1)
    assert enc.board[ch.gate_color(right, blue)][0, 3] == 1
    assert enc.board[ch.gate_locked(right)][0, 3] == 1
    assert enc.board[ch.gate_active(right)][0, 3] == 0
    active_state = State(level, (level.blocks[0],), total_blocks=2)  # cleared 1
    enc2 = encode_state(ENV, active_state, CFG)
    assert enc2.board[ch.gate_active(right)][0, 3] == 1
    assert enc2.board[ch.gate_locked(right)][0, 3] == 0


def test_overlapping_same_edge_gates_ok():
    # Two top gates of different colors overlapping a column is representable.
    level = lvl(cols=4, rows=4, blocks=[{"color": "red", "cells": [[1, 1]]}],
                exits=[{"edge": "top", "start": 0, "length": 2, "color": "purple"},
                       {"edge": "top", "start": 1, "length": 2, "color": "red"}])
    enc = encode_state(ENV, ENV.initial_state(level), CFG)
    ch = _ChannelMap(CFG)
    up = 0
    assert enc.board[ch.gate_color(up, CFG.colors.index("purple"))][0, 1] == 1
    assert enc.board[ch.gate_color(up, CFG.colors.index("red"))][0, 1] == 1


def test_corner_gates_different_colors_and_directions_lossless():
    # A top-red gate and a right-blue gate share corner (0, cols-1); per-direction
    # planes keep them distinct (no error, no collapse).
    level = lvl(cols=4, rows=4, blocks=[{"color": "red", "cells": [[2, 2]]}],
                exits=[{"edge": "top", "start": 3, "length": 1, "color": "red"},
                       {"edge": "right", "start": 0, "length": 1, "color": "blue"}])
    enc = encode_state(ENV, ENV.initial_state(level), CFG)
    ch = _ChannelMap(CFG)
    assert enc.board[ch.gate_color(0, CFG.colors.index("red"))][0, 3] == 1   # up/red
    assert enc.board[ch.gate_color(3, CFG.colors.index("blue"))][0, 3] == 1  # right/blue


def test_terminal_state_encoding():
    level = lvl(cols=3, rows=3, blocks=[{"color": "red", "cells": [[0, 0]]}],
                exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}])
    terminal = State(level, (), total_blocks=1)
    enc = encode_state(ENV, terminal, CFG)
    ch = _ChannelMap(CFG)
    assert enc.board[ch.occupancy0:ch.occupancy0 + CFG.num_colors].sum() == 0
    assert enc.board[ch.remaining_plane].sum() == 0  # remaining 0


# --------------------------------------------------------------------------
# Action encoding
# --------------------------------------------------------------------------

def test_all_legal_actions_round_trip_no_collision():
    for i in range(4):
        state = ENV.initial_state(handcrafted(i))
        seen = set()
        for action in ENV.legal_actions(state):
            idx = ae.action_index(state, action, CFG)
            assert idx not in seen, "index collision"
            seen.add(idx)
            decoded = ae.decode_action(ENV, state, idx, CFG)
            a1 = ENV.apply_action(state, action)
            a2 = ENV.apply_action(state, decoded)
            assert ENV.canonical_key(a1) == ENV.canonical_key(a2)


def test_exit_and_slide_distinct():
    level = lvl(cols=5, rows=5, blocks=[{"color": "red", "cells": [[2, 0]]}],
                exits=[{"edge": "right", "start": 2, "length": 1, "color": "red"}])
    state = ENV.initial_state(level)
    slide = exit_a = None
    for a in ENV.legal_actions(state):
        if a.direction.value == "right" and a.exit:
            exit_a = a
        elif a.direction.value == "right" and not a.exit:
            slide = a
    assert slide is not None and exit_a is not None
    assert ae.action_index(state, slide, CFG) != ae.action_index(state, exit_a, CFG)


def test_illegal_and_padded_actions_masked():
    state = ENV.initial_state(handcrafted(0))
    mask = ae.build_legal_action_mask(ENV, state, CFG)
    assert mask.sum() == len(ENV.legal_actions(state))
    # A padded anchor (outside the real board) is always masked.
    padded_index = ae.index_from_parts(7, 7, 0, 0, CFG)
    assert mask[padded_index] == 0


def test_block_order_does_not_change_action_index():
    level = lvl(cols=5, rows=5,
                blocks=[{"color": "red", "cells": [[0, 0]]},
                        {"color": "blue", "cells": [[4, 4]]}],
                exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}])
    state = ENV.initial_state(level)
    reordered = State(level, tuple(reversed(state.blocks)), state.total_blocks)
    # The red block's "up" exit/slide should encode to the same index regardless
    # of block ordering, since the anchor is positional.
    for a in ENV.legal_actions(state):
        block = state.blocks[a.block_index]
        # find equivalent action in reordered
        ridx = next(i for i, b in enumerate(reordered.blocks) if b.cells == block.cells)
        from blocksort.actions import Action
        ra = Action(ridx, a.direction, a.distance, a.exit)
        assert ae.action_index(state, a, CFG) == ae.action_index(reordered, ra, CFG)


def test_max_slide_distance_enforced():
    small = EncodingConfig(max_slide_distance=1)
    level = lvl(cols=5, rows=5, blocks=[{"color": "red", "cells": [[0, 0]]}],
                exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}])
    state = ENV.initial_state(level)
    # The block can slide right up to distance 4 (> max 1) -> error.
    with pytest.raises(EncodingError):
        ae.build_legal_action_mask(ENV, state, small)


def test_sparse_policy_target_alignment():
    records = json.loads((REPO_ROOT / "data" / "training" / "pv_examples.jsonl")
                         .read_text().splitlines()[0])
    target = ae.encode_sparse_policy_target(records, CFG)
    legal = ae.legal_mask_from_record(records, CFG)
    assert target.sum() == pytest.approx(1.0, abs=1e-5)
    assert float((target * (1 - legal)).sum()) == 0.0  # no mass on illegal
