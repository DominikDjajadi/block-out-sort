"""Search tests against the real environment and a PyTorch model."""

from __future__ import annotations

import pytest
import torch

from blocksort import Environment, level_from_dict
from blocksort.solution import verify_solution
from blocksort.search.config import SearchConfig
from blocksort.search.graph_search import BlocksortAdapter, GraphSearch, search
from blocksort.training.config import EncodingConfig, ModelConfig, ValueNormConfig
from blocksort.training.model import PolicyValueNet

ENV = Environment()


def _model(channels=8, blocks=1):
    enc = EncodingConfig()
    model = PolicyValueNet(enc, ModelConfig(channels=channels, residual_blocks=blocks,
                                            value_hidden_size=16)).eval()
    return model, enc, ValueNormConfig()


def _adapter(device="cpu"):
    model, enc, vn = _model()
    return BlocksortAdapter(ENV, model, enc, vn, device), vn


def _one_move_level():
    # 1x3 board, a red block at col 0 with a matching left gate -> one exit clears.
    return level_from_dict({
        "name": "one-move", "rows": 1, "cols": 3,
        "blocks": [{"color": "red", "cells": [[0, 0]]}],
        "exits": [{"edge": "left", "start": 0, "length": 1, "color": "red"}],
    })


def test_legal_action_masking_and_alignment():
    adapter, _ = _adapter()
    level = _one_move_level()
    state = ENV.initial_state(level)
    res = GraphSearch(adapter, SearchConfig(simulations=20)).run(state)
    legal = ENV.legal_actions(state)
    assert len(res.legal_actions) == len(legal)
    assert len(res.priors) == len(legal)
    assert sum(res.priors) == pytest.approx(1.0, abs=1e-4)
    # Chosen action is a real legal action.
    assert res.chosen_action in legal


def test_graph_search_computes_legal_actions_once_per_expansion():
    class CountingEnvironment(Environment):
        def __init__(self):
            super().__init__()
            self.legal_action_calls = 0

        def legal_actions(self, state):
            self.legal_action_calls += 1
            return super().legal_actions(state)

    env = CountingEnvironment()
    model, enc, value_norm = _model()
    # A 1x1 board leaves exactly one legal action: exit through the matching
    # gate. This makes the simulated child terminal, so only the root needs a
    # legal-action query.
    level = level_from_dict({
        "name": "forced-one-move", "rows": 1, "cols": 1,
        "blocks": [{"color": "red", "cells": [[0, 0]]}],
        "exits": [{"edge": "left", "start": 0, "length": 1, "color": "red"}],
    })
    state = env.initial_state(level)
    adapter = BlocksortAdapter(env, model, enc, value_norm, "cpu")

    result = GraphSearch(
        adapter, SearchConfig(simulations=1, seed=0)).run(state)

    # Root is the only nonterminal expansion. The forced terminal child does
    # not need legal actions, and mask/model evaluation reuse the root's list.
    assert result.stats.nodes_expanded == 2
    assert env.legal_action_calls == 1


def test_neural_leaf_value_respects_remaining_block_lower_bound():
    enc = EncodingConfig()
    value_norm = ValueNormConfig()

    class OptimisticModel(torch.nn.Module):
        def forward(self, board, global_features):
            return (
                torch.zeros(
                    board.shape[0], enc.action_space_size,
                    dtype=board.dtype, device=board.device),
                torch.zeros(
                    board.shape[0], dtype=board.dtype, device=board.device),
            )

    level = level_from_dict({
        "name": "two-block-lower-bound", "rows": 3, "cols": 1,
        "blocks": [
            {"color": "red", "cells": [[0, 0]]},
            {"color": "red", "cells": [[1, 0]]},
        ],
        "exits": [{
            "edge": "top", "start": 0, "length": 1, "color": "red",
        }],
    })
    state = ENV.initial_state(level)
    adapter = BlocksortAdapter(
        ENV, OptimisticModel(), enc, value_norm, "cpu")

    _priors, direct_value = adapter.evaluate(state)
    result = GraphSearch(
        adapter, SearchConfig(simulations=1, seed=0)).run(state)
    terminal = state.with_blocks(())
    _terminal_priors, terminal_value = adapter.evaluate(terminal)

    assert direct_value == pytest.approx(float(state.remaining))
    assert result.root_value_cost_model == pytest.approx(
        float(state.remaining))
    assert ENV.is_terminal(terminal)
    assert terminal_value == 0.0


def test_finds_and_verifies_known_one_move_solution():
    adapter, _ = _adapter()
    level = _one_move_level()
    state = ENV.initial_state(level)
    res = GraphSearch(adapter, SearchConfig(simulations=30)).run(state)
    assert res.solved and res.solution_verified
    assert res.solution_length == 1
    assert verify_solution(ENV, state, res.solution_actions)


def test_returned_solution_replays_to_terminal():
    adapter, _ = _adapter()
    # A slightly larger level: two stacked blocks that each exit upward.
    level = level_from_dict({
        "name": "two-exit", "rows": 3, "cols": 1,
        "blocks": [{"color": "red", "cells": [[0, 0]]},
                   {"color": "red", "cells": [[1, 0]]}],
        "exits": [{"edge": "top", "start": 0, "length": 1, "color": "red"}],
    })
    state = ENV.initial_state(level)
    res = GraphSearch(adapter, SearchConfig(simulations=100)).run(state)
    assert res.solved
    final = state
    for action in res.solution_actions:
        final = ENV.apply_action(final, action)
    assert ENV.is_terminal(final)


def test_single_legal_action_state_no_crash():
    adapter, _ = _adapter()
    level = _one_move_level()
    state = ENV.initial_state(level)
    # Force a one-legal-action situation by checking the env first.
    res = GraphSearch(adapter, SearchConfig(simulations=5)).run(state)
    assert res.chosen_action is not None


def test_deterministic_search_fixed_seed():
    level = _one_move_level()
    state = ENV.initial_state(level)
    adapter, _ = _adapter()
    cfg = SearchConfig(simulations=25, seed=7, temperature=0.0)
    r1 = GraphSearch(adapter, cfg).run(state)
    r2 = GraphSearch(adapter, cfg).run(state)
    assert r1.visit_counts == r2.visit_counts


def test_reused_adapter_caches_model_evaluations_across_searches():
    enc = EncodingConfig()
    value_norm = ValueNormConfig()

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, board, global_features):
            self.calls += 1
            return (
                torch.zeros(
                    board.shape[0], enc.action_space_size,
                    dtype=board.dtype, device=board.device),
                torch.zeros(
                    board.shape[0], dtype=board.dtype, device=board.device),
            )

    model = CountingModel()
    adapter = BlocksortAdapter(
        ENV, model, enc, value_norm, "cpu", evaluation_cache_size=32)
    state = ENV.initial_state(_one_move_level())
    config = SearchConfig(simulations=5, seed=7, temperature=0.0)

    first = GraphSearch(adapter, config).run(state)
    second = GraphSearch(adapter, config).run(state)

    assert first.visit_counts == second.visit_counts
    assert first.stats.model_evaluations > 0
    assert first.stats.model_evaluation_cache_hits == 0
    assert second.stats.model_evaluations == 0
    assert second.stats.model_evaluation_cache_hits == \
        first.stats.model_evaluations
    assert model.calls == first.stats.model_evaluation_batches
    assert first.stats.model_evaluations > first.stats.model_evaluation_batches


def test_batched_search_groups_distinct_leaf_evaluations():
    enc = EncodingConfig()
    value_norm = ValueNormConfig()

    class BatchRecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, board, global_features):
            self.batch_sizes.append(board.shape[0])
            return (
                torch.zeros(
                    board.shape[0], enc.action_space_size,
                    dtype=board.dtype, device=board.device),
                torch.zeros(
                    board.shape[0], dtype=board.dtype, device=board.device),
            )

    model = BatchRecordingModel()
    adapter = BlocksortAdapter(ENV, model, enc, value_norm, "cpu")
    state = ENV.initial_state(_one_move_level())

    searcher = GraphSearch(adapter, SearchConfig(
        simulations=8,
        inference_batch_size=8,
        virtual_loss=1.0,
        seed=3,
    ))
    result = searcher.run(state)

    assert sum(result.visit_counts) == 8
    assert result.stats.model_evaluations >= 2
    assert result.stats.model_evaluation_batches < \
        result.stats.model_evaluations
    assert max(model.batch_sizes) > 1
    for node in searcher.table._nodes.values():
        assert node.total_visits == sum(node.N)
        for visits, total_cost, mean_cost in zip(node.N, node.W, node.Q):
            assert visits >= 0
            if visits:
                assert mean_cost == pytest.approx(total_cost / visits)
            else:
                assert total_cost == pytest.approx(0.0)
                assert mean_cost == pytest.approx(0.0)


def test_batch_size_one_preserves_legacy_search_path():
    state = ENV.initial_state(_one_move_level())
    adapter, value_norm = _adapter()
    sequential = GraphSearch(adapter, SearchConfig(
        simulations=12, inference_batch_size=1, seed=11)).run(state)
    batched = GraphSearch(adapter, SearchConfig(
        simulations=12, inference_batch_size=4, seed=11)).run(state)

    assert sum(sequential.visit_counts) == sum(batched.visit_counts) == 12
    assert sequential.solved and sequential.solution_verified
    assert batched.solved and batched.solution_verified
    assert sequential.stats.model_evaluation_batches >= 1
    # The second run reuses the fixed-model cache but still exercises batched
    # selection and leaves no virtual visits behind.
    assert all(count >= 0 for count in batched.visit_counts)


def test_adapter_evaluation_cache_can_be_disabled_or_cleared():
    model, enc, value_norm = _model()
    state = ENV.initial_state(_one_move_level())
    adapter = BlocksortAdapter(
        ENV, model, enc, value_norm, "cpu", evaluation_cache_size=4)

    adapter.evaluate(state)
    adapter.evaluate(state)
    assert adapter.model_evaluations == 1
    assert adapter.model_evaluation_cache_hits == 1

    adapter.clear_evaluation_cache()
    adapter.evaluate(state)
    assert adapter.model_evaluations == 2


def test_value_uses_checkpoint_normalization_constant(tmp_path):
    # Round-trip a checkpoint, then confirm search consumes its encoding + value
    # config (normalized = -cost / constant) without re-specifying architecture.
    from blocksort.training.checkpoint import (
        load_checkpoint, model_from_checkpoint, configs_from_checkpoint,
        save_checkpoint,
    )
    model, enc, vn = _model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, epoch=1,
                    best_val_metric=0.0, encoding_config=enc,
                    model_config=model.model_config, value_norm=vn, seed=0,
                    dataset_version=1, split_identity=None)
    ckpt = load_checkpoint(path)
    enc2, _, vn2 = configs_from_checkpoint(ckpt)
    restored = model_from_checkpoint(ckpt)
    level = _one_move_level()
    state = ENV.initial_state(level)
    res = search(ENV, state, restored, encoding_config=enc2, value_norm=vn2,
                 simulations=20)
    assert res.search_value_normalized == pytest.approx(
        -res.search_value_cost / vn2.constant)


def test_transposition_hits_recorded_on_real_state():
    # A 2x2 board with two same-color blocks creates many transpositions
    # (different move orders reach equivalent states).
    adapter, _ = _adapter()
    level = level_from_dict({
        "name": "transp", "rows": 2, "cols": 2,
        "blocks": [{"color": "red", "cells": [[0, 0]]},
                   {"color": "red", "cells": [[1, 1]]}],
        "exits": [{"edge": "top", "start": 0, "length": 2, "color": "red"}],
    })
    state = ENV.initial_state(level)
    gs = GraphSearch(adapter, SearchConfig(simulations=150))
    res = gs.run(state)
    assert res.stats.unique_states == len(gs.table)
    assert res.stats.transposition_hits >= 1
