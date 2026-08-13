from __future__ import annotations

from dataclasses import replace

import pytest

from blocksort.cotraining.search_utility import (
    SearchUtilityConfig,
    strict_preference_pairs,
)


def test_strict_preference_pairs_ignores_missing_values_and_ties():
    assert strict_preference_pairs([1.0, 0.5, 1.0, None]) == 2
    assert strict_preference_pairs([0.5, 0.5, None]) == 0


def test_search_utility_config_requires_normalized_aligned_budget_weights(
        tmp_path):
    sample = tmp_path / "sample.jsonl"
    checkpoint = tmp_path / "model.pt"
    sample.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    config = SearchUtilityConfig(
        sample_jsonl=str(sample),
        teacher_checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "out"),
    )
    config.validate()

    with pytest.raises(ValueError, match="must align"):
        replace(config, budget_weights=(1.0,)).validate()
    with pytest.raises(ValueError, match="sum to 1"):
        replace(config, budget_weights=(0.2, 0.3, 0.4)).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config, budgets=(4, 4, 16)).validate()
    with pytest.raises(ValueError, match="max unique states"):
        replace(config, max_unique_states=0).validate()
    with pytest.raises(ValueError, match="preferred iteration"):
        replace(config, preferred_iteration=-1).validate()
