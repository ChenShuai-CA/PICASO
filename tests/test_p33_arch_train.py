"""P3.3.3 architecture-mode training contracts (pure budget/early-stopping logic)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from research_tasks import train_p33_nominal as train  # noqa: E402


def _config():
    return {
        "training": {
            "micro_batch_size": 16,
            "target_effective_batch_size": 256,
            "max_epochs": 30,
            "early_stopping_metric": "source_macro_minade_at_6",
            "early_stopping_patience": 5,
            "formal_seeds": [7, 17, 27],
            "t1_sampling": {"num_samples": 6, "temperature": 1.0, "top_p": 0.95},
        }
    }


def test_budget_derivation_matches_frozen_training_config():
    budget = train.derive_architecture_budget(_config())
    assert budget["batch_per_source"] == 8
    assert budget["micro_batches_per_step"] == 16  # 256 effective / 16 micro
    assert budget["max_epochs"] == 30
    assert budget["patience"] == 5
    assert budget["seed"] == 7
    assert budget["early_stopping_metric"] == "source_macro_minade_at_6"


def test_budget_rejects_indivisible_effective_batch():
    bad = _config()
    bad["training"]["target_effective_batch_size"] = 250
    with pytest.raises(ValueError):
        train.derive_architecture_budget(bad)


def test_early_stopping_tracks_best_and_stagnation():
    best, stagnant, improved = train.update_early_stopping(float("inf"), 0, 10.0, 5)
    assert (best, stagnant, improved) == (10.0, 0, True)
    best, stagnant, improved = train.update_early_stopping(best, stagnant, 11.0, 5)
    assert (best, stagnant, improved) == (10.0, 1, False)
    best, stagnant, improved = train.update_early_stopping(best, stagnant, 9.5, 5)
    assert (best, stagnant, improved) == (9.5, 0, True)
    for _ in range(5):
        best, stagnant, improved = train.update_early_stopping(best, stagnant, 9.6, 5)
    assert stagnant == 5 and not improved  # stop condition reached
