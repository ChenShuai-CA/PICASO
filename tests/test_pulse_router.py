import numpy as np
import pytest

from scenario_lab.pulse_router import (
    fit_router, history_features, paired_bootstrap, portfolio_scores,
    predict_router, ranking)


def test_history_features_never_truth_fills_masked_tokens():
    tokens = np.arange(2 * 1 * 1 * 2 * 3, dtype=float).reshape(2, 1, 1, 2, 3)
    histories = {
        'tokens': tokens,
        'token_mask': np.array([[[[True, False]]], [[[True, False]]]]),
        'actor_mask': np.ones((2, 1, 1)),
    }
    changed = {key: value.copy() for key, value in histories.items()}
    changed['tokens'][:, :, :, 1] += 10000
    assert np.array_equal(history_features(histories), history_features(changed))


def test_router_learns_condition_dependent_head_order():
    x = np.arange(30, dtype=float)[:, None]
    y = np.column_stack((x[:, 0] < 15, x[:, 0] >= 15)).astype(float)
    split = np.where(np.arange(30) % 5 == 0, 'val', 'train')
    model = fit_router(x, y, split)
    order = ranking(predict_router(model, x))
    assert np.mean(order[:10, 0] == 0) > .8
    assert np.mean(order[-10:, 0] == 1) > .8


def test_portfolio_scores_keep_perturbations_paired():
    outcomes = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=bool)
    order = np.array([[1, 0, 2]])
    assert portfolio_scores(outcomes, order, 1)[0] == pytest.approx(1 / 3)
    assert portfolio_scores(outcomes, order, 2)[0] == pytest.approx(2 / 3)
    assert paired_bootstrap(np.array([1., 1.]), np.array([0., 0.]))[0] == 1.0
