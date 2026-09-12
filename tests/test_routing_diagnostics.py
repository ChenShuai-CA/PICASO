import numpy as np

from scenario_lab.routing_diagnostics import (
    condition_signal_reliability, fit_rbf_router, oracle_pair_scores,
    pair_outcomes, predict_rbf_router, prototype_pairs)


def test_oracle_pair_leave_one_out_preserves_stable_pair_preference():
    outcomes = np.zeros((2, 4, 5), dtype=bool)
    outcomes[0, 0] = True
    outcomes[1, 2] = True
    result = oracle_pair_scores(outcomes, fixed_pair=(0, 1))
    assert result['optimistic_scores'].tolist() == [1.0, 1.0]
    assert result['loo_scores'].tolist() == [1.0, 1.0]
    assert np.all(result['loo_tie_average_scores'] == 1.0)
    assert np.all(result['loo_pair_consistency'] == 1.0)
    assert result['fixed_scores'].tolist() == [1.0, 0.0]


def test_pair_outcomes_returns_all_six_pairs_for_four_heads():
    outcomes = np.zeros((3, 4, 2), dtype=bool)
    pairs, paired = pair_outcomes(outcomes)
    assert np.array_equal(pairs, prototype_pairs(4))
    assert paired.shape == (3, 6, 2)


def test_condition_signal_reliability_separates_deterministic_conditions():
    outcomes = np.zeros((4, 1, 5), dtype=bool)
    outcomes[2:, 0] = True
    result = condition_signal_reliability(outcomes)
    assert result['estimated_draw_noise_variance'][0] == 0.0
    assert result['reliability'][0] == 1.0


def test_rbf_router_fits_and_predicts_nonlinear_probe():
    x = np.arange(30, dtype=float)[:, None]
    y = ((x[:, 0] < 8) | (x[:, 0] > 21)).astype(float)[:, None]
    split = np.where(np.arange(30) % 5 == 0, 'val', 'train')
    model = fit_rbf_router(x, y, split)
    prediction = predict_rbf_router(model, x)
    assert prediction.shape == y.shape
    assert np.mean((prediction[:, 0] >= .5) == y[:, 0]) > .8
