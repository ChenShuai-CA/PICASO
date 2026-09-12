import numpy as np

from scripts.analyze_abd_braking_source import (
    auc_score,
    balanced_accuracy,
    fit_threshold,
)


def test_auc_score_handles_perfect_ranking_and_ties():
    assert auc_score([0, 0, 1, 1], [0.0, 1.0, 2.0, 3.0]) == 1.0
    assert auc_score([0, 1], [1.0, 1.0]) == 0.5


def test_fit_threshold_detects_low_values_as_aeb():
    y = np.asarray([1, 1, 0, 0])
    values = np.asarray([0.0, 1.0, 3.0, 4.0])
    threshold, direction = fit_threshold(y, values)
    assert direction == 'aeb_if_low'
    pred = (values <= threshold).astype(int)
    assert balanced_accuracy(y, pred)['balanced_accuracy'] == 1.0
