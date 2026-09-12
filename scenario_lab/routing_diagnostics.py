"""Diagnostics for condition-dependent routing over frozen candidate libraries."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def prototype_pairs(n_heads: int) -> np.ndarray:
    """Return every unordered two-head portfolio in stable lexical order."""
    if n_heads < 2:
        raise ValueError('at least two heads are required')
    return np.asarray(list(combinations(range(n_heads), 2)), dtype=int)


def pair_outcomes(outcomes: np.ndarray, pairs: np.ndarray | None = None):
    """Return paired success arrays with shape condition x pair x perturbation."""
    outcomes = np.asarray(outcomes, dtype=bool)
    if outcomes.ndim != 3:
        raise ValueError('outcomes must have shape condition x head x perturbation')
    pairs = prototype_pairs(outcomes.shape[1]) if pairs is None else np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('pairs must have shape pair x 2')
    if np.any(pairs < 0) or np.any(pairs >= outcomes.shape[1]):
        raise ValueError('pair contains an invalid head index')
    paired = np.stack([outcomes[:, pair].any(axis=1) for pair in pairs], axis=1)
    return pairs, paired


def oracle_pair_scores(outcomes: np.ndarray, fixed_pair):
    """Compute optimistic and leave-one-perturbation-out pair-selection scores.

    The optimistic score chooses and evaluates a pair on the same perturbation
    draws.  The leave-one-out score selects on all other draws and evaluates
    only the held draw, exposing whether condition-level head preference is
    stable rather than an artefact of a small perturbation sample.
    """
    pairs, paired = pair_outcomes(outcomes)
    fixed = tuple(sorted(map(int, fixed_pair)))
    pair_lookup = {tuple(map(int, pair)): index for index, pair in enumerate(pairs)}
    if fixed not in pair_lookup:
        raise ValueError('fixed_pair is not a valid two-head portfolio')
    fixed_index = pair_lookup[fixed]
    pair_means = paired.mean(axis=2)
    optimistic_index = np.argmax(pair_means, axis=1)
    optimistic = np.take_along_axis(
        pair_means, optimistic_index[:, None], axis=1)[:, 0]
    fixed_scores = paired[:, fixed_index].mean(axis=1)

    n_conditions, _, n_perturbations = paired.shape
    if n_perturbations < 2:
        raise ValueError('leave-one-perturbation-out requires at least two draws')
    loo_values = np.zeros((n_conditions, n_perturbations), dtype=float)
    loo_tie_average = np.zeros((n_conditions, n_perturbations), dtype=float)
    loo_tie_worst = np.zeros((n_conditions, n_perturbations), dtype=float)
    loo_pair_indices = np.zeros((n_conditions, n_perturbations), dtype=int)
    loo_tie_counts = np.zeros((n_conditions, n_perturbations), dtype=int)
    total = paired.sum(axis=2)
    condition_index = np.arange(n_conditions)
    for held in range(n_perturbations):
        training_rate = (total - paired[:, :, held]) / (n_perturbations - 1)
        selected = np.argmax(training_rate, axis=1)
        loo_pair_indices[:, held] = selected
        loo_values[:, held] = paired[condition_index, selected, held]
        best = training_rate.max(axis=1, keepdims=True)
        tied = np.isclose(training_rate, best)
        loo_tie_counts[:, held] = tied.sum(axis=1)
        held_values = paired[:, :, held]
        loo_tie_average[:, held] = (held_values * tied).sum(axis=1) / tied.sum(axis=1)
        loo_tie_worst[:, held] = np.where(tied, held_values, True).min(axis=1)
    loo = loo_values.mean(axis=1)
    consistency = np.asarray([
        np.bincount(row, minlength=len(pairs)).max() / n_perturbations
        for row in loo_pair_indices
    ])
    return {
        'pairs': pairs,
        'pair_outcomes': paired,
        'pair_rates': paired.mean(axis=(0, 2)),
        'fixed_pair_index': fixed_index,
        'fixed_scores': fixed_scores,
        'optimistic_scores': optimistic,
        'optimistic_pair_indices': optimistic_index,
        'loo_scores': loo,
        'loo_tie_average_scores': loo_tie_average.mean(axis=1),
        'loo_tie_worst_scores': loo_tie_worst.mean(axis=1),
        'loo_pair_indices': loo_pair_indices,
        'loo_tie_counts': loo_tie_counts,
        'loo_pair_consistency': consistency,
    }


def condition_signal_reliability(outcomes: np.ndarray):
    """Estimate head-wise between-condition signal beyond finite-draw noise."""
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim != 3 or outcomes.shape[2] < 2:
        raise ValueError('outcomes require at least two perturbation draws')
    rates = outcomes.mean(axis=2)
    observed_variance = rates.var(axis=0, ddof=1)
    # y(1-y)/(P-1) is unbiased for Var(sample mean | condition).
    noise_variance = np.mean(
        rates * (1.0 - rates) / (outcomes.shape[2] - 1), axis=0)
    signal_variance = np.maximum(observed_variance - noise_variance, 0.0)
    reliability = np.divide(
        signal_variance, observed_variance,
        out=np.zeros_like(signal_variance), where=observed_variance > 0)
    return {
        'observed_variance': observed_variance,
        'estimated_draw_noise_variance': noise_variance,
        'estimated_condition_signal_variance': signal_variance,
        'reliability': reliability,
    }


def permutation_alignment(outcomes, orders, budget=2, rounds=5000, seed=2029):
    """Test whether condition-to-order alignment beats permuted order assignments."""
    from scenario_lab.pulse_router import portfolio_scores

    outcomes = np.asarray(outcomes, dtype=bool)
    orders = np.asarray(orders, dtype=int)
    observed_scores = portfolio_scores(outcomes, orders, budget)
    observed = float(observed_scores.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(rounds, dtype=float)
    for index in range(rounds):
        permuted = orders[rng.permutation(len(orders))]
        null[index] = portfolio_scores(outcomes, permuted, budget).mean()
    p_value = (1.0 + np.sum(null >= observed)) / (rounds + 1.0)
    return {
        'observed_scores': observed_scores,
        'observed_rate': observed,
        'permutation_mean': float(null.mean()),
        'delta_vs_permutation_mean': float(observed - null.mean()),
        'permutation_ci95': np.quantile(null, [.025, .975]).tolist(),
        'permutation_p_one_sided': float(p_value),
    }


def _squared_distance(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    distance = (left * left).sum(1)[:, None] + (right * right).sum(1)[None, :] - 2 * left @ right.T
    return np.maximum(distance, 0.0)


def fit_rbf_router(x, y, split, gamma_multipliers=(.25, 1., 4.),
                   alphas=(.1, 1., 10., 100.)):
    """Fit a fixed-grid RBF kernel-ridge probe using only the training subset."""
    x, y, split = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(split)
    train, val = split == 'train', split == 'val'
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError('x and y must be aligned two-dimensional arrays')
    if not train.any() or not val.any():
        raise ValueError('nonempty train and validation subsets are required')
    mean_x, scale_x = x[train].mean(0), x[train].std(0)
    scale_x[scale_x < 1e-8] = 1.0
    train_x = (x[train] - mean_x) / scale_x
    val_x = (x[val] - mean_x) / scale_x
    train_distance = _squared_distance(train_x, train_x)
    positive = train_distance[np.triu_indices(len(train_x), 1)]
    positive = positive[positive > 1e-12]
    if not len(positive):
        raise ValueError('RBF probe requires non-identical training features')
    inverse_median = 1.0 / float(np.median(positive))
    mean_y = y[train].mean(0)
    centered_y = y[train] - mean_y
    validation = []
    best = None
    for multiplier in gamma_multipliers:
        gamma = float(multiplier) * inverse_median
        kernel_train = np.exp(-gamma * train_distance)
        kernel_val = np.exp(-gamma * _squared_distance(val_x, train_x))
        for alpha in alphas:
            dual = np.linalg.solve(
                kernel_train + float(alpha) * np.eye(len(train_x)), centered_y)
            prediction = np.clip(kernel_val @ dual + mean_y, 0.0, 1.0)
            mse = float(np.mean((prediction - y[val]) ** 2))
            row = {
                'gamma_multiplier': float(multiplier), 'gamma': gamma,
                'alpha': float(alpha), 'validation_mse': mse,
            }
            validation.append(row)
            if best is None or mse < best[0]:
                best = (mse, row, dual)
    _, selected, dual = best
    return {
        'mean_x': mean_x, 'scale_x': scale_x, 'train_x': train_x,
        'mean_y': mean_y, 'dual': dual, 'gamma': selected['gamma'],
        'gamma_multiplier': selected['gamma_multiplier'],
        'alpha': selected['alpha'], 'validation_mse': selected['validation_mse'],
        'validation_grid': validation,
    }


def predict_rbf_router(model, x):
    x = (np.asarray(x, dtype=float) - model['mean_x']) / model['scale_x']
    kernel = np.exp(-model['gamma'] * _squared_distance(x, model['train_x']))
    return np.clip(kernel @ model['dual'] + model['mean_y'], 0.0, 1.0)
