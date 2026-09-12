"""Condition-level routing over a frozen pulse prototype library."""
from __future__ import annotations

import numpy as np


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def history_features(histories):
    """Flatten actor-visible histories while preserving visibility masks.

    Hidden entities remain zero because ``observable_history`` obtains these
    arrays through the actor observation boundary.  The masks are included so
    the scenario-level router can use changes in visibility without truth-fill.
    """
    tokens = np.asarray(histories['tokens'], dtype=np.float64)
    token_mask = np.asarray(histories['token_mask'], dtype=bool)
    actor_mask = np.asarray(histories['actor_mask'], dtype=np.float64)
    if tokens.shape[:-1] != token_mask.shape:
        raise ValueError('token and token-mask shapes differ')
    masked = tokens * token_mask[..., None]
    return np.concatenate((masked.reshape(len(tokens), -1),
                           token_mask.reshape(len(tokens), -1),
                           actor_mask.reshape(len(tokens), -1)), axis=1)


def _ridge_weights(x, y, alpha):
    """Dual ridge solution, efficient when histories have more columns than rows."""
    gram = x @ x.T
    return x.T @ np.linalg.solve(
        gram + float(alpha) * np.eye(len(x), dtype=np.float64), y)


def fit_router(x, y, split, alphas=ALPHAS):
    """Select ridge strength on a fixed validation split and refit on train only."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    split = np.asarray(split)
    train, val = split == 'train', split == 'val'
    if not train.any() or not val.any():
        raise ValueError('router requires nonempty train and validation subsets')
    mean_x, scale_x = x[train].mean(0), x[train].std(0)
    scale_x[scale_x < 1e-8] = 1.0
    mean_y = y[train].mean(0)
    train_x = (x[train] - mean_x) / scale_x
    val_x = (x[val] - mean_x) / scale_x
    scores = []
    for alpha in alphas:
        weights = _ridge_weights(train_x, y[train] - mean_y, alpha)
        prediction = np.clip(val_x @ weights + mean_y, 0.0, 1.0)
        scores.append(float(np.mean((prediction - y[val]) ** 2)))
    selected = int(np.argmin(scores))
    alpha = float(alphas[selected])
    weights = _ridge_weights(train_x, y[train] - mean_y, alpha)
    return {
        'mean_x': mean_x, 'scale_x': scale_x, 'mean_y': mean_y,
        'weights': weights, 'alpha': alpha,
        'validation_mse': scores[selected], 'validation_mse_by_alpha': scores,
        'alphas': list(map(float, alphas)),
    }


def predict_router(model, x):
    x = np.asarray(x, dtype=np.float64)
    standardized = (x - model['mean_x']) / model['scale_x']
    return np.clip(standardized @ model['weights'] + model['mean_y'], 0.0, 1.0)


def ranking(scores):
    """Stable descending head order for every condition."""
    return np.argsort(-np.asarray(scores), axis=1, kind='stable')


def portfolio_scores(outcomes, orders, budget):
    """Per-condition success fraction across paired perturbation draws."""
    outcomes = np.asarray(outcomes, dtype=bool)
    orders = np.asarray(orders, dtype=int)
    if outcomes.ndim != 3 or orders.shape != (len(outcomes), outcomes.shape[1]):
        raise ValueError('outcomes must be N x heads x perturbations and orders N x heads')
    if not 1 <= budget <= outcomes.shape[1]:
        raise ValueError('invalid portfolio budget')
    selected = np.take_along_axis(
        outcomes, orders[:, :budget, None], axis=1)
    return selected.any(axis=1).mean(axis=1)


def paired_bootstrap(left, right, rounds=5000, seed=2028):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError('paired bootstrap requires equal nonempty vectors')
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), (rounds, len(difference)))
    estimates = difference[indices].mean(1)
    return float(difference.mean()), np.quantile(estimates, [.025, .975]).tolist()
