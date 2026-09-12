# P2.9 routing ceiling and observability diagnosis

This diagnostic reuses only the frozen P2.8 training and screen attempt matrices. It performs no new rollout and does not read fresh development or heldout conditions. The perturbations retain sensitivity-domain provenance and are not treated as an AEB-calibrated distribution.

| branch | fixed-2 | optimistic oracle-2 | oracle delta [95% CI] | LOO oracle-2 | LOO delta [95% CI] | signal reliability | ridge-2 / perm delta / p | RBF-2 / perm delta / p | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| single | 0.324 | 0.417 | 0.093 [0.049, 0.144] | 0.406 | 0.082 [0.037, 0.134] | 0.966 | 0.365 / 0.049 / 0.0058 | 0.357 / 0.051 / 0.0092 | aligned_routing_signal_requires_new_screen |
| dual | 0.232 | 0.343 | 0.111 [0.063, 0.164] | 0.333 | 0.101 [0.053, 0.154] | 0.947 | 0.301 / 0.037 / 0.0108 | 0.307 / 0.040 / 0.0062 | aligned_routing_signal_requires_new_screen |

## Decision

The existing legal feature representation contains aligned routing signal. Freeze the simplest aligned model and test it once on a new preregistered screen.

## single evidence

- Fixed order: [1, 3, 0, 2]; head rates: 0.120, 0.315, 0.093, 0.212.
- Optimistic oracle pair frequencies: {'0+1': 87, '0+2': 16, '0+3': 0, '1+2': 0, '1+3': 0, '2+3': 0}.
- Mean leave-one-perturbation-out pair consistency: 0.988.
- Pair-selection ties occur in 1.000 of LOO folds; tie-averaged LOO rate is 0.411 and worst-tie rate is 0.404.
- Ridge screen MSE / constant MSE ratio: 0.983; RBF ratio: 0.918.
- Gates: {'prototype_opportunity': True, 'stable_condition_preference': True, 'ridge_feature_alignment': True, 'rbf_feature_alignment': True, 'nonlinear_gain_over_ridge': False}.

## dual evidence

- Fixed order: [1, 3, 0, 2]; head rates: 0.152, 0.226, 0.125, 0.180.
- Optimistic oracle pair frequencies: {'0+1': 82, '0+2': 19, '0+3': 0, '1+2': 0, '1+3': 0, '2+3': 0}.
- Mean leave-one-perturbation-out pair consistency: 0.990.
- Pair-selection ties occur in 1.000 of LOO folds; tie-averaged LOO rate is 0.335 and worst-tie rate is 0.325.
- Ridge screen MSE / constant MSE ratio: 0.922; RBF ratio: 0.861.
- Gates: {'prototype_opportunity': True, 'stable_condition_preference': True, 'ridge_feature_alignment': True, 'rbf_feature_alignment': True, 'nonlinear_gain_over_ridge': False}.

The optimistic oracle is an upper ceiling because it selects and evaluates on the same five perturbations. The leave-one-perturbation-out result is the relevant stability test. All model comparisons on the already-consumed P2.8 screen are diagnostic; any revised mechanism requires a new preregistered screen before a confirmatory claim.
