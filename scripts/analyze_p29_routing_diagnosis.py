"""Diagnose P2.8 routing ceiling, perturbation stability, and observability."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_lab.pulse_router import (  # noqa: E402
    history_features, paired_bootstrap, portfolio_scores, predict_router, ranking)
from scenario_lab.routing_diagnostics import (  # noqa: E402
    condition_signal_reliability, fit_rbf_router, oracle_pair_scores,
    permutation_alignment, predict_rbf_router)


BRANCHES = ('single', 'dual')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_ridge(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def safe_correlations(prediction, target):
    values = []
    for head in range(target.shape[1]):
        if np.std(prediction[:, head]) < 1e-12 or np.std(target[:, head]) < 1e-12:
            values.append(None)
        else:
            values.append(float(np.corrcoef(prediction[:, head], target[:, head])[0, 1]))
    return values


def model_report(name, prediction, outcomes, fixed_scores, permutation_rounds, seed):
    order = ranking(prediction)
    scores = portfolio_scores(outcomes, order, 2)
    alignment = permutation_alignment(
        outcomes, order, budget=2, rounds=permutation_rounds, seed=seed)
    delta_fixed, ci_fixed = paired_bootstrap(
        scores, fixed_scores, rounds=5000, seed=seed + 100)
    return order, scores, {
        'model': name,
        'top2_rate': float(scores.mean()),
        'delta_vs_fixed2': delta_fixed,
        'delta_vs_fixed2_ci95': ci_fixed,
        'top2_minus_permutation_mean': alignment['delta_vs_permutation_mean'],
        'permutation_mean': alignment['permutation_mean'],
        'permutation_ci95': alignment['permutation_ci95'],
        'permutation_p_one_sided': alignment['permutation_p_one_sided'],
        'distinct_top2_orders': int(len({tuple(row[:2]) for row in order})),
        'modal_top2_share': float(max(
            sum(tuple(other[:2]) == tuple(row[:2]) for other in order)
            for row in order) / len(order)),
        'prediction_std_by_head': prediction.std(axis=0).tolist(),
    }


def head_overlap(outcomes):
    flattened = np.asarray(outcomes, dtype=bool).transpose(1, 0, 2).reshape(outcomes.shape[1], -1)
    matrix = np.zeros((outcomes.shape[1], outcomes.shape[1]), dtype=float)
    for left in range(outcomes.shape[1]):
        for right in range(outcomes.shape[1]):
            union = np.logical_or(flattened[left], flattened[right]).sum()
            intersection = np.logical_and(flattened[left], flattened[right]).sum()
            matrix[left, right] = intersection / union if union else 0.0
    return matrix


def diagnose_branch(branch, train, screen, p28_summary, prereg, condition_rows):
    train_select = train['branch'] == branch
    screen_select = screen['branch'] == branch
    train_local = {key: value[train_select] for key, value in train.items()}
    screen_local = {key: value[screen_select] for key, value in screen.items()}
    training_rows = train_local['split'] == 'train'
    train_target = train_local['outcomes'].mean(axis=2)
    screen_target = screen_local['outcomes'].mean(axis=2)
    fixed_order = np.argsort(
        -train_target[training_rows].mean(axis=0), kind='stable')
    expected_fixed = next(
        row['fixed_order'] for row in p28_summary['screen_results']
        if row['branch'] == branch)
    if fixed_order.tolist() != expected_fixed:
        raise ValueError(f'{branch} fixed order differs from frozen P2.8 report')

    oracle = oracle_pair_scores(screen_local['outcomes'], fixed_order[:2])
    rounds = int(prereg['inference']['condition_bootstrap_rounds'])
    seed = int(prereg['inference']['seed']) + (0 if branch == 'single' else 1000)
    optimistic_delta, optimistic_ci = paired_bootstrap(
        oracle['optimistic_scores'], oracle['fixed_scores'], rounds=rounds, seed=seed)
    loo_delta, loo_ci = paired_bootstrap(
        oracle['loo_scores'], oracle['fixed_scores'], rounds=rounds, seed=seed + 1)
    tie_average_delta, tie_average_ci = paired_bootstrap(
        oracle['loo_tie_average_scores'], oracle['fixed_scores'],
        rounds=rounds, seed=seed + 2)
    tie_worst_delta, tie_worst_ci = paired_bootstrap(
        oracle['loo_tie_worst_scores'], oracle['fixed_scores'],
        rounds=rounds, seed=seed + 3)
    reliability = condition_signal_reliability(screen_local['outcomes'])

    screen_features = history_features(screen_local)
    train_features = history_features(train_local)
    ridge_path = ROOT / 'runs/20260912_p28_conditional_router' / f'router_{branch}.npz'
    ridge_prediction = predict_router(load_ridge(ridge_path), screen_features)
    ridge_order, ridge_scores, ridge = model_report(
        'P2.8 ridge', ridge_prediction, screen_local['outcomes'],
        oracle['fixed_scores'], int(prereg['inference']['permutation_rounds']), seed + 10)

    rbf_model = fit_rbf_router(
        train_features, train_target, train_local['split'],
        gamma_multipliers=prereg['rbf_probe']['gamma_multipliers_over_inverse_median_squared_distance'],
        alphas=prereg['rbf_probe']['alphas'])
    rbf_prediction = predict_rbf_router(rbf_model, screen_features)
    rbf_order, rbf_scores, rbf = model_report(
        'RBF kernel ridge probe', rbf_prediction, screen_local['outcomes'],
        oracle['fixed_scores'], int(prereg['inference']['permutation_rounds']), seed + 20)
    rbf_ridge_delta, rbf_ridge_ci = paired_bootstrap(
        rbf_scores, ridge_scores, rounds=rounds, seed=seed + 30)

    constant = train_target[training_rows].mean(axis=0)
    constant_mse = float(np.mean((screen_target - constant) ** 2))
    ridge_mse = float(np.mean((screen_target - ridge_prediction) ** 2))
    rbf_mse = float(np.mean((screen_target - rbf_prediction) ** 2))
    ridge.update(
        screen_mse=ridge_mse,
        screen_mse_relative_to_training_constant=ridge_mse / constant_mse,
        screen_head_correlations=safe_correlations(ridge_prediction, screen_target),
    )
    rbf.update(
        screen_mse=rbf_mse,
        screen_mse_relative_to_training_constant=rbf_mse / constant_mse,
        screen_head_correlations=safe_correlations(rbf_prediction, screen_target),
        selected_gamma_multiplier=rbf_model['gamma_multiplier'],
        selected_gamma=rbf_model['gamma'], selected_alpha=rbf_model['alpha'],
        training_validation_mse=rbf_model['validation_mse'],
        rbf_top2_minus_ridge2=rbf_ridge_delta,
        rbf_top2_minus_ridge2_ci95=rbf_ridge_ci,
    )

    threshold = prereg['inference']
    opportunity = (
        optimistic_delta >= threshold['material_oracle_delta'] and optimistic_ci[0] > 0)
    stable = (
        loo_delta >= threshold['minimum_practical_delta'] and loo_ci[0] > 0)
    ridge_alignment = (
        ridge['top2_minus_permutation_mean'] >= threshold['minimum_practical_delta']
        and ridge['permutation_p_one_sided'] <= threshold['alpha'])
    rbf_alignment = (
        rbf['top2_minus_permutation_mean'] >= threshold['minimum_practical_delta']
        and rbf['permutation_p_one_sided'] <= threshold['alpha'])
    nonlinear_gain = (
        rbf_ridge_delta >= threshold['minimum_practical_delta'] and rbf_ridge_ci[0] > 0)
    if not opportunity:
        diagnosis = 'prototype_complementarity_ceiling'
    elif not stable:
        diagnosis = 'perturbation_outcome_instability'
    elif rbf_alignment and nonlinear_gain and not ridge_alignment:
        diagnosis = 'ridge_capacity_limitation'
    elif not ridge_alignment and not rbf_alignment:
        diagnosis = 'actor_visible_observability_limitation'
    else:
        diagnosis = 'aligned_routing_signal_requires_new_screen'

    pairs = oracle['pairs']
    head_rates = screen_local['outcomes'].mean(axis=(0, 2))
    pair_rows = []
    for index, pair in enumerate(pairs):
        pair_rate = float(oracle['pair_rates'][index])
        pair_rows.append({
            'pair': pair.tolist(), 'rate': pair_rate,
            'unique_gain_over_better_member': float(pair_rate - max(head_rates[pair])),
        })
    optimistic_frequency = {
        f'{pairs[index, 0]}+{pairs[index, 1]}': int(
            np.sum(oracle['optimistic_pair_indices'] == index))
        for index in range(len(pairs))
    }
    for local_index, uid in enumerate(screen_local['uid']):
        best_pair = pairs[oracle['optimistic_pair_indices'][local_index]]
        condition_rows.append({
            'uid': str(uid), 'branch': branch,
            'fixed2_rate': float(oracle['fixed_scores'][local_index]),
            'optimistic_oracle2_rate': float(oracle['optimistic_scores'][local_index]),
            'loo_oracle2_rate': float(oracle['loo_scores'][local_index]),
            'optimistic_pair': f'{best_pair[0]}+{best_pair[1]}',
            'loo_pair_consistency': float(oracle['loo_pair_consistency'][local_index]),
            **{f'head{head}_rate': float(screen_target[local_index, head])
               for head in range(screen_target.shape[1])},
        })

    return {
        'branch': branch, 'conditions': int(screen_select.sum()),
        'perturbations': int(screen_local['outcomes'].shape[2]),
        'fixed_order': fixed_order.tolist(),
        'fixed2_rate': float(oracle['fixed_scores'].mean()),
        'head_rates': head_rates.tolist(),
        'pair_diagnostics': pair_rows,
        'head_success_jaccard': head_overlap(screen_local['outcomes']).tolist(),
        'optimistic_oracle2_rate': float(oracle['optimistic_scores'].mean()),
        'optimistic_oracle2_minus_fixed2': optimistic_delta,
        'optimistic_oracle2_minus_fixed2_ci95': optimistic_ci,
        'optimistic_pair_frequency': optimistic_frequency,
        'loo_oracle2_rate': float(oracle['loo_scores'].mean()),
        'loo_oracle2_minus_fixed2': loo_delta,
        'loo_oracle2_minus_fixed2_ci95': loo_ci,
        'loo_tie_average_rate': float(oracle['loo_tie_average_scores'].mean()),
        'loo_tie_average_minus_fixed2': tie_average_delta,
        'loo_tie_average_minus_fixed2_ci95': tie_average_ci,
        'loo_tie_worst_rate': float(oracle['loo_tie_worst_scores'].mean()),
        'loo_tie_worst_minus_fixed2': tie_worst_delta,
        'loo_tie_worst_minus_fixed2_ci95': tie_worst_ci,
        'loo_folds_with_pair_ties_fraction': float(np.mean(oracle['loo_tie_counts'] > 1)),
        'mean_loo_tied_pair_count': float(oracle['loo_tie_counts'].mean()),
        'mean_loo_pair_consistency': float(oracle['loo_pair_consistency'].mean()),
        'condition_signal_reliability': {
            key: value.tolist() for key, value in reliability.items()
        },
        'mean_condition_signal_reliability': float(reliability['reliability'].mean()),
        'constant_screen_mse': constant_mse,
        'ridge': ridge, 'rbf_probe': rbf,
        'gates': {
            'prototype_opportunity': bool(opportunity),
            'stable_condition_preference': bool(stable),
            'ridge_feature_alignment': bool(ridge_alignment),
            'rbf_feature_alignment': bool(rbf_alignment),
            'nonlinear_gain_over_ridge': bool(nonlinear_gain),
        },
        'diagnosis': diagnosis,
    }


def fmt_ci(value, ci):
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def write_report(path, results, recommendation):
    lines = [
        '# P2.9 routing ceiling and observability diagnosis', '',
        'This diagnostic reuses only the frozen P2.8 training and screen attempt matrices. '
        'It performs no new rollout and does not read fresh development or heldout conditions. '
        'The perturbations retain sensitivity-domain provenance and are not treated as an '
        'AEB-calibrated distribution.', '',
        '| branch | fixed-2 | optimistic oracle-2 | oracle delta [95% CI] | LOO oracle-2 | '
        'LOO delta [95% CI] | signal reliability | ridge-2 / perm delta / p | '
        'RBF-2 / perm delta / p | diagnosis |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in results:
        ridge, rbf = row['ridge'], row['rbf_probe']
        lines.append(
            f"| {row['branch']} | {row['fixed2_rate']:.3f} | "
            f"{row['optimistic_oracle2_rate']:.3f} | "
            f"{fmt_ci(row['optimistic_oracle2_minus_fixed2'], row['optimistic_oracle2_minus_fixed2_ci95'])} | "
            f"{row['loo_oracle2_rate']:.3f} | "
            f"{fmt_ci(row['loo_oracle2_minus_fixed2'], row['loo_oracle2_minus_fixed2_ci95'])} | "
            f"{row['mean_condition_signal_reliability']:.3f} | "
            f"{ridge['top2_rate']:.3f} / {ridge['top2_minus_permutation_mean']:.3f} / "
            f"{ridge['permutation_p_one_sided']:.4f} | "
            f"{rbf['top2_rate']:.3f} / {rbf['top2_minus_permutation_mean']:.3f} / "
            f"{rbf['permutation_p_one_sided']:.4f} | {row['diagnosis']} |")
    lines += ['', '## Decision', '', recommendation, '']
    for row in results:
        lines += [f"## {row['branch']} evidence", '',
                  f"- Fixed order: {row['fixed_order']}; head rates: "
                  + ', '.join(f"{value:.3f}" for value in row['head_rates']) + '.',
                  f"- Optimistic oracle pair frequencies: {row['optimistic_pair_frequency']}.",
                  f"- Mean leave-one-perturbation-out pair consistency: "
                  f"{row['mean_loo_pair_consistency']:.3f}.",
                  f"- Pair-selection ties occur in {row['loo_folds_with_pair_ties_fraction']:.3f} "
                  f"of LOO folds; tie-averaged LOO rate is {row['loo_tie_average_rate']:.3f} "
                  f"and worst-tie rate is {row['loo_tie_worst_rate']:.3f}.",
                  f"- Ridge screen MSE / constant MSE ratio: "
                  f"{row['ridge']['screen_mse_relative_to_training_constant']:.3f}; "
                  f"RBF ratio: {row['rbf_probe']['screen_mse_relative_to_training_constant']:.3f}.",
                  f"- Gates: {row['gates']}.", '']
    lines += [
        'The optimistic oracle is an upper ceiling because it selects and evaluates on the same '
        'five perturbations. The leave-one-perturbation-out result is the relevant stability test. '
        'All model comparisons on the already-consumed P2.8 screen are diagnostic; any revised '
        'mechanism requires a new preregistered screen before a confirmatory claim.', ''
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p28', default='runs/20260912_p28_conditional_router')
    parser.add_argument('--output', default='runs/20260912_p29_routing_diagnosis')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    p28, output = (ROOT / args.p28).resolve(), (ROOT / args.output).resolve()
    if 'heldout' in str(p28).casefold() or 'heldout' in str(output).casefold():
        raise ValueError('P2.9 must not access heldout paths')
    prereg_path = output / 'preregistration.json'
    if not prereg_path.exists():
        raise FileNotFoundError('write and freeze the P2.9 preregistration before analysis')
    prereg = read_json(prereg_path)
    if not prereg.get('created_before_p29_metric_computation'):
        raise ValueError('P2.9 preregistration is not marked as pre-outcome')
    for item in ('training_dataset', 'screen_dataset', 'ridge_single', 'ridge_dual'):
        source = ROOT / prereg['inputs'][item]['path']
        if sha256(source) != prereg['inputs'][item]['sha256']:
            raise ValueError(f'P2.9 frozen input hash mismatch: {item}')
    context_hashes = {
        'preregistration.json': prereg['inputs']['p28_preregistration_sha256'],
        'frozen_library.json': prereg['inputs']['p28_frozen_library_sha256'],
        'summary.json': prereg['inputs']['p28_summary_sha256'],
    }
    for filename, expected in context_hashes.items():
        if sha256(p28 / filename) != expected:
            raise ValueError(f'P2.9 frozen P2.8 context hash mismatch: {filename}')
    p28_summary = read_json(p28 / 'summary.json')
    if p28_summary['fresh_development_evaluated'] or p28_summary['heldout_read']:
        raise ValueError('P2.9 input boundary violated')
    train = load_dataset(p28 / 'training_dataset.npz')
    screen = load_dataset(p28 / 'screen_dataset.npz')
    condition_rows = []
    results = [
        diagnose_branch(branch, train, screen, p28_summary, prereg, condition_rows)
        for branch in BRANCHES
    ]
    dual = next(row for row in results if row['branch'] == 'dual')
    recommendations = {
        'prototype_complementarity_ceiling': (
            'The dual frozen library lacks enough material pair-selection opportunity. Build a '
            'dual-specialized complementary prototype library before another router experiment.'),
        'perturbation_outcome_instability': (
            'Dual has an optimistic selection ceiling, but its preferred pair does not transfer '
            'across perturbation draws. The next experiment should improve robust outcome estimation '
            'with more paired training-only perturbations or a robust success target before changing '
            'the router architecture.'),
        'ridge_capacity_limitation': (
            'Stable dual preference is visible and the nonlinear probe materially beats ridge. '
            'Pre-register the nonlinear router on a new training/screen split.'),
        'actor_visible_observability_limitation': (
            'Stable dual preference exists but neither legal feature model aligns with it. Redesign '
            'the actor-visible history or causal geometry features without hidden-target truth-fill.'),
        'aligned_routing_signal_requires_new_screen': (
            'The existing legal feature representation contains aligned routing signal. Freeze the '
            'simplest aligned model and test it once on a new preregistered screen.'),
    }
    recommendation = recommendations[dual['diagnosis']]
    summary = {
        'protocol': prereg['protocol'],
        'preregistration_sha256': sha256(prereg_path),
        'input_hashes_verified': True,
        'new_rollouts': 0, 'fresh_development_read': False, 'heldout_read': False,
        'results': results, 'dual_driven_recommendation': recommendation,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'condition_diagnostics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    write_report(output / 'REPORT.md', results, recommendation)
    (output / 'completed.json').write_text(json.dumps({
        'protocol': prereg['protocol'], 'dual_diagnosis': dual['diagnosis'],
        'preregistration_sha256': sha256(prereg_path),
        'new_rollouts': 0, 'fresh_development_read': False, 'heldout_read': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({
        'results': [{
            'branch': row['branch'], 'fixed2': row['fixed2_rate'],
            'optimistic_oracle2': row['optimistic_oracle2_rate'],
            'loo_oracle2': row['loo_oracle2_rate'],
            'ridge2': row['ridge']['top2_rate'], 'rbf2': row['rbf_probe']['top2_rate'],
            'diagnosis': row['diagnosis'],
        } for row in results],
        'new_rollouts': 0, 'fresh_development_read': False, 'heldout_read': False,
    }, indent=2))


if __name__ == '__main__':
    main()
