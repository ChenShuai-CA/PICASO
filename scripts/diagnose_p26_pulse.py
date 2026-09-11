"""Post-hoc parameter-regression diagnostics for completed P2.6."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


SEEDS = (7, 17, 27, 37, 47)
DIMS = ('amplitude', 'start', 'duration')
RIDGE_ALPHAS = (.01, .1, 1., 10., 100.)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def predict(predictor, data, device):
    with torch.no_grad():
        return predictor(
            torch.as_tensor(data['tokens'], device=device),
            torch.as_tensor(data['token_mask'], device=device),
            torch.as_tensor(data['actor_mask'], device=device)).cpu().numpy()


def metrics(prediction, data, branch, split):
    select = (data['branch'] == branch) & (data['split'] == split)
    target = data['target'][select]
    prediction = prediction[select]
    active = data['actor_mask'][select, -1].astype(bool)
    target_values = target[active]
    predicted_values = prediction[active]
    role_target_std, role_prediction_std = [], []
    for role in range(2):
        role_active = active[:, role]
        if role_active.any():
            role_target_std.append(target[role_active, role].std(0).tolist())
            role_prediction_std.append(prediction[role_active, role].std(0).tolist())
        else:
            role_target_std.append(None)
            role_prediction_std.append(None)
    return {
        'branch': branch, 'split': split, 'examples': int(select.sum()),
        'mse': float(np.square(predicted_values - target_values).sum(1).mean()),
        'target_mean': target_values.mean(0).tolist(),
        'target_std': target_values.std(0).tolist(),
        'prediction_mean': predicted_values.mean(0).tolist(),
        'prediction_std': predicted_values.std(0).tolist(),
        'target_std_by_role': role_target_std,
        'prediction_std_by_role': role_prediction_std,
        'dimension_mse': np.square(predicted_values - target_values).mean(0).tolist(),
    }


def spec_features(manifest):
    fields = ('ego_speed', 'crossing_x', 'pedestrian_y', 'pedestrian_speed',
              'pedestrian_delay', 'occluder_x', 'occluder_y', 'occluder_speed',
              'horizon', 'controller_threshold', 'brake_deceleration',
              'response_delay', 'action_delay_steps', 'target_accel_scale',
              'observation_noise')
    return np.asarray([[float(row['spec'][field]) for field in fields]
                       + [float(row['branch'] == 'dual')] for row in manifest])


def observable_features(data, role):
    tokens = data['tokens'][:, :, role].reshape(len(data['tokens']), -1)
    masks = data['token_mask'][:, :, role].reshape(len(data['tokens']), -1)
    return np.concatenate((tokens, masks.astype(np.float32)), axis=1)


def ridge_predict(train_x, train_y, test_x, alpha):
    mean_x, scale_x = train_x.mean(0), train_x.std(0)
    scale_x[scale_x < 1e-8] = 1.
    x = (train_x - mean_x) / scale_x
    test = (test_x - mean_x) / scale_x
    mean_y = train_y.mean(0)
    # Dual form stays well conditioned when observable history has more columns
    # than the small number of teacher conditions.
    weights = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)),
                                    train_y - mean_y)
    return np.clip(test @ weights + mean_y, -1., 1.)


def linear_predictability(train, screen, train_manifest, screen_manifest):
    results, screen_predictions = [], {}
    for role in range(2):
        active_train = train['actor_mask'][:, -1, role].astype(bool)
        active_screen = screen['actor_mask'][:, -1, role].astype(bool)
        fit = active_train & (train['split'] == 'train')
        val = active_train & (train['split'] == 'val')
        target = train['target'][:, role]
        screen_target = screen['target'][active_screen, role]
        for feature_kind, train_x, screen_x in (
                ('actor_visible_history', observable_features(train, role),
                 observable_features(screen, role)),
                ('full_scenario_spec_diagnostic_ceiling', spec_features(train_manifest),
                 spec_features(screen_manifest))):
            validation = []
            for alpha in RIDGE_ALPHAS:
                prediction = ridge_predict(train_x[fit], target[fit], train_x[val], alpha)
                validation.append((float(np.square(prediction - target[val]).sum(1).mean()),
                                   alpha))
            val_mse, alpha = min(validation)
            # The independent screen remains untouched during alpha selection.
            refit = active_train
            prediction = ridge_predict(train_x[refit], target[refit],
                                       screen_x[active_screen], alpha)
            baseline = np.broadcast_to(target[refit].mean(0), screen_target.shape)
            results.append({
                'role': role, 'feature_kind': feature_kind,
                'train_examples': int(fit.sum()), 'val_examples': int(val.sum()),
                'screen_examples': int(active_screen.sum()), 'selected_alpha': alpha,
                'validation_mse': val_mse,
                'screen_mse': float(np.square(prediction - screen_target).sum(1).mean()),
                'screen_train_mean_baseline_mse': float(
                    np.square(baseline - screen_target).sum(1).mean()),
            })
            screen_predictions.setdefault(feature_kind, np.zeros_like(screen['target']))
            screen_predictions[feature_kind][active_screen, role] = prediction
    return results, screen_predictions


def closed_loop_predictions(predictions, screen, screen_manifest, role_means):
    from scenario_lab.pulse import PulsePolicy
    from scenario_lab.evaluate import run_episode
    from scenario_lab.schema import ScenarioSpec

    predictions = {
        **predictions,
        'train_role_mean': np.broadcast_to(role_means, screen['target'].shape).copy(),
        'selected_cem_teacher_oracle': screen['target'].copy(),
    }
    results = []

    class FixedPredictor(torch.nn.Module):
        def __init__(self, parameters):
            super().__init__()
            self.register_buffer('fixed_parameters', torch.as_tensor(parameters))

        def forward(self, tokens, token_mask, actor_mask):
            return self.fixed_parameters[None].expand(tokens.shape[0], -1, -1)

    for method, values in predictions.items():
        rows = []
        for index, source in enumerate(screen_manifest):
            spec = ScenarioSpec(**source['spec'])
            spec.role_action_mode = 'lane_locked'
            policy = PulsePolicy(FixedPredictor(values[index]), device='cpu')
            outcome = run_episode(policy, spec, int(source['replay_seed']))
            rows.append(outcome)
        for branch in ('single', 'dual'):
            subset = [row for row in rows if row['branch'] == branch]
            results.append({
                'method': method, 'branch': branch, 'conditions': len(subset),
                'dangerous_rate': float(np.mean([row['dangerous'] for row in subset])),
                'valid_rate': float(np.mean([row['valid'] for row in subset])),
                'role_invalid': sum(reason in ('pedestrian_role', 'occluder_role')
                                    for row in subset for reason in row['invalid_reasons']),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from scenario_lab.pulse import PulsePredictor
    from scenario_lab.runtime import resolve_device

    root = args.root.resolve()
    completed = json.loads((root / 'completed.json').read_text(encoding='utf-8'))
    if completed.get('heldout_read') is not False or completed.get('dev_evaluated') is not False:
        raise ValueError('expected completed screen-only P2.6 artifacts')
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    train = load_npz(root / 'train_corpus' / 'pulse.npz')
    screen = load_npz(root / 'screen_corpus' / 'pulse.npz')
    train_manifest = json.loads(
        (root / 'train_corpus' / 'manifest.json').read_text(encoding='utf-8'))
    screen_manifest = json.loads(
        (root / 'screen_corpus' / 'manifest.json').read_text(encoding='utf-8'))

    train_active = train['actor_mask'][train['split'] == 'train', -1].astype(bool)
    train_targets = train['target'][train['split'] == 'train']
    role_means = []
    for role in range(2):
        role_active = train_active[:, role]
        role_means.append(train_targets[role_active, role].mean(0))
    role_means = np.asarray(role_means)
    regressions = []
    for baseline, data, split in (
            ('zero', train, 'val'), ('zero', screen, 'screen'),
            ('train_role_mean', train, 'val'),
            ('train_role_mean', screen, 'screen')):
        prediction = np.zeros_like(data['target'])
        if baseline == 'train_role_mean':
            prediction[:] = role_means
        for branch in ('single', 'dual'):
            regressions.append({'model': baseline, **metrics(prediction, data, branch, split)})

    for seed in SEEDS:
        bundle = torch.load(root / f'pulse_s{seed}' / 'pulse.pt', map_location=device,
                            weights_only=True)
        predictor = PulsePredictor(bundle['hidden']).to(device).eval()
        predictor.load_state_dict(bundle['predictor'])
        for data, split in ((train, 'val'), (screen, 'screen')):
            prediction = predict(predictor, data, device)
            for branch in ('single', 'dual'):
                regressions.append({'model': f'pulse_s{seed}',
                                    **metrics(prediction, data, branch, split)})

    consensus = []
    for branch in ('single', 'dual'):
        outcomes = {}
        for seed in SEEDS:
            rows = [json.loads(line) for line in
                    (root / f'eval_screen_s{seed}' / 'episodes.jsonl').read_text().splitlines()]
            for row in rows:
                if row['branch'] == branch:
                    outcomes.setdefault(row['scenario_id'], []).append(bool(row['dangerous']))
        counts = {hits: sum(sum(values) == hits for values in outcomes.values())
                  for hits in range(6)}
        consensus.append({'branch': branch, 'conditions': len(outcomes),
                          'dangerous_seed_count_histogram': counts})

    predictability, ridge_predictions = linear_predictability(
        train, screen, train_manifest, screen_manifest)
    result = {
        'kind': 'post_hoc_parameter_regression_diagnostic',
        'formal_gate_impact': 'none', 'dev_read': False, 'heldout_read': False,
        'train_role_parameter_means': role_means.tolist(),
        'regressions': regressions, 'screen_outcome_consensus': consensus,
        'linear_predictability': predictability,
        'post_hoc_closed_loop_predictions': closed_loop_predictions(
            ridge_predictions, screen, screen_manifest, role_means),
    }
    diagnostics = root / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    (diagnostics / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = [
        '# P2.6 post-hoc pulse regression diagnostic', '',
        'Training/screen artifacts only; this does not alter the formal mechanism gate. '
        'Development and heldout were not read.', '',
        '## Parameter regression', '',
        '| model | split | branch | MSE | target std by role (amp/start/duration) | '
        'prediction std by role (amp/start/duration) |',
        '|---|---|---|---:|---|---|',
    ]
    for row in regressions:
        def role_text(values):
            return '; '.join('NA' if value is None else '/'.join(
                f'{x:.3f}' for x in value) for value in values)
        lines.append(f"| {row['model']} | {row['split']} | {row['branch']} | "
                     f"{row['mse']:.4f} | {role_text(row['target_std_by_role'])} | "
                     f"{role_text(row['prediction_std_by_role'])} |")
    lines += ['', '## Screen outcome agreement across policy seeds', '',
              '| branch | conditions | dangerous under 0/1/2/3/4/5 seeds |',
              '|---|---:|---|']
    for row in consensus:
        histogram = row['dangerous_seed_count_histogram']
        lines.append(f"| {row['branch']} | {row['conditions']} | "
                     f"{'/'.join(str(histogram[str(i)] if str(i) in histogram else histogram[i]) for i in range(6))} |")
    lines += ['', '## Linear parameter predictability', '',
              '| role | features | train/val/screen | selected alpha | val MSE | '
              'screen MSE | screen mean baseline |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for row in result['linear_predictability']:
        lines.append(f"| {row['role']} | {row['feature_kind']} | "
                     f"{row['train_examples']}/{row['val_examples']}/{row['screen_examples']} | "
                     f"{row['selected_alpha']:.2g} | {row['validation_mse']:.4f} | "
                     f"{row['screen_mse']:.4f} | "
                     f"{row['screen_train_mean_baseline_mse']:.4f} |")
    lines += ['', '## Post-hoc closed-loop parameter predictors', '',
              '| method | branch | dangerous rate | valid | role invalid |',
              '|---|---|---:|---:|---:|']
    for row in result['post_hoc_closed_loop_predictions']:
        lines.append(f"| {row['method']} | {row['branch']} | "
                     f"{row['dangerous_rate']:.3f} | {row['valid_rate']:.3f} | "
                     f"{row['role_invalid']} |")
    (diagnostics / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(diagnostics), 'dev_read': False,
                      'heldout_read': False}, indent=2))


if __name__ == '__main__':
    main()
