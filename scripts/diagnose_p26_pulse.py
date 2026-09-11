"""Post-hoc parameter-regression diagnostics for completed P2.6."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


SEEDS = (7, 17, 27, 37, 47)
DIMS = ('amplitude', 'start', 'duration')


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

    result = {
        'kind': 'post_hoc_parameter_regression_diagnostic',
        'formal_gate_impact': 'none', 'dev_read': False, 'heldout_read': False,
        'train_role_parameter_means': role_means.tolist(),
        'regressions': regressions, 'screen_outcome_consensus': consensus,
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
    (diagnostics / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(diagnostics), 'dev_read': False,
                      'heldout_read': False}, indent=2))


if __name__ == '__main__':
    main()
