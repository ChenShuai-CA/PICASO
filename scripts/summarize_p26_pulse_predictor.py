"""Summarize P2.6 incremental pulse prediction and its conditional dev gate."""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_p22_architecture import load_rows, scenario_values, two_way_paired_delta


SEEDS = (7, 17, 27, 37, 47)
BRANCHES = ('single', 'dual')
THRESHOLDS = {'single': .50, 'dual': .25}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def two_way_rate(rows_by_seed, branch, rounds, seed):
    scenario_ids = sorted({row['scenario_id'] for rows in rows_by_seed.values()
                           for row in rows if row['branch'] == branch})
    matrices = []
    for policy_seed in SEEDS:
        mapping = {row['scenario_id']: float(row['dangerous'])
                   for row in rows_by_seed[policy_seed] if row['branch'] == branch}
        if set(mapping) != set(scenario_ids):
            raise ValueError(f'screen condition mismatch for {branch} seed {policy_seed}')
        matrices.append([mapping[key] for key in scenario_ids])
    matrix = np.asarray(matrices)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(rounds):
        sampled_seeds = rng.integers(0, len(SEEDS), len(SEEDS))
        sampled_scenarios = rng.integers(0, len(scenario_ids), len(scenario_ids))
        boot.append(matrix[np.ix_(sampled_seeds, sampled_scenarios)].mean())
    return {'rate': float(matrix.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'conditions': len(scenario_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--p22', type=Path, default=Path('runs/20260911_p22_architecture'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    prereg = read_json(args.root / 'preregistration.json')
    if prereg.get('heldout_read') is not False:
        raise ValueError('heldout-read certificate missing')
    train_report = read_json(args.root / 'train_corpus' / 'report.json')
    screen_report = read_json(args.root / 'screen_corpus' / 'report.json')
    if any(report.get('heldout_read') is not False
           for report in (train_report, screen_report)):
        raise ValueError('corpus heldout-read certificate missing')

    rows_by_seed = {
        seed: load_rows(args.root / f'eval_screen_s{seed}' / 'episodes.jsonl')
        for seed in SEEDS}
    mechanism = []
    for branch in BRANCHES:
        stat = two_way_rate(rows_by_seed, branch, args.bootstrap, args.seed)
        seed_rows = [[row for row in rows_by_seed[seed] if row['branch'] == branch]
                     for seed in SEEDS]
        min_valid = min(float(np.mean([row['valid'] for row in rows]))
                        for rows in seed_rows)
        role_invalid = sum(
            reason in ('pedestrian_role', 'occluder_role')
            for rows in seed_rows for row in rows for reason in row['invalid_reasons'])
        threshold = THRESHOLDS[branch]
        passed = (stat['conditions'] >= 15 and stat['rate'] > threshold
                  and stat['ci95'][0] > threshold and min_valid >= .8
                  and role_invalid == 0)
        mechanism.append({
            'branch': branch, 'dangerous_rate': stat['rate'],
            'ci_lo': stat['ci95'][0], 'ci_hi': stat['ci95'][1],
            'conditions': stat['conditions'], 'p25_upper_threshold': threshold,
            'min_valid_rate': min_valid, 'role_invalid': role_invalid,
            'mechanism_pass': passed,
        })
    eligible_for_dev = all(row['mechanism_pass'] for row in mechanism)
    gate = {'eligible_for_dev': eligible_for_dev, 'mechanism': mechanism,
            'bootstrap_rounds': args.bootstrap, 'bootstrap_seed': args.seed,
            'heldout_read': False}
    (args.root / 'mechanism_gate.json').write_text(
        json.dumps(gate, indent=2), encoding='utf-8')

    training = []
    for seed in SEEDS:
        bundle_path = args.root / f'pulse_s{seed}' / 'pulse.pt'
        import torch
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
        history = read_json(args.root / f'pulse_s{seed}' / 'training.json')
        chosen = next(row for row in history
                      if row['epoch'] == bundle['config']['selected_epoch'])
        training.append({
            'seed': seed, 'selected_epoch': bundle['config']['selected_epoch'],
            'val_single_mse': chosen['val_single_mse'],
            'val_dual_mse': chosen['val_dual_mse'],
            'val_branch_balanced_mse': chosen['val_branch_balanced_mse'],
            'parameters': sum(value.numel() for value in bundle['predictor'].values()),
        })

    dev_effects = None
    dev_directories = [args.root / f'eval_dev_s{seed}' for seed in SEEDS]
    if all((directory / 'episodes.jsonl').exists() for directory in dev_directories):
        script_rows = load_rows(args.p22 / 'eval_script' / 'episodes.jsonl')
        dev_effects = []
        for branch in BRANCHES:
            left = {seed: scenario_values(
                load_rows(args.root / f'eval_dev_s{seed}' / 'episodes.jsonl'),
                branch, 'dangerous') for seed in SEEDS}
            script = scenario_values(script_rows, branch, 'dangerous')
            right = {seed: script for seed in SEEDS}
            stat = two_way_paired_delta(left, right, args.bootstrap, args.seed)
            policy_rows = [[row for row in load_rows(
                args.root / f'eval_dev_s{seed}' / 'episodes.jsonl')
                            if row['branch'] == branch] for seed in SEEDS]
            min_valid = min(float(np.mean([row['valid'] for row in rows]))
                            for rows in policy_rows)
            role_invalid = sum(
                reason in ('pedestrian_role', 'occluder_role')
                for rows in policy_rows for row in rows for reason in row['invalid_reasons'])
            dev_effects.append({
                'branch': branch, 'dangerous_delta_vs_script': stat['delta'],
                'ci_lo': stat['ci95'][0], 'ci_hi': stat['ci95'][1],
                'min_valid_rate': min_valid, 'role_invalid': role_invalid,
                'dev_pass': (stat['delta'] > 0 and stat['ci95'][0] > 0
                             and min_valid >= .8 and role_invalid == 0),
            })

    summary = {
        'protocol': 'P2.6 incremental pulse parameter prediction',
        'train_corpus': train_report, 'screen_corpus': screen_report,
        'training': training, 'mechanism': mechanism,
        'eligible_for_dev': eligible_for_dev, 'dev_effects': dev_effects,
        'dev_evaluated': dev_effects is not None, 'heldout_read': False,
    }
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# P2.6 incremental pulse parameter prediction', '',
        f"Training corpus: {train_report['examples']} examples. Independent screen: "
        f"{screen_report['examples']} examples. Heldout was not read.", '',
        '## Mechanism gate', '',
        '| branch | screen conditions | dangerous rate [95% CI] | P2.5 threshold | '
        'min valid | role invalid | pass |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in mechanism:
        lines.append(f"| {row['branch']} | {row['conditions']} | "
                     f"{row['dangerous_rate']:.3f} [{row['ci_lo']:.3f}, "
                     f"{row['ci_hi']:.3f}] | >{row['p25_upper_threshold']:.2f} | "
                     f"{row['min_valid_rate']:.3f} | {row['role_invalid']} | "
                     f"{row['mechanism_pass']} |")
    lines += ['', '## Selected checkpoints', '',
              '| seed | epoch | single val MSE | dual val MSE | balanced MSE | params |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in training:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | "
                     f"{row['val_single_mse']:.5f} | {row['val_dual_mse']:.5f} | "
                     f"{row['val_branch_balanced_mse']:.5f} | {row['parameters']} |")
    lines += ['', f'Eligible for conditional dev evaluation: **{eligible_for_dev}**.', '']
    if dev_effects is not None:
        lines += ['## Conditional dev result', '',
                  '| branch | dangerous delta versus script [95% CI] | pass |',
                  '|---|---:|---:|']
        for row in dev_effects:
            lines.append(f"| {row['branch']} | {row['dangerous_delta_vs_script']:.3f} "
                         f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] | "
                         f"{row['dev_pass']} |")
        lines.append('')
    else:
        lines += ['The preregistered mechanism gate failed, so P2.6 did not read or '
                  'evaluate the frozen development conditions.', '']
    (args.root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'eligible_for_dev': eligible_for_dev,
                      'mechanism': mechanism, 'dev_evaluated': dev_effects is not None,
                      'heldout_read': False}, indent=2))


if __name__ == '__main__':
    main()
