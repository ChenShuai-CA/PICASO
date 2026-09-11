"""Summarize P2.5 independent CEM-teacher transfer on the frozen dev set."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from summarize_p22_architecture import load_rows, scenario_values, two_way_paired_delta


SEEDS = (7, 17, 27, 37, 47)
BRANCHES = ('single', 'dual')
METHODS = ('script', 'pure_mappo', 'bc', 'bc_mappo')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--p22', type=Path, default=Path('runs/20260911_p22_architecture'))
    parser.add_argument('--p23', type=Path, default=Path('runs/20260911_p23_equal_branch'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    completed = read_json(args.root / 'completed.json')
    prereg = read_json(args.root / 'preregistration.json')
    teacher_report = read_json(args.root / 'teacher_corpus' / 'report.json')
    if completed.get('heldout_read') is not False or prereg.get('heldout_read') is not False:
        raise ValueError('heldout-read certificate missing')
    if completed['dev_condition_set_version'] != prereg['dev_condition_set_version']:
        raise ValueError('development condition version differs from preregistration')
    if teacher_report.get('condition_fingerprint_overlap') != 0:
        raise ValueError('teacher/development condition overlap detected')

    rows = {'script': load_rows(args.p22 / 'eval_script' / 'episodes.jsonl')}
    for seed in SEEDS:
        rows[f'pure_mappo_s{seed}'] = load_rows(
            args.p23 / f'eval_mixed_s{seed}' / 'episodes.jsonl')
        rows[f'bc_s{seed}'] = load_rows(args.root / f'eval_bc_s{seed}' / 'episodes.jsonl')
        rows[f'bc_mappo_s{seed}'] = load_rows(
            args.root / f'eval_bc_mappo_s{seed}' / 'episodes.jsonl')

    values = {}
    for branch in BRANCHES:
        for field in ('dangerous', 'valid', 'risk'):
            script = scenario_values(rows['script'], branch, field)
            values[('script', branch, field)] = {seed: script for seed in SEEDS}
            for method in ('pure_mappo', 'bc', 'bc_mappo'):
                values[(method, branch, field)] = {
                    seed: scenario_values(rows[f'{method}_s{seed}'], branch, field)
                    for seed in SEEDS}

    per_seed = []
    bc_validation = []
    for seed in SEEDS:
        pretraining = read_json(args.root / f'bc_s{seed}' / 'pretraining.json')
        bundle = torch.load(args.root / f'bc_s{seed}' / 'prior.pt',
                            map_location='cpu', weights_only=True)
        bc_validation.append({
            'seed': seed, 'final_train_weighted_mse': pretraining[-1]['train_weighted_mse'],
            'final_val_mse': pretraining[-1]['val_mse'],
            'best_val_mse': min(row['val_mse'] for row in pretraining),
            'actor_parameters': sum(value.numel() for value in bundle['actor'].values()),
        })
        training = load_rows(args.root / f'train_bc_mappo_s{seed}' / 'training.jsonl')
        training_invalid = Counter(reason for row in training
                                   for reason, count in row['invalid_reasons'].items()
                                   for _ in range(count))
        for method in ('pure_mappo', 'bc', 'bc_mappo'):
            for branch in BRANCHES:
                subset = [row for row in rows[f'{method}_s{seed}'] if row['branch'] == branch]
                reasons = Counter(reason for row in subset for reason in row['invalid_reasons'])
                script_subset = [row for row in rows['script'] if row['branch'] == branch]
                per_seed.append({
                    'method': method, 'seed': seed, 'branch': branch,
                    'valid_rate': float(np.mean([row['valid'] for row in subset])),
                    'dangerous_rate': float(np.mean([row['dangerous'] for row in subset])),
                    'mean_risk': float(np.mean([row['risk'] for row in subset])),
                    'mean_abs_action': float(np.mean([row['mean_abs_action'] for row in subset])),
                    'outcomes_changed_vs_script': sum(
                        left['dangerous'] != right['dangerous'] or left['valid'] != right['valid']
                        for left, right in zip(subset, script_subset)),
                    'role_invalid': reasons['pedestrian_role'] + reasons['occluder_role'],
                    'target_target_collision': reasons['target_target_collision'],
                    'bc_mappo_training_target_collision': (
                        training_invalid['target_target_collision'] if method == 'bc_mappo' else ''),
                })
    with (args.root / 'per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader(); writer.writerows(per_seed)
    with (args.root / 'bc_validation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bc_validation[0]))
        writer.writeheader(); writer.writerows(bc_validation)

    comparisons = [
        ('bc-script', 'bc', 'script'),
        ('bc_mappo-script', 'bc_mappo', 'script'),
        ('bc_mappo-pure_mappo', 'bc_mappo', 'pure_mappo'),
        ('bc_mappo-bc', 'bc_mappo', 'bc'),
        ('pure_mappo-script', 'pure_mappo', 'script'),
    ]
    effects = []
    for label, left, right in comparisons:
        for branch in BRANCHES:
            result = {'comparison': label, 'left': left, 'right': right, 'branch': branch}
            for field in ('dangerous', 'valid', 'risk'):
                stat = two_way_paired_delta(values[(left, branch, field)],
                                            values[(right, branch, field)],
                                            args.bootstrap, args.seed)
                result[f'{field}_delta'] = stat['delta']
                result[f'{field}_ci_lo'], result[f'{field}_ci_hi'] = stat['ci95']
            effects.append(result)
    with (args.root / 'paired_effects.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)

    decisions = []
    for branch in BRANCHES:
        effect = next(row for row in effects
                      if row['comparison'] == 'bc_mappo-script' and row['branch'] == branch)
        relevant = [row for row in per_seed
                    if row['method'] == 'bc_mappo' and row['branch'] == branch]
        stable = min(row['valid_rate'] for row in relevant) >= .8
        role_invalid = sum(row['role_invalid'] for row in relevant)
        passed = (effect['dangerous_delta'] > 0 and effect['dangerous_ci_lo'] > 0
                  and stable and role_invalid == 0)
        decisions.append({
            'branch': branch, 'dangerous_delta_vs_script': effect['dangerous_delta'],
            'ci_lo': effect['dangerous_ci_lo'], 'ci_hi': effect['dangerous_ci_hi'],
            'stable': stable, 'role_invalid': role_invalid, 'primary_pass': passed,
        })
    transfer = [row for row in effects if row['comparison'] == 'bc_mappo-pure_mappo']
    bc_effect = [row for row in effects if row['comparison'] == 'bc-script']
    primary_all = all(row['primary_pass'] for row in decisions)
    transfer_all = all(row['dangerous_ci_lo'] > 0 for row in transfer)
    bc_all = all(row['dangerous_ci_lo'] > 0 for row in bc_effect)
    if primary_all and transfer_all:
        diagnosis = 'teacher_transfer_success_on_development'
    elif bc_all and not primary_all:
        diagnosis = 'behavior_cloning_signal_present_online_finetuning_erases_teacher'
    elif primary_all:
        diagnosis = 'development_superiority_pass_transfer_increment_uncertain'
    else:
        diagnosis = 'teacher_distillation_or_observability_insufficient'

    summary = {
        'protocol': 'P2.5 independent CEM teacher transfer',
        'heldout_read': False, 'teacher': teacher_report,
        'bc_validation': bc_validation, 'decisions': decisions,
        'effects': effects, 'diagnosis': diagnosis,
        'eligible_for_heldout_decision': primary_all,
    }
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# P2.5 independent CEM-teacher transfer', '',
        f"Teacher corpus: {teacher_report['examples']} replay-verified episodes; "
        f"training/development fingerprint overlap={teacher_report['condition_fingerprint_overlap']}. "
        'Development conditions only; heldout was not read.', '',
        '## Primary gate: BC-to-MAPPO versus script', '',
        '| branch | dangerous delta [95% CI] | stable | role invalid | pass |',
        '|---|---|---:|---:|---:|',
    ]
    for row in decisions:
        lines.append(f"| {row['branch']} | {row['dangerous_delta_vs_script']:.4f} "
                     f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] | {row['stable']} | "
                     f"{row['role_invalid']} | {row['primary_pass']} |")
    lines += ['', '## Mechanism effects', '',
              '| comparison | branch | dangerous delta [95% CI] | risk delta [95% CI] |',
              '|---|---|---|---|']
    for row in effects:
        lines.append(f"| {row['comparison']} | {row['branch']} | "
                     f"{row['dangerous_delta']:.4f} [{row['dangerous_ci_lo']:.4f}, "
                     f"{row['dangerous_ci_hi']:.4f}] | {row['risk_delta']:.4f} "
                     f"[{row['risk_ci_lo']:.4f}, {row['risk_ci_hi']:.4f}] |")
    lines += ['', '## BC fit', '',
              '| seed | final weighted train MSE | final val MSE | best val MSE | actor params |',
              '|---:|---:|---:|---:|---:|']
    for row in bc_validation:
        lines.append(f"| {row['seed']} | {row['final_train_weighted_mse']:.6f} | "
                     f"{row['final_val_mse']:.6f} | {row['best_val_mse']:.6f} | "
                     f"{row['actor_parameters']} |")
    lines += ['', f'Diagnosis: **{diagnosis}**.', '',
              f"Eligible for a heldout decision under the preregistered development gate: "
              f'**{primary_all}**. No heldout data were read.', '']
    (args.root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'decisions': decisions, 'diagnosis': diagnosis,
                      'eligible_for_heldout_decision': primary_all}, indent=2))


if __name__ == '__main__':
    main()
