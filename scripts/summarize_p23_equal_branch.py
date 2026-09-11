"""Summarize P2.3 equal-per-branch mixed-policy comparison."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from summarize_p22_architecture import (load_rows, scenario_values,
                                        two_way_paired_delta)


SEEDS = (7, 17, 27, 37, 47)
MARGIN = .05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--p22', type=Path, default=Path('runs/20260911_p22_architecture'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    completed = json.loads((args.root / 'completed.json').read_text())
    p22_complete = json.loads((args.p22 / 'completed.json').read_text())
    if completed.get('heldout_read') is not False or p22_complete.get('heldout_read') is not False:
        raise ValueError('heldout-read certificate missing')
    if completed['condition_set_version'] != p22_complete['condition_set_version']:
        raise ValueError('condition versions differ')

    rows = {}
    per_seed = []
    for seed in SEEDS:
        key = f'mixed12_s{seed}'
        rows[key] = load_rows(args.root / f'eval_mixed_s{seed}' / 'episodes.jsonl')
        history = load_rows(args.root / f'train_mixed_s{seed}' / 'training.jsonl')
        totals = Counter()
        invalid = Counter()
        for item in history:
            totals.update(item['branch_steps'])
            invalid.update(item['invalid_reasons'])
        for branch in ('single', 'dual'):
            subset = [row for row in rows[key] if row['branch'] == branch]
            reasons = Counter(reason for row in subset for reason in row['invalid_reasons'])
            per_seed.append({
                'seed': seed, 'branch': branch,
                'training_branch_steps': totals[branch],
                'training_total_target_collision': invalid['target_target_collision'],
                'valid_rate': float(np.mean([row['valid'] for row in subset])),
                'dangerous_rate': float(np.mean([row['dangerous'] for row in subset])),
                'mean_risk': float(np.mean([row['risk'] for row in subset])),
                'mean_abs_action': float(np.mean([row['mean_abs_action'] for row in subset])),
                'target_target_collision': reasons['target_target_collision'],
                'role_invalid': reasons['pedestrian_role'] + reasons['occluder_role'],
            })
        rows[f'mixed6_s{seed}'] = load_rows(
            args.p22 / f'eval_mixed_s{seed}' / 'episodes.jsonl')
        rows[f'single_s{seed}'] = load_rows(
            args.p22 / f'eval_single_s{seed}' / 'episodes.jsonl')
        rows[f'dual_s{seed}'] = load_rows(
            args.p22 / f'eval_dual_s{seed}' / 'episodes.jsonl')
    script_rows = load_rows(args.p22 / 'eval_script' / 'episodes.jsonl')

    with (args.root / 'per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader(); writer.writerows(per_seed)

    comparisons = [
        ('mixed12-single6', 'mixed12', 'single', 'single'),
        ('mixed12-dual6', 'mixed12', 'dual', 'dual'),
        ('mixed12-mixed6-single', 'mixed12', 'mixed6', 'single'),
        ('mixed12-mixed6-dual', 'mixed12', 'mixed6', 'dual'),
        ('mixed12-script-single', 'mixed12', 'script', 'single'),
        ('mixed12-script-dual', 'mixed12', 'script', 'dual'),
    ]
    effects = []
    for label, left, right, branch in comparisons:
        result = {'comparison': label, 'left': left, 'right': right, 'branch': branch}
        for field in ('dangerous', 'valid', 'risk'):
            left_values = {seed: scenario_values(rows[f'{left}_s{seed}'], branch, field)
                           for seed in SEEDS}
            if right == 'script':
                script = scenario_values(script_rows, branch, field)
                right_values = {seed: script for seed in SEEDS}
            else:
                right_values = {seed: scenario_values(rows[f'{right}_s{seed}'], branch, field)
                                for seed in SEEDS}
            stat = two_way_paired_delta(left_values, right_values,
                                        args.bootstrap, args.seed)
            result[f'{field}_delta'] = stat['delta']
            result[f'{field}_ci_lo'], result[f'{field}_ci_hi'] = stat['ci95']
        effects.append(result)
    with (args.root / 'paired_effects.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)

    decisions = []
    for specialized, branch in (('single', 'single'), ('dual', 'dual')):
        effect = next(row for row in effects if row['right'] == specialized)
        relevant = [row for row in per_seed if row['branch'] == branch]
        stable = min(row['valid_rate'] for row in relevant) >= .8
        role_invalid = sum(row['role_invalid'] for row in relevant)
        decisions.append({
            'branch': branch, 'comparison': effect['comparison'], 'margin': MARGIN,
            'dangerous_delta': effect['dangerous_delta'],
            'ci_lo': effect['dangerous_ci_lo'], 'ci_hi': effect['dangerous_ci_hi'],
            'stable': stable, 'role_invalid': role_invalid,
            'noninferiority_pass': (effect['dangerous_ci_lo'] >= -MARGIN
                                    and stable and role_invalid == 0),
        })
    summary = {
        'protocol': 'P2.3 equal-per-branch budget screen', 'heldout_read': False,
        'branch_interaction_budget': completed['branch_interaction_budget'],
        'decisions': decisions, 'effects': effects,
    }
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# P2.3 equal per-branch budget screen', '',
        f"Mixed policies received exactly {completed['branch_interaction_budget']} training "
        'decision steps in each branch; specialized comparison policies received 6000 steps '
        'in their trained branch. Development conditions only; heldout was not read.', '',
        '## Noninferiority', '',
        '| comparison | branch | dangerous delta [95% CI] | stable | role invalid | pass |',
        '|---|---|---|---:|---:|---:|']
    for row in decisions:
        lines.append(f"| {row['comparison']} | {row['branch']} | "
                     f"{row['dangerous_delta']:.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] | "
                     f"{row['stable']} | {row['role_invalid']} | {row['noninferiority_pass']} |")
    lines += ['', '## All paired effects', '',
              '| comparison | branch | dangerous delta [95% CI] | risk delta [95% CI] |',
              '|---|---|---|---|']
    for row in effects:
        lines.append(f"| {row['comparison']} | {row['branch']} | "
                     f"{row['dangerous_delta']:.4f} [{row['dangerous_ci_lo']:.4f}, "
                     f"{row['dangerous_ci_hi']:.4f}] | {row['risk_delta']:.4f} "
                     f"[{row['risk_ci_lo']:.4f}, {row['risk_ci_hi']:.4f}] |")
    lines += ['', 'This closes the budget comparison on the development set. It does not turn '
              'a comparison against specialized policies into superiority over the script.', '']
    (args.root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(args.root), 'decisions': decisions}, indent=2))


if __name__ == '__main__':
    main()
