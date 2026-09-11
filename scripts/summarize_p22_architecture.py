"""Summarize P2.2 shared-vs-specialized development results."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


SEEDS = (7, 17, 27, 37, 47)
MODES = ('mixed', 'single', 'dual')
MARGIN = 0.05


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def scenario_values(rows, branch, field):
    subset = [row for row in rows if row['branch'] == branch]
    groups = sorted({row['scenario_id'] for row in subset})
    return {group: float(np.mean([row[field] for row in subset
                                  if row['scenario_id'] == group])) for group in groups}


def two_way_paired_delta(left_by_seed, right_by_seed, rounds=2000, seed=2026):
    seeds = sorted(left_by_seed)
    if seeds != sorted(right_by_seed) or not seeds:
        raise ValueError('seed grids differ')
    scenarios = sorted(left_by_seed[seeds[0]])
    if not scenarios or any(sorted(left_by_seed[s]) != scenarios
                            or sorted(right_by_seed[s]) != scenarios for s in seeds):
        raise ValueError('scenario grids differ')
    delta = np.asarray([[left_by_seed[s][g] - right_by_seed[s][g]
                         for g in scenarios] for s in seeds])
    rng = np.random.default_rng(seed)
    draws = np.empty(rounds)
    for i in range(rounds):
        si = rng.integers(0, len(seeds), len(seeds))
        gi = rng.integers(0, len(scenarios), len(scenarios))
        draws[i] = delta[np.ix_(si, gi)].mean()
    return {'delta': float(delta.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'seed_count': len(seeds), 'scenario_count': len(scenarios)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    completed = json.loads((args.root / 'completed.json').read_text(encoding='utf-8'))
    if completed.get('heldout_read') is not False:
        raise ValueError('heldout-read certificate missing')

    eval_rows = {'script': load_rows(args.root / 'eval_script' / 'episodes.jsonl')}
    per_seed = []
    actor_params = {}
    for mode in MODES:
        for seed in SEEDS:
            key = f'{mode}_s{seed}'
            rows = load_rows(args.root / f'eval_{key}' / 'episodes.jsonl')
            history = load_rows(args.root / f'train_{key}' / 'training.jsonl')
            eval_rows[key] = rows
            import torch
            bundle = torch.load(args.root / f'train_{key}' / 'policy.pt',
                                map_location='cpu', weights_only=True)
            actor_params[key] = sum(value.numel() for value in bundle['actor'].values())
            trained_steps = Counter()
            trained_episodes = Counter()
            training_invalid = Counter()
            for row in history:
                trained_steps.update(row['branch_steps'])
                trained_episodes.update(row['branch_episodes'])
                training_invalid.update(row['invalid_reasons'])
            for branch in ('single', 'dual'):
                subset = [row for row in rows if row['branch'] == branch]
                invalid = Counter(reason for row in subset for reason in row['invalid_reasons'])
                script = [row for row in eval_rows['script'] if row['branch'] == branch]
                changed = sum(a['dangerous'] != b['dangerous'] or a['valid'] != b['valid']
                              for a, b in zip(subset, script))
                per_seed.append({
                    'mode': mode, 'seed': seed, 'branch': branch,
                    'actor_parameters': actor_params[key],
                    'training_total_steps': history[-1]['steps'],
                    'training_branch_steps': trained_steps[branch],
                    'training_branch_episodes': trained_episodes[branch],
                    'training_mean_valid_rate': float(np.mean(
                        [row['valid_rate'] for row in history])),
                    'training_total_target_target_collision': training_invalid['target_target_collision'],
                    'valid_rate': float(np.mean([row['valid'] for row in subset])),
                    'dangerous_rate': float(np.mean([row['dangerous'] for row in subset])),
                    'mean_risk': float(np.mean([row['risk'] for row in subset])),
                    'mean_abs_action': float(np.mean([row['mean_abs_action'] for row in subset])),
                    'mean_total_effort': float(np.mean([row['total_effort'] for row in subset])),
                    'outcomes_changed_vs_script': changed,
                    'target_target_collision': invalid['target_target_collision'],
                    'pedestrian_role': invalid['pedestrian_role'],
                    'occluder_role': invalid['occluder_role'],
                })
    with (args.root / 'per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    values = {}
    for branch in ('single', 'dual'):
        for field in ('dangerous', 'valid', 'risk'):
            script = scenario_values(eval_rows['script'], branch, field)
            values[('script', branch, field)] = {seed: script for seed in SEEDS}
            for mode in MODES:
                values[(mode, branch, field)] = {
                    seed: scenario_values(eval_rows[f'{mode}_s{seed}'], branch, field)
                    for seed in SEEDS}

    comparisons = [
        ('mixed-single', 'mixed', 'single', 'single'),
        ('mixed-dual', 'mixed', 'dual', 'dual'),
        ('mixed-script-single', 'mixed', 'script', 'single'),
        ('mixed-script-dual', 'mixed', 'script', 'dual'),
        ('single-script', 'single', 'script', 'single'),
        ('dual-script', 'dual', 'script', 'dual'),
    ]
    effects = []
    for label, left, right, branch in comparisons:
        result = {'comparison': label, 'left': left, 'right': right, 'branch': branch}
        for field in ('dangerous', 'valid', 'risk'):
            stat = two_way_paired_delta(values[(left, branch, field)],
                                        values[(right, branch, field)],
                                        args.bootstrap, args.seed)
            result[f'{field}_delta'] = stat['delta']
            result[f'{field}_ci_lo'] = stat['ci95'][0]
            result[f'{field}_ci_hi'] = stat['ci95'][1]
        effects.append(result)
    with (args.root / 'paired_effects.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)

    decisions = []
    for label, specialized, branch in (
            ('RQ1 single: mixed vs single-only', 'single', 'single'),
            ('RQ2/RQ3 dual: mixed vs dual-only', 'dual', 'dual')):
        effect = next(row for row in effects
                      if row['left'] == 'mixed' and row['right'] == specialized
                      and row['branch'] == branch)
        relevant = [row for row in per_seed
                    if row['mode'] in ('mixed', specialized) and row['branch'] == branch]
        stable = min(row['valid_rate'] for row in relevant) >= .8
        role_invalid = sum(row['pedestrian_role'] + row['occluder_role'] for row in relevant)
        noninferior = effect['dangerous_ci_lo'] >= -MARGIN
        changed = sum(row['outcomes_changed_vs_script'] for row in relevant)
        activity = max(row['mean_abs_action'] for row in relevant)
        informative = changed > 0 or activity >= .01
        decisions.append({
            'claim': label, 'branch': branch, 'margin': MARGIN,
            'dangerous_delta': effect['dangerous_delta'],
            'ci_lo': effect['dangerous_ci_lo'], 'ci_hi': effect['dangerous_ci_hi'],
            'stable': stable, 'role_invalid': role_invalid,
            'noninferiority_pass': noninferior and stable and role_invalid == 0,
            'activity_gate': informative,
            'interpretation': ('architecture_noninferiority_development_support'
                               if informative and noninferior and stable
                               else ('vacuous_or_low_activity' if noninferior and stable
                                     else 'development_fail')),
        })
    with (args.root / 'go_no_go.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)

    method_effects = [row for row in effects if row['right'] == 'script'
                      and row['left'] in ('mixed', 'single', 'dual')]
    summary = {
        'protocol': 'P2.2 equal-total-budget architecture screen',
        'heldout_read': False, 'interaction_budget_per_model': completed['interaction_budget'],
        'seeds': list(SEEDS), 'actor_parameter_counts': sorted(set(actor_params.values())),
        'noninferiority_margin': MARGIN, 'decisions': decisions, 'effects': effects,
        'method_superiority_vs_script': [
            dict(comparison=row['comparison'], branch=row['branch'],
                 dangerous_delta=row['dangerous_delta'],
                 ci95=[row['dangerous_ci_lo'], row['dangerous_ci_hi']],
                 superiority_pass=row['dangerous_ci_lo'] > 0)
            for row in method_effects],
    }
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# P2.2 shared vs specialized architecture screen', '',
        'Development-set screen; heldout was not read. All models have the same actor '
        f"parameter count and exactly {completed['interaction_budget']} total training decision "
        'steps. Mixed training has no single-only warmup. The comparison is equal total budget; '
        'branch-specific training exposure is reported and is intentionally lower for mixed.', '',
        '## Go/no-go', '',
        '| claim | delta [95% CI] | stable | role invalid | NI pass | activity gate | interpretation |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for row in decisions:
        lines.append(f"| {row['claim']} | {row['dangerous_delta']:.4f} "
                     f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] | {row['stable']} | "
                     f"{row['role_invalid']} | {row['noninferiority_pass']} | "
                     f"{row['activity_gate']} | {row['interpretation']} |")
    lines += ['', '## Per-seed primary slices', '',
              '| mode | seed | branch | train branch steps | train total target collision | valid | dangerous | '
              'mean abs action | changed vs script | target-target invalid |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in per_seed:
        if (row['mode'] == 'mixed'
                or (row['mode'] == 'single' and row['branch'] == 'single')
                or (row['mode'] == 'dual' and row['branch'] == 'dual')):
            lines.append(f"| {row['mode']} | {row['seed']} | {row['branch']} | "
                         f"{row['training_branch_steps']} | "
                         f"{row['training_total_target_target_collision']} | {row['valid_rate']:.3f} | "
                         f"{row['dangerous_rate']:.3f} | {row['mean_abs_action']:.4f} | "
                         f"{row['outcomes_changed_vs_script']} | "
                         f"{row['target_target_collision']} |")
    lines += ['', '## Method effect versus script', '',
              '| comparison | branch | dangerous delta [95% CI] | superiority pass |',
              '|---|---|---|---:|']
    for row in method_effects:
        lines.append(f"| {row['comparison']} | {row['branch']} | "
                     f"{row['dangerous_delta']:.4f} [{row['dangerous_ci_lo']:.4f}, "
                     f"{row['dangerous_ci_hi']:.4f}] | {row['dangerous_ci_lo'] > 0} |")
    lines += ['', 'A noninferiority pass is not treated as support when the activity gate fails: '
              'a shared and specialized policy that both reproduce the script outcome do not '
              'demonstrate useful learned scenario generation. Passing architecture '
              'noninferiority also does not establish method superiority over the script.', '']
    (args.root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(args.root), 'decisions': decisions}, indent=2))


if __name__ == '__main__':
    main()
