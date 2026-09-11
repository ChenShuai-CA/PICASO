"""Summarize the preregistered P2.4 lane-locked CEM feasibility experiment."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from summarize_p22_architecture import load_rows, scenario_values, two_way_paired_delta


SEEDS = (7, 17, 27, 37, 47)
KINDS = ('parameters', 'trajectory')
BRANCHES = ('single', 'dual')
DELTA_GATE = .10


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def first_success_steps(rows):
    cumulative = 0
    for row in rows:
        cumulative += row['decision_steps']
        if row['dangerous'] and row['valid'] and not row['budget_truncated']:
            return cumulative
    return None


def effect(left, right, rounds, seed):
    stat = two_way_paired_delta(left, right, rounds, seed)
    return stat['delta'], stat['ci95'][0], stat['ci95'][1]


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
    if completed.get('heldout_read') is not False or prereg.get('heldout_read') is not False:
        raise ValueError('heldout-read certificate missing')
    if completed['condition_set_version'] != prereg['condition_set_version']:
        raise ValueError('condition set version differs from preregistration')

    script_rows = load_rows(args.p22 / 'eval_script' / 'episodes.jsonl')
    script_values = {branch: scenario_values(script_rows, branch, 'dangerous')
                     for branch in BRANCHES}
    mixed_values = {
        branch: {seed: scenario_values(
            load_rows(args.p23 / f'eval_mixed_s{seed}' / 'episodes.jsonl'),
            branch, 'dangerous') for seed in SEEDS}
        for branch in BRANCHES
    }
    search_values = {}
    per_run = []
    per_condition = []
    role_invalid_totals = Counter()
    for kind in KINDS:
        for branch in BRANCHES:
            search_values[(kind, branch)] = {}
            seed_support = defaultdict(int)
            seed_rows = {}
            for seed in SEEDS:
                directory = args.root / f'{kind}_{branch}_s{seed}'
                summary = read_json(directory / 'conditions_search.json')
                attempts = load_rows(directory / 'attempts.jsonl')
                if (summary['role_action_mode'] != 'lane_locked'
                        or summary['interaction_budget_per_condition']
                        != completed['interaction_budget_per_condition']):
                    raise ValueError(f'protocol mismatch in {directory}')
                grouped = defaultdict(list)
                for row in attempts:
                    grouped[row['scenario_id']].append(row)
                values = {}
                success_steps = []
                selected_rows = []
                reasons = Counter(reason for row in attempts for reason in row['invalid_reasons'])
                role_invalid = reasons['pedestrian_role'] + reasons['occluder_role']
                role_invalid_totals[(kind, branch)] += role_invalid
                for condition in summary['per_condition']:
                    scenario_id = condition['scenario_id']
                    successful = bool(condition['best_valid'] and condition['best_dangerous'])
                    values[scenario_id] = float(successful)
                    seed_support[scenario_id] += int(successful)
                    steps = first_success_steps(grouped[scenario_id])
                    if steps is not None:
                        success_steps.append(steps)
                    eligible = [row for row in grouped[scenario_id]
                                if row['valid'] and not row['budget_truncated']]
                    if eligible:
                        selected_rows.append(max(eligible, key=lambda row: row['score']))
                    per_condition.append({
                        'kind': kind, 'branch': branch, 'seed': seed,
                        'scenario_id': scenario_id, 'success': successful,
                        'first_success_steps': '' if steps is None else steps,
                        'evaluations': condition['evaluations'],
                        'completed_attempts': condition['completed_attempts'],
                        'completed_valid_attempts': condition['completed_valid_attempts'],
                    })
                search_values[(kind, branch)][seed] = values
                seed_rows[seed] = values
                per_run.append({
                    'kind': kind, 'branch': branch, 'seed': seed,
                    'success_rate': float(np.mean(list(values.values()))),
                    'total_evaluations': summary['total_evaluations'],
                    'completed_attempts': sum(not row['budget_truncated'] for row in attempts),
                    'budget_truncated_attempts': sum(row['budget_truncated'] for row in attempts),
                    'completed_valid_rate_all_attempts': float(np.mean([
                        row['valid'] and not row['budget_truncated'] for row in attempts])),
                    'completed_dangerous_rate_all_attempts': float(np.mean([
                        row['dangerous'] and not row['budget_truncated'] for row in attempts])),
                    'role_invalid': role_invalid,
                    'target_target_collision': reasons['target_target_collision'],
                    'median_first_success_steps': (float(np.median(success_steps))
                                                   if success_steps else ''),
                    'mean_selected_total_effort': float(np.mean(
                        [row['total_effort'] for row in selected_rows])),
                    'mean_selected_abs_action': float(np.mean(
                        [row['mean_abs_action'] for row in selected_rows])),
                    'mean_evaluations_per_condition': summary['total_evaluations'] / 40,
                    'elapsed_s': summary['elapsed_s'],
                })
            support = list(seed_support.values())
            per_run.append({
                'kind': kind, 'branch': branch, 'seed': 'aggregate',
                'success_rate': float(np.mean([value for values in seed_rows.values()
                                               for value in values.values()])),
                'total_evaluations': sum(row['total_evaluations'] for row in per_run
                                         if row['kind'] == kind and row['branch'] == branch
                                         and row['seed'] != 'aggregate'),
                'completed_attempts': '', 'budget_truncated_attempts': '',
                'completed_valid_rate_all_attempts': '',
                'completed_dangerous_rate_all_attempts': '',
                'role_invalid': role_invalid_totals[(kind, branch)],
                'target_target_collision': sum(row['target_target_collision'] for row in per_run
                                               if row['kind'] == kind and row['branch'] == branch
                                               and row['seed'] != 'aggregate'),
                'median_first_success_steps': '', 'mean_evaluations_per_condition': '',
                'mean_selected_total_effort': '', 'mean_selected_abs_action': '',
                'elapsed_s': '', 'all_five_seed_success_fraction': float(np.mean(
                    [count == len(SEEDS) for count in support])),
                'median_seed_support': float(np.median(support)),
            })

    with (args.root / 'per_run.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = sorted({key for row in per_run for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(per_run)
    with (args.root / 'success_by_condition.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_condition[0]))
        writer.writeheader(); writer.writerows(per_condition)

    effects = []
    decisions = []
    for kind in KINDS:
        for branch in BRANCHES:
            left = search_values[(kind, branch)]
            script = {seed: script_values[branch] for seed in SEEDS}
            delta, lo, hi = effect(left, script, args.bootstrap, args.seed)
            role_invalid = role_invalid_totals[(kind, branch)]
            passed = delta >= DELTA_GATE and lo > 0 and role_invalid == 0
            decisions.append({
                'kind': kind, 'branch': branch, 'coverage_delta_vs_script': delta,
                'ci_lo': lo, 'ci_hi': hi, 'delta_gate': DELTA_GATE,
                'role_invalid': role_invalid, 'feasibility_pass': passed,
            })
            effects.append({'comparison': f'{kind}-script', 'branch': branch,
                            'delta': delta, 'ci_lo': lo, 'ci_hi': hi})
            delta, lo, hi = effect(left, mixed_values[branch], args.bootstrap, args.seed)
            effects.append({'comparison': f'{kind}-mixed12', 'branch': branch,
                            'delta': delta, 'ci_lo': lo, 'ci_hi': hi})
    for branch in BRANCHES:
        delta, lo, hi = effect(search_values[('trajectory', branch)],
                               search_values[('parameters', branch)],
                               args.bootstrap, args.seed)
        effects.append({'comparison': 'trajectory-parameters', 'branch': branch,
                        'delta': delta, 'ci_lo': lo, 'ci_hi': hi})

    with (args.root / 'paired_effects.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)

    parameter_all = all(row['feasibility_pass'] for row in decisions
                        if row['kind'] == 'parameters')
    trajectory_all = all(row['feasibility_pass'] for row in decisions
                         if row['kind'] == 'trajectory')
    if parameter_all:
        diagnosis = 'low_dimensional_search_feasible_learning_or_optimization_bottleneck'
    elif trajectory_all:
        diagnosis = 'trajectory_search_feasible_low_dimensional_parameterization_bottleneck'
    else:
        diagnosis = 'branch_specific_or_search_space_objective_bottleneck'
    result = {
        'protocol': 'P2.4 lane-locked CEM feasibility',
        'condition_set_version': completed['condition_set_version'],
        'interaction_budget_per_condition': completed['interaction_budget_per_condition'],
        'heldout_read': False, 'decisions': decisions, 'effects': effects,
        'diagnosis': diagnosis,
        'interpretation_boundary': 'best-of-search diagnostic; not equal-cost method superiority',
    }
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = [
        '# P2.4 lane-locked CEM feasibility', '',
        f"Development conditions only; heldout was not read. Each search used exactly "
        f"{completed['interaction_budget_per_condition']} decision steps per condition, "
        'including invalid and final truncated attempts. A truncated episode could not be a solution.', '',
        '## Preregistered feasibility decisions', '',
        '| search | branch | coverage delta vs script [95% CI] | role invalid | pass |',
        '|---|---|---|---:|---:|',
    ]
    for row in decisions:
        lines.append(f"| {row['kind']} | {row['branch']} | "
                     f"{row['coverage_delta_vs_script']:.4f} [{row['ci_lo']:.4f}, "
                     f"{row['ci_hi']:.4f}] | {row['role_invalid']} | "
                     f"{row['feasibility_pass']} |")
    lines += ['', '## Search-seed stability and cost', '',
              '| search | branch | seed | success rate | valid completed / all attempts | '
              'target collision | median steps to success | selected effort | evaluations |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in per_run:
        if row['seed'] == 'aggregate':
            continue
        lines.append(f"| {row['kind']} | {row['branch']} | {row['seed']} | "
                     f"{row['success_rate']:.3f} | "
                     f"{row['completed_valid_rate_all_attempts']:.3f} | "
                     f"{row['target_target_collision']} | "
                     f"{row['median_first_success_steps']} | "
                     f"{row['mean_selected_total_effort']:.3f} | "
                     f"{row['total_evaluations']} |")
    lines += ['', '## Diagnostic comparisons', '',
              '| comparison | branch | dangerous-condition coverage delta [95% CI] |',
              '|---|---|---|']
    for row in effects:
        lines.append(f"| {row['comparison']} | {row['branch']} | {row['delta']:.4f} "
                     f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] |")
    lines += ['', f"Diagnosis: **{diagnosis}**.", '',
              'CEM receives many simulator interactions per condition. These results diagnose whether '
              'the constrained action space contains solutions; they do not establish equal-cost method '
              'superiority or justify reading heldout.', '']
    (args.root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
