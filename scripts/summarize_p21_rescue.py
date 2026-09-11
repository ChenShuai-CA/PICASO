"""Summarize the preregistered P2.1 stability rescue matrix.

Inference uses a paired two-way bootstrap: training seeds and shared scenarios
are both resampled. This keeps the common-condition design while representing
the seed instability that the original P2 scenario-only interval omitted.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

SEEDS = (7, 17, 27, 37, 47)
FACTORS = {
    'pr': dict(prior=True, robust=True),
    'pn': dict(prior=True, robust=False),
    'nr': dict(prior=False, robust=True),
    'nn': dict(prior=False, robust=False),
}


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def scenario_values(rows, branch, field):
    subset = [row for row in rows if row['branch'] == branch]
    groups = sorted({row['scenario_id'] for row in subset})
    return {group: float(np.mean([row[field] for row in subset
                                  if row['scenario_id'] == group])) for group in groups}


def two_way_paired_delta(left_by_seed, right_by_seed, b_rounds=2000, seed=2026):
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if set(left_by_seed) != set(right_by_seed) or not seeds:
        raise ValueError('paired comparison requires identical non-empty seed sets')
    scenarios = sorted(set(left_by_seed[seeds[0]]) & set(right_by_seed[seeds[0]]))
    expected = set(scenarios)
    if not scenarios or any(set(left_by_seed[s]) != expected or set(right_by_seed[s]) != expected
                            for s in seeds):
        raise ValueError('paired comparison requires identical scenario sets for every seed')
    delta = np.array([[left_by_seed[s][g] - right_by_seed[s][g] for g in scenarios]
                      for s in seeds])
    rng = np.random.default_rng(seed)
    draws = np.empty(b_rounds)
    for i in range(b_rounds):
        seed_idx = rng.integers(0, len(seeds), len(seeds))
        scenario_idx = rng.integers(0, len(scenarios), len(scenarios))
        draws[i] = delta[np.ix_(seed_idx, scenario_idx)].mean()
    return dict(delta=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                seeds=len(seeds), scenarios=len(scenarios), b_rounds=b_rounds, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.root)
    completed = json.loads((root / 'completed.json').read_text(encoding='utf-8'))
    if completed.get('heldout_read') is not False:
        raise ValueError('matrix completion record does not certify heldout_read=false')

    eval_rows = {'script': load_rows(root / 'eval_script' / 'episodes.jsonl')}
    per_seed = []
    for factor in FACTORS:
        for seed in SEEDS:
            name = f'{factor}_s{seed}'
            rows = load_rows(root / f'eval_{name}' / 'episodes.jsonl')
            eval_rows[name] = rows
            history = load_rows(root / f'train_{name}' / 'training.jsonl')
            max_kl = max(row['reference_kl'] for row in history)
            train_invalid = sum((Counter(row['invalid_reasons']) for row in history), Counter())
            for branch in ('single', 'dual'):
                subset = [row for row in rows if row['branch'] == branch]
                invalid = Counter(reason for row in subset for reason in row['invalid_reasons'])
                per_seed.append(dict(
                    factor=factor, prior=FACTORS[factor]['prior'], robust=FACTORS[factor]['robust'],
                    seed=seed, branch=branch, training_steps=history[-1]['steps'],
                    training_updates=len(history), max_reference_kl=max_kl,
                    mean_training_valid=float(np.mean([row['valid_rate'] for row in history])),
                    training_target_collisions=train_invalid['target_target_collision'],
                    mean_projection_event_rate=float(np.mean([row['projection_event_rate']
                                                               for row in history])),
                    valid_rate=float(np.mean([row['valid'] for row in subset])),
                    dangerous_rate=float(np.mean([row['dangerous'] for row in subset])),
                    dangerous_per_10k_steps=(sum(row['dangerous'] for row in subset)
                                             / sum(row['decision_steps'] for row in subset) * 10000),
                    collision_rate=float(np.mean([row['collision'] for row in subset])),
                    valid_mean_min_clearance=float(np.mean([row['min_clearance'] for row in subset
                                                            if row['valid']])),
                    mean_collision_speed=float(np.mean([row['collision_speed'] for row in subset])),
                    mean_risk=float(np.mean([row['risk'] for row in subset])),
                    target_target_collision=invalid['target_target_collision'],
                    pedestrian_role=invalid['pedestrian_role'],
                    occluder_role=invalid['occluder_role'],
                    evaluation_steps=sum(row['decision_steps'] for row in subset)))

    with (root / 'per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    gates = []
    for factor in FACTORS:
        rows = [row for row in per_seed if row['factor'] == factor]
        failures = [f"s{row['seed']}/{row['branch']}={row['valid_rate']:.3f}"
                    for row in rows if row['valid_rate'] < .8]
        role_invalid = sum(row['pedestrian_role'] + row['occluder_role'] for row in rows)
        stable = not failures and role_invalid == 0
        gates.append(dict(factor=factor, stable=stable,
                          min_valid_rate=min(row['valid_rate'] for row in rows),
                          median_valid_rate=float(np.median([row['valid_rate'] for row in rows])),
                          role_invalid=role_invalid,
                          failed_slices=';'.join(failures)))
    with (root / 'go_no_go.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gates[0]))
        writer.writeheader()
        writer.writerows(gates)

    # Replicate the deterministic script row across training seeds so factor-vs-script
    # intervals include factor seed variation while preserving paired conditions.
    values = {}
    for branch in ('single', 'dual'):
        for field in ('dangerous', 'valid', 'risk'):
            script = scenario_values(eval_rows['script'], branch, field)
            values[('script', branch, field)] = {seed: script for seed in SEEDS}
            for factor in FACTORS:
                values[(factor, branch, field)] = {
                    seed: scenario_values(eval_rows[f'{factor}_s{seed}'], branch, field)
                    for seed in SEEDS}

    comparisons = [
        ('pr-script', 'pr', 'script'), ('pn-script', 'pn', 'script'),
        ('nr-script', 'nr', 'script'), ('nn-script', 'nn', 'script'),
        ('prior|robust', 'pr', 'nr'), ('prior|no_robust', 'pn', 'nn'),
        ('robust|prior', 'pr', 'pn'), ('robust|no_prior', 'nr', 'nn'),
    ]
    effects = []
    for label, left, right in comparisons:
        for branch in ('single', 'dual'):
            stat = two_way_paired_delta(values[(left, branch, 'dangerous')],
                                        values[(right, branch, 'dangerous')],
                                        args.bootstrap, args.seed)
            valid = two_way_paired_delta(values[(left, branch, 'valid')],
                                         values[(right, branch, 'valid')],
                                         args.bootstrap, args.seed)
            risk = two_way_paired_delta(values[(left, branch, 'risk')],
                                        values[(right, branch, 'risk')],
                                        args.bootstrap, args.seed)
            effects.append(dict(comparison=label, left=left, right=right, branch=branch,
                                dangerous_delta=stat['delta'], ci_lo=stat['ci95'][0],
                                ci_hi=stat['ci95'][1], valid_delta=valid['delta'],
                                valid_ci_lo=valid['ci95'][0], valid_ci_hi=valid['ci95'][1],
                                risk_delta=risk['delta'], risk_ci_lo=risk['ci95'][0],
                                risk_ci_hi=risk['ci95'][1], seeds=stat['seeds'],
                                scenarios=stat['scenarios'], bootstrap=stat['b_rounds'],
                                bootstrap_seed=stat['seed']))
    with (root / 'paired_effects.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)

    script_summary = {}
    for branch in ('single', 'dual'):
        subset = [row for row in eval_rows['script'] if row['branch'] == branch]
        script_summary[branch] = dict(valid_rate=float(np.mean([r['valid'] for r in subset])),
                                      dangerous_rate=float(np.mean([r['dangerous'] for r in subset])))
    summary = dict(protocol='P2.1 stability rescue', heldout_read=False,
                   interaction_budget=completed['interaction_budget'], seeds=list(SEEDS),
                   gates=gates, script=script_summary, paired_effects=effects)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = ['# P2.1 stability rescue', '',
             'Development-set diagnosis; heldout was not read. Every model trained for exactly '
             f"{completed['interaction_budget']} decision steps. Paired intervals resample both "
             f"5 training seeds and 40 shared scenarios (B={args.bootstrap}, seed={args.seed}).", '',
             '## Go/no-go', '',
             '| factor | prior | robust | min valid | median valid | role invalid | stable | failed slices |',
             '|---|---:|---:|---:|---:|---:|---|---|']
    for gate in gates:
        cfg = FACTORS[gate['factor']]
        lines.append(f"| {gate['factor']} | {cfg['prior']} | {cfg['robust']} | "
                     f"{gate['min_valid_rate']:.3f} | {gate['median_valid_rate']:.3f} | "
                     f"{gate['role_invalid']} | {gate['stable']} | {gate['failed_slices']} |")
    lines += ['', '## Per-seed evaluation', '',
              '| factor | seed | branch | valid | dangerous | danger/10k steps | '
              'valid mean clearance | target-target invalid | max KL |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in per_seed:
        lines.append(f"| {row['factor']} | {row['seed']} | {row['branch']} | "
                     f"{row['valid_rate']:.3f} | {row['dangerous_rate']:.3f} | "
                     f"{row['dangerous_per_10k_steps']:.1f} | "
                     f"{row['valid_mean_min_clearance']:.3f} | "
                     f"{row['target_target_collision']} | {row['max_reference_kl']:.4f} |")
    lines += ['', '## Paired dangerous-rate effects', '',
              '| comparison (left-right) | branch | dangerous delta [95% CI] | '
              'valid delta [95% CI] | continuous risk delta [95% CI] |',
              '|---|---|---|---|---|']
    for row in effects:
        lines.append(f"| {row['comparison']} | {row['branch']} | {row['dangerous_delta']:.4f} "
                     f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] | "
                     f"{row['valid_delta']:.4f} [{row['valid_ci_lo']:.4f}, "
                     f"{row['valid_ci_hi']:.4f}] | {row['risk_delta']:.4f} "
                     f"[{row['risk_ci_lo']:.4f}, {row['risk_ci_hi']:.4f}] |")
    lines += ['', '## Interpretation boundary', '',
              'Lane locking can remove role-axis violations but does not guarantee separation between '
              'the learned targets. A factor is unstable if any seed/branch valid rate is below 0.80. '
              'The public prior still contains mixed pedestrian/bicycle labels and is diagnostic only.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=str(root), gates=gates), indent=2))


if __name__ == '__main__':
    main()
