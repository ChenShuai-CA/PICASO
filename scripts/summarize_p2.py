"""P2 comparison tables: interaction-budget accounting and preregistered verdicts.

Reads evaluate/search outputs produced under the P2 protocol (GLM_CHANGELOG P2
R2-R4) and writes comparison.csv, sensitivity.csv, avoidability.csv, REPORT.md
and comparison.png. Cluster bootstrap for merged multi-seed rows recomputes at
the preregistered B/seed; single-method CIs come straight from each summary.json
(already B=2000, seed=2026 via evaluate.summarize). Old pilot artifacts are never
read or recomputed here.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_rows(eval_dir):
    path = Path(eval_dir) / 'episodes.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cluster_stats(rows, b_rounds, seed):
    """Scenario-grouped dangerous-rate estimate + cluster bootstrap CI (P2 R2)."""
    groups = sorted({r['scenario_id'] for r in rows})
    rates = np.array([np.mean([r['dangerous'] for r in rows if r['scenario_id'] == g])
                      for g in groups])
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(rates, len(groups), replace=True).mean()
                      for _ in range(b_rounds)])
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(rate=float(np.mean([r['dangerous'] for r in rows])), scenarios=len(groups),
                attempts=len(rows), ci=[float(lo), float(hi)])


def per_scenario_std(rows):
    """Within-scenario spread of dangerous over execution perturbations."""
    out = {}
    for branch in ('single', 'dual'):
        subset = [r for r in rows if r['branch'] == branch]
        if not subset:
            continue
        groups = sorted({r['scenario_id'] for r in subset})
        stds = [float(np.std([r['dangerous'] for r in subset if r['scenario_id'] == g]))
                for g in groups]
        out[branch] = dict(median_std=float(np.median(stds)), scenarios=len(groups))
    return out


def parse_pair(item):
    name, _, value = item.partition('=')
    return name, value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--eval', action='append', default=[], help='name=eval_dir (repeatable)')
    p.add_argument('--train', action='append', default=[], help='name=train_dir (repeatable)')
    p.add_argument('--search', action='append', default=[], help='name=search_dir (repeatable)')
    p.add_argument('--merged', action='append', default=[],
                   help='name=dir1,dir2,... merge episodes across seeds (repeatable)')
    p.add_argument('--sensitivity', action='append', default=[],
                   help='name=eval_dir with --perturbations>1 rows (repeatable)')
    p.add_argument('--avoidability', action='append', default=[],
                   help='name=stopping_dir,ttc_dir counterfactual pair (repeatable)')
    p.add_argument('--verdict', action='append', default=[],
                   help='belowName,aboveName preregistered non-overlap check (repeatable)')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--label', default='dev set')
    a = p.parse_args()

    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    train_steps = {}
    for item in a.train:
        name, directory = parse_pair(item)
        lines = (Path(directory) / 'training.jsonl').read_text().splitlines()
        train_steps[name] = json.loads(lines[-1])['steps']

    rows, metrics = {}, {}  # metrics[name][branch] -> dict from summary.json
    for item in a.eval:
        name, directory = parse_pair(item)
        summary = json.loads((Path(directory) / 'summary.json').read_text())
        metrics[name] = summary['metrics']
        rows[name] = load_rows(directory)

    merged_metrics = {}
    for item in a.merged:
        name, _, dirs = item.partition('=')
        merged_rows = [r for d in dirs.split(',') for r in load_rows(d)]
        rows[name] = merged_rows
        merged_metrics[name] = {b: cluster_stats([r for r in merged_rows if r['branch'] == b],
                                                 a.bootstrap, a.seed)
                                for b in ('single', 'dual')
                                if any(r['branch'] == b for r in merged_rows)}

    comparison = []
    for name in metrics:
        for branch, m in metrics[name].items():
            comparison.append(dict(method=name, branch=branch, kind='single-sample',
                                   dangerous_rate=m['dangerous_rate'],
                                   ci_lo=m['dangerous_rate_cluster_ci95'][0],
                                   ci_hi=m['dangerous_rate_cluster_ci95'][1],
                                   valid_rate=m['valid_rate'], collision_rate=m['collision_rate'],
                                   unique_valid_dangerous=m['unique_valid_dangerous'],
                                   attempts=m['attempts'], scenarios=m['independent_scenarios'],
                                   eval_steps=m['total_steps'], steps_mean=m['steps_mean'],
                                   training_steps=train_steps.get(name, '')))
    for name, per_branch in merged_metrics.items():
        for branch, m in per_branch.items():
            comparison.append(dict(method=name, branch=branch, kind='merged-seeds',
                                   dangerous_rate=m['rate'], ci_lo=m['ci'][0], ci_hi=m['ci'][1],
                                   valid_rate='', collision_rate='', unique_valid_dangerous='',
                                   attempts=m['attempts'], scenarios=m['scenarios'],
                                   eval_steps=sum(r['decision_steps']
                                                  for r in rows[name] if r['branch'] == branch),
                                   steps_mean='', training_steps=''))

    searches = {}
    for item in a.search:
        name, directory = parse_pair(item)
        searches[name] = json.loads((Path(directory) / 'conditions_search.json').read_text())

    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    verdicts = []
    for pair in a.verdict:
        below, above = pair.split(',')

        def stat(name):
            for row in comparison:
                if row['method'] == name and row['branch'] == branch:
                    return row
            raise KeyError(name)

        for branch in ('single', 'dual'):
            b, t = stat(below), stat(above)
            separated = b['ci_lo'] > t['ci_hi']
            verdicts.append(dict(comparison=f'{below} > {above}', branch=branch,
                                 below=f"{b['dangerous_rate']:.4f} [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]",
                                 above=f"{t['dangerous_rate']:.4f} [{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]",
                                 non_overlap=bool(separated),
                                 verdict='PASS' if separated else 'FAIL'))
    with (output / 'verdicts.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['comparison', 'branch', 'below', 'above',
                                               'non_overlap', 'verdict'])
        writer.writeheader()
        writer.writerows(verdicts)

    sensitivity = []
    for item in a.sensitivity:
        name, directory = parse_pair(item)
        r = load_rows(directory)
        stats = per_scenario_std(r)
        for branch, s in stats.items():
            rate = float(np.mean([x['dangerous'] for x in r if x['branch'] == branch]))
            sensitivity.append(dict(method=name, branch=branch, perturb_attempts=len(
                [x for x in r if x['branch'] == branch]), scenarios=s['scenarios'],
                median_within_scenario_std=s['median_std'], merged_dangerous_rate=rate))
    if sensitivity:
        with (output / 'sensitivity.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(sensitivity[0]))
            writer.writeheader()
            writer.writerows(sensitivity)

    avoidability = []
    for item in a.avoidability:
        name, _, dirs = item.partition('=')
        stopping = {(r['branch'], r['scenario_id']): r['dangerous'] for r in load_rows(dirs.split(',')[0])}
        ttc = {(r['branch'], r['scenario_id']): r['dangerous'] for r in load_rows(dirs.split(',')[1])}
        for branch in ('single', 'dual'):
            danger_keys = [k for k, d in stopping.items() if d and k[0] == branch]
            if not danger_keys:
                continue
            avoided = sum(1 for k in danger_keys if k in ttc and not ttc[k])
            avoidability.append(dict(model=name, branch=branch, dangerous_stopping=len(danger_keys),
                                     avoided_under_ttc=avoided,
                                     avoidable_fraction=avoided / len(danger_keys),
                                     note='counterfactual swaps only the ego brake controller'))
    if avoidability:
        with (output / 'avoidability.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(avoidability[0]))
            writer.writeheader()
            writer.writerows(avoidability)

    # CI bar chart: single-sample and merged methods per branch.
    methods = list(dict.fromkeys(row['method'] for row in comparison))
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    y = np.arange(len(methods))
    for j, branch in enumerate(('single', 'dual')):
        subset = [next((r for r in comparison if r['method'] == m and r['branch'] == branch), None)
                  for m in methods]
        rates = [r['dangerous_rate'] for r in subset]
        errs = [[r['dangerous_rate'] - r['ci_lo'] if r else 0 for r in subset],
                [r['ci_hi'] - r['dangerous_rate'] if r else 0 for r in subset]]
        ax.errorbar([r if r is not None else 0 for r in rates], y + (j - .5) * .36,
                    xerr=errs, fmt='o', capsize=3, label=branch)
    ax.set(yticks=y, yticklabels=methods, xlabel='Valid dangerous rate (cluster bootstrap 95% CI)',
           title=f'P2 {a.label}: matched conditions, B={a.bootstrap}')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.legend()
    fig.savefig(output / 'comparison.png', dpi=160)
    fig.savefig(output / 'comparison.svg')
    plt.close(fig)

    lines = [f'# P2 comparison ({a.label})', '',
             f'Cluster bootstrap B={a.bootstrap}, seed={a.seed}; scenario-level grouping; '
             'physics v2 / sampler v2. Development diagnostics, not paper evidence.', '',
             '## Comparison (single-sample + merged)', '',
             '| method | branch | dangerous [95% CI] | valid | collision | unique | attempts | '
             'scenarios | steps mean | eval steps | train steps |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"| {r['method']} | {r['branch']} | {r['dangerous_rate']:.4f} "
                     f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['valid_rate']} | "
                     f"{r['collision_rate']} | {r['unique_valid_dangerous']} | {r['attempts']} | "
                     f"{r['scenarios']} | {r['steps_mean']} | {r['eval_steps']} | {r['training_steps']} |")
    if verdicts:
        lines += ['', '## Preregistered verdicts', '',
                  '| comparison | branch | below | above | non-overlap | verdict |',
                  '|---|---|---|---|---|---|']
        for v in verdicts:
            lines.append(f"| {v['comparison']} | {v['branch']} | {v['below']} | {v['above']} | "
                         f"{v['non_overlap']} | {v['verdict']} |")
    if searches:
        lines += ['', '## CEM per-condition search (cost accounting; best-of-search, never '
                      'ranked against single-sample rows)', '',
                  '| search | conditions | budget/cond | total evals | total steps | '
                  'mean steps/cond | valid attempts |', '|---|---:|---:|---:|---:|---:|---:|']
        for name, s in searches.items():
            lines.append(f"| {name} | {s['n_conditions']} | {s['budget_per_condition']} | "
                         f"{s['total_evaluations']} | {s['total_interaction_steps']} | "
                         f"{s['mean_steps_per_condition']:.1f} | {s['valid_attempts']} |")
    if sensitivity:
        lines += ['', '## Perturbation sensitivity (within-scenario std over execution '
                      'perturbations)', '',
                  '| method | branch | attempts | scenarios | median within-scenario std | '
                  'merged dangerous rate |', '|---|---|---:|---:|---:|---:|']
        for s in sensitivity:
            lines.append(f"| {s['method']} | {s['branch']} | {s['perturb_attempts']} | "
                         f"{s['scenarios']} | {s['median_within_scenario_std']:.4f} | "
                         f"{s['merged_dangerous_rate']:.4f} |")
    if avoidability:
        lines += ['', '## Avoidability diagnostic (counterfactual ego brake controller)', '',
                  '| model | branch | dangerous@stopping | avoided@ttc | fraction |', '|---|---|---:|---:|---:|']
        for r in avoidability:
            lines.append(f"| {r['model']} | {r['branch']} | {r['dangerous_stopping']} | "
                         f"{r['avoided_under_ttc']} | {r['avoidable_fraction']:.2%} |")
    lines += ['', '## Interpretation boundary', '',
              'All rows are development-set diagnostics under the P2 preregistration. '
              '"Better than baseline" claims require the one-shot heldout evaluation, '
              'which has not been run at the time of this report.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=str(output), comparison_rows=len(comparison),
                          verdicts={f"{v['comparison']}|{v['branch']}": v['verdict'] for v in verdicts}),
                     indent=2))


if __name__ == '__main__':
    main()
