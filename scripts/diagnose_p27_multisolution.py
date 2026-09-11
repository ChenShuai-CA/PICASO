"""Post-hoc head diversity diagnostics for completed P2.7 artifacts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np


SEEDS = (7, 17, 27, 37, 47)
BRANCHES = ('single', 'dual')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def outcome_diagnostics(root, split):
    result = []
    for branch in BRANCHES:
        for seed in SEEDS:
            rows = [row for row in load_rows(
                root / f'{split}_evaluation' / f'learned_s{seed}.jsonl')
                    if row['branch'] == branch]
            by_head = {head: [row for row in rows if row['candidate_index'] == head]
                       for head in range(4)}
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['uid']].append(row)
            head_rates = [float(np.mean([row['valid'] and row['dangerous']
                                         for row in by_head[head]])) for head in range(4)]
            any_rate = float(np.mean([any(row['valid'] and row['dangerous']
                                         for row in attempts)
                                      for attempts in grouped.values()]))
            result.append({
                'split': split, 'branch': branch, 'seed': seed,
                'conditions': len(grouped), 'head_rates': head_rates,
                'any4_rate': any_rate, 'best_fixed_head_rate': max(head_rates),
                'any4_gain_over_best_fixed_head': any_rate - max(head_rates),
            })
    return result


def action_diversity(root, split):
    from scenario_lab.pulse_set_data import action_representation
    rows = []
    for seed in SEEDS:
        rows.extend(load_rows(root / f'{split}_evaluation' / f'learned_s{seed}.jsonl'))
    result = []
    for branch in BRANCHES:
        grouped = defaultdict(list)
        for row in rows:
            if row['branch'] == branch:
                grouped[(row['policy_seed'], row['uid'])].append(row)
        distances, duplicate_groups = [], 0
        for attempts in grouped.values():
            attempts.sort(key=lambda row: row['candidate_index'])
            reps = [action_representation(np.asarray(row['parameters'])[:1 if branch == 'single' else 2],
                                          branch) for row in attempts]
            local = [float(np.sqrt(np.square(left - right).mean()))
                     for index, left in enumerate(reps) for right in reps[index + 1:]]
            distances.extend(local)
            duplicate_groups += int(any(value < 1e-8 for value in local))
        result.append({
            'split': split, 'branch': branch, 'condition_seed_groups': len(grouped),
            'pairwise_action_rms_median': float(np.median(distances)),
            'pairwise_action_rms_q25_q75': np.quantile(distances, [.25, .75]).tolist(),
            'groups_with_any_exact_duplicate': duplicate_groups,
        })
    return result


def conditional_variation(root, split):
    """Separate input-conditioned variation within a head from head-library variation."""
    from scenario_lab.pulse_set_data import action_representation
    result = []
    for branch in BRANCHES:
        seed_shares, within_rms, between_rms = [], [], []
        for seed in SEEDS:
            rows = [row for row in load_rows(
                root / f'{split}_evaluation' / f'learned_s{seed}.jsonl')
                    if row['branch'] == branch]
            by_head = []
            for head in range(4):
                ordered = sorted((row for row in rows if row['candidate_index'] == head),
                                 key=lambda row: row['uid'])
                by_head.append(np.stack([action_representation(
                    np.asarray(row['parameters'])[:1 if branch == 'single' else 2], branch)
                                         for row in ordered]))
            values = np.stack(by_head)
            head_centers = values.mean(1)
            global_center = values.reshape(-1, values.shape[-1]).mean(0)
            within = float(np.square(values - head_centers[:, None]).sum())
            between = float(values.shape[1] * np.square(head_centers - global_center).sum())
            seed_shares.append(within / max(within + between, 1e-15))
            within_rms.append(float(np.sqrt(np.square(values - head_centers[:, None]).mean())))
            between_rms.append(float(np.sqrt(np.square(head_centers - global_center).mean())))
        result.append({
            'split': split, 'branch': branch,
            'condition_dependent_variance_share_mean': float(np.mean(seed_shares)),
            'within_head_across_condition_rms_mean': float(np.mean(within_rms)),
            'between_head_centroid_rms_mean': float(np.mean(between_rms)),
        })
    return result


def label_density(manifest):
    result = []
    for branch in BRANCHES:
        rows = [row for row in manifest if row['branch'] == branch]
        eligible = np.asarray([row['eligible_successes'] for row in rows])
        selected = np.asarray([len(row['selected_candidates']) for row in rows])
        result.append({
            'branch': branch, 'conditions': len(rows),
            'eligible_successes_median': float(np.median(eligible)),
            'selected_candidates_median': float(np.median(selected)),
            'selected_candidate_histogram': dict(Counter(map(int, selected))),
            'conditions_with_selected_ge2': int(np.sum(selected >= 2)),
            'conditions_with_selected_ge4': int(np.sum(selected >= 4)),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    root = args.root.resolve()
    completed = read_json(root / 'completed.json')
    if completed.get('heldout_read') is not False or not completed.get('dev_evaluated'):
        raise ValueError('expected completed P2.7 dev artifacts with heldout sealed')
    diagnostics = {
        'kind': 'post_hoc_p27_head_diversity_diagnostic',
        'outcomes': outcome_diagnostics(root, 'screen') + outcome_diagnostics(root, 'dev'),
        'action_diversity': action_diversity(root, 'screen') + action_diversity(root, 'dev'),
        'conditional_variation': (conditional_variation(root, 'screen')
                                  + conditional_variation(root, 'dev')),
        'label_density': {
            'train': label_density(read_json(root / 'train_corpus' / 'manifest.json')),
            'screen': label_density(read_json(root / 'screen_corpus' / 'manifest.json')),
        },
        'changes_preregistered_gate': False, 'heldout_read': False,
    }
    output = root / 'diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    lines = ['# P2.7 post-hoc head diagnostics', '',
             'Descriptive follow-up only; it does not change the preregistered gate. Heldout was not read.',
             '', '## Outcome coverage', '',
             '| split | branch | seed | head 0/1/2/3 | any 4 | best fixed head | gain |',
             '|---|---|---:|---|---:|---:|---:|']
    for row in diagnostics['outcomes']:
        heads = '/'.join(f'{value:.3f}' for value in row['head_rates'])
        lines.append(f"| {row['split']} | {row['branch']} | {row['seed']} | {heads} | "
                     f"{row['any4_rate']:.3f} | {row['best_fixed_head_rate']:.3f} | "
                     f"{row['any4_gain_over_best_fixed_head']:.3f} |")
    lines += ['', '## Learned action diversity', '',
              '| split | branch | groups | pairwise action RMS median [IQR] | exact-duplicate groups |',
              '|---|---|---:|---:|---:|']
    for row in diagnostics['action_diversity']:
        lines.append(f"| {row['split']} | {row['branch']} | {row['condition_seed_groups']} | "
                     f"{row['pairwise_action_rms_median']:.3f} "
                     f"[{row['pairwise_action_rms_q25_q75'][0]:.3f}, "
                     f"{row['pairwise_action_rms_q25_q75'][1]:.3f}] | "
                     f"{row['groups_with_any_exact_duplicate']} |")
    lines += ['', '## Condition dependence versus fixed head library', '',
              '| split | branch | condition-dependent variance share | within-head condition RMS | between-head centroid RMS |',
              '|---|---|---:|---:|---:|']
    for row in diagnostics['conditional_variation']:
        lines.append(f"| {row['split']} | {row['branch']} | "
                     f"{row['condition_dependent_variance_share_mean']:.4f} | "
                     f"{row['within_head_across_condition_rms_mean']:.4f} | "
                     f"{row['between_head_centroid_rms_mean']:.4f} |")
    lines += ['', '## Label density after preregistered selection', '',
              '| split | branch | conditions | eligible median | selected median | selected >=2 | selected >=4 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for split, rows in diagnostics['label_density'].items():
        for row in rows:
            lines.append(f"| {split} | {row['branch']} | {row['conditions']} | "
                         f"{row['eligible_successes_median']:.1f} | "
                         f"{row['selected_candidates_median']:.1f} | "
                         f"{row['conditions_with_selected_ge2']} | "
                         f"{row['conditions_with_selected_ge4']} |")
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'heldout_read': False}, indent=2))


if __name__ == '__main__':
    main()
