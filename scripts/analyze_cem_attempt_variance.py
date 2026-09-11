"""Post-hoc variance audit of all CEM attempts, stratified by compatible protocol."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def classify(path):
    text = path.as_posix()
    if '20260911_p24_cem_feasibility/' in text:
        return 'p24_formal'
    if '20260911_p25_cem_teacher/' in text:
        return 'p25_formal'
    if '20260911_p26_pulse_predictor/' in text:
        return 'p26_formal'
    if '20260911_p2_search_' in text:
        return 'p2_legacy_unconstrained'
    if '20260911_p24_smoke/' in text:
        return 'p24_smoke'
    if '20260911_gpu_pilot/' in text:
        return 'gpu_pilot'
    return 'other'


def metadata(directory):
    for name in ('conditions_search.json', 'search.json', 'summary.json'):
        path = directory / name
        if path.exists():
            return read_json(path), name
    return {}, None


def cohort_key(path, summary, rows):
    protocol = classify(path)
    if protocol not in ('p24_formal', 'p25_formal', 'p26_formal'):
        return None
    kind = summary.get('kind') or ('trajectory' if 'trajectory' in path.name else 'parameters')
    branch = ((summary.get('branches') or [None])[0]
              or (rows[0].get('branch') if rows else None))
    return f'{protocol}_{kind}_{branch}'


def action_representation(parameters, kind, branch):
    parameters = np.asarray(parameters, dtype=float)
    actors = 1 if branch == 'single' else 2
    if kind == 'parameters':
        rows = parameters.reshape(actors, 3)
        result = np.zeros((80, actors))
        t = 0.
        for step in range(80):
            for actor, (amplitude, start, duration) in enumerate(rows):
                if ((start + 1.) * 2. <= t
                        < (start + 1.) * 2. + (duration + 1.) * 1.5):
                    result[step, actor] = amplitude
            t += .1
        return result.ravel()
    knots = parameters.reshape(8, actors)
    return np.repeat(knots, 10, axis=0).ravel()


def variance_decomposition(groups, representation):
    eligible = {key: values for key, values in groups.items() if len(values) >= 2}
    if len(eligible) < 2:
        return None
    arrays = {key: np.stack([representation(row) for row in values])
              for key, values in eligible.items()}
    all_values = np.concatenate(list(arrays.values()))
    dimension = all_values.shape[1]
    global_mean = all_values.mean(0)
    within_ss = sum(np.square(values - values.mean(0)).sum()
                    for values in arrays.values())
    between_ss = sum(len(values) * np.square(values.mean(0) - global_mean).sum()
                     for values in arrays.values())
    denom = len(all_values) * dimension
    total = (within_ss + between_ss) / denom

    centroids = np.stack([values.mean(0) for values in arrays.values()])
    equal_global = centroids.mean(0)
    within_equal = float(np.mean([
        np.square(values - values.mean(0)).mean() for values in arrays.values()]))
    between_equal = float(np.square(centroids - equal_global).mean())
    centroid_distances = [float(np.sqrt(np.square(left - right).mean()))
                          for left, right in itertools.combinations(centroids, 2)]
    within_distances = [float(np.sqrt(np.square(value - values.mean(0)).mean()))
                        for values in arrays.values() for value in values]
    return {
        'conditions_ge2': len(arrays), 'solutions': len(all_values),
        'dimension': dimension,
        'within_variance': within_ss / denom,
        'between_variance': between_ss / denom,
        'total_variance': total,
        'within_share': float(within_ss / max(within_ss + between_ss, 1e-15)),
        'condition_icc': float(between_ss / max(within_ss + between_ss, 1e-15)),
        'equal_condition_within_variance': within_equal,
        'equal_condition_between_variance': between_equal,
        'equal_condition_within_share': (
            within_equal / max(within_equal + between_equal, 1e-15)),
        'median_rms_to_own_centroid': float(np.median(within_distances)),
        'median_rms_between_condition_centroids': float(np.median(centroid_distances)),
    }


def top_quartile(groups):
    result = {}
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda row: row['score'], reverse=True)
        keep = min(len(ordered), max(2, math.ceil(len(ordered) * .25)))
        result[key] = ordered[:keep]
    return result


def sign_multimodality(groups, kind, branch):
    actors = 1 if branch == 'single' else 2
    eligible = 0
    multimodal = 0
    for rows in groups.values():
        if len(rows) < 5:
            continue
        eligible += 1
        values = np.stack([row['search_parameters'] for row in rows])
        if kind == 'parameters':
            coordinates = values.reshape(len(values), actors, 3)[:, :, 0]
        else:
            coordinates = values.reshape(len(values), 8 * actors)
        for coordinate in coordinates.T:
            if np.mean(coordinate > .1) >= .2 and np.mean(coordinate < -.1) >= .2:
                multimodal += 1
                break
    return {'eligible_conditions': eligible, 'sign_multimodal_conditions': multimodal,
            'fraction': multimodal / eligible if eligible else None}


def best_solution_analysis(directories, kind, branch):
    groups = defaultdict(list)
    for directory in directories:
        summary = read_json(directory / 'conditions_search.json')
        for row in summary['per_condition']:
            if row.get('best_valid') and row.get('best_dangerous') \
                    and row.get('best_parameters') is not None:
                groups[row['scenario_id']].append({
                    'search_parameters': row['best_parameters'],
                    'score': row.get('best_score', 0.)})
    raw = variance_decomposition(
        groups, lambda row: np.asarray(row['search_parameters'], dtype=float))
    action = variance_decomposition(
        groups, lambda row: action_representation(
            row['search_parameters'], kind, branch))
    return {'conditions_with_success': len(groups),
            'conditions_with_multiple_search_seed_best': sum(len(v) >= 2 for v in groups.values()),
            'raw_parameters': raw, 'executed_actions': action,
            'sign_multimodality': sign_multimodality(groups, kind, branch)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if 'heldout' in str(args.output).casefold():
        raise ValueError('variance audit output must not target heldout')
    attempt_paths = sorted(args.runs.rglob('attempts.jsonl'))
    inventory, cohort_rows, cohort_directories = [], defaultdict(list), defaultdict(list)
    for path in attempt_paths:
        summary, metadata_file = metadata(path.parent)
        rows = load_rows(path)
        key = cohort_key(path.parent, summary, rows)
        inventory.append({
            'path': path.as_posix(), 'classification': classify(path.parent),
            'included_in_formal_variance': key is not None,
            'metadata_file': metadata_file, 'kind': summary.get('kind'),
            'branches': summary.get('branches'),
            'role_action_mode': summary.get('role_action_mode'),
            'condition_set_version': summary.get('condition_set_version'),
            'attempts': len(rows),
        })
        if key is not None:
            cohort_rows[key].extend(rows)
            cohort_directories[key].append(path.parent)

    results = []
    incremental_manifests = {
        'p25_formal': args.runs / '20260911_p26_pulse_predictor'
        / 'train_corpus' / 'manifest.json',
        'p26_formal': args.runs / '20260911_p26_pulse_predictor'
        / 'screen_corpus' / 'manifest.json',
    }
    for key, rows in sorted(cohort_rows.items()):
        protocol, kind, branch = key.rsplit('_', 2)
        successful = [row for row in rows if row.get('valid') and row.get('dangerous')
                      and row.get('terminated', True) and not row.get('budget_truncated', False)]
        groups = defaultdict(list)
        for row in successful:
            groups[row['scenario_id']].append(row)
        top = top_quartile(groups)
        representation = lambda row: np.asarray(row['search_parameters'], dtype=float)
        actions = lambda row: action_representation(row['search_parameters'], kind, branch)
        result = {
            'cohort': key, 'protocol': protocol, 'kind': kind, 'branch': branch,
            'attempt_files': len(cohort_directories[key]), 'attempts': len(rows),
            'successful_attempts': len(successful),
            'success_attempt_rate': len(successful) / len(rows) if rows else 0.,
            'conditions': len({row['scenario_id'] for row in rows}),
            'conditions_with_success': len(groups),
            'all_success_raw_parameters': variance_decomposition(groups, representation),
            'all_success_executed_actions': variance_decomposition(groups, actions),
            'top_quartile_raw_parameters': variance_decomposition(top, representation),
            'top_quartile_executed_actions': variance_decomposition(top, actions),
            'all_success_sign_multimodality': sign_multimodality(groups, kind, branch),
        }
        if protocol == 'p24_formal':
            result['search_seed_best'] = best_solution_analysis(
                cohort_directories[key], kind, branch)
        if protocol in incremental_manifests and incremental_manifests[protocol].exists():
            selected = {row['scenario_id'] for row in read_json(
                incremental_manifests[protocol]) if row['branch'] == branch}
            incremental = {name: values for name, values in groups.items() if name in selected}
            incremental_top = top_quartile(incremental)
            result['incremental_selected'] = {
                'conditions': len(incremental),
                'successful_attempts': sum(map(len, incremental.values())),
                'median_successes_per_condition': float(np.median(
                    [len(values) for values in incremental.values()])),
                'all_raw_parameters': variance_decomposition(incremental, representation),
                'all_executed_actions': variance_decomposition(incremental, actions),
                'top_quartile_raw_parameters': variance_decomposition(
                    incremental_top, representation),
                'top_quartile_executed_actions': variance_decomposition(
                    incremental_top, actions),
            }
        results.append(result)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    summary = {
        'kind': 'post_hoc_cem_attempt_variance_audit',
        'attempt_files_total': len(inventory),
        'attempt_files_formally_comparable': sum(
            row['included_in_formal_variance'] for row in inventory),
        'classifications': dict(Counter(row['classification'] for row in inventory)),
        'cohorts': results, 'dev_outcomes_reused': True,
        'heldout_read': False,
        'limitations': [
            'CEM candidates are adaptive and not independent samples.',
            'Variance decomposition is descriptive and stratified by dimension/protocol.',
            'A successful parameter vector is not a unique ground-truth label.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# CEM attempt variance audit', '',
        f"Inventoried {len(inventory)} attempt files; "
        f"{summary['attempt_files_formally_comparable']} compatible formal files entered "
        'the variance analysis. Existing dev/training search artifacts only; heldout was not read.',
        '', '## All complete valid dangerous attempts', '',
        '| cohort | attempts | successful | successful conditions | raw within share / ICC | '
        'action within share / ICC | sign-multimodal conditions |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        raw, action = row['all_success_raw_parameters'], row['all_success_executed_actions']
        signs = row['all_success_sign_multimodality']
        lines.append(f"| {row['cohort']} | {row['attempts']} | "
                     f"{row['successful_attempts']} | {row['conditions_with_success']} | "
                     f"{raw['within_share']:.3f} / {raw['condition_icc']:.3f} | "
                     f"{action['within_share']:.3f} / {action['condition_icc']:.3f} | "
                     f"{signs['sign_multimodal_conditions']}/{signs['eligible_conditions']} |")
    lines += ['', '## Top-quartile successful attempts within each condition', '',
              '| cohort | raw within share / ICC | action within share / ICC | '
              'within/action centroid RMS ratio |',
              '|---|---:|---:|---:|']
    for row in results:
        raw, action = row['top_quartile_raw_parameters'], row['top_quartile_executed_actions']
        ratio = (action['median_rms_to_own_centroid']
                 / max(action['median_rms_between_condition_centroids'], 1e-15))
        lines.append(f"| {row['cohort']} | {raw['within_share']:.3f} / "
                     f"{raw['condition_icc']:.3f} | {action['within_share']:.3f} / "
                     f"{action['condition_icc']:.3f} | {ratio:.3f} |")
    lines += ['', '## P2.4 best solution across five search seeds', '',
              '| cohort | repeated-success conditions | raw within share / ICC | '
              'action within share / ICC |',
              '|---|---:|---:|---:|']
    for row in results:
        if 'search_seed_best' not in row:
            continue
        best = row['search_seed_best']
        raw, action = best['raw_parameters'], best['executed_actions']
        lines.append(f"| {row['cohort']} | "
                     f"{best['conditions_with_multiple_search_seed_best']} | "
                     f"{raw['within_share']:.3f} / {raw['condition_icc']:.3f} | "
                     f"{action['within_share']:.3f} / {action['condition_icc']:.3f} |")
    lines += ['', '## P2.6 incremental-teacher subset', '',
              '| source cohort | selected conditions | successful attempts (median/condition) | '
              'all action within share / ICC | top-quartile action within share / ICC |',
              '|---|---:|---:|---:|---:|']
    for row in results:
        if 'incremental_selected' not in row:
            continue
        selected = row['incremental_selected']
        all_action = selected['all_executed_actions']
        top_action = selected['top_quartile_executed_actions']
        lines.append(f"| {row['cohort']} | {selected['conditions']} | "
                     f"{selected['successful_attempts']} "
                     f"({selected['median_successes_per_condition']:.1f}) | "
                     f"{all_action['within_share']:.3f} / "
                     f"{all_action['condition_icc']:.3f} | "
                     f"{top_action['within_share']:.3f} / "
                     f"{top_action['condition_icc']:.3f} |")
    lines += ['', 'Within share is the fraction of successful-solution variance that remains '
              'inside the same condition; ICC is the fraction explained by condition identity. '
              'These are descriptive because CEM samples adapt over iterations.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'attempt_files': len(inventory),
                      'formal_files': summary['attempt_files_formally_comparable'],
                      'cohorts': len(results), 'heldout_read': False}, indent=2))


if __name__ == '__main__':
    main()
