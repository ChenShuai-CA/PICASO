"""Run preregistered P2.7 multi-success pulse-set mechanism screening."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


SEEDS = (7, 17, 27, 37, 47)
BRANCHES = ('single', 'dual')
THRESHOLDS = {'single': .50, 'dual': .25}
TRAIN_SEED = 62000
TRAIN_COUNT = 160
TRAIN_SEARCH_SEEDS = (137, 157)
SCREEN_SEED = 71000
SCREEN_COUNT = 120
SCREEN_SEARCH_SEED = 177
SEARCH_BUDGET = 1000
POPULATION = 8
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2026


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def search_complete(directory, version, branch, search_seed, count):
    directory = Path(directory)
    summary_path, attempts_path = directory / 'conditions_search.json', directory / 'attempts.jsonl'
    if not summary_path.exists() or not attempts_path.exists():
        return False
    summary = read_json(summary_path)
    return (summary.get('kind') == 'parameters'
            and summary.get('seed') == search_seed
            and summary.get('branches') == [branch]
            and summary.get('role_action_mode') == 'lane_locked'
            and summary.get('condition_set_version') == version
            and summary.get('population') == POPULATION
            and summary.get('interaction_budget_per_condition') == SEARCH_BUDGET
            and summary.get('n_conditions') == count
            and summary.get('total_interaction_steps') == count * SEARCH_BUDGET
            and all(row.get('interaction_steps') == SEARCH_BUDGET
                    for row in summary.get('per_condition', []))
            and len(attempts_path.read_text(encoding='utf-8').splitlines())
            == summary.get('total_evaluations'))


def model_complete(directory, seed, corpus_hash):
    directory = Path(directory)
    if not (directory / 'model.pt').exists() or not (directory / 'training.json').exists():
        return False
    bundle = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
    return (bundle.get('schema_version') == 'pulse-set-v1'
            and bundle.get('config', {}).get('seed') == seed
            and bundle.get('config', {}).get('epochs') == 100
            and bundle.get('config', {}).get('hidden') == 16
            and bundle.get('config', {}).get('heads') == 4
            and bundle.get('corpus_sha256') == corpus_hash
            and bundle.get('heldout_read') is False
            and len(read_json(directory / 'training.json')) == 100)


class FixedPredictor(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.register_buffer('fixed_parameters',
                             torch.as_tensor(parameters, dtype=torch.float32))

    def forward(self, tokens, token_mask, actor_mask):
        return self.fixed_parameters[None].expand(tokens.shape[0], -1, -1)


def run_parameters(parameters, source):
    from scenario_lab.evaluate import run_episode
    from scenario_lab.pulse import PulsePolicy
    from scenario_lab.schema import ScenarioSpec
    spec = ScenarioSpec(**source['spec'])
    spec.role_action_mode = 'lane_locked'
    policy = PulsePolicy(FixedPredictor(parameters), device='cpu')
    return run_episode(policy, spec, int(source['replay_seed']))


def predict_heads(model_path, data, device):
    from scenario_lab.pulse_set import load_pulse_set
    model, _ = load_pulse_set(model_path, device)
    with torch.no_grad():
        return model(
            torch.as_tensor(data['tokens'], device=device),
            torch.as_tensor(data['token_mask'], device=device),
            torch.as_tensor(data['actor_mask'], device=device)).cpu().numpy()


def evaluate_learned(output, seed, data, manifest, device):
    path = output / 'screen_evaluation' / f'learned_s{seed}.jsonl'
    expected = len(manifest) * 4
    if path.exists() and len(load_rows(path)) == expected:
        return load_rows(path)
    predictions = predict_heads(output / f'pulse_set_s{seed}' / 'model.pt', data, device)
    rows = []
    for condition_index, source in enumerate(manifest):
        for head in range(4):
            row = run_parameters(predictions[condition_index, head], source)
            row.update(uid=source['uid'], condition_index=condition_index,
                       candidate_index=head, method='learned4', policy_seed=seed,
                       parameters=predictions[condition_index, head].tolist())
            rows.append(row)
    write_rows(path, rows)
    return rows


def evaluate_baselines(output, data, manifest, ridge_predictions):
    baseline_path = output / 'screen_evaluation' / 'baselines.jsonl'
    oracle_path = output / 'screen_evaluation' / 'oracle.jsonl'
    if baseline_path.exists() and len(load_rows(baseline_path)) == len(manifest) * 5:
        baselines = load_rows(baseline_path)
    else:
        rng, baselines = np.random.default_rng(BOOTSTRAP_SEED), []
        for condition_index, source in enumerate(manifest):
            ridge = run_parameters(ridge_predictions[condition_index], source)
            ridge.update(uid=source['uid'], condition_index=condition_index,
                         candidate_index=0, method='ridge1', policy_seed=None,
                         parameters=ridge_predictions[condition_index].tolist())
            baselines.append(ridge)
            active = 1 if source['branch'] == 'single' else 2
            for candidate in range(4):
                parameters = np.zeros((2, 3), dtype=np.float32)
                parameters[:active, 0] = rng.uniform(-1., 1., active)
                parameters[:active, 1] = rng.uniform(-.75, 1., active)
                parameters[:active, 2] = rng.uniform(-1., 1., active)
                row = run_parameters(parameters, source)
                row.update(uid=source['uid'], condition_index=condition_index,
                           candidate_index=candidate, method='random4', policy_seed=None,
                           parameters=parameters.tolist())
                baselines.append(row)
        write_rows(baseline_path, baselines)

    expected_oracle = sum(len(source['selected_candidates']) for source in manifest)
    if oracle_path.exists() and len(load_rows(oracle_path)) == expected_oracle:
        oracle = load_rows(oracle_path)
    else:
        oracle = []
        for condition_index, source in enumerate(manifest):
            for candidate_index, candidate in enumerate(source['selected_candidates']):
                replay_source = dict(source, replay_seed=candidate['replay_seed'])
                active = 1 if source['branch'] == 'single' else 2
                parameters = np.zeros((2, 3), dtype=np.float32)
                parameters[:active] = np.asarray(candidate['parameters']).reshape(active, 3)
                row = run_parameters(parameters, replay_source)
                row.update(uid=source['uid'], condition_index=condition_index,
                           candidate_index=candidate_index, method='teacher_oracle',
                           policy_seed=None, parameters=parameters.tolist())
                oracle.append(row)
        write_rows(oracle_path, oracle)
    return baselines, oracle


def coverage(rows, branch, policy_seed=None):
    selected = [row for row in rows if row['branch'] == branch
                and (policy_seed is None or row.get('policy_seed') == policy_seed)]
    grouped = {}
    for row in selected:
        grouped.setdefault(row['uid'], []).append(row)
    return {uid: float(any(row['valid'] and row['dangerous'] for row in attempts))
            for uid, attempts in grouped.items()}


def two_way_absolute(matrix, rounds=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(rounds):
        seeds = rng.integers(0, matrix.shape[0], matrix.shape[0])
        conditions = rng.integers(0, matrix.shape[1], matrix.shape[1])
        estimates.append(matrix[np.ix_(seeds, conditions)].mean())
    return float(matrix.mean()), np.quantile(estimates, [.025, .975]).tolist()


def two_way_paired(matrix, baseline, rounds=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    difference = matrix - baseline[None]
    rate, ci = two_way_absolute(difference, rounds, seed)
    return rate, ci


def cumulative_cost(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.get('policy_seed'), row['uid']), []).append(row)
    values = []
    for attempts in grouped.values():
        attempts.sort(key=lambda row: row['candidate_index'])
        total = 0
        for row in attempts:
            total += int(row['decision_steps'])
            if row['valid'] and row['dangerous']:
                break
        values.append(total)
    return {'total_all_candidates': int(sum(row['decision_steps'] for row in rows)),
            'mean_steps_until_success_or_exhaustion': float(np.mean(values)),
            'median_steps_until_success_or_exhaustion': float(np.median(values))}


def summarize_screen(output, train_report, screen_report, ridge_report,
                     learned_by_seed, baselines, oracle):
    training = []
    for seed in SEEDS:
        bundle = torch.load(output / f'pulse_set_s{seed}' / 'model.pt',
                            map_location='cpu', weights_only=True)
        training.append({
            'seed': seed, 'selected_epoch': bundle['config']['selected_epoch'],
            'selected_val_set_loss': next(
                row['val_branch_balanced_set_loss'] for row in
                read_json(output / f'pulse_set_s{seed}' / 'training.json')
                if row['epoch'] == bundle['config']['selected_epoch']),
            'parameters': sum(value.numel() for value in bundle['model'].values()),
        })
    mechanism = []
    baseline_methods = {name: [row for row in baselines if row['method'] == name]
                        for name in ('ridge1', 'random4')}
    for branch in BRANCHES:
        ids = sorted(coverage(learned_by_seed[SEEDS[0]], branch, SEEDS[0]))
        matrix = np.asarray([[coverage(learned_by_seed[seed], branch, seed)[uid]
                              for uid in ids] for seed in SEEDS])
        ridge_map = coverage(baseline_methods['ridge1'], branch)
        random_map = coverage(baseline_methods['random4'], branch)
        ridge = np.asarray([ridge_map[uid] for uid in ids])
        random = np.asarray([random_map[uid] for uid in ids])
        rate, absolute_ci = two_way_absolute(matrix)
        ridge_delta, ridge_ci = two_way_paired(matrix, ridge)
        random_delta, random_ci = two_way_paired(matrix, random)
        attempts = [row for seed in SEEDS for row in learned_by_seed[seed]
                    if row['branch'] == branch]
        valid_rate = float(np.mean([row['valid'] for row in attempts]))
        role_invalid = sum(reason in ('pedestrian_role', 'occluder_role')
                           for row in attempts for reason in row['invalid_reasons'])
        passed = (len(ids) >= 15 and absolute_ci[0] > THRESHOLDS[branch]
                  and ridge_ci[0] > 0 and random_ci[0] >= -.05
                  and valid_rate >= .8 and role_invalid == 0)
        mechanism.append({
            'branch': branch, 'conditions': len(ids), 'learned4_rate': rate,
            'learned4_ci95': absolute_ci, 'absolute_threshold': THRESHOLDS[branch],
            'ridge1_rate': float(ridge.mean()), 'delta_vs_ridge1': ridge_delta,
            'delta_vs_ridge1_ci95': ridge_ci, 'random4_rate': float(random.mean()),
            'delta_vs_random4': random_delta, 'delta_vs_random4_ci95': random_ci,
            'candidate_valid_rate': valid_rate, 'role_invalid': role_invalid,
            'mechanism_pass': passed, 'learned_cost': cumulative_cost(attempts),
            'ridge_cost': cumulative_cost([row for row in baseline_methods['ridge1']
                                           if row['branch'] == branch]),
            'random_cost': cumulative_cost([row for row in baseline_methods['random4']
                                            if row['branch'] == branch]),
        })
    oracle_rate = {}
    for branch in BRANCHES:
        oracle_rate[branch] = float(np.mean(list(coverage(oracle, branch).values())))
    eligible = all(row['mechanism_pass'] for row in mechanism)
    gate = {'eligible_for_dev': eligible, 'mechanism': mechanism,
            'teacher_oracle_condition_coverage': oracle_rate,
            'bootstrap_rounds': BOOTSTRAP, 'bootstrap_seed': BOOTSTRAP_SEED,
            'heldout_read': False}
    (output / 'mechanism_gate.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')
    summary = {'protocol': 'P2.7 multi-success solution-set supervision',
               'train_corpus': train_report, 'screen_corpus': screen_report,
               'training': training, 'ridge': ridge_report, **gate,
               'dev_evaluated': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        '# P2.7 multi-success solution-set supervision', '',
        f"Training corpus: {train_report['examples']} conditions, median "
        f"{train_report['candidate_count_median']:.1f} selected solutions. Independent "
        f"screen: {screen_report['examples']} eligible script-safe conditions. Heldout was not read.",
        '', '## Preregistered mechanism gate', '',
        '| branch | N | learned-4 [95% CI] | ridge-1 | delta [95% CI] | random-4 | delta [95% CI] | valid | role invalid | pass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in mechanism:
        lines.append(
            f"| {row['branch']} | {row['conditions']} | {row['learned4_rate']:.3f} "
            f"[{row['learned4_ci95'][0]:.3f}, {row['learned4_ci95'][1]:.3f}] | "
            f"{row['ridge1_rate']:.3f} | {row['delta_vs_ridge1']:.3f} "
            f"[{row['delta_vs_ridge1_ci95'][0]:.3f}, {row['delta_vs_ridge1_ci95'][1]:.3f}] | "
            f"{row['random4_rate']:.3f} | {row['delta_vs_random4']:.3f} "
            f"[{row['delta_vs_random4_ci95'][0]:.3f}, {row['delta_vs_random4_ci95'][1]:.3f}] | "
            f"{row['candidate_valid_rate']:.3f} | {row['role_invalid']} | {row['mechanism_pass']} |")
    lines += ['', '## Interaction cost', '',
              '| branch | learned-4 total / mean until success | ridge-1 total | random-4 total / mean until success |',
              '|---|---:|---:|---:|']
    for row in mechanism:
        lines.append(f"| {row['branch']} | {row['learned_cost']['total_all_candidates']} / "
                     f"{row['learned_cost']['mean_steps_until_success_or_exhaustion']:.1f} | "
                     f"{row['ridge_cost']['total_all_candidates']} | "
                     f"{row['random_cost']['total_all_candidates']} / "
                     f"{row['random_cost']['mean_steps_until_success_or_exhaustion']:.1f} |")
    lines += ['', f"Exact selected-teacher replay condition coverage: single "
              f"{oracle_rate['single']:.3f}, dual {oracle_rate['dual']:.3f}.", '',
              f'Eligible for conditional dev evaluation: **{eligible}**.', '']
    if not eligible:
        lines += ['The preregistered screen gate failed, so P2.7 did not read or evaluate '
                  'the frozen development conditions.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return gate


def conditional_dev(output, dev_path, p22, train_data, device, audit_condition_paths):
    """Read and evaluate development conditions only after the screen gate passes."""
    from scenario_lab.pulse_set_data import fit_ridge, observable_history, predict_ridge
    from scenario_lab.sampling import load_conditions
    from scenario_lab.teacher import spec_fingerprint

    dev_specs, dev_header = load_conditions(dev_path)
    if dev_header.get('purpose') != 'development':
        raise ValueError('conditional P2.7 evaluation requires development conditions')
    sets = {}
    for path in [*map(Path, audit_condition_paths), dev_path]:
        specs, header = load_conditions(path)
        sets[header['condition_set_version']] = {spec_fingerprint(spec) for spec in specs}
    overlaps = {}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[f'{left}__{right}'] = len(sets[left] & sets[right])
    if any(overlaps.values()):
        raise ValueError(f'P2.7 condition fingerprint overlap: {overlaps}')
    (output / 'dev_fingerprint_audit.json').write_text(
        json.dumps({'overlaps': overlaps}, indent=2), encoding='utf-8')

    script_rows = load_rows(p22 / 'eval_script' / 'episodes.jsonl')
    script_map = {(row['branch'], row['scenario_id']): row for row in script_rows}
    histories, manifest, aligned_script = [], [], []
    for index, spec in enumerate(dev_specs):
        baseline = script_map[(spec.branch, spec.scenario_id)]
        history = observable_history(spec, int(baseline['seed']))
        histories.append(history)
        uid = f"{dev_header['condition_set_version']}::{spec.branch}::{spec.scenario_id}"
        manifest.append({'uid': uid, 'branch': spec.branch, 'scenario_id': spec.scenario_id,
                         'replay_seed': int(baseline['seed']), 'spec': spec.to_dict()})
        aligned_script.append({**baseline, 'uid': uid, 'method': 'script',
                               'candidate_index': 0, 'policy_seed': None})
    dev_data = {key: np.stack([row[key] for row in histories])
                for key in ('tokens', 'token_mask', 'actor_mask')}
    dev_data.update(branch=np.asarray([row['branch'] for row in manifest]),
                    split=np.asarray(['dev'] * len(manifest)))
    dummy = dict(dev_data)
    dummy['target'] = np.zeros((len(manifest), 8, 2, 3), dtype=np.float32)
    dummy['candidate_mask'] = np.ones((len(manifest), 8), dtype=bool)
    _, _, ridge_models = fit_ridge(train_data, dummy)
    ridge_predictions = predict_ridge(dev_data, ridge_models)

    learned_by_seed = {}
    for seed in SEEDS:
        path = output / 'dev_evaluation' / f'learned_s{seed}.jsonl'
        if path.exists() and len(load_rows(path)) == len(manifest) * 4:
            learned_by_seed[seed] = load_rows(path)
            continue
        predictions = predict_heads(output / f'pulse_set_s{seed}' / 'model.pt',
                                    dev_data, device)
        rows = []
        for condition_index, source in enumerate(manifest):
            for head in range(4):
                row = run_parameters(predictions[condition_index, head], source)
                row.update(uid=source['uid'], condition_index=condition_index,
                           candidate_index=head, method='learned4', policy_seed=seed,
                           parameters=predictions[condition_index, head].tolist())
                rows.append(row)
        write_rows(path, rows)
        learned_by_seed[seed] = rows

    baseline_path = output / 'dev_evaluation' / 'baselines.jsonl'
    if baseline_path.exists() and len(load_rows(baseline_path)) == len(manifest) * 5:
        baselines = load_rows(baseline_path)
    else:
        rng, baselines = np.random.default_rng(BOOTSTRAP_SEED), []
        for condition_index, source in enumerate(manifest):
            row = run_parameters(ridge_predictions[condition_index], source)
            row.update(uid=source['uid'], condition_index=condition_index,
                       candidate_index=0, method='ridge1', policy_seed=None,
                       parameters=ridge_predictions[condition_index].tolist())
            baselines.append(row)
            active = 1 if source['branch'] == 'single' else 2
            for candidate in range(4):
                parameters = np.zeros((2, 3), dtype=np.float32)
                parameters[:active, 0] = rng.uniform(-1., 1., active)
                parameters[:active, 1] = rng.uniform(-.75, 1., active)
                parameters[:active, 2] = rng.uniform(-1., 1., active)
                row = run_parameters(parameters, source)
                row.update(uid=source['uid'], condition_index=condition_index,
                           candidate_index=candidate, method='random4', policy_seed=None,
                           parameters=parameters.tolist())
                baselines.append(row)
        write_rows(baseline_path, baselines)
    write_rows(output / 'dev_evaluation' / 'script.jsonl', aligned_script)

    results = []
    for branch in BRANCHES:
        ids = sorted(coverage(learned_by_seed[SEEDS[0]], branch, SEEDS[0]))
        learned = np.asarray([[coverage(learned_by_seed[seed], branch, seed)[uid]
                               for uid in ids] for seed in SEEDS])
        methods = {}
        for method, rows in (('script', aligned_script), ('ridge1', baselines),
                             ('random4', baselines)):
            subset = rows if method == 'script' else [row for row in rows if row['method'] == method]
            mapping = coverage(subset, branch)
            methods[method] = np.asarray([mapping[uid] for uid in ids])
        rate, ci = two_way_absolute(learned)
        delta, delta_ci = two_way_paired(learned, methods['script'])
        attempts = [row for seed in SEEDS for row in learned_by_seed[seed]
                    if row['branch'] == branch]
        valid_rate = float(np.mean([row['valid'] for row in attempts]))
        role_invalid = sum(reason in ('pedestrian_role', 'occluder_role')
                           for row in attempts for reason in row['invalid_reasons'])
        results.append({
            'branch': branch, 'conditions': len(ids), 'learned4_rate': rate,
            'learned4_ci95': ci, 'script_rate': float(methods['script'].mean()),
            'ridge1_rate': float(methods['ridge1'].mean()),
            'random4_rate': float(methods['random4'].mean()),
            'delta_vs_script': delta, 'delta_vs_script_ci95': delta_ci,
            'candidate_valid_rate': valid_rate, 'role_invalid': role_invalid,
            'dev_superiority_pass': (delta > 0 and delta_ci[0] > 0
                                     and valid_rate >= .8 and role_invalid == 0),
            'learned_cost': cumulative_cost(attempts),
            'script_cost': cumulative_cost([row for row in aligned_script
                                            if row['branch'] == branch]),
            'ridge_cost': cumulative_cost([row for row in baselines
                                           if row['branch'] == branch
                                           and row['method'] == 'ridge1']),
            'random_cost': cumulative_cost([row for row in baselines
                                            if row['branch'] == branch
                                            and row['method'] == 'random4']),
        })
    summary = read_json(output / 'summary.json')
    summary.update(dev_evaluated=True, dev_condition_set_version=dev_header['condition_set_version'],
                   dev_results=results)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'REPORT.md').open('a', encoding='utf-8') as report:
        report.write('\n## Conditional development result\n\n')
        report.write('| branch | learned-4 | script | ridge-1 | random-4 | delta vs script [95% CI] | valid | pass |\n')
        report.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for row in results:
            report.write(f"| {row['branch']} | {row['learned4_rate']:.3f} | "
                         f"{row['script_rate']:.3f} | {row['ridge1_rate']:.3f} | "
                         f"{row['random4_rate']:.3f} | {row['delta_vs_script']:.3f} "
                         f"[{row['delta_vs_script_ci95'][0]:.3f}, "
                         f"{row['delta_vs_script_ci95'][1]:.3f}] | "
                         f"{row['candidate_valid_rate']:.3f} | "
                         f"{row['dev_superiority_pass']} |\n")
        report.write('\nThe learned result uses four candidate rollouts per condition. '
                     'All five training seeds are retained in the inference report; '
                     'ridge and script use one rollout, while random uses four.\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/20260911_p27_multisolution')
    parser.add_argument('--p25', default='runs/20260911_p25_cem_teacher')
    parser.add_argument('--p26', default='runs/20260911_p26_pulse_predictor')
    parser.add_argument('--p22', default='runs/20260911_p22_architecture')
    parser.add_argument('--dev-conditions', default='runs/20260911_p2_conditions/dev_seed31000.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if 'heldout' in str(args.dev_conditions).casefold():
        raise ValueError('P2.7 must not use heldout conditions')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from scenario_lab.pulse_set import train_pulse_set
    from scenario_lab.pulse_set_data import build_pulse_set_corpus, fit_ridge
    from scenario_lab.runtime import resolve_device
    from scenario_lab.sampling import export_conditions

    output = (project / args.output).resolve()
    p25, p26 = (project / args.p25).resolve(), (project / args.p26).resolve()
    p22 = (project / args.p22).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for completed in (p25 / 'completed.json', p26 / 'completed.json'):
        if read_json(completed).get('heldout_read') is not False:
            raise ValueError(f'heldout-read certificate missing: {completed}')
    p25_conditions = p25 / 'training_conditions.json'
    p26_conditions = p26 / 'screen_conditions.json'
    preregistration = {
        'protocol': 'P2.7 multi-success solution-set supervision',
        'created_before_new_conditions_and_searches': True,
        'reused_training_versions': [read_json(p25_conditions)['condition_set_version'],
                                     read_json(p26_conditions)['condition_set_version']],
        'new_dual_training_seed': TRAIN_SEED, 'new_dual_training_count': TRAIN_COUNT,
        'new_dual_search_seeds': list(TRAIN_SEARCH_SEEDS),
        'screen_seed': SCREEN_SEED, 'screen_count_per_branch': SCREEN_COUNT,
        'screen_search_seed': SCREEN_SEARCH_SEED, 'search_population': POPULATION,
        'search_interaction_budget_per_condition': SEARCH_BUDGET,
        'role_action_mode': 'lane_locked', 'history_steps': 5,
        'candidate_selection': 'score top half then action-RMS farthest-first maximum 8',
        'model': {'hidden': 16, 'heads': 4, 'seeds': list(SEEDS), 'epochs': 100,
                  'batch_size': 16, 'learning_rate': 3e-4,
                  'loss': 'coverage_chamfer_plus_0.25_precision_branch_weighted'},
        'baselines': {'ridge_alphas': [.01, .1, 1., 10., 100.],
                      'ridge_target': 'joint successful action medoid',
                      'random_candidates': 4, 'random_seed': BOOTSTRAP_SEED},
        'thresholds': THRESHOLDS, 'minimum_screen_conditions_per_branch': 15,
        'bootstrap_rounds': BOOTSTRAP, 'bootstrap_seed': BOOTSTRAP_SEED,
        'dev_read_only_if_both_branch_screen_gates_pass': True,
        'dev_path_recorded_but_not_read': str(args.dev_conditions), 'heldout_read': False,
    }
    prereg_path = output / 'preregistration.json'
    if prereg_path.exists() and read_json(prereg_path) != preregistration:
        raise ValueError('existing P2.7 preregistration differs; use a fresh output')
    prereg_path.write_text(json.dumps(preregistration, indent=2), encoding='utf-8')

    train_conditions = output / 'training_conditions_62000.json'
    if not train_conditions.exists():
        export_conditions(train_conditions, seed=TRAIN_SEED, count=TRAIN_COUNT,
                          branches=('dual',), sampler_version=2, role='reference',
                          purpose='training')
    screen_conditions = output / 'screen_conditions_71000.json'
    if not screen_conditions.exists():
        export_conditions(screen_conditions, seed=SCREEN_SEED, count=SCREEN_COUNT,
                          branches=BRANCHES, sampler_version=2, role='reference',
                          purpose='training')
    train_version = read_json(train_conditions)['condition_set_version']
    screen_version = read_json(screen_conditions)['condition_set_version']

    jobs = []
    for search_seed in TRAIN_SEARCH_SEEDS:
        jobs.append((f'train62000_dual_s{search_seed}', train_conditions, 'dual',
                     search_seed, TRAIN_COUNT, train_version))
    for branch in BRANCHES:
        jobs.append((f'screen71000_{branch}_s{SCREEN_SEARCH_SEED}', screen_conditions,
                     branch, SCREEN_SEARCH_SEED, SCREEN_COUNT, screen_version))
    pending = []
    for name, conditions, branch, search_seed, count, version in jobs:
        directory = output / name
        if search_complete(directory, version, branch, search_seed, count):
            print(f'SKIP verified {name}', flush=True)
            continue
        command = [sys.executable, '-m', 'scenario_lab', 'search', '--conditions', conditions,
                   '--branch', branch, '--kind', 'parameters', '--interaction-budget',
                   str(SEARCH_BUDGET), '--population', str(POPULATION),
                   '--role-action-mode', 'lane_locked', '--seed', str(search_seed),
                   '--output', directory]
        log = (output / f'{name}.log').open('w', encoding='utf-8')
        process = subprocess.Popen(list(map(str, command)), cwd=project, stdout=log,
                                   stderr=subprocess.STDOUT)
        pending.append((name, process, log, directory, version, branch, search_seed, count))
        print(f'START {name} pid={process.pid}', flush=True)
    for name, process, log, directory, version, branch, search_seed, count in pending:
        returncode = process.wait(); log.close()
        if returncode or not search_complete(directory, version, branch, search_seed, count):
            raise RuntimeError(f'{name} failed or artifact verification failed')
        print(f'PASS {name}', flush=True)

    train_dir = output / 'train_corpus'
    if not (train_dir / 'pulse_set.npz').exists():
        build_pulse_set_corpus([
            {'name': 'p25_seed51000', 'conditions': p25_conditions,
             'searches': [p25 / 'teacher_search_single', p25 / 'teacher_search_dual']},
            {'name': 'p26_seed61000', 'conditions': p26_conditions,
             'searches': [p26 / 'screen_search_single', p26 / 'screen_search_dual']},
            {'name': 'p27_seed62000_dual', 'conditions': train_conditions,
             'searches': [output / f'train62000_dual_s{seed}' for seed in TRAIN_SEARCH_SEEDS]},
        ], train_dir, 'train_val', audit_conditions=[screen_conditions])
    screen_dir = output / 'screen_corpus'
    if not (screen_dir / 'pulse_set.npz').exists():
        build_pulse_set_corpus([
            {'name': 'p27_seed71000_screen', 'conditions': screen_conditions,
             'searches': [output / f'screen71000_{branch}_s{SCREEN_SEARCH_SEED}'
                          for branch in BRANCHES]},
        ], screen_dir, 'screen',
            audit_conditions=[p25_conditions, p26_conditions, train_conditions])
    train_report, screen_report = read_json(train_dir / 'report.json'), read_json(screen_dir / 'report.json')
    train_data, screen_data = load_npz(train_dir / 'pulse_set.npz'), load_npz(screen_dir / 'pulse_set.npz')
    corpus_hash = train_report['corpus_sha256']
    for seed in SEEDS:
        directory = output / f'pulse_set_s{seed}'
        if model_complete(directory, seed, corpus_hash):
            print(f'SKIP verified pulse_set_s{seed}', flush=True)
            continue
        result = train_pulse_set(train_dir / 'pulse_set.npz', directory, seed=seed,
                                 epochs=100, hidden=16, heads=4, batch_size=16,
                                 device=args.device)
        if not model_complete(directory, seed, corpus_hash):
            raise RuntimeError(f'pulse_set_s{seed} artifact verification failed')
        print(json.dumps({'model_seed': seed, **result}), flush=True)

    ridge_predictions, ridge_report, _ = fit_ridge(train_data, screen_data)
    np.save(output / 'ridge_screen_predictions.npy', ridge_predictions)
    (output / 'ridge_report.json').write_text(json.dumps(ridge_report, indent=2), encoding='utf-8')
    device = resolve_device(args.device)
    manifest = read_json(screen_dir / 'manifest.json')
    learned_by_seed = {}
    for seed in SEEDS:
        learned_by_seed[seed] = evaluate_learned(output, seed, screen_data, manifest, device)
        print(f'PASS screen learned seed {seed}', flush=True)
    baselines, oracle = evaluate_baselines(output, screen_data, manifest, ridge_predictions)
    gate = summarize_screen(output, train_report, screen_report, ridge_report,
                            learned_by_seed, baselines, oracle)
    dev_evaluated = False
    if gate['eligible_for_dev']:
        conditional_dev(
            output, (project / args.dev_conditions).resolve(), p22, train_data, device,
            [p25_conditions, p26_conditions, train_conditions, screen_conditions])
        dev_evaluated = True
    completed = {
        'status': 'completed', 'policy_seeds': list(SEEDS),
        'training_condition_versions': [row['condition_set_version'] for row in train_report['sources']],
        'screen_condition_set_version': screen_version,
        'train_examples': train_report['examples'], 'screen_examples': screen_report['examples'],
        'mechanism_gate_passed': gate['eligible_for_dev'], 'dev_evaluated': dev_evaluated,
        'heldout_read': False,
        'results_kind': ('training-domain mechanism screen plus conditional development'
                         if dev_evaluated else 'training-domain multi-solution mechanism screen'),
    }
    (output / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == '__main__':
    main()
