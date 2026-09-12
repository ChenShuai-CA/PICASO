"""Run P2.10 independent confirmation of the frozen P2.8 ridge router."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_lab.pulse_router import (  # noqa: E402
    history_features, paired_bootstrap, predict_router, ranking)
from scenario_lab.routing_diagnostics import permutation_alignment  # noqa: E402
from run_p28_conditional_router import (  # noqa: E402
    compact_manifest, method_metrics, prepare_dataset)


BRANCHES = ('single', 'dual')
SCREEN_SEED = 75000
DEVELOPMENT_SEED = 74000
COUNT_PER_BRANCH = 160
PERTURBATIONS = 5


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_router(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def local_candidate_rows(candidate_rows, select):
    old_to_new = {
        int(old): new for new, old in enumerate(np.flatnonzero(select))
    }
    return [
        {**row, 'condition_index': old_to_new[row['condition_index']]}
        for row in candidate_rows if row['condition_index'] in old_to_new
    ]


def evaluate_split(data, candidate_rows, models, fixed_orders, prereg, split_name):
    features = history_features(data)
    rounds = int(prereg['inference']['condition_bootstrap_rounds'])
    permutation_rounds = int(prereg['inference']['permutation_rounds'])
    base_seed = int(prereg['inference']['seed'])
    gate_spec = prereg['gate_each_branch_for_screen_and_development']
    results = []
    for branch_index, branch in enumerate(BRANCHES):
        select = data['branch'] == branch
        local = {
            key: value[select] for key, value in data.items()
            if isinstance(value, np.ndarray) and len(value) == len(data['branch'])
        }
        rows = local_candidate_rows(candidate_rows, select)
        prediction = predict_router(models[branch], features[select])
        router_order = ranking(prediction)
        fixed_order = np.tile(
            np.asarray(fixed_orders[branch], dtype=int), (int(select.sum()), 1))
        router1_scores, router1 = method_metrics(local, rows, router_order, 1)
        router2_scores, router2 = method_metrics(local, rows, router_order, 2)
        fixed2_scores, fixed2 = method_metrics(local, rows, fixed_order, 2)
        script_scores = local['script_outcomes'].mean(axis=1)
        seed = base_seed + branch_index * 1000 + (0 if split_name == 'screen' else 10000)
        delta_fixed, ci_fixed = paired_bootstrap(
            router2_scores, fixed2_scores, rounds=rounds, seed=seed)
        delta_script, ci_script = paired_bootstrap(
            router2_scores, script_scores, rounds=rounds, seed=seed + 1)
        alignment = permutation_alignment(
            local['outcomes'], router_order, budget=2,
            rounds=permutation_rounds, seed=seed + 2)
        top2 = [tuple(map(int, row[:2])) for row in router_order]
        counts = Counter(top2)
        modal_share = max(counts.values()) / len(top2)
        gates = {
            'minimum_conditions': int(select.sum()) >= gate_spec['minimum_eligible_conditions'],
            'router2_vs_fixed2': ci_fixed[0] > 0,
            'router2_vs_script': ci_script[0] > 0,
            'permutation_alignment': (
                alignment['delta_vs_permutation_mean']
                >= gate_spec['router2_minus_permutation_mean_at_least']
                and alignment['permutation_p_one_sided']
                <= gate_spec['permutation_p_one_sided_at_most']),
            'candidate_validity': (
                router2['candidate_valid_rate']
                >= gate_spec['router2_selected_candidate_valid_rate_at_least']),
            'role_validity': router2['role_invalid'] == gate_spec['router2_role_invalid_equals'],
            'order_diversity': (
                len(counts) >= gate_spec['distinct_router_top2_orders_at_least']
                and modal_share <= gate_spec['modal_router_top2_share_at_most']),
        }
        results.append({
            'split': split_name, 'branch': branch,
            'conditions': int(select.sum()),
            'router1': router1, 'router2': router2, 'fixed2': fixed2,
            'script_rate': float(script_scores.mean()),
            'router2_minus_fixed2': delta_fixed,
            'router2_minus_fixed2_ci95': ci_fixed,
            'router2_minus_script': delta_script,
            'router2_minus_script_ci95': ci_script,
            'router2_minus_permutation_mean': alignment['delta_vs_permutation_mean'],
            'permutation_mean': alignment['permutation_mean'],
            'permutation_ci95': alignment['permutation_ci95'],
            'permutation_p_one_sided': alignment['permutation_p_one_sided'],
            'distinct_router_top2_orders': len(counts),
            'modal_router_top2_share': modal_share,
            'top2_order_counts': {f'{left}+{right}': count
                                  for (left, right), count in sorted(counts.items())},
            'prediction_std_by_head': prediction.std(axis=0).tolist(),
            'gates': gates, 'pass': all(gates.values()),
        })
    return results


def fmt(value, ci):
    return f'{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'


def report_table(lines, title, results):
    lines += [f'## {title}', '',
              '| branch | N | router-1 | router-2 | fixed-2 | script | '
              'router2-fixed2 [95% CI] | router2-script [95% CI] | '
              'perm delta | perm p | valid | pass |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append(
            f"| {row['branch']} | {row['conditions']} | {row['router1']['rate']:.3f} | "
            f"{row['router2']['rate']:.3f} | {row['fixed2']['rate']:.3f} | "
            f"{row['script_rate']:.3f} | "
            f"{fmt(row['router2_minus_fixed2'], row['router2_minus_fixed2_ci95'])} | "
            f"{fmt(row['router2_minus_script'], row['router2_minus_script_ci95'])} | "
            f"{row['router2_minus_permutation_mean']:.3f} | "
            f"{row['permutation_p_one_sided']:.4f} | "
            f"{row['router2']['candidate_valid_rate']:.3f} | {row['pass']} |")
    lines += ['', 'Router-2 interaction cost (mean decision steps until success or two-candidate '
              'exhaustion): ' + '; '.join(
                  f"{row['branch']} {row['router2']['mean_steps_until_success_or_exhaustion']:.1f}"
                  for row in results) + '.', '']


def write_report(path, screen_results, development_results, final_eligible):
    lines = [
        '# P2.10 frozen-router independent confirmation', '',
        'The P2.7 four-prototype library and P2.8 branch-specific ridge routers are frozen. '
        'This run uses a new seed-75000 confirmatory screen and a 5000-permutation alignment '
        'test. The numerical perturbation envelope is retained only as a reproducible '
        'sensitivity domain; this is not AEB-calibrated validation.', ''
    ]
    report_table(lines, 'Independent screen', screen_results)
    screen_pass = all(row['pass'] for row in screen_results)
    lines += [f'Both-branch screen gate: **{screen_pass}**.', '']
    if development_results is None:
        lines += [
            'The screen gate failed, so the fresh-development condition file was not read or '
            'evaluated.', ''
        ]
    else:
        report_table(lines, 'Conditionally opened fresh development', development_results)
        lines += [
            f"Both-branch fresh-development gate: **{all(row['pass'] for row in development_results)}**.",
            ''
        ]
    if final_eligible:
        lines += [
            'The frozen two-candidate router passed both independent stages and is eligible for '
            'a later, explicitly authorized one-time heldout evaluation. P2.10 did not read heldout.',
            ''
        ]
    else:
        lines += [
            'The frozen router is not eligible for heldout evaluation under the P2.10 rules. '
            'The failed split must not be used for tuning.', ''
        ]
    lines += [
        'The result concerns dangerous-and-valid coverage with up to two candidate rollouts per '
        'condition. It is not evidence that one closed-loop policy execution beats the script.', ''
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/20260912_p210_confirmatory_router')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    output = (ROOT / args.output).resolve()
    if 'heldout' in str(output).casefold():
        raise ValueError('P2.10 must not read or write heldout paths')
    prereg_path = output / 'preregistration.json'
    if not prereg_path.exists():
        raise FileNotFoundError('freeze P2.10 preregistration before generating the screen')
    prereg = read_json(prereg_path)
    if not prereg.get('created_before_p210_screen_conditions_and_outcomes'):
        raise ValueError('P2.10 preregistration is not marked as pre-screen')

    frozen = prereg['frozen_inputs']
    for key in ('prototype_library', 'single_ridge', 'dual_ridge', 'perturbation_config'):
        source = ROOT / frozen[key]['path']
        if file_hash(source) != frozen[key]['sha256']:
            raise ValueError(f'frozen input hash mismatch: {key}')
    p29 = ROOT / 'runs/20260912_p29_routing_diagnosis'
    for filename, key in (
            ('preregistration.json', 'p29_preregistration_sha256'),
            ('summary.json', 'p29_summary_sha256'),
            ('completed.json', 'p29_completed_sha256')):
        if file_hash(p29 / filename) != frozen[key]:
            raise ValueError(f'P2.9 provenance hash mismatch: {filename}')

    from scenario_lab.sampling import export_conditions, load_conditions
    from scenario_lab.teacher import spec_fingerprint
    from scenario_lab.train import load_perturb_config

    library_file = read_json(ROOT / frozen['prototype_library']['path'])
    library = {branch: np.asarray(library_file['parameters'][branch], dtype=np.float32)
               for branch in BRANCHES}
    models = {
        branch: load_router(ROOT / frozen[f'{branch}_ridge']['path'])
        for branch in BRANCHES
    }
    fixed_orders = prereg['frozen_methods']['fixed_orders']
    perturb_config = load_perturb_config(ROOT / frozen['perturbation_config']['path'])
    if perturb_config['version'] != 'abd_supported_v1':
        raise ValueError('P2.10 numerical sensitivity config version changed')

    condition_dir = output / 'conditions'
    condition_dir.mkdir(parents=True, exist_ok=True)
    screen_path = condition_dir / 'confirmatory_screen_seed75000.json'
    if not screen_path.exists():
        export_conditions(
            screen_path, SCREEN_SEED, COUNT_PER_BRANCH, BRANCHES,
            sampler_version=2, role='reference', purpose='training')

    p28_conditions = ROOT / 'runs/20260912_p28_conditional_router/conditions'
    prior_paths = [
        p28_conditions / 'training_seed72000.json',
        p28_conditions / 'screen_seed73000.json',
    ]
    prior_fingerprints = set()
    prior_counts = {}
    for path in prior_paths:
        specs, _ = load_conditions(path)
        fingerprints = {spec_fingerprint(spec) for spec in specs}
        prior_counts[path.name] = len(fingerprints)
        prior_fingerprints.update(fingerprints)
    screen_specs, screen_header = load_conditions(screen_path)
    screen_fingerprints = {spec_fingerprint(spec) for spec in screen_specs}
    screen_overlap = len(screen_fingerprints & prior_fingerprints)
    if screen_overlap:
        raise ValueError('P2.10 screen overlaps P2.8 training/screen conditions')

    screen_data, screen_rows, screen_script_rows, screen_manifest = prepare_dataset(
        screen_path, 'confirmatory_screen', SCREEN_SEED, library,
        perturb_config, PERTURBATIONS, output)
    screen_results = evaluate_split(
        screen_data, screen_rows, models, fixed_orders, prereg, 'screen')
    screen_gate = all(row['pass'] for row in screen_results)
    audit = {
        'prior_condition_counts': prior_counts,
        'screen_condition_set_version': screen_header['condition_set_version'],
        'screen_conditions': len(screen_fingerprints),
        'screen_overlap_with_p28_train_screen': screen_overlap,
        'fresh_development_accessed_after_screen_gate': False,
        'heldout_checked_or_read': False,
    }

    development_results = None
    development_manifest = None
    if screen_gate:
        development_path = ROOT / prereg['conditional_fresh_development']['path']
        if not development_path.exists():
            export_conditions(
                development_path, DEVELOPMENT_SEED, COUNT_PER_BRANCH, BRANCHES,
                sampler_version=2, role='reference', purpose='development')
        development_specs, development_header = load_conditions(development_path)
        development_fingerprints = {spec_fingerprint(spec) for spec in development_specs}
        development_overlap = len(
            development_fingerprints & (prior_fingerprints | screen_fingerprints))
        if development_overlap:
            raise ValueError('P2.10 fresh development overlaps prior conditions')
        audit.update(
            fresh_development_accessed_after_screen_gate=True,
            fresh_development_condition_set_version=development_header['condition_set_version'],
            fresh_development_conditions=len(development_fingerprints),
            fresh_development_overlap=development_overlap,
            fresh_development_sha256=file_hash(development_path),
        )
        development_data, development_rows, development_script_rows, development_manifest = prepare_dataset(
            development_path, 'fresh_development', DEVELOPMENT_SEED, library,
            perturb_config, PERTURBATIONS, output)
        development_results = evaluate_split(
            development_data, development_rows, models, fixed_orders, prereg,
            'development')
    else:
        development_rows, development_script_rows = [], []

    development_gate = (
        development_results is not None and all(row['pass'] for row in development_results))
    final_eligible = screen_gate and development_gate
    summary = {
        'protocol': prereg['protocol'],
        'preregistration_sha256': file_hash(prereg_path),
        'frozen_input_hashes_verified': True,
        'screen_manifest': compact_manifest(screen_manifest),
        'screen_results': screen_results,
        'both_branch_screen_pass': screen_gate,
        'fresh_development_evaluated': development_results is not None,
        'development_manifest': (
            compact_manifest(development_manifest) if development_manifest else None),
        'development_results': development_results,
        'both_branch_development_pass': development_gate,
        'eligible_for_later_one_time_heldout': final_eligible,
        'episode_evaluations': {
            'screen': len(screen_rows) + len(screen_script_rows),
            'fresh_development': len(development_rows) + len(development_script_rows),
            'total': (len(screen_rows) + len(screen_script_rows)
                      + len(development_rows) + len(development_script_rows)),
        },
        'heldout_read': False,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'condition_fingerprint_audit.json').write_text(
        json.dumps(audit, indent=2), encoding='utf-8')
    write_report(output / 'REPORT.md', screen_results, development_results, final_eligible)
    raw_artifacts = [screen_path]
    raw_artifacts += [
        output / f'{split}_{suffix}'
        for split in ('confirmatory_screen', 'fresh_development')
        for suffix in ('dataset.npz', 'eligible.json', 'candidate_attempts.jsonl',
                       'script_attempts.jsonl')
        if (output / f'{split}_{suffix}').exists()
    ]
    (output / 'artifact_manifest.json').write_text(json.dumps({
        'raw_artifacts': [{
            'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': file_hash(path),
        } for path in raw_artifacts],
        'large_attempt_jsonl_committed': False,
    }, indent=2), encoding='utf-8')
    (output / 'completed.json').write_text(json.dumps({
        'protocol': prereg['protocol'], 'both_branch_screen_pass': screen_gate,
        'fresh_development_evaluated': development_results is not None,
        'both_branch_development_pass': development_gate,
        'eligible_for_later_one_time_heldout': final_eligible,
        'heldout_read': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({
        'screen': [{
            'branch': row['branch'], 'router2': row['router2']['rate'],
            'fixed2': row['fixed2']['rate'],
            'permutation_delta': row['router2_minus_permutation_mean'],
            'permutation_p': row['permutation_p_one_sided'], 'pass': row['pass'],
        } for row in screen_results],
        'both_branch_screen_pass': screen_gate,
        'development_evaluated': development_results is not None,
        'development': ([{
            'branch': row['branch'], 'router2': row['router2']['rate'],
            'fixed2': row['fixed2']['rate'],
            'permutation_delta': row['router2_minus_permutation_mean'],
            'permutation_p': row['permutation_p_one_sided'], 'pass': row['pass'],
        } for row in development_results] if development_results else None),
        'eligible_for_later_one_time_heldout': final_eligible,
        'heldout_read': False,
    }, indent=2))


if __name__ == '__main__':
    main()
