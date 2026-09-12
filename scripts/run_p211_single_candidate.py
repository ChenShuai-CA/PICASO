"""Run P2.11 frozen single-candidate conditional-router confirmation."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_lab.pulse_router import (  # noqa: E402
    history_features, paired_bootstrap, portfolio_scores, predict_router, ranking)
from scenario_lab.routing_diagnostics import permutation_alignment  # noqa: E402
from run_p28_conditional_router import (  # noqa: E402
    base_seed, load_rows, nominal_script_safe, parameters_policy, write_rows)


BRANCHES = ('single', 'dual')
SCREEN_SEED = 76000
COUNT_PER_BRANCH = 360
PERTURBATIONS = 5
FIXED_HEAD = {'single': 1, 'dual': 1}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_router(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compact_manifest(manifest):
    return {key: value for key, value in manifest.items() if key != 'eligible'}


def prepare_screen(condition_path, library, models, perturb_config, output):
    from scenario_lab.evaluate import ScriptPolicy, run_episode
    from scenario_lab.pulse_set_data import observable_history
    from scenario_lab.sampling import load_conditions
    from scenario_lab.train import perturb_spec

    dataset_path = output / 'screen_dataset.npz'
    candidate_path = output / 'candidate_attempts.jsonl'
    script_path = output / 'script_attempts.jsonl'
    manifest_path = output / 'screen_eligible.json'
    if all(path.exists() for path in (
            dataset_path, candidate_path, script_path, manifest_path)):
        with np.load(dataset_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        return (data, load_rows(candidate_path), load_rows(script_path),
                read_json(manifest_path))

    specs, header = load_conditions(condition_path)
    histories, eligible, nominal_rows = [], [], []
    for source_index, spec in enumerate(specs):
        seed = base_seed(SCREEN_SEED, spec.branch, source_index)
        safe, nominal = nominal_script_safe(spec, seed)
        nominal_rows.append(nominal)
        if not safe:
            continue
        histories.append(observable_history(spec, seed))
        eligible.append({
            'uid': f"{header['condition_set_version']}::{spec.branch}::{spec.scenario_id}",
            'branch': spec.branch, 'scenario_id': spec.scenario_id,
            'source_index': source_index, 'spec': spec.to_dict(),
        })
    arrays = {key: np.stack([history[key] for history in histories])
              for key in ('tokens', 'token_mask', 'actor_mask')}
    arrays.update(
        branch=np.asarray([row['branch'] for row in eligible]),
        uid=np.asarray([row['uid'] for row in eligible]),
        source_index=np.asarray([row['source_index'] for row in eligible]),
    )
    features = history_features(arrays)
    order = np.zeros((len(eligible), 4), dtype=np.int8)
    for branch in BRANCHES:
        select = arrays['branch'] == branch
        order[select] = ranking(predict_router(models[branch], features[select]))
    selected_head = order[:, 0].astype(np.int8)

    n = len(eligible)
    outcomes = np.zeros((n, 4, PERTURBATIONS), dtype=bool)
    valid = np.zeros_like(outcomes)
    steps = np.zeros((n, 4, PERTURBATIONS), dtype=np.int16)
    script_outcomes = np.zeros((n, PERTURBATIONS), dtype=bool)
    candidate_rows, script_rows = [], []
    for condition_index, source in enumerate(eligible):
        spec = type(specs[0])(**source['spec'])
        active = 1 if spec.branch == 'single' else 2
        for perturbation in range(PERTURBATIONS):
            seed = base_seed(
                SCREEN_SEED, spec.branch, source['source_index'], perturbation + 1)
            perturbed = perturb_spec(
                spec, np.random.default_rng(seed), calibrated=perturb_config)
            perturbed.role_action_mode = 'lane_locked'
            script = run_episode(ScriptPolicy(), perturbed, seed)
            script.update(
                uid=source['uid'], condition_index=condition_index,
                source_index=source['source_index'], perturbation=perturbation,
                method='script')
            script_rows.append(script)
            script_outcomes[condition_index, perturbation] = (
                script['valid'] and script['dangerous'])

            for head in range(4):
                candidate = run_episode(
                    parameters_policy(library[spec.branch][head], active),
                    perturbed, seed)
                candidate.update(
                    uid=source['uid'], condition_index=condition_index,
                    source_index=source['source_index'], perturbation=perturbation,
                    method='frozen_prototype_matrix', head=head,
                    selected_by_router1=(head == int(selected_head[condition_index])),
                    fixed1=(head == FIXED_HEAD[spec.branch]),
                    parameters=library[spec.branch][head].tolist())
                candidate_rows.append(candidate)
                outcomes[condition_index, head, perturbation] = (
                    candidate['valid'] and candidate['dangerous'])
                valid[condition_index, head, perturbation] = candidate['valid']
                steps[condition_index, head, perturbation] = candidate['decision_steps']

    arrays.update(
        router_order=order, selected_head=selected_head,
        outcomes=outcomes, valid=valid, steps=steps, script_outcomes=script_outcomes,
    )
    np.savez_compressed(dataset_path, **arrays)
    write_rows(candidate_path, candidate_rows)
    write_rows(script_path, script_rows)
    manifest = {
        'condition_set_version': header['condition_set_version'],
        'generated_per_branch': COUNT_PER_BRANCH,
        'eligible_total': len(eligible),
        'eligible_by_branch': dict(Counter(row['branch'] for row in eligible)),
        'eligibility': 'complete valid nominal script-safe only; outcomes unused',
        'perturbations': PERTURBATIONS,
        'perturbation_source': 'numerical sensitivity domain',
        'nominal_episode_executions': len(nominal_rows),
        'perturbed_script_episode_executions': len(script_rows),
        'candidate_matrix_episode_executions': len(candidate_rows),
        'reported_method_budget': 1,
        'eligible': eligible,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return arrays, candidate_rows, script_rows, manifest


def invalid_reason_counts(rows):
    return dict(Counter(
        reason for row in rows for reason in row.get('invalid_reasons', [])))


def evaluate(data, candidate_rows, prereg):
    results = []
    rounds = prereg['inference']['condition_bootstrap_rounds']
    permutations = prereg['inference']['permutation_rounds']
    gate = prereg['gate_each_branch']
    for branch_index, branch in enumerate(BRANCHES):
        select = data['branch'] == branch
        local_outcomes = data['outcomes'][select]
        local_order = data['router_order'][select]
        router_condition = portfolio_scores(local_outcomes, local_order, 1)
        fixed_condition = local_outcomes[:, FIXED_HEAD[branch], :].mean(axis=1)
        script_condition = data['script_outcomes'][select].mean(axis=1)
        seed = prereg['inference']['seed'] + branch_index * 1000
        delta_fixed, ci_fixed = paired_bootstrap(
            router_condition, fixed_condition, rounds=rounds, seed=seed)
        delta_script, ci_script = paired_bootstrap(
            router_condition, script_condition, rounds=rounds, seed=seed + 1)
        alignment = permutation_alignment(
            local_outcomes, local_order, budget=1,
            rounds=permutations, seed=seed + 2)
        observed = alignment['observed_rate']
        permutation_mean = alignment['permutation_mean']
        permutation_delta = alignment['delta_vs_permutation_mean']
        permutation_p = alignment['permutation_p_one_sided']

        global_selected = {
            int(index): int(data['selected_head'][index])
            for index in np.flatnonzero(select)
        }
        branch_router_rows = [
            row for row in candidate_rows
            if row['condition_index'] in global_selected
            and row['head'] == global_selected[row['condition_index']]
        ]
        branch_fixed_rows = [
            row for row in candidate_rows
            if row['condition_index'] in global_selected
            and row['head'] == FIXED_HEAD[branch]
        ]
        top_counts = Counter(map(int, data['selected_head'][select]))
        modal_share = max(top_counts.values()) / int(select.sum())
        role_invalid = sum(
            reason in ('pedestrian_role', 'occluder_role')
            for row in branch_router_rows for reason in row['invalid_reasons'])
        gates = {
            'minimum_conditions': int(select.sum()) >= gate['minimum_eligible_conditions'],
            'router1_vs_fixed1': ci_fixed[0] > 0,
            'router1_vs_script': ci_script[0] > 0,
            'permutation_alignment': (
                permutation_delta >= gate['router1_minus_permutation_mean_at_least']
                and permutation_p <= gate['permutation_p_one_sided_at_most']),
            'candidate_validity': (
                float(np.take_along_axis(
                    data['valid'][select], local_order[:, :1, None], axis=1).mean())
                >= gate['router_candidate_valid_rate_at_least']),
            'role_validity': role_invalid == gate['router_role_invalid_equals'],
            'head_diversity': (
                len(top_counts) >= gate['distinct_router_top1_heads_at_least']
                and modal_share <= gate['modal_router_top1_share_at_most']),
        }
        results.append({
            'branch': branch, 'conditions': int(select.sum()),
            'router1_rate': observed,
            'fixed1_rate': float(fixed_condition.mean()),
            'script_rate': float(script_condition.mean()),
            'router1_minus_fixed1': delta_fixed,
            'router1_minus_fixed1_ci95': ci_fixed,
            'router1_minus_script': delta_script,
            'router1_minus_script_ci95': ci_script,
            'router1_minus_permutation_mean': permutation_delta,
            'permutation_mean': permutation_mean,
            'permutation_ci95': alignment['permutation_ci95'],
            'permutation_p_one_sided': permutation_p,
            'candidate_valid_rate': float(np.take_along_axis(
                data['valid'][select], local_order[:, :1, None], axis=1).mean()),
            'role_invalid': role_invalid,
            'router_invalid_reasons': invalid_reason_counts(branch_router_rows),
            'fixed_invalid_reasons': invalid_reason_counts(branch_fixed_rows),
            'mean_router_decision_steps': float(np.take_along_axis(
                data['steps'][select], local_order[:, :1, None], axis=1).mean()),
            'total_router_decision_steps': int(np.take_along_axis(
                data['steps'][select], local_order[:, :1, None], axis=1).sum()),
            'top1_head_counts': {str(key): value for key, value in sorted(top_counts.items())},
            'modal_top1_share': modal_share,
            'gates': gates, 'pass': all(gates.values()),
        })
    return results


def fmt(value, ci):
    return f'{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'


def write_report(path, results, eligible_for_heldout):
    lines = [
        '# P2.11 frozen single-candidate router confirmation', '',
        'The frozen ridge router observes the first five actor-visible frames, selects one '
        'frozen prototype, and executes exactly one candidate episode. This is a single-candidate '
        'conditional selection experiment, not a continuously reactive actor. The perturbations '
        'form a numerical sensitivity domain and are not an AEB-calibrated distribution.', '',
        '| branch | N | router-1 | fixed-1 | script | router1-fixed1 [95% CI] | '
        'router1-script [95% CI] | perm delta | perm p | valid | pass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        lines.append(
            f"| {row['branch']} | {row['conditions']} | {row['router1_rate']:.3f} | "
            f"{row['fixed1_rate']:.3f} | {row['script_rate']:.3f} | "
            f"{fmt(row['router1_minus_fixed1'], row['router1_minus_fixed1_ci95'])} | "
            f"{fmt(row['router1_minus_script'], row['router1_minus_script_ci95'])} | "
            f"{row['router1_minus_permutation_mean']:.3f} | "
            f"{row['permutation_p_one_sided']:.4f} | "
            f"{row['candidate_valid_rate']:.3f} | {row['pass']} |")
    lines += ['', f'Both-branch gate: **{eligible_for_heldout}**.', '']
    if eligible_for_heldout:
        lines += [
            'The one-candidate router becomes the preferred final method and is eligible for a '
            'later user-authorized one-time heldout evaluation. Heldout was not read in P2.11.', ''
        ]
    else:
        lines += [
            'The one-candidate router does not replace the already-qualified P2.10 two-candidate '
            'method. This screen must not be used to tune another P2.11 attempt.', ''
        ]
    lines += [
        'The primary comparison is equal-budget router-1 versus fixed-1. The script comparison '
        'also uses one episode. Results remain limited to dangerous-and-valid scenario coverage.', ''
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/20260912_p211_single_candidate')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    output = (ROOT / args.output).resolve()
    if 'heldout' in str(output).casefold():
        raise ValueError('P2.11 must not read or write heldout paths')
    prereg_path = output / 'preregistration.json'
    if not prereg_path.exists():
        raise FileNotFoundError('freeze P2.11 preregistration before generating conditions')
    prereg = read_json(prereg_path)
    if not prereg.get('created_before_p211_conditions_and_outcomes'):
        raise ValueError('P2.11 preregistration is not marked pre-outcome')

    frozen = prereg['frozen_inputs']
    for key in ('prototype_library', 'single_ridge', 'dual_ridge', 'perturbation_config'):
        source = ROOT / frozen[key]['path']
        if file_hash(source) != frozen[key]['sha256']:
            raise ValueError(f'frozen input hash mismatch: {key}')
    p210 = ROOT / 'runs/20260912_p210_confirmatory_router'
    for filename, key in (
            ('preregistration.json', 'p210_preregistration_sha256'),
            ('summary.json', 'p210_summary_sha256'),
            ('completed.json', 'p210_completed_sha256'),
            ('condition_fingerprint_audit.json', 'p210_fingerprint_audit_sha256')):
        if file_hash(p210 / filename) != frozen[key]:
            raise ValueError(f'P2.10 provenance hash mismatch: {filename}')

    from scenario_lab.sampling import export_conditions, load_conditions
    from scenario_lab.teacher import spec_fingerprint
    from scenario_lab.train import load_perturb_config

    library_file = read_json(ROOT / frozen['prototype_library']['path'])
    library = {branch: np.asarray(library_file['parameters'][branch], dtype=np.float32)
               for branch in BRANCHES}
    models = {branch: load_router(ROOT / frozen[f'{branch}_ridge']['path'])
              for branch in BRANCHES}
    perturb_config = load_perturb_config(ROOT / frozen['perturbation_config']['path'])

    condition_dir = output / 'conditions'
    condition_dir.mkdir(parents=True, exist_ok=True)
    screen_path = condition_dir / 'single_candidate_screen_seed76000.json'
    if not screen_path.exists():
        export_conditions(
            screen_path, SCREEN_SEED, COUNT_PER_BRANCH, BRANCHES,
            sampler_version=2, role='reference', purpose='training')

    prior_paths = [
        ROOT / 'runs/20260912_p28_conditional_router/conditions/training_seed72000.json',
        ROOT / 'runs/20260912_p28_conditional_router/conditions/screen_seed73000.json',
        ROOT / 'runs/20260912_p28_conditional_router/conditions/development_seed74000.json',
        ROOT / 'runs/20260912_p210_confirmatory_router/conditions/confirmatory_screen_seed75000.json',
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
    overlap = len(screen_fingerprints & prior_fingerprints)
    if overlap:
        raise ValueError('P2.11 screen overlaps a prior condition set')

    data, candidate_rows, script_rows, manifest = prepare_screen(
        screen_path, library, models, perturb_config, output)
    results = evaluate(data, candidate_rows, prereg)
    passed = all(row['pass'] for row in results)
    summary = {
        'protocol': prereg['protocol'],
        'preregistration_sha256': file_hash(prereg_path),
        'paper_scope': prereg['paper_scope'],
        'frozen_input_hashes_verified': True,
        'screen_manifest': compact_manifest(manifest),
        'results': results, 'both_branch_pass': passed,
        'preferred_final_budget': 1 if passed else 2,
        'eligible_for_later_one_time_heldout': True,
        'heldout_method': 'P2.11 router-1' if passed else 'P2.10 router-2',
        'heldout_read': False,
    }
    audit = {
        'prior_condition_counts': prior_counts,
        'screen_condition_set_version': screen_header['condition_set_version'],
        'screen_conditions': len(screen_fingerprints),
        'overlap_with_all_prior_nonheldout_conditions': overlap,
        'heldout_checked_or_read': False,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'condition_fingerprint_audit.json').write_text(
        json.dumps(audit, indent=2), encoding='utf-8')
    write_report(output / 'REPORT.md', results, passed)
    raw = [screen_path, output / 'screen_dataset.npz', output / 'screen_eligible.json',
           output / 'candidate_attempts.jsonl', output / 'script_attempts.jsonl']
    (output / 'artifact_manifest.json').write_text(json.dumps({
        'raw_artifacts': [{
            'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': file_hash(path),
        } for path in raw],
        'large_attempt_jsonl_committed': False,
    }, indent=2), encoding='utf-8')
    (output / 'completed.json').write_text(json.dumps({
        'protocol': prereg['protocol'], 'both_branch_pass': passed,
        'preferred_final_budget': 1 if passed else 2,
        'heldout_method': 'P2.11 router-1' if passed else 'P2.10 router-2',
        'heldout_read': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({
        'results': [{
            'branch': row['branch'], 'conditions': row['conditions'],
            'router1': row['router1_rate'], 'fixed1': row['fixed1_rate'],
            'script': row['script_rate'],
            'permutation_delta': row['router1_minus_permutation_mean'],
            'permutation_p': row['permutation_p_one_sided'], 'pass': row['pass'],
        } for row in results],
        'both_branch_pass': passed,
        'heldout_method': 'P2.11 router-1' if passed else 'P2.10 router-2',
        'heldout_read': False,
    }, indent=2))


if __name__ == '__main__':
    main()
