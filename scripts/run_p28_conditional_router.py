"""Run formal P2.8 conditional routing over the frozen P2.7 prototype library."""
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
import torch


BRANCHES = ('single', 'dual')
TRAIN_SEED, SCREEN_SEED, DEV_SEED = 72000, 73000, 74000
COUNT_PER_BRANCH = 160
TRAIN_PERTURBATIONS, EVAL_PERTURBATIONS = 3, 5
BOOTSTRAP, BOOTSTRAP_SEED = 5000, 2028
MIN_SCREEN = 60


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def compact_manifest(manifest):
    """Keep summary metadata small; full eligible specs remain in their own artifact."""
    return {key: value for key, value in manifest.items() if key != 'eligible'}


def select_p27_library(p27):
    """Use P2.7 validation loss only to select one already-trained model."""
    from scenario_lab.pulse_set import load_pulse_set
    values = []
    for seed in (7, 17, 27, 37, 47):
        bundle = torch.load(p27 / f'pulse_set_s{seed}/model.pt', map_location='cpu',
                            weights_only=True)
        selected = bundle['config']['selected_epoch']
        history = read_json(p27 / f'pulse_set_s{seed}/training.json')
        metric = next(row['val_branch_balanced_set_loss'] for row in history
                      if row['epoch'] == selected)
        values.append((float(metric), seed))
    _, selected_seed = min(values)
    with np.load(p27 / 'train_corpus/pulse_set.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    model, _ = load_pulse_set(p27 / f'pulse_set_s{selected_seed}/model.pt', 'cpu')
    with torch.no_grad():
        prediction = model(torch.as_tensor(data['tokens']),
                           torch.as_tensor(data['token_mask']),
                           torch.as_tensor(data['actor_mask'])).numpy()
    library = {}
    drift = {}
    for branch in BRANCHES:
        subset = prediction[data['branch'] == branch]
        library[branch] = np.median(subset, axis=0).astype(np.float32)
        centered = subset - library[branch]
        drift[branch] = float(np.sqrt(np.mean(centered ** 2)))
    return selected_seed, values, library, drift


def parameters_policy(parameters, active):
    from scenario_lab.evaluate import ParamPolicy
    return ParamPolicy(np.asarray(parameters)[:active], active_actors=active)


def base_seed(split_seed, branch, source_index, perturbation=0):
    return int(split_seed + (100000 if branch == 'dual' else 0)
               + source_index * 100 + perturbation)


def nominal_script_safe(spec, seed):
    from scenario_lab.evaluate import ScriptPolicy, run_episode
    candidate = deepcopy(spec)
    candidate.role_action_mode = 'lane_locked'
    row = run_episode(ScriptPolicy(), candidate, seed)
    return bool(row['terminated'] and row['valid'] and not row['dangerous']), row


def prepare_dataset(condition_path, split_name, split_seed, library, perturb_config,
                    perturbations, output):
    """Filter only on nominal script safety, then evaluate every frozen head."""
    from scenario_lab.evaluate import ScriptPolicy, run_episode
    from scenario_lab.pulse_set_data import observable_history
    from scenario_lab.sampling import load_conditions
    from scenario_lab.train import perturb_spec

    dataset_path = output / f'{split_name}_dataset.npz'
    candidate_path = output / f'{split_name}_candidate_attempts.jsonl'
    script_path = output / f'{split_name}_script_attempts.jsonl'
    manifest_path = output / f'{split_name}_eligible.json'
    if all(path.exists() for path in (dataset_path, candidate_path, script_path, manifest_path)):
        with np.load(dataset_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        return data, load_rows(candidate_path), load_rows(script_path), read_json(manifest_path)

    specs, header = load_conditions(condition_path)
    histories, eligible, nominal_rows = [], [], []
    for source_index, spec in enumerate(specs):
        seed = base_seed(split_seed, spec.branch, source_index)
        safe, nominal = nominal_script_safe(spec, seed)
        nominal.update(source_index=source_index, split=split_name,
                       nominal_script_safe=safe)
        nominal_rows.append(nominal)
        if not safe:
            continue
        history = observable_history(spec, seed)
        histories.append(history)
        eligible.append({
            'uid': f"{header['condition_set_version']}::{spec.branch}::{spec.scenario_id}",
            'branch': spec.branch, 'scenario_id': spec.scenario_id,
            'source_index': source_index, 'spec': spec.to_dict(),
        })

    candidate_rows, script_rows = [], []
    n, heads = len(eligible), 4
    outcomes = np.zeros((n, heads, perturbations), dtype=bool)
    valid = np.zeros_like(outcomes)
    steps = np.zeros((n, heads, perturbations), dtype=np.int16)
    script_outcomes = np.zeros((n, perturbations), dtype=bool)
    for condition_index, source in enumerate(eligible):
        spec = type(specs[0])(**source['spec'])
        active = 1 if spec.branch == 'single' else 2
        for perturbation in range(perturbations):
            seed = base_seed(split_seed, spec.branch, source['source_index'], perturbation + 1)
            perturbed = perturb_spec(spec, np.random.default_rng(seed), calibrated=perturb_config)
            perturbed.role_action_mode = 'lane_locked'
            script = run_episode(ScriptPolicy(), perturbed, seed)
            script.update(uid=source['uid'], condition_index=condition_index,
                          source_index=source['source_index'], perturbation=perturbation,
                          method='script')
            script_rows.append(script)
            script_outcomes[condition_index, perturbation] = script['valid'] and script['dangerous']
            for head in range(heads):
                row = run_episode(parameters_policy(library[spec.branch][head], active),
                                  perturbed, seed)
                row.update(uid=source['uid'], condition_index=condition_index,
                           source_index=source['source_index'], perturbation=perturbation,
                           head=head, method='frozen_p27_prototype',
                           parameters=library[spec.branch][head].tolist())
                candidate_rows.append(row)
                outcomes[condition_index, head, perturbation] = row['valid'] and row['dangerous']
                valid[condition_index, head, perturbation] = row['valid']
                steps[condition_index, head, perturbation] = row['decision_steps']

    arrays = {key: np.stack([history[key] for history in histories])
              for key in ('tokens', 'token_mask', 'actor_mask')}
    arrays.update(
        branch=np.asarray([row['branch'] for row in eligible]),
        uid=np.asarray([row['uid'] for row in eligible]),
        source_index=np.asarray([row['source_index'] for row in eligible]),
        split=np.asarray(['val' if row['source_index'] % 5 == 0 else 'train'
                          for row in eligible]),
        outcomes=outcomes, valid=valid, steps=steps,
        script_outcomes=script_outcomes,
    )
    np.savez_compressed(dataset_path, **arrays)
    write_rows(candidate_path, candidate_rows)
    write_rows(script_path, script_rows)
    manifest = {
        'condition_set_version': header['condition_set_version'],
        'split': split_name, 'generated_per_branch': COUNT_PER_BRANCH,
        'eligible_total': len(eligible),
        'eligible_by_branch': dict(Counter(row['branch'] for row in eligible)),
        'eligibility': 'complete valid nominal script-safe only; candidate outcomes unused',
        'perturbations': perturbations,
        'perturbation_source': f"{perturb_config['version']}_partial",
        'eligible': eligible, 'nominal_counts': {
            'total': len(nominal_rows),
            'safe': sum(row['nominal_script_safe'] for row in nominal_rows),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return arrays, candidate_rows, script_rows, manifest


def fit_models(train_data):
    from scenario_lab.pulse_router import fit_router, history_features
    features = history_features(train_data)
    models, reports = {}, {}
    for branch in BRANCHES:
        select = train_data['branch'] == branch
        y = train_data['outcomes'][select].mean(axis=2)
        model = fit_router(features[select], y, train_data['split'][select])
        models[branch] = model
        reports[branch] = {
            'alpha': model['alpha'], 'validation_mse': model['validation_mse'],
            'validation_mse_by_alpha': model['validation_mse_by_alpha'],
            'train_conditions': int(np.sum(train_data['split'][select] == 'train')),
            'val_conditions': int(np.sum(train_data['split'][select] == 'val')),
            'head_train_rates': y[train_data['split'][select] == 'train'].mean(0).tolist(),
        }
    return models, reports


def orders_for(data, models, train_data):
    from scenario_lab.pulse_router import history_features, predict_router, ranking
    features, train_features = history_features(data), history_features(train_data)
    router = np.zeros((len(features), 4), dtype=int)
    fixed = np.zeros_like(router)
    shuffled = np.zeros_like(router)
    predictions = np.zeros((len(features), 4), dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fixed_by_branch = {}
    for branch in BRANCHES:
        select = data['branch'] == branch
        train_select = train_data['branch'] == branch
        score = predict_router(models[branch], features[select])
        predictions[select] = score
        router[select] = ranking(score)
        train_rows = train_data['split'][train_select] == 'train'
        mean = train_data['outcomes'][train_select][train_rows].mean(axis=(0, 2))
        fixed_order = np.argsort(-mean, kind='stable')
        fixed[select] = fixed_order
        fixed_by_branch[branch] = fixed_order.tolist()
        permuted = score[rng.permutation(len(score))]
        shuffled[select] = ranking(permuted)
    return router, fixed, shuffled, predictions, fixed_by_branch


def method_metrics(data, candidate_rows, order, budget):
    from scenario_lab.pulse_router import portfolio_scores
    scores = portfolio_scores(data['outcomes'], order, budget)
    selected_valid, selected_steps, role_invalid = [], [], 0
    row_map = {(row['condition_index'], row['head'], row['perturbation']): row
               for row in candidate_rows}
    costs = []
    for condition in range(len(order)):
        for perturbation in range(data['outcomes'].shape[2]):
            cost = 0
            for head in order[condition, :budget]:
                row = row_map[(condition, int(head), perturbation)]
                selected_valid.append(row['valid'])
                selected_steps.append(row['decision_steps'])
                role_invalid += sum(reason in ('pedestrian_role', 'occluder_role')
                                    for reason in row['invalid_reasons'])
                cost += row['decision_steps']
                if row['valid'] and row['dangerous']:
                    break
            costs.append(cost)
    return scores, {
        'rate': float(scores.mean()), 'candidate_valid_rate': float(np.mean(selected_valid)),
        'role_invalid': int(role_invalid),
        'mean_steps_until_success_or_exhaustion': float(np.mean(costs)),
        'total_selected_attempt_steps': int(np.sum(selected_steps)),
    }


def summarize(data, candidate_rows, router, fixed, shuffled, predictions, fixed_by_branch):
    from scenario_lab.pulse_router import paired_bootstrap
    results = []
    for branch in BRANCHES:
        select = data['branch'] == branch
        local_rows = []
        old_to_new = {}
        for new, old in enumerate(np.flatnonzero(select)):
            old_to_new[int(old)] = new
        for row in candidate_rows:
            if row['branch'] == branch:
                local_rows.append({**row, 'condition_index': old_to_new[row['condition_index']]})
        local = {key: value[select] for key, value in data.items()
                 if isinstance(value, np.ndarray) and len(value) == len(data['branch'])}
        metrics, scores = {}, {}
        for name, order in (('router', router[select]), ('fixed', fixed[select]),
                            ('shuffled', shuffled[select])):
            for budget in (1, 2):
                values, report = method_metrics(local, local_rows, order, budget)
                scores[f'{name}{budget}'], metrics[f'{name}{budget}'] = values, report
        script = local['script_outcomes'].mean(1)
        any4 = local['outcomes'].any(axis=1).mean(axis=1)
        delta_fixed, ci_fixed = paired_bootstrap(scores['router2'], scores['fixed2'])
        delta_shuffled, ci_shuffled = paired_bootstrap(scores['router2'], scores['shuffled2'])
        delta_script, ci_script = paired_bootstrap(scores['router2'], script)
        top2 = [tuple(map(int, row[:2])) for row in router[select]]
        counts = Counter(top2)
        modal_share = max(counts.values()) / len(top2)
        passed = (
            int(select.sum()) >= MIN_SCREEN
            and ci_fixed[0] > 0 and ci_shuffled[0] > 0 and ci_script[0] > 0
            and metrics['router2']['candidate_valid_rate'] >= .8
            and metrics['router2']['role_invalid'] == 0
            and len(counts) >= 2 and modal_share <= .9)
        results.append({
            'branch': branch, 'conditions': int(select.sum()),
            'methods': metrics, 'script_rate': float(script.mean()),
            'all_four_oracle_rate': float(any4.mean()),
            'router2_delta_vs_fixed2': delta_fixed,
            'router2_delta_vs_fixed2_ci95': ci_fixed,
            'router2_delta_vs_shuffled2': delta_shuffled,
            'router2_delta_vs_shuffled2_ci95': ci_shuffled,
            'router2_delta_vs_script': delta_script,
            'router2_delta_vs_script_ci95': ci_script,
            'fixed_order': fixed_by_branch[branch],
            'distinct_router_top2_orders': len(counts),
            'modal_router_top2_share': modal_share,
            'prediction_std_by_head': predictions[select].std(0).tolist(),
            'mechanism_pass': passed,
        })
    return results


def write_report(output, prereg, training, results, dev_results=None):
    lines = [
        '# P2.8 conditional prototype routing', '',
        'The P2.7 four-prototype library is frozen. A scenario-level ridge router uses only '
        'the first five actor-visible observation frames to rank those prototypes. All outcome '
        'labels and evaluations use `abd_supported_v1_partial`; heldout was not read.', '',
        f"Selected P2.7 seed: {prereg['frozen_library']['selected_seed']}. Training eligible: "
        f"single {training['eligible_by_branch'].get('single', 0)}, dual "
        f"{training['eligible_by_branch'].get('dual', 0)}.", '',
        '## Formal screen', '',
        '| branch | N | router-1 | fixed-1 | router-2 | fixed-2 | shuffled-2 | script | '
        'delta router2-fixed2 [95% CI] | delta router2-shuffled2 [95% CI] | pass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        m = row['methods']
        lines.append(
            f"| {row['branch']} | {row['conditions']} | {m['router1']['rate']:.3f} | "
            f"{m['fixed1']['rate']:.3f} | {m['router2']['rate']:.3f} | "
            f"{m['fixed2']['rate']:.3f} | {m['shuffled2']['rate']:.3f} | "
            f"{row['script_rate']:.3f} | {row['router2_delta_vs_fixed2']:.3f} "
            f"[{row['router2_delta_vs_fixed2_ci95'][0]:.3f}, "
            f"{row['router2_delta_vs_fixed2_ci95'][1]:.3f}] | "
            f"{row['router2_delta_vs_shuffled2']:.3f} "
            f"[{row['router2_delta_vs_shuffled2_ci95'][0]:.3f}, "
            f"{row['router2_delta_vs_shuffled2_ci95'][1]:.3f}] | "
            f"{row['mechanism_pass']} |")
    passed = all(row['mechanism_pass'] for row in results)
    lines += ['', f'Both-branch mechanism gate: **{passed}**.', '']
    if dev_results is None:
        lines += ['The gate did not pass both branches, so the preregistered fresh development '
                  'set was not evaluated.', '']
    else:
        lines += ['## Conditional fresh-development result', '']
        for row in dev_results:
            lines.append(
                f"- {row['branch']}: router-2 {row['methods']['router2']['rate']:.3f}, "
                f"fixed-2 {row['methods']['fixed2']['rate']:.3f}, delta "
                f"{row['router2_delta_vs_fixed2']:.3f} "
                f"[{row['router2_delta_vs_fixed2_ci95'][0]:.3f}, "
                f"{row['router2_delta_vs_fixed2_ci95'][1]:.3f}].")
        lines.append('')
    lines += [
        'The router is a centralized scenario-level candidate selector over actor-visible '
        'histories. It is not a decentralized closed-loop actor policy. Success rates estimate '
        'dangerous-and-valid coverage over five paired perturbation draws per condition.', ''
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/20260912_p28_conditional_router')
    parser.add_argument('--p27', default='runs/20260911_p27_multisolution')
    parser.add_argument('--perturb-config',
                        default='runs/20260912_abd_calibration/abd_supported_v1.json')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from scenario_lab.sampling import export_conditions, load_conditions
    from scenario_lab.teacher import spec_fingerprint
    from scenario_lab.train import load_perturb_config

    output = (project / args.output).resolve()
    p27 = (project / args.p27).resolve()
    config_path = (project / args.perturb_config).resolve()
    if 'heldout' in str(output).casefold() or 'heldout' in str(config_path).casefold():
        raise ValueError('P2.8 formal screen must not read or write heldout paths')
    output.mkdir(parents=True, exist_ok=True)
    perturb_config = load_perturb_config(config_path)
    if perturb_config['version'] != 'abd_supported_v1':
        raise ValueError('formal P2.8 requires reviewed abd_supported_v1')
    selected_seed, selection, library, drift = select_p27_library(p27)
    library_serialized = {branch: library[branch].tolist() for branch in BRANCHES}
    library_hash = hashlib.sha256(
        json.dumps(library_serialized, sort_keys=True).encode()).hexdigest()
    prereg = {
        'protocol': 'P2.8 conditional routing over frozen P2.7 prototypes',
        'created_before_p28_condition_outcomes': True,
        'condition_sets': {
            'training': {'seed': TRAIN_SEED, 'count_per_branch': COUNT_PER_BRANCH,
                         'purpose': 'training'},
            'screen': {'seed': SCREEN_SEED, 'count_per_branch': COUNT_PER_BRANCH,
                       'purpose': 'training'},
            'fresh_development': {'seed': DEV_SEED, 'count_per_branch': COUNT_PER_BRANCH,
                                  'purpose': 'development',
                                  'evaluate_only_after_both_branch_screen_pass': True},
        },
        'sampler_version': 2, 'role_action_mode': 'lane_locked',
        'eligibility': 'complete valid nominal script-safe; candidate outcomes never filter',
        'frozen_library': {
            'source': 'P2.7 model selected by P2.7 validation set loss only',
            'selected_seed': selected_seed,
            'selection_metrics': [{'seed': seed, 'val_set_loss': metric}
                                  for metric, seed in sorted(selection, key=lambda item: item[1])],
            'aggregation': 'per-head median prediction over P2.7 train corpus by branch',
            'library_sha256': library_hash, 'p27_model_sha256': file_hash(
                p27 / f'pulse_set_s{selected_seed}/model.pt'),
            'cross_condition_prediction_rms': drift,
        },
        'router': {
            'type': 'separate branch ridge outcome regressors',
            'input': 'flattened first 5 actor-visible frames plus visibility/presence masks',
            'information_scope': 'centralized scenario-level selector; no privileged truth-fill',
            'targets': 'per-head dangerous-and-valid fraction over 3 ABD-supported draws',
            'alphas': [0.01, 0.1, 1.0, 10.0, 100.0],
            'split': 'source_index modulo 5 equals zero is validation',
        },
        'evaluation': {
            'perturb_config': str(config_path.relative_to(project)),
            'perturb_config_sha256': file_hash(config_path),
            'train_perturbations': TRAIN_PERTURBATIONS,
            'screen_perturbations': EVAL_PERTURBATIONS,
            'budgets': [1, 2],
            'comparators': ['fixed training-best order', 'deterministically shuffled-input router',
                            'zero-action script', 'all-four oracle diagnostic'],
            'bootstrap_conditions': BOOTSTRAP, 'bootstrap_seed': BOOTSTRAP_SEED,
        },
        'screen_gate_each_branch': {
            'minimum_conditions': MIN_SCREEN,
            'router2_minus_fixed2_ci95_lower_strictly_above': 0,
            'router2_minus_shuffled2_ci95_lower_strictly_above': 0,
            'router2_minus_script_ci95_lower_strictly_above': 0,
            'selected_candidate_valid_rate_at_least': .8,
            'role_invalid_equals': 0,
            'distinct_top2_orders_at_least': 2,
            'modal_top2_share_at_most': .9,
        },
        'heldout_read': False,
    }
    prereg_path = output / 'preregistration.json'
    if prereg_path.exists() and read_json(prereg_path) != prereg:
        raise ValueError('existing P2.8 preregistration differs; use a fresh output')
    prereg_path.write_text(json.dumps(prereg, indent=2), encoding='utf-8')
    (output / 'frozen_library.json').write_text(json.dumps(
        {'sha256': library_hash, 'selected_seed': selected_seed,
         'parameters': library_serialized}, indent=2), encoding='utf-8')

    condition_dir = output / 'conditions'
    condition_dir.mkdir(exist_ok=True)
    train_path = condition_dir / 'training_seed72000.json'
    screen_path = condition_dir / 'screen_seed73000.json'
    dev_path = condition_dir / 'development_seed74000.json'
    if not train_path.exists():
        export_conditions(train_path, TRAIN_SEED, COUNT_PER_BRANCH, BRANCHES,
                          sampler_version=2, role='reference', purpose='training')
    if not screen_path.exists():
        export_conditions(screen_path, SCREEN_SEED, COUNT_PER_BRANCH, BRANCHES,
                          sampler_version=2, role='reference', purpose='training')
    if not dev_path.exists():
        export_conditions(dev_path, DEV_SEED, COUNT_PER_BRANCH, BRANCHES,
                          sampler_version=2, role='reference', purpose='development')

    train_data, train_rows, _, train_manifest = prepare_dataset(
        train_path, 'training', TRAIN_SEED, library, perturb_config,
        TRAIN_PERTURBATIONS, output)
    screen_data, screen_rows, _, screen_manifest = prepare_dataset(
        screen_path, 'screen', SCREEN_SEED, library, perturb_config,
        EVAL_PERTURBATIONS, output)
    train_fingerprints = {spec_fingerprint(spec) for spec in load_conditions(train_path)[0]}
    screen_fingerprints = {spec_fingerprint(spec) for spec in load_conditions(screen_path)[0]}
    train_screen_overlap = len(train_fingerprints & screen_fingerprints)
    if train_screen_overlap:
        raise ValueError('P2.8 train/screen condition overlap')
    fingerprint_audit = {
        'training_conditions': len(train_fingerprints),
        'screen_conditions': len(screen_fingerprints),
        'train_screen_overlap': train_screen_overlap,
        'development_checked_after_gate': False,
        'heldout_checked': False,
    }

    models, training_report = fit_models(train_data)
    for branch, model in models.items():
        np.savez_compressed(output / f'router_{branch}.npz',
                            mean_x=model['mean_x'], scale_x=model['scale_x'],
                            mean_y=model['mean_y'], weights=model['weights'],
                            alpha=model['alpha'])
    (output / 'training_report.json').write_text(
        json.dumps(training_report, indent=2), encoding='utf-8')
    router, fixed, shuffled, predictions, fixed_by_branch = orders_for(
        screen_data, models, train_data)
    results = summarize(screen_data, screen_rows, router, fixed, shuffled,
                        predictions, fixed_by_branch)
    gate = all(row['mechanism_pass'] for row in results)
    summary = {
        'protocol': prereg['protocol'], 'training': training_report,
        'training_manifest': compact_manifest(train_manifest),
        'screen_manifest': compact_manifest(screen_manifest),
        'screen_results': results, 'both_branch_screen_pass': gate,
        'fresh_development_evaluated': False, 'heldout_read': False,
    }
    dev_results = None
    if gate:
        dev_data, dev_rows, _, dev_manifest = prepare_dataset(
            dev_path, 'development', DEV_SEED, library, perturb_config,
            EVAL_PERTURBATIONS, output)
        dev_fingerprints = {spec_fingerprint(spec) for spec in load_conditions(dev_path)[0]}
        dev_overlap = len(dev_fingerprints & (train_fingerprints | screen_fingerprints))
        if dev_overlap:
            raise ValueError('P2.8 development condition overlap')
        fingerprint_audit.update(development_checked_after_gate=True,
                                 development_conditions=len(dev_fingerprints),
                                 development_overlap=dev_overlap)
        router_d, fixed_d, shuffled_d, predictions_d, fixed_d_by_branch = orders_for(
            dev_data, models, train_data)
        dev_results = summarize(dev_data, dev_rows, router_d, fixed_d, shuffled_d,
                                predictions_d, fixed_d_by_branch)
        summary.update(fresh_development_evaluated=True,
                       development_manifest=compact_manifest(dev_manifest),
                       development_results=dev_results)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'condition_fingerprint_audit.json').write_text(
        json.dumps(fingerprint_audit, indent=2), encoding='utf-8')
    write_report(output, prereg, train_manifest, results, dev_results)
    (output / 'completed.json').write_text(json.dumps({
        'protocol': prereg['protocol'], 'both_branch_screen_pass': gate,
        'fresh_development_evaluated': dev_results is not None,
        'heldout_read': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'screen_results': results, 'both_branch_pass': gate,
                      'development_evaluated': dev_results is not None,
                      'heldout_read': False}, indent=2))


if __name__ == '__main__':
    main()
