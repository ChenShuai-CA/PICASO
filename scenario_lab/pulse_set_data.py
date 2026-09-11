"""Build and evaluate multi-solution pulse supervision without heldout access."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from .env import DECISION_DT, ScenarioEnv
from .evaluate import ScriptPolicy, run_episode
from .sampling import load_conditions
from .teacher import spec_fingerprint
from .pulse_set import SCHEMA


HISTORY_STEPS = 5
MAX_CANDIDATES = 8


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def action_representation(parameters, branch, steps=80):
    actors = 1 if branch == 'single' else 2
    rows = np.asarray(parameters, dtype=np.float32).reshape(actors, 3)
    result = np.zeros((steps, actors), dtype=np.float32)
    for step in range(steps):
        t = step * DECISION_DT
        for actor, (amplitude, start, duration) in enumerate(rows):
            onset = (float(start) + 1.) * 2.
            if onset <= t < onset + (float(duration) + 1.) * 1.5:
                result[step, actor] = amplitude
    return result.ravel()


def select_diverse_candidates(rows, branch, maximum=MAX_CANDIDATES):
    """Top score half followed by deterministic farthest-first action selection."""
    ordered = sorted(rows, key=lambda row: (-float(row['score']),
                                             int(row['_search_seed']),
                                             int(row.get('cem_iteration', 0)),
                                             int(row.get('cem_candidate', 0))))
    pool = ordered[:max(1, math.ceil(len(ordered) / 2))]
    chosen = [pool.pop(0)]
    representations = {id(row): action_representation(row['search_parameters'], branch)
                       for row in ordered}
    while pool and len(chosen) < maximum:
        def priority(row):
            distance = min(float(np.sqrt(np.square(
                representations[id(row)] - representations[id(other)]).mean()))
                           for other in chosen)
            return distance, float(row['score']), -int(row['_search_seed'])
        selected = max(pool, key=priority)
        chosen.append(selected)
        pool.remove(selected)
    return chosen


def _zero_history(spec, seed):
    locked = deepcopy(spec)
    locked.role_action_mode = 'lane_locked'
    env = ScenarioEnv()
    obs = env.reset(locked, seed)
    values = {key: [] for key in ('tokens', 'token_mask', 'actor_mask')}
    while not env.done:
        if len(values['tokens']) < HISTORY_STEPS:
            for key in values:
                values[key].append(np.asarray(obs[key]).copy())
        obs, _, _, info = env.step(np.zeros((2, 2), dtype=np.float32))
    if len(values['tokens']) != HISTORY_STEPS:
        raise ValueError('condition ended before the fixed observation history')
    return {key: np.asarray(value) for key, value in values.items()}, info


def _condition_map(specs):
    result = {}
    for spec in specs:
        key = (spec.branch, spec.scenario_id)
        if key in result:
            raise ValueError(f'duplicate condition {key}')
        result[key] = spec
    return result


def build_pulse_set_corpus(sources, output, split_kind, audit_conditions=()):
    """Aggregate successful attempts into condition-level target sets.

    ``sources`` is a sequence of dictionaries containing ``name``, ``conditions``
    and ``searches``. Search seeds may differ within a source.
    """
    if split_kind not in ('train_val', 'screen'):
        raise ValueError('split_kind must be train_val or screen')
    output = Path(output)
    if (output / 'pulse_set.npz').exists():
        raise FileExistsError('pulse-set corpus exists; use a fresh output')

    loaded, fingerprint_owners = [], {}
    source_reports = []
    for source in sources:
        conditions = Path(source['conditions'])
        specs, manifest = load_conditions(conditions)
        if manifest.get('purpose') != 'training':
            raise ValueError('pulse-set sources must have purpose=training')
        by_key = _condition_map(specs)
        for spec in specs:
            fingerprint = spec_fingerprint(spec)
            owner = fingerprint_owners.setdefault(fingerprint, source['name'])
            if owner != source['name']:
                raise ValueError(f'training source fingerprint overlap: {owner}/{source["name"]}')
        attempts = defaultdict(list)
        search_records = []
        for directory in map(Path, source['searches']):
            summary_path = directory / 'conditions_search.json'
            attempts_path = directory / 'attempts.jsonl'
            summary = read_json(summary_path)
            if (summary.get('kind') != 'parameters'
                    or summary.get('role_action_mode') != 'lane_locked'
                    or summary.get('condition_set_version') != manifest['condition_set_version']
                    or len(summary.get('branches', [])) != 1
                    or summary.get('seed') is None):
                raise ValueError(f'incompatible pulse-set search {directory}')
            branch, search_seed = summary['branches'][0], int(summary['seed'])
            for row in load_rows(attempts_path):
                if row['branch'] != branch:
                    raise ValueError(f'branch mismatch in {attempts_path}')
                row['_search_seed'] = search_seed
                attempts[(branch, row['scenario_id'])].append(row)
            search_records.append({
                'path': directory.as_posix(), 'branch': branch, 'seed': search_seed,
                'summary_sha256': sha256(summary_path),
                'attempts_sha256': sha256(attempts_path),
            })
        loaded.append((source, manifest, by_key, attempts))
        source_reports.append({
            'name': source['name'], 'condition_set_version': manifest['condition_set_version'],
            'conditions_sha256': sha256(conditions), 'searches': search_records,
        })

    audit = {}
    own_fingerprints = set(fingerprint_owners)
    for path in map(Path, audit_conditions):
        specs, manifest = load_conditions(path)
        overlap = own_fingerprints & {spec_fingerprint(spec) for spec in specs}
        audit[manifest['condition_set_version']] = len(overlap)
        if overlap:
            raise ValueError(f'condition leakage with {path}: {len(overlap)} fingerprints')

    examples, provenance, rejection = [], [], Counter()
    for source, manifest, by_key, attempts in loaded:
        for key, spec in by_key.items():
            candidates = []
            active = 1 if spec.branch == 'single' else 2
            for row in attempts.get(key, []):
                parameters = np.asarray(row.get('search_parameters', []), dtype=np.float32)
                if parameters.size != active * 3:
                    rejection['wrong_parameter_dimension'] += 1
                    continue
                onset = (parameters.reshape(active, 3)[:, 1] + 1.) * 2.
                if not (row.get('valid') and row.get('dangerous')
                        and row.get('terminated', True)
                        and not row.get('budget_truncated', False)):
                    continue
                if np.any(onset < HISTORY_STEPS * DECISION_DT - 1e-8):
                    rejection['pulse_before_history_complete'] += 1
                    continue
                candidates.append(row)
            if not candidates:
                rejection['no_eligible_success'] += 1
                continue
            chosen = select_diverse_candidates(candidates, spec.branch)
            replay_seed = int(chosen[0]['_search_seed']) + int(chosen[0]['condition_index'])
            history, scripted = _zero_history(spec, replay_seed)
            if not scripted['valid']:
                rejection['script_invalid'] += 1
                continue
            if scripted['dangerous']:
                rejection['script_already_dangerous'] += 1
                continue
            target = np.zeros((MAX_CANDIDATES, 2, 3), dtype=np.float32)
            mask = np.zeros(MAX_CANDIDATES, dtype=bool)
            selected_rows = []
            for index, row in enumerate(chosen):
                target[index, :active] = np.asarray(row['search_parameters'], dtype=np.float32).reshape(active, 3)
                mask[index] = True
                selected_rows.append({
                    'parameters': row['search_parameters'], 'score': row['score'],
                    'search_seed': int(row['_search_seed']),
                    'replay_seed': int(row['_search_seed']) + int(row['condition_index']),
                    'decision_steps': int(row['decision_steps']),
                    'outcome': {key: row.get(key) for key in
                                ('valid', 'dangerous', 'collision', 'min_clearance',
                                 'invalid_reasons', 'terminated', 'budget_truncated')},
                })
            condition_index = int(chosen[0]['condition_index'])
            split = ('screen' if split_kind == 'screen'
                     else ('val' if condition_index % 5 == 0 else 'train'))
            uid = f"{manifest['condition_set_version']}::{spec.branch}::{spec.scenario_id}"
            examples.append({**history, 'target': target, 'candidate_mask': mask,
                             'branch': spec.branch, 'split': split, 'uid': uid})
            provenance.append({
                'uid': uid, 'source': source['name'], 'branch': spec.branch,
                'scenario_id': spec.scenario_id, 'condition_index': condition_index,
                'condition_set_version': manifest['condition_set_version'],
                'condition_fingerprint': spec_fingerprint(spec), 'split': split,
                'replay_seed': replay_seed, 'spec': spec.to_dict(),
                'eligible_successes': len(candidates), 'selected_candidates': selected_rows,
                'script_replay': scripted,
            })
    if not examples:
        raise ValueError('no eligible pulse-set conditions')
    for branch in ('single', 'dual'):
        if not any(row['branch'] == branch for row in examples):
            raise ValueError(f'no eligible {branch} conditions')
        if split_kind == 'train_val':
            for split in ('train', 'val'):
                if not any(row['branch'] == branch and row['split'] == split for row in examples):
                    raise ValueError(f'empty {branch} {split} split')

    arrays = {key: np.stack([row[key] for row in examples])
              for key in ('tokens', 'token_mask', 'actor_mask', 'target', 'candidate_mask')}
    arrays.update({
        'branch': np.asarray([row['branch'] for row in examples]),
        'split': np.asarray([row['split'] for row in examples]),
        'uid': np.asarray([row['uid'] for row in examples]),
        'schema_version': np.asarray(SCHEMA),
    })
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / 'pulse_set.npz'
    np.savez_compressed(corpus_path, **arrays)
    (output / 'manifest.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = Counter((row['branch'], row['split']) for row in examples)
    candidate_counts = [int(row['candidate_mask'].sum()) for row in examples]
    report = {
        'schema_version': SCHEMA, 'split_kind': split_kind, 'examples': len(examples),
        'counts': {f'{branch}_{split}': count
                   for (branch, split), count in sorted(counts.items())},
        'history_steps': HISTORY_STEPS, 'max_candidates': MAX_CANDIDATES,
        'conditions_with_fewer_than_two_candidates': sum(value < 2 for value in candidate_counts),
        'candidate_count_min': min(candidate_counts),
        'candidate_count_median': float(np.median(candidate_counts)),
        'candidate_count_max': max(candidate_counts), 'rejections': dict(rejection),
        'sources': source_reports, 'audit_fingerprint_overlap': audit,
        'selection': ('complete valid dangerous onset>=0.5s attempts; top score half; '
                      'farthest-first 80-step action RMS; valid safe script'),
        'heldout_read': False,
    }
    report['corpus_sha256'] = sha256(corpus_path)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def medoid_targets(data):
    """Return the joint action-space medoid of each condition's target set."""
    result = np.zeros((len(data['target']), 2, 3), dtype=np.float32)
    for index, branch in enumerate(data['branch']):
        candidates = data['target'][index, data['candidate_mask'][index]]
        reps = np.stack([action_representation(
            candidate[:1 if branch == 'single' else 2], str(branch))
                         for candidate in candidates])
        distances = np.sqrt(np.square(reps[:, None] - reps[None]).mean(-1))
        result[index] = candidates[int(np.argmin(distances.sum(1)))]
    return result


def observable_features(data, role):
    tokens = data['tokens'][:, :, role].reshape(len(data['tokens']), -1)
    masks = data['token_mask'][:, :, role].reshape(len(data['tokens']), -1)
    return np.concatenate((tokens, masks.astype(np.float32)), axis=1)


def fit_ridge(train, screen, alphas=(.01, .1, 1., 10., 100.)):
    """Select actor-local ridge regularization on train-val and predict screen."""
    targets = medoid_targets(train)
    predictions = np.zeros((len(screen['tokens']), 2, 3), dtype=np.float32)
    report, models = [], []
    for role in range(2):
        features = observable_features(train, role)
        screen_features = observable_features(screen, role)
        active = train['actor_mask'][:, -1, role].astype(bool)
        fit = active & (train['split'] == 'train')
        val = active & (train['split'] == 'val')

        def solve(indices, test, alpha):
            x, y = features[indices], targets[indices, role]
            mean_x, scale_x = x.mean(0), x.std(0)
            scale_x[scale_x < 1e-8] = 1.
            standardized = (x - mean_x) / scale_x
            mean_y = y.mean(0)
            weights = standardized.T @ np.linalg.solve(
                standardized @ standardized.T + alpha * np.eye(len(standardized)),
                y - mean_y)
            prediction = np.clip((test - mean_x) / scale_x @ weights + mean_y, -1., 1.)
            return prediction, (mean_x, scale_x, mean_y, weights)

        choices = []
        for alpha in alphas:
            predicted, _ = solve(fit, features[val], alpha)
            choices.append((float(np.square(predicted - targets[val, role]).sum(1).mean()), alpha))
        val_mse, alpha = min(choices)
        predicted, model = solve(active, screen_features, alpha)
        screen_active = screen['actor_mask'][:, -1, role].astype(bool)
        predicted[:, 1] = np.maximum(predicted[:, 1], -.75)
        predictions[screen_active, role] = predicted[screen_active]
        report.append({'role': role, 'selected_alpha': alpha,
                       'train_examples': int(fit.sum()), 'val_examples': int(val.sum()),
                       'validation_parameter_mse': val_mse})
        models.append(model)
    return predictions, report, models


def predict_ridge(data, models):
    predictions = np.zeros((len(data['tokens']), 2, 3), dtype=np.float32)
    for role, (mean_x, scale_x, mean_y, weights) in enumerate(models):
        features = observable_features(data, role)
        values = np.clip((features - mean_x) / scale_x @ weights + mean_y, -1., 1.)
        values[:, 1] = np.maximum(values[:, 1], -.75)
        active = data['actor_mask'][:, -1, role].astype(bool)
        predictions[active, role] = values[active]
    return predictions


def observable_history(spec, seed):
    history, _ = _zero_history(spec, seed)
    return history
