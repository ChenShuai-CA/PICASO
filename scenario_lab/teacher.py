"""CEM teacher-corpus construction and recurrent behavior cloning."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .env import OBS_DIM, ScenarioEnv
from .evaluate import ParamPolicy
from .policy import Actor, save_bundle
from .sampling import load_conditions


TEACHER_SCHEMA_VERSION = 'cem-parameter-v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def spec_fingerprint(spec):
    """Hash physical/controller inputs while ignoring set-local scenario names."""
    values = spec.to_dict()
    values.pop('scenario_id', None)
    payload = json.dumps(values, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def collect_parameter_teacher(spec, parameters, seed):
    """Replay one open-loop pulse solution and retain only actor-visible inputs."""
    spec = deepcopy(spec)
    spec.role_action_mode = 'lane_locked'
    active_actors = 1 if spec.branch == 'single' else 2
    policy = ParamPolicy(parameters, active_actors=active_actors)
    env = ScenarioEnv()
    obs = env.reset(spec, seed)
    policy.reset()
    sequences = {key: [] for key in ('tokens', 'token_mask', 'actor_mask', 'target')}
    while not env.done:
        action = policy.act(obs).astype(np.float32)
        for key in ('tokens', 'token_mask', 'actor_mask'):
            sequences[key].append(obs[key])
        sequences['target'].append(action)
        obs, _, _, info = env.step(action)
    return {key: np.asarray(value) for key, value in sequences.items()}, info


def _condition_map(specs):
    result = {}
    for spec in specs:
        key = (spec.branch, spec.scenario_id)
        if key in result:
            raise ValueError(f'duplicate condition key {key}')
        result[key] = spec
    return result


def build_teacher_corpus(training_conditions, dev_conditions, search_directories, output):
    """Build a padded sequence corpus from independent training-only CEM solutions."""
    training_conditions, dev_conditions = Path(training_conditions), Path(dev_conditions)
    train_specs, train_manifest = load_conditions(training_conditions)
    dev_specs, dev_manifest = load_conditions(dev_conditions)
    if train_manifest['purpose'] != 'training':
        raise ValueError('teacher conditions must have purpose=training')
    if dev_manifest['purpose'] != 'development':
        raise ValueError('validation conditions must have purpose=development')
    if train_manifest['condition_set_version'] == dev_manifest['condition_set_version']:
        raise ValueError('teacher and development condition sets must differ')
    train_fingerprints = {spec_fingerprint(spec) for spec in train_specs}
    dev_fingerprints = {spec_fingerprint(spec) for spec in dev_specs}
    overlap = train_fingerprints & dev_fingerprints
    if overlap:
        raise ValueError(f'teacher/development condition leakage: {len(overlap)} fingerprints')

    by_key = _condition_map(train_specs)
    examples, provenance = [], []
    search_hashes = {}
    for directory in map(Path, search_directories):
        summary_path = directory / 'conditions_search.json'
        attempts_path = directory / 'attempts.jsonl'
        summary = read_json(summary_path)
        if (summary.get('kind') != 'parameters'
                or summary.get('role_action_mode') != 'lane_locked'
                or summary.get('condition_set_version')
                != train_manifest['condition_set_version']
                or len(summary.get('branches', [])) != 1):
            raise ValueError(f'incompatible teacher search {directory}')
        if summary.get('seed') is None:
            raise ValueError(f'teacher search seed missing in {directory}')
        branch = summary['branches'][0]
        search_hashes[branch] = {
            'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            'attempts_sha256': hashlib.sha256(attempts_path.read_bytes()).hexdigest(),
        }
        for condition in summary['per_condition']:
            if not (condition.get('best_valid') and condition.get('best_dangerous')
                    and condition.get('best_parameters') is not None):
                continue
            key = (branch, condition['scenario_id'])
            if key not in by_key:
                raise ValueError(f'search condition absent from training manifest: {key}')
            spec = by_key[key]
            replay_seed = int(summary['seed']) + int(condition['condition_index'])
            sequence, replay = collect_parameter_teacher(
                spec, condition['best_parameters'], replay_seed)
            if not replay['valid'] or not replay['dangerous']:
                raise ValueError(f'teacher solution replay mismatch for {key}')
            condition_index = int(condition['condition_index'])
            split = 'val' if condition_index % 5 == 0 else 'train'
            example = dict(sequence, split=split, branch=branch,
                           scenario_id=spec.scenario_id,
                           fingerprint=spec_fingerprint(spec))
            examples.append(example)
            provenance.append({
                'branch': branch, 'scenario_id': spec.scenario_id,
                'condition_index': condition_index, 'split': split,
                'condition_fingerprint': example['fingerprint'],
                'search_seed': int(summary['seed']), 'replay_seed': replay_seed,
                'best_parameters': condition['best_parameters'],
                'decision_steps': int(len(sequence['target'])),
                'replay': replay,
            })
    if not examples:
        raise ValueError('no complete valid dangerous teacher solutions')
    if {example['branch'] for example in examples} != {'single', 'dual'}:
        raise ValueError('teacher corpus must contain both formal branches')
    if {example['split'] for example in examples} != {'train', 'val'}:
        raise ValueError('teacher corpus must contain train and val examples')

    max_steps = max(len(example['target']) for example in examples)
    count = len(examples)
    arrays = {
        'tokens': np.zeros((count, max_steps, 2, 3, OBS_DIM), dtype=np.float32),
        'token_mask': np.zeros((count, max_steps, 2, 3), dtype=bool),
        'actor_mask': np.zeros((count, max_steps, 2), dtype=np.float32),
        'target': np.zeros((count, max_steps, 2, 2), dtype=np.float32),
        'time_mask': np.zeros((count, max_steps), dtype=bool),
    }
    for index, example in enumerate(examples):
        steps = len(example['target'])
        for key in ('tokens', 'token_mask', 'actor_mask', 'target'):
            arrays[key][index, :steps] = example[key]
        arrays['time_mask'][index, :steps] = True
    arrays.update({
        'split': np.asarray([example['split'] for example in examples]),
        'branch': np.asarray([example['branch'] for example in examples]),
        'scenario_id': np.asarray([example['scenario_id'] for example in examples]),
        'condition_fingerprint': np.asarray([example['fingerprint'] for example in examples]),
        'schema_version': np.asarray(TEACHER_SCHEMA_VERSION),
    })
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / 'teacher.npz'
    if corpus_path.exists():
        raise FileExistsError('teacher corpus exists; use a fresh output directory')
    np.savez_compressed(corpus_path, **arrays)
    (output / 'manifest.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = defaultdict(int)
    for example in examples:
        counts[(example['branch'], example['split'])] += 1
    report = {
        'schema_version': TEACHER_SCHEMA_VERSION,
        'examples': count, 'max_steps': max_steps,
        'counts': {f'{branch}_{split}': counts[(branch, split)]
                   for branch in ('single', 'dual') for split in ('train', 'val')},
        'training_condition_set_version': train_manifest['condition_set_version'],
        'development_condition_set_version': dev_manifest['condition_set_version'],
        'training_conditions_sha256': hashlib.sha256(training_conditions.read_bytes()).hexdigest(),
        'development_conditions_sha256': hashlib.sha256(dev_conditions.read_bytes()).hexdigest(),
        'condition_fingerprint_overlap': 0,
        'search_hashes': search_hashes,
        'selection': 'complete valid dangerous parameter-CEM best solutions only',
        'scope': 'training-only CEM teacher; development conditions used only for leakage audit',
    }
    report['corpus_sha256'] = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def pretrain_teacher(corpus, output, epochs=20, hidden=64, seed=7,
                     device='cpu', batch_size=16):
    """Fit the shared recurrent actor to full teacher episodes."""
    from .runtime import resolve_device, record_runtime
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    device = resolve_device(device)
    output = Path(output)
    if (output / 'prior.pt').exists():
        raise FileExistsError('teacher pretraining output exists; use a fresh directory')
    record_runtime(output, device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(corpus, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data.get('schema_version')) != TEACHER_SCHEMA_VERSION:
        raise ValueError('unsupported teacher corpus schema')
    for key in ('tokens', 'actor_mask', 'target'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'non-finite teacher array: {key}')
    if np.max(np.abs(data['target'])) > 1 + 1e-6:
        raise ValueError('teacher actions exceed normalized bounds')
    memberships = defaultdict(set)
    for fingerprint, split in zip(data['condition_fingerprint'], data['split']):
        memberships[str(fingerprint)].add(str(split))
    if any(len(splits) > 1 for splits in memberships.values()):
        raise ValueError('teacher condition appears in multiple corpus splits')
    train_indices = np.flatnonzero(data['split'] == 'train')
    val_indices = np.flatnonzero(data['split'] == 'val')
    if not len(train_indices) or not len(val_indices):
        raise ValueError('teacher corpus needs nonempty train and val splits')

    actor = Actor(hidden).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    role_totals = (data['actor_mask'][train_indices]
                   * data['time_mask'][train_indices, :, None]).sum(axis=(0, 1))
    role_weights = role_totals.sum() / (np.maximum(role_totals, 1.)
                                        * max(np.count_nonzero(role_totals), 1))
    role_weights = torch.as_tensor(role_weights, dtype=torch.float32, device=device)

    def batch_loss(indices, weighted):
        x = {key: torch.as_tensor(data[key][indices], device=device)
             for key in ('tokens', 'token_mask', 'actor_mask', 'target', 'time_mask')}
        dist, _ = actor(x['tokens'], x['token_mask'], x['actor_mask'])
        error = (dist.mean.tanh() - x['target']).square().sum(-1)
        mask = x['actor_mask'] * x['time_mask'][..., None]
        weights = mask * role_weights
        if weighted:
            weights = weights * (1 + 2 * x['target'][..., 0].abs())
        return (error * weights).sum() / weights.sum().clamp_min(1.)

    history = []
    for epoch in range(epochs):
        actor.train()
        losses = []
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(train_indices), batch_size):
            indices = shuffled[start:start + batch_size]
            loss = batch_loss(indices, weighted=True)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise FloatingPointError('non-finite teacher pretraining update')
            optimizer.step()
            losses.append(float(loss.detach()))
        actor.eval()
        with torch.no_grad():
            val_mse = float(batch_loss(val_indices, weighted=False))
        row = {'epoch': epoch + 1, 'train_weighted_mse': float(np.mean(losses)),
               'val_mse': val_mse}
        history.append(row)
        print(json.dumps(row), flush=True)
    config = {
        'seed': seed, 'epochs': epochs, 'hidden': hidden,
        'corpus': str(corpus), 'device': device, 'batch_size': batch_size,
        'role_action_mode': 'lane_locked',
        'pretraining_kind': 'parameter_cem_teacher_behavior_cloning',
        'loss': 'role-balanced action-weighted sequence MSE',
    }
    save_bundle(output / 'prior.pt', actor, config,
                extra={'scope': 'training-only parameter-CEM behavior cloning',
                       'corpus_sha256': hashlib.sha256(Path(corpus).read_bytes()).hexdigest()})
    (output / 'pretraining.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return history
