"""Incremental CEM pulse corpus, parameter predictor, and closed-loop runner."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .env import DECISION_DT, OBS_DIM, ScenarioEnv
from .evaluate import ParamPolicy, ScriptPolicy, run_episode
from .sampling import load_conditions
from .teacher import spec_fingerprint


PULSE_SCHEMA_VERSION = 'incremental-pulse-v1'
HISTORY_STEPS = 5


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class PulsePredictor(nn.Module):
    """Predict each role's pulse using only that role's observable history."""
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.token_encoder = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh())
        self.role_embedding = nn.Embedding(2, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, 3), nn.Linear(hidden, 3)])
        for head in self.heads:
            nn.init.orthogonal_(head.weight, .01)
            nn.init.zeros_(head.bias)

    def forward(self, tokens, token_mask, actor_mask):
        batch, length, actors, entities, _ = tokens.shape
        x = self.token_encoder(tokens).reshape(-1, entities, self.hidden)
        mask = token_mask.reshape(-1, entities).clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True
        roles = torch.arange(actors, device=tokens.device).view(1, 1, actors)
        roles = roles.expand(batch, length, actors)
        query = self.role_embedding(roles.reshape(-1)).unsqueeze(1)
        pooled, _ = self.attention(query, x, x, key_padding_mask=~mask,
                                   need_weights=False)
        pooled = pooled.reshape(batch, length, actors, self.hidden)
        sequence = pooled.permute(0, 2, 1, 3).reshape(
            batch * actors, length, self.hidden)
        sequence, _ = self.memory(sequence)
        final = sequence[:, -1].reshape(batch, actors, self.hidden)
        parameters = torch.stack(
            [self.heads[actor](final[:, actor]) for actor in range(actors)], dim=1)
        return parameters.tanh() * actor_mask[:, -1, :, None]


class PulsePolicy:
    """Observe a zero-action prefix, predict once, then execute on absolute time."""
    def __init__(self, predictor, device='cpu', history_steps=HISTORY_STEPS):
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.history_steps = history_steps
        self.reset()

    def reset(self):
        self.history = []
        self.parameters = None
        self.step = 0
        self.t = 0.

    @torch.no_grad()
    def _predict(self):
        tokens = torch.as_tensor(
            np.stack([obs['tokens'] for obs in self.history]), device=self.device)[None]
        token_mask = torch.as_tensor(
            np.stack([obs['token_mask'] for obs in self.history]), device=self.device)[None]
        actor_mask = torch.as_tensor(
            np.stack([obs['actor_mask'] for obs in self.history]), device=self.device)[None]
        return self.predictor(tokens, token_mask, actor_mask)[0].cpu().numpy()

    def act(self, obs):
        if len(self.history) < self.history_steps:
            self.history.append({key: np.asarray(obs[key]).copy()
                                 for key in ('tokens', 'token_mask', 'actor_mask')})
            self.step += 1
            self.t += DECISION_DT
            return np.zeros((2, 2), dtype=np.float32)
        if self.parameters is None:
            self.parameters = self._predict()
        action = np.zeros((2, 2), dtype=np.float32)
        for actor, (amplitude, start, duration) in enumerate(self.parameters):
            onset = (float(start) + 1.) * 2.
            end = onset + (float(duration) + 1.) * 1.5
            if onset <= self.t < end:
                action[actor, 0] = amplitude
        self.step += 1
        self.t += DECISION_DT
        return action * obs['actor_mask'][:, None]


def save_pulse_bundle(path, predictor, config, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'schema_version': PULSE_SCHEMA_VERSION, 'hidden': predictor.hidden,
        'predictor': predictor.state_dict(), 'config': config, 'extra': extra or {},
    }, path)


def load_pulse_bundle(path, device='cpu'):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get('schema_version') != PULSE_SCHEMA_VERSION:
        raise ValueError('unsupported pulse bundle schema')
    predictor = PulsePredictor(bundle['hidden'])
    predictor.load_state_dict(bundle['predictor'])
    return PulsePolicy(predictor, device=device), bundle


def _condition_map(specs):
    result = {}
    for spec in specs:
        key = (spec.branch, spec.scenario_id)
        if key in result:
            raise ValueError(f'duplicate condition key {key}')
        result[key] = spec
    return result


def _teacher_history(spec, parameters, seed):
    spec = deepcopy(spec)
    spec.role_action_mode = 'lane_locked'
    active = 1 if spec.branch == 'single' else 2
    policy = ParamPolicy(parameters, active_actors=active)
    env = ScenarioEnv()
    obs = env.reset(spec, seed)
    history = {key: [] for key in ('tokens', 'token_mask', 'actor_mask')}
    prefix_actions = []
    while not env.done:
        action = policy.act(obs).astype(np.float32)
        if len(prefix_actions) < HISTORY_STEPS:
            for key in history:
                history[key].append(obs[key])
            prefix_actions.append(action)
        obs, _, _, info = env.step(action)
    return {key: np.asarray(value) for key, value in history.items()}, \
        np.asarray(prefix_actions), info


def build_incremental_pulse_corpus(conditions, search_directories, output,
                                   split_kind='train_val', audit_conditions=()):
    """Select script-safe replayable teachers with an observable zero-action prefix."""
    conditions = Path(conditions)
    specs, condition_manifest = load_conditions(conditions)
    if condition_manifest.get('purpose') != 'training':
        raise ValueError('pulse corpus requires purpose=training conditions')
    if split_kind not in ('train_val', 'screen'):
        raise ValueError('split_kind must be train_val or screen')
    own_fingerprints = {spec_fingerprint(spec) for spec in specs}
    audit = {}
    for path in map(Path, audit_conditions):
        other_specs, other_manifest = load_conditions(path)
        overlap = own_fingerprints & {spec_fingerprint(spec) for spec in other_specs}
        audit[other_manifest['condition_set_version']] = len(overlap)
        if overlap:
            raise ValueError(f'condition leakage with {path}: {len(overlap)} fingerprints')

    by_key = _condition_map(specs)
    examples, provenance = [], []
    rejection = Counter()
    search_hashes = {}
    for directory in map(Path, search_directories):
        summary_path = directory / 'conditions_search.json'
        summary = read_json(summary_path)
        if (summary.get('kind') != 'parameters'
                or summary.get('role_action_mode') != 'lane_locked'
                or summary.get('condition_set_version')
                != condition_manifest['condition_set_version']
                or summary.get('seed') is None
                or len(summary.get('branches', [])) != 1):
            raise ValueError(f'incompatible pulse search {directory}')
        branch = summary['branches'][0]
        search_hashes[branch] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        for condition in summary['per_condition']:
            if not (condition.get('best_valid') and condition.get('best_dangerous')
                    and condition.get('best_parameters') is not None):
                rejection['no_valid_dangerous_teacher'] += 1
                continue
            key = (branch, condition['scenario_id'])
            if key not in by_key:
                raise ValueError(f'search condition absent from manifest: {key}')
            spec = by_key[key]
            index = int(condition['condition_index'])
            replay_seed = int(summary['seed']) + index
            parameters = np.asarray(condition['best_parameters'], dtype=np.float32)
            active = 1 if branch == 'single' else 2
            parameter_rows = parameters.reshape(active, 3)
            if np.any((parameter_rows[:, 1] + 1.) * 2. < HISTORY_STEPS * DECISION_DT):
                rejection['pulse_before_history_complete'] += 1
                continue
            history, prefix_actions, replay = _teacher_history(spec, parameters, replay_seed)
            if len(prefix_actions) != HISTORY_STEPS or np.any(np.abs(prefix_actions) > 1e-7):
                raise ValueError(f'teacher prefix is not zero-action for {key}')
            if not replay['valid'] or not replay['dangerous']:
                raise ValueError(f'teacher replay mismatch for {key}')
            locked = deepcopy(spec)
            locked.role_action_mode = 'lane_locked'
            scripted = run_episode(ScriptPolicy(), locked, replay_seed)
            if not scripted['valid']:
                rejection['script_invalid'] += 1
                continue
            if scripted['dangerous']:
                rejection['script_already_dangerous'] += 1
                continue
            target = np.zeros((2, 3), dtype=np.float32)
            target[:active] = parameter_rows
            split = ('screen' if split_kind == 'screen'
                     else ('val' if index % 5 == 0 else 'train'))
            examples.append({**history, 'target': target, 'branch': branch, 'split': split})
            provenance.append({
                'branch': branch, 'scenario_id': spec.scenario_id,
                'condition_index': index, 'split': split,
                'condition_fingerprint': spec_fingerprint(spec),
                'replay_seed': replay_seed, 'best_parameters': parameters.tolist(),
                'spec': spec.to_dict(), 'teacher_replay': replay,
                'script_replay': scripted,
            })
    if not examples:
        raise ValueError('no eligible incremental pulse teachers')
    for branch in ('single', 'dual'):
        if not any(example['branch'] == branch for example in examples):
            raise ValueError(f'no eligible {branch} pulse teachers')
    if split_kind == 'train_val':
        for branch in ('single', 'dual'):
            for split in ('train', 'val'):
                if not any(example['branch'] == branch and example['split'] == split
                           for example in examples):
                    raise ValueError(f'empty {branch} {split} pulse split')

    arrays = {
        key: np.stack([example[key] for example in examples])
        for key in ('tokens', 'token_mask', 'actor_mask', 'target')}
    arrays.update({
        'branch': np.asarray([example['branch'] for example in examples]),
        'split': np.asarray([example['split'] for example in examples]),
        'scenario_id': np.asarray([row['scenario_id'] for row in provenance]),
        'schema_version': np.asarray(PULSE_SCHEMA_VERSION),
    })
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / 'pulse.npz'
    if corpus_path.exists():
        raise FileExistsError('pulse corpus exists; use a fresh output directory')
    np.savez_compressed(corpus_path, **arrays)
    (output / 'manifest.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = Counter((example['branch'], example['split']) for example in examples)
    report = {
        'schema_version': PULSE_SCHEMA_VERSION, 'examples': len(examples),
        'history_steps': HISTORY_STEPS,
        'counts': {f'{branch}_{split}': count
                   for (branch, split), count in sorted(counts.items())},
        'rejections': dict(rejection),
        'condition_set_version': condition_manifest['condition_set_version'],
        'conditions_sha256': hashlib.sha256(conditions.read_bytes()).hexdigest(),
        'audit_fingerprint_overlap': audit, 'search_hashes': search_hashes,
        'selection': ('complete valid dangerous parameter-CEM teacher; valid safe script; '
                      'all active pulse onsets after zero-action history'),
        'split_kind': split_kind, 'heldout_read': False,
    }
    report['corpus_sha256'] = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def train_pulse_predictor(corpus, output, epochs=50, hidden=64, seed=7,
                          device='cpu', batch_size=16):
    """Fit pulse parameters and select the branch-balanced teacher-val minimum."""
    from .runtime import resolve_device, record_runtime
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    device = resolve_device(device)
    output = Path(output)
    if (output / 'pulse.pt').exists():
        raise FileExistsError('pulse predictor exists; use a fresh output directory')
    record_runtime(output, device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(corpus, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data.get('schema_version')) != PULSE_SCHEMA_VERSION:
        raise ValueError('unsupported pulse corpus schema')
    train_indices = np.flatnonzero(data['split'] == 'train')
    val_indices = np.flatnonzero(data['split'] == 'val')
    if not len(train_indices) or not len(val_indices):
        raise ValueError('pulse training requires nonempty train and val splits')

    predictor = PulsePredictor(hidden).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-4)
    role_totals = data['actor_mask'][train_indices, -1].sum(0)
    role_weights = role_totals.sum() / (np.maximum(role_totals, 1.)
                                        * max(np.count_nonzero(role_totals), 1))
    role_weights = torch.as_tensor(role_weights, dtype=torch.float32, device=device)

    def batch_loss(indices):
        x = {key: torch.as_tensor(data[key][indices], device=device)
             for key in ('tokens', 'token_mask', 'actor_mask', 'target')}
        prediction = predictor(x['tokens'], x['token_mask'], x['actor_mask'])
        mask = x['actor_mask'][:, -1]
        error = (prediction - x['target']).square().sum(-1)
        weights = mask * role_weights
        return (error * weights).sum() / weights.sum().clamp_min(1.)

    def validation():
        values = {}
        with torch.no_grad():
            for branch in ('single', 'dual'):
                indices = val_indices[data['branch'][val_indices] == branch]
                values[branch] = float(batch_loss(indices))
        return values, float(np.mean(list(values.values())))

    history, best = [], None
    for epoch in range(1, epochs + 1):
        predictor.train()
        losses = []
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            loss = batch_loss(shuffled[start:start + batch_size])
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), .5)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise FloatingPointError('non-finite pulse predictor update')
            optimizer.step()
            losses.append(float(loss.detach()))
        predictor.eval()
        val_by_branch, val_balanced = validation()
        row = {'epoch': epoch, 'train_mse': float(np.mean(losses)),
               'val_single_mse': val_by_branch['single'],
               'val_dual_mse': val_by_branch['dual'],
               'val_branch_balanced_mse': val_balanced}
        history.append(row)
        if best is None or val_balanced < best['metric']:
            best = {'epoch': epoch, 'metric': val_balanced,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in predictor.state_dict().items()}}
        print(json.dumps(row), flush=True)
    predictor.load_state_dict(best['state'])
    config = {
        'seed': seed, 'epochs': epochs, 'selected_epoch': best['epoch'],
        'hidden': hidden, 'batch_size': batch_size, 'learning_rate': 3e-4,
        'history_steps': HISTORY_STEPS, 'role_action_mode': 'lane_locked',
        'selection_metric': 'branch-balanced teacher-val parameter MSE',
    }
    corpus_hash = hashlib.sha256(Path(corpus).read_bytes()).hexdigest()
    save_pulse_bundle(output / 'pulse.pt', predictor, config,
                      {'corpus_sha256': corpus_hash, 'heldout_read': False})
    (output / 'training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return {'selected_epoch': best['epoch'], 'selected_val_mse': best['metric'],
            'epochs': epochs}


def evaluate_pulse_corpus(policy, corpus_directory, output):
    """Evaluate on selected teacher conditions using their exact replay seeds."""
    corpus_directory, output = Path(corpus_directory), Path(output)
    report = read_json(corpus_directory / 'report.json')
    if report.get('split_kind') != 'screen' or report.get('heldout_read') is not False:
        raise ValueError('pulse mechanism evaluation requires a screen corpus')
    provenance = read_json(corpus_directory / 'manifest.json')
    rows = []
    for source in provenance:
        from .schema import ScenarioSpec
        spec = ScenarioSpec(**source['spec'])
        spec.role_action_mode = 'lane_locked'
        result = run_episode(policy, spec, int(source['replay_seed']))
        result.update(condition_index=source['condition_index'], split='screen',
                      teacher_dangerous=True, script_dangerous=False)
        rows.append(result)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'episodes.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    summary = {
        'kind': 'incremental_pulse_screen', 'episodes': len(rows),
        'condition_set_version': report['condition_set_version'],
        'corpus_sha256': report['corpus_sha256'], 'heldout_read': False,
        'counts': dict(Counter(row['branch'] for row in rows)),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
