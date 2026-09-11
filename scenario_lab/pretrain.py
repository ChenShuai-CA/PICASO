"""Small, traceable motion-prior pretraining; not a full traffic foundation model."""
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import torch
from .env import OBS_DIM
from .policy import Actor, save_bundle


def track_examples(track, length=8):
    if track['kind'] not in ('vehicle', 'pedestrian'):
        return
    t, xy = np.asarray(track['t']), np.asarray(track['xy'])
    velocity, heading = np.asarray(track['velocity']), np.asarray(track['heading'])
    if len(t) < length + 2:
        return
    dt = np.diff(t)
    if not np.isfinite(dt).all() or np.any(dt <= 0) or np.any(dt > .15):
        return
    speed = np.linalg.norm(velocity, axis=-1)
    accel = np.diff(speed) / dt
    yawrate = np.diff(np.unwrap(heading)) / dt
    ped = track['kind'] == 'pedestrian'
    role = 0 if ped else 1
    if ped:
        turns = yawrate / .45
    else:
        steer = np.arctan(2.7 * yawrate / np.maximum(speed[:-1], .5))
        turns = np.r_[np.diff(steer) / dt[:-1], 0.] / .2
    actions = np.stack((accel / (1.5 if ped else 3.), turns), axis=-1)
    for start in range(0, len(actions) - length + 1, length):
        stop = start + length
        a = actions[start:stop]
        if not np.isfinite(a).all() or np.any(np.abs(a) > 1.) or np.mean(speed[start:stop]) < .3:
            continue
        tokens = np.zeros((length, 2, 3, OBS_DIM), dtype=np.float32)
        masks = np.zeros((length, 2, 3), dtype=bool)
        active = np.zeros((length, 2), dtype=np.float32)
        target = np.zeros((length, 2, 2), dtype=np.float32)
        rotate = (math.pi / 2 if ped else 0.) - heading[start]
        c, s = math.cos(rotate), math.sin(rotate)
        rotation = np.array([[c, -s], [s, c]])
        transformed_velocity = velocity[start:stop] @ rotation.T
        h = heading[start:stop] + rotate
        # Self-only motion prior is intentional. Interaction pretraining requires
        # aligned neighbors and is recorded as pending, never fabricated here.
        tokens[:, role, role + 1, 2:4] = transformed_velocity / 15
        tokens[:, role, role + 1, 4] = np.cos(h)
        tokens[:, role, role + 1, 5] = np.sin(h)
        tokens[:, role, role + 1, 6] = 1
        tokens[:, role, role + 1, 8] = 1
        tokens[:, role, role + 1, 9] = not ped
        tokens[:, role, role + 1, 10] = ped
        tokens[:, role, role + 1, 11] = 1
        masks[:, role, role + 1] = True
        active[:, role] = 1
        target[:, role] = a
        yield dict(tokens=tokens, token_mask=masks, actor_mask=active, target=target,
                   group_id=track['group_id'], source=track['source'], split=track['split'],
                   kind=track['kind'], track_id=str(track['track_id']), start_time=float(t[start]),
                   end_time=float(t[stop]),
                   window_hash=hashlib.sha256(np.c_[t[start:stop], xy[start:stop], velocity[start:stop], heading[start:stop]].astype('<f8').tobytes()).hexdigest(),
                   provenance=track['provenance'])


def select_public_files(root, source, max_files):
    """Bounded deterministic train/val coverage, without mixing INTERACTION releases."""
    from .data import _location_and_split, split_group
    root = Path(root)
    if max_files < 1:
        raise ValueError('max_files must be positive')
    paths = sorted((root / ('INTERACTION' if source == 'interaction' else 'Waymo')).rglob(
        '*.csv' if source == 'interaction' else '*.tfrecord*'))
    if source == 'interaction':
        originals = [p for p in paths if 'recorded_trackfiles' in p.parts and p.name.startswith('vehicle_tracks_')]
        if originals:
            paths = originals  # avoid mixing recut challenge exports with original recordings
    buckets = {'train': [], 'val': [], 'test': []}
    for path in paths:
        if source == 'interaction':
            location, official, observed = _location_and_split(path)
            split = 'test' if observed else (official or split_group(source, location))
        else:
            labels = [p.lower() for p in path.parts[-2:]]
            split = 'test' if any('test' in p for p in labels) else ('val' if any('val' in p for p in labels) else 'train')
        buckets[split].append(path)
    for bucket in buckets.values():
        bucket.sort(key=lambda p: (p.name, str(p.parent)))  # diversify recordings before taking further chunks
    chosen = []
    # Interleave train and validation files; test data is never fallback training.
    while len(chosen) < max_files and any(buckets[s] for s in ('train', 'val')):
        for split in ('train', 'val'):
            if buckets[split] and len(chosen) < max_files:
                chosen.append(buckets[split].pop(0))
    return chosen


def prepare_public(root, output, max_files=2, records_per_file=8, max_examples=10000):
    from .data import iter_interaction_tracks
    from .waymo import iter_waymo_tracks
    root, output = Path(root), Path(output)
    if max_examples < 4 or records_per_file < 1:
        raise ValueError('max_examples >=4 and records_per_file >=1 required')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'motion_prior.npz').exists():
        raise FileExistsError('Corpus already exists; use a fresh output directory')
    examples, errors, selected_files = [], [], []
    duplicate_count = 0
    fingerprints = set()
    # Separate budgets prevent the first source from consuming all capacity.
    for source in ('interaction', 'waymo'):
        paths = select_public_files(root, source, max_files)
        per_file = max(1, (max_examples // 2) // max(len(paths), 1))
        for path in paths:
            kind_counts = {'vehicle': 0, 'pedestrian': 0}
            kind_caps = {'vehicle': max(1, per_file * 3 // 4), 'pedestrian': max(1, per_file // 4)}
            selected_files.append(str(path))
            try:
                iterator = iter_interaction_tracks(path) if source == 'interaction' else iter_waymo_tracks(path, records_per_file)
                for track in iterator:
                    kind = track['kind']
                    if kind not in kind_caps or kind_counts[kind] >= kind_caps[kind]:
                        continue
                    for example in track_examples(track):
                        if kind_counts[kind] >= kind_caps[kind]:
                            break
                        key = (source, example['group_id'], example['track_id'], example['start_time'], example['end_time'], example['window_hash'], example['split'])
                        if key in fingerprints:
                            duplicate_count += 1
                            continue
                        fingerprints.add(key)
                        examples.append(example)
                        kind_counts[kind] += 1
            except (ValueError, KeyError, OSError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    groups = defaultdict(set)
    for e in examples:
        groups[(e['source'], e['group_id'])].add(e['split'])
    ambiguous = {group for group, splits in groups.items() if len(splits) > 1}
    examples = [e for e in examples if (e['source'], e['group_id']) not in ambiguous]
    if not examples:
        raise ValueError(f'no valid public examples; errors={errors}')
    arrays = {k: np.stack([e[k] for e in examples]) for k in ('tokens', 'token_mask', 'actor_mask', 'target')}
    for key in ('split', 'source', 'group_id', 'kind'):
        arrays[key] = np.array([e[key] for e in examples])
    np.savez_compressed(output / 'motion_prior.npz', **arrays)
    manifest = [{k: e[k] for k in ('group_id', 'track_id', 'start_time', 'end_time', 'window_hash', 'source', 'split', 'kind', 'provenance')} for e in examples]
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = lambda key, values: {s: int(np.sum(arrays[key] == s)) for s in values}
    report = dict(examples=len(examples), sources=counts('source', ('interaction', 'waymo')),
                  splits=counts('split', ('train', 'val', 'test')), kinds=counts('kind', ('vehicle', 'pedestrian')),
                  independent_groups=len({(e['source'], e['group_id']) for e in examples}),
                  cross_split_groups_quarantined=len(ambiguous), duplicate_windows_skipped=duplicate_count,
                  selected_files=selected_files, errors=errors,
                  sampling='source/file/type quotas; original INTERACTION release preferred; train/val files interleaved',
                  scope='self-motion prior only; aligned interaction/map pretraining pending',
                  role_sources='INTERACTION vehicle exports only; unambiguous pedestrians from Waymo')
    report['ready_for_joint_pilot'] = (all(report['sources'].values()) and report['splits']['train'] > 0
                                      and report['splits']['val'] > 0 and all(report['kinds'].values()))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def pretrain(corpus, output, epochs=5, hidden=64, seed=7, device='cpu'):
    from .runtime import resolve_device, record_runtime
    device = resolve_device(device)
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if (Path(output) / 'prior.pt').exists():
        raise FileExistsError('Pretraining output exists; use a fresh directory')
    record_runtime(output, device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(corpus, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}  # decode compressed arrays once, not per minibatch
    memberships = defaultdict(set)
    for source, group, split in zip(data['source'], data['group_id'], data['split']):
        memberships[(str(source), str(group))].add(str(split))
    if any(len(splits) > 1 for splits in memberships.values()):
        raise ValueError('Training corpus has cross-split source groups')
    train_indices = np.flatnonzero(data['split'] == 'train')
    val_indices = np.flatnonzero(data['split'] == 'val')
    if len(train_indices) == 0:
        raise ValueError('no training examples; official val/test data never used as substitute')
    actor = Actor(hidden).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    role_totals = data['actor_mask'][train_indices].sum(axis=(0, 1))
    role_weights = role_totals.sum() / (np.maximum(role_totals, 1.) * max(np.count_nonzero(role_totals), 1))
    role_weights = torch.as_tensor(role_weights, device=device, dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        shuffled = rng.permutation(train_indices)
        losses = []
        actor.train()
        for start in range(0, len(shuffled), 64):
            idx = shuffled[start:start + 64]
            x = {k: torch.as_tensor(data[k][idx], device=device) for k in ('tokens', 'token_mask', 'actor_mask', 'target')}
            dist, _ = actor(x['tokens'], x['token_mask'], x['actor_mask'])
            error = (dist.mean.tanh() - x['target']).square().sum(-1)
            weighted_mask = x['actor_mask'] * role_weights
            loss = (error * weighted_mask).sum() / weighted_mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = None
        if len(val_indices):
            actor.eval()
            with torch.no_grad():
                idx = val_indices[:1024]
                x = {k: torch.as_tensor(data[k][idx], device=device) for k in ('tokens', 'token_mask', 'actor_mask', 'target')}
                dist, _ = actor(x['tokens'], x['token_mask'], x['actor_mask'])
                val_loss = float(((dist.mean.tanh() - x['target']).square().sum(-1) * x['actor_mask']).sum() / x['actor_mask'].sum())
        history.append(dict(epoch=epoch + 1, train_mse=float(np.mean(losses)), val_mse=val_loss))
        print(json.dumps(history[-1]), flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_bundle(output / 'prior.pt', actor, dict(seed=seed, epochs=epochs, corpus=str(corpus), device=device,
                     loss_weighting='inverse training role frequency'),
                extra=dict(scope='self-motion prior; not full interaction model',
                           corpus_sha256=hashlib.sha256(Path(corpus).read_bytes()).hexdigest()))
    (output / 'pretraining.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return history
