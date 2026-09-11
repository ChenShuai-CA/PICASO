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

CORPUS_VERSION = 'v5'
FEATURE_VERSION = 'neighbor-v1'
# Pre-registered 2026-09-11 (GLM_CHANGELOG P1.2 R2), fixed before any v5 training:
# INTERACTION locations map to splits explicitly (sha256('interaction_holdout_v5\x1f<location>')
# ascending, last two -> val) because the location-level hash produced zero val locations and the
# official validation-set-list shipped with the dataset is an empty file. split_group itself is
# untouched; this map overrides at the corpus-preparation layer only.
LOCATION_SPLITS_V5 = {
    'DR_USA_Roundabout_EP': 'train', 'DR_USA_Intersection_EP1': 'train',
    'DR_USA_Intersection_GL': 'train', 'DR_USA_Roundabout_FT': 'train',
    'DR_USA_Intersection_MA': 'train', 'DR_USA_Intersection_EP0': 'train',
    'DR_USA_Roundabout_SR': 'val', 'DR_DEU_Roundabout_OF': 'val',
}
_NEIGHBOR_RADIUS = 40.  # meters; matches the /40 position normalisation in env.observe


def _frame_keys(t):
    """Integer frame keys on the shared 100 ms grid (exact match, no interpolation)."""
    return np.round(np.asarray(t) * 10).astype(np.int64)


def _neighbor_candidates(anchor, start, stop, others):
    """Window-level top-1 neighbor per kind among same-group track segments.

    A candidate must cover the anchor window on every frame key (>= length shared
    timestamps by construction); the winner is fixed for the whole window so the
    neighbor identity never flickers frame to frame. Only spec parameters of the
    recording are read; the anchor's own future is never touched.
    """
    anchor_keys = _frame_keys(anchor['t'][start:stop])
    best = {}
    for other in others:
        if other is anchor or other['kind'] not in ('vehicle', 'pedestrian'):
            continue
        if str(other['track_id']) == str(anchor['track_id']) and other['source'] == anchor['source']:
            continue  # a different segment of the same recorded track is not a neighbor
        lookup = dict(zip(_frame_keys(other['t']).tolist(), range(len(other['t']))))
        idx = [lookup.get(int(key)) for key in anchor_keys]
        if any(i is None for i in idx):
            continue
        idx = np.asarray(idx)
        # Same finiteness bar the anchor window itself must pass: INTERACTION pedestrian
        # headings are velocity-derived and stay NaN at low speed, and a NaN neighbor
        # feature would poison training even when the anchor window is clean.
        frames = (other['xy'][idx], other['velocity'][idx], other['heading'][idx])
        if not all(np.isfinite(f).all() for f in frames):
            continue
        mean_dist = float(np.linalg.norm(other['xy'][idx] - anchor['xy'][start:stop], axis=-1).mean())
        if mean_dist > _NEIGHBOR_RADIUS:
            continue
        if other['kind'] not in best or mean_dist < best[other['kind']][0]:
            best[other['kind']] = (mean_dist, other, idx)
    return best


def _fill_neighbor_tokens(tokens, masks, role, other, idx, anchor, start, stop, rotate):
    """Encode one neighbor into its kind slot, mirroring the anchor's rotation."""
    c, s = math.cos(rotate), math.sin(rotate)
    rotation = np.array([[c, -s], [s, c]])
    rel_xy = (other['xy'][idx] - anchor['xy'][start:stop]) @ rotation.T
    rel_velocity = other['velocity'][idx] @ rotation.T
    h = other['heading'][idx] + rotate
    ped = other['kind'] == 'pedestrian'
    slot = 1 if ped else 2
    tokens[:, role, slot, 0] = rel_xy[:, 0] / 40
    tokens[:, role, slot, 1] = rel_xy[:, 1] / 10
    tokens[:, role, slot, 2:4] = rel_velocity / 15
    tokens[:, role, slot, 4] = np.cos(h)
    tokens[:, role, slot, 5] = np.sin(h)
    tokens[:, role, slot, 6] = 1  # visible
    tokens[:, role, slot, 8] = 1  # present
    tokens[:, role, slot, 9] = not ped
    tokens[:, role, slot, 10] = ped
    masks[:, role, slot] = True


def track_examples(track, length=8, group_tracks=None):
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
    others = [o for o in (group_tracks or ()) if o is not track]
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
        neighbor_ids = {}
        for kind, (_, other, idx) in _neighbor_candidates(track, start, stop, others).items():
            _fill_neighbor_tokens(tokens, masks, role, other, idx, track, start, stop, rotate)
            neighbor_ids[kind] = str(other['track_id'])
        yield dict(tokens=tokens, token_mask=masks, actor_mask=active, target=target,
                   group_id=track['group_id'], source=track['source'], split=track['split'],
                   kind=track['kind'], track_id=str(track['track_id']), start_time=float(t[start]),
                   end_time=float(t[stop]),
                   window_hash=hashlib.sha256(np.c_[t[start:stop], xy[start:stop], velocity[start:stop], heading[start:stop]].astype('<f8').tobytes()).hexdigest(),
                   neighbors=neighbor_ids,
                   provenance=track['provenance'])


def select_public_files(root, source, max_files, include_pedestrians=False, per_location=None,
                        location_splits=None):
    """Bounded deterministic train/val coverage, without mixing INTERACTION releases."""
    from .data import _location_and_split, split_group
    root = Path(root)
    if max_files < 1:
        raise ValueError('max_files must be positive')
    paths = sorted((root / ('INTERACTION' if source == 'interaction' else 'Waymo')).rglob(
        '*.csv' if source == 'interaction' else '*.tfrecord*'))
    if source == 'interaction':
        prefixes = ('vehicle_tracks_',) + (('pedestrian_tracks_',) if include_pedestrians else ())
        originals = [p for p in paths if 'recorded_trackfiles' in p.parts
                     and p.name.startswith(prefixes)]
        if originals:
            paths = originals  # avoid mixing recut challenge exports with original recordings
        if location_splits:
            # The pre-registered map doubles as the participation list: locations outside
            # it keep their hash splits and must not crowd the selection budget.
            paths = [p for p in paths if _location_and_split(p)[0] in location_splits]
        if per_location:
            # Split the per-location cap evenly across the two file families; a plain
            # name sort would let whichever family has more chunks take the whole cap.
            by_location = defaultdict(list)
            for p in paths:
                by_location[p.parent.name].append(p)
            paths = [p for loc in sorted(by_location)
                     for prefix in ('pedestrian_tracks_', 'vehicle_tracks_')
                     for p in sorted((q for q in by_location[loc] if q.name.startswith(prefix)),
                                     key=lambda q: q.name)[:max(1, per_location // 2)]]
    buckets = {'train': [], 'val': [], 'test': []}
    for path in paths:
        if source == 'interaction':
            location, official, observed = _location_and_split(path)
            if location_splits and location in location_splits:
                split = location_splits[location]  # pre-registered v5 holdout map
            else:
                split = 'test' if observed else (official or split_group(source, location))
        else:
            labels = [p.lower() for p in path.parts[-2:]]
            split = 'test' if any('test' in p for p in labels) else ('val' if any('val' in p for p in labels) else 'train')
        buckets[split].append(path)
    for bucket in buckets.values():
        # Location-major order interleaves pedestrian and vehicle exports of the same
        # recording; a name-major sort would let one file family consume the budget.
        bucket.sort(key=lambda p: (str(p.parent), p.name))
    chosen = []
    # Interleave train and validation files; test data is never fallback training.
    while len(chosen) < max_files and any(buckets[s] for s in ('train', 'val')):
        for split in ('train', 'val'):
            if buckets[split] and len(chosen) < max_files:
                chosen.append(buckets[split].pop(0))
    return chosen


def prepare_public(root, output, max_files=2, records_per_file=8, max_examples=10000,
                   include_pedestrians=False, per_location=None, location_splits=None,
                   max_files_interaction=None):
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
    neighbor_windows = 0
    # Separate budgets prevent the first source from consuming all capacity.
    for source in ('interaction', 'waymo'):
        limit = max_files_interaction if (source == 'interaction' and max_files_interaction) else max_files
        paths = select_public_files(root, source, limit, include_pedestrians
                                    if source == 'interaction' else False,
                                    per_location if source == 'interaction' else None,
                                    location_splits if source == 'interaction' else None)
        for path in paths:
            selected_files.append(str(path))
        groups = defaultdict(list)
        for path in paths:
            try:
                iterator = (iter_interaction_tracks(path) if source == 'interaction'
                            else iter_waymo_tracks(path, records_per_file))
                for track in iterator:
                    if track['kind'] in ('vehicle', 'pedestrian'):
                        # Pre-registered explicit split overrides the location-level hash.
                        location = str(track['group_id']).split('::')[0]
                        if location_splits and location in location_splits:
                            track['split'] = location_splits[location]
                        groups[(str(track['source']), str(track['group_id']))].append(track)
            except (ValueError, KeyError, OSError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
        # Quotas are per independent group, so one prolific recording cannot consume
        # the whole source budget; pedestrians get priority within each group's cap.
        per_group = max(1, (max_examples // 2) // max(len(groups), 1))
        kind_caps = {'vehicle': max(1, per_group * 3 // 5), 'pedestrian': max(1, per_group * 2 // 5)}
        for (track_source, group_id), tracks in sorted(groups.items()):
            kind_counts = {'vehicle': 0, 'pedestrian': 0}
            for track in tracks:
                kind = track['kind']
                if kind not in kind_caps or kind_counts[kind] >= kind_caps[kind]:
                    continue
                for example in track_examples(track, group_tracks=tracks):
                    if kind_counts[kind] >= kind_caps[kind]:
                        break
                    key = (source, example['group_id'], example['track_id'], example['start_time'], example['end_time'], example['window_hash'], example['split'])
                    if key in fingerprints:
                        duplicate_count += 1
                        continue
                    fingerprints.add(key)
                    if example['neighbors']:
                        neighbor_windows += 1
                    examples.append(example)
                    kind_counts[kind] += 1
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
    np.savez_compressed(output / 'motion_prior.npz', feature_version=np.array(FEATURE_VERSION), **arrays)
    manifest = [{k: e[k] for k in ('group_id', 'track_id', 'start_time', 'end_time', 'window_hash',
                                   'source', 'split', 'kind', 'neighbors', 'provenance')} for e in examples]
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = lambda key, values: {s: int(np.sum(arrays[key] == s)) for s in values}
    report = dict(examples=len(examples), sources=counts('source', ('interaction', 'waymo')),
                  splits=counts('split', ('train', 'val', 'test')), kinds=counts('kind', ('vehicle', 'pedestrian')),
                  independent_groups=len({(e['source'], e['group_id']) for e in examples}),
                  cross_split_groups_quarantined=len(ambiguous), duplicate_windows_skipped=duplicate_count,
                  windows_with_neighbors=neighbor_windows,
                  corpus_version=CORPUS_VERSION, feature_version=FEATURE_VERSION,
                  include_pedestrians=include_pedestrians, per_location=per_location,
                  location_splits=location_splits or None,
                  selected_files=selected_files, errors=errors,
                  sampling='per-group quotas (vehicle 3/5, pedestrian 2/5); original INTERACTION '
                           'release preferred; train/val files interleaved',
                  scope='self-motion prior plus aligned same-group neighbors; map pretraining pending',
                  role_sources=('INTERACTION vehicle and pedestrian exports; '
                                'unambiguous pedestrians from Waymo') if include_pedestrians
                               else 'INTERACTION vehicle exports only; unambiguous pedestrians from Waymo')
    report['ready_for_joint_pilot'] = (all(report['sources'].values()) and report['splits']['train'] > 0
                                      and report['splits']['val'] > 0 and all(report['kinds'].values()))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def pretrain(corpus, output, epochs=5, hidden=64, seed=7, device='cpu', use_neighbors=True):
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
    if not np.isfinite(data['tokens']).all():
        bad = int((~np.isfinite(data['tokens'])).sum())
        raise ValueError(f'corpus tokens contain {bad} non-finite values; refusing to train')
    if not use_neighbors:
        # Self-only ablation on the very same corpus: mask every slot except each
        # role's own (role r lives in slot r+1), so neighbor tokens stay present in
        # the arrays but are invisible to attention.
        self_only = np.zeros_like(data['token_mask'])
        for role in (0, 1):
            self_only[:, :, role, role + 1] = data['token_mask'][:, :, role, role + 1]
        data['token_mask'] = self_only
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
                     use_neighbors=use_neighbors,
                     loss_weighting='inverse training role frequency'),
                extra=dict(scope='self-motion prior; not full interaction model',
                           corpus_sha256=hashlib.sha256(Path(corpus).read_bytes()).hexdigest()))
    (output / 'pretraining.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return history
