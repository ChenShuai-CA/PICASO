import copy
import math
import numpy as np
import pytest
from scenario_lab.pretrain import (track_examples, prepare_public, select_public_files,
                                   LOCATION_SPLITS_V5)


def make_track(kind, track_id, t, xy, velocity, heading, group_id='G', split='train'):
    return dict(kind=kind, track_id=track_id, group_id=group_id, source='interaction',
                split=split, t=np.asarray(t, float), xy=np.asarray(xy, float),
                velocity=np.asarray(velocity, float), heading=np.asarray(heading, float),
                provenance={})


def steady_vehicle(track_id='v0', frames=33, speed=2.):
    t = np.arange(frames) * .1
    xy = np.c_[t * speed, np.zeros(frames)]
    return make_track('vehicle', track_id, t, xy, np.tile([speed, 0.], (frames, 1)), np.zeros(frames))


def examples_of(anchor, group):
    return list(track_examples(anchor, group_tracks=group))


def test_neighbor_future_tamper_leaves_window_input_untouched():
    anchor, neighbor = steady_vehicle(), make_track(
        'pedestrian', 'p0', np.arange(33) * .1, np.tile([5., 3.], (33, 1)),
        np.tile([.4, 0.], (33, 1)), np.zeros(33))
    group = [anchor, neighbor]
    before = examples_of(anchor, group)
    tampered = copy.deepcopy(group)
    # Corrupt the pedestrian only AFTER the anchor's last window (frame 32+, t >= 3.2 s).
    tampered[1]['xy'][32:] += 500.
    tampered[1]['velocity'][32:] = 40.
    after = examples_of(tampered[0], tampered)
    assert len(before) == len(after) == 4
    for b, a in zip(before, after):
        assert np.array_equal(b['tokens'], a['tokens'])
        assert np.array_equal(b['token_mask'], a['token_mask'])
        assert np.array_equal(b['target'], a['target'])


def test_anchor_future_tamper_only_visible_in_nothing():
    anchor = steady_vehicle()
    group = [anchor, make_track('pedestrian', 'p0', np.arange(33) * .1,
                                np.tile([5., 3.], (33, 1)), np.tile([.4, 0.], (33, 1)), np.zeros(33))]
    before = examples_of(group[0], group)
    tampered = copy.deepcopy(group)
    tampered[0]['xy'][32] += 9.  # anchor's own post-window frame
    after = examples_of(tampered[0], tampered)
    for b, a in zip(before, after):
        assert np.array_equal(b['tokens'], a['tokens'])
        assert np.array_equal(b['target'], a['target'])


def test_neighbor_rotation_geometry():
    # Vehicle anchor heading pi/4 (northeast); pedestrian 5 m due north.
    frames = 33
    t = np.arange(frames) * .1
    diag = np.tile([math.cos(math.pi / 4) * 2., math.sin(math.pi / 4) * 2.], (frames, 1))
    xy = np.cumsum(np.c_[diag[:, 0] * .1, diag[:, 1] * .1], axis=0)
    anchor = make_track('vehicle', 'v0', t, xy, diag, np.full(frames, math.pi / 4))
    rel_north = np.array([0., 5.])
    neighbor = make_track('pedestrian', 'p0', t, xy + rel_north, np.tile([0., .4], (frames, 1)),
                          np.full(frames, math.pi / 2))
    (example,) = examples_of(anchor, [anchor, neighbor])[:1]
    # rotate = 0 - pi/4; R(-pi/4) @ (0, 5) = (5/sqrt(2), 5/sqrt(2)).
    expect_x, expect_y = 5 / math.sqrt(2), 5 / math.sqrt(2)
    assert example['neighbors'] == {'pedestrian': 'p0'}
    tok = example['tokens'][:, 1, 1]  # vehicle role, pedestrian slot
    assert np.allclose(tok[:, 0], expect_x / 40)
    assert np.allclose(tok[:, 1], expect_y / 10)
    assert tok[0, 9] == 0 and tok[0, 10] == 1  # kind one-hot: pedestrian
    assert tok[0, 11] == 0  # is_self stays false for a neighbor


def test_window_picks_nearest_neighbor_per_kind_and_records_it():
    frames = 33
    t = np.arange(frames) * .1
    anchor = steady_vehicle()
    near = make_track('pedestrian', 'near', t, np.c_[np.full(frames, 5.), 4. + t * 0],
                      np.tile([0., .4], (frames, 1)), np.full(frames, math.pi / 2))
    far = make_track('pedestrian', 'far', t, np.tile([25., 30.], (frames, 1)),
                     np.tile([0., .4], (frames, 1)), np.full(frames, math.pi / 2))
    (example,) = examples_of(anchor, [anchor, near, far])[:1]
    assert example['neighbors']['pedestrian'] == 'near'
    tok = example['tokens'][:, 1, 1]
    assert np.allclose(tok[:, 1], 4. / 10)  # the near track's fixed lateral offset


def test_beyond_radius_neighbor_is_not_filled():
    anchor = steady_vehicle()
    distant = make_track('pedestrian', 'far', np.arange(33) * .1, np.tile([60., 0.], (33, 1)),
                         np.tile([.4, 0.], (33, 1)), np.zeros(33))
    (example,) = examples_of(anchor, [anchor, distant])[:1]
    assert example['neighbors'] == {}
    assert not example['token_mask'][:, 1, 1].any()


def _fake_interaction_by_location(paths_frames=40):
    def fake(path):
        location = path.parent.name
        for index, kind in enumerate(('vehicle', 'pedestrian')):
            frames = paths_frames
            t = np.arange(frames) * .1
            xy = np.c_[t * 2. + index * 3., np.full(frames, index * 2.)]
            yield dict(kind=kind, track_id=f'{index}', group_id=location, source='interaction',
                       split='train', t=t, xy=xy, velocity=np.tile([2., 0.], (frames, 1)),
                       heading=np.zeros(frames), provenance={})
    return fake


def test_holdout_locations_never_leak_into_train(tmp_path, monkeypatch):
    heldout = [loc for loc, split in LOCATION_SPLITS_V5.items() if split == 'val']
    train_locs = [loc for loc, split in LOCATION_SPLITS_V5.items() if split == 'train']
    for loc in heldout + train_locs:
        for kind in ('vehicle_tracks_000.csv', 'pedestrian_tracks_000.csv'):
            p = tmp_path / 'INTERACTION' / 'recorded_trackfiles' / loc / kind
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
    monkeypatch.setattr('scenario_lab.data.iter_interaction_tracks', _fake_interaction_by_location())
    monkeypatch.setattr('scenario_lab.waymo.iter_waymo_tracks', lambda path, limit: iter(()))
    report = prepare_public(tmp_path, tmp_path / 'corpus', max_files=32,
                            include_pedestrians=True, location_splits=LOCATION_SPLITS_V5)
    import json
    manifest = json.loads((tmp_path / 'corpus' / 'manifest.json').read_text(encoding='utf-8'))
    by_split = {}
    for entry in manifest:
        by_split.setdefault(entry['split'], set()).add(entry['group_id'])
    assert by_split['val'] == set(heldout)
    assert by_split['train'] <= set(train_locs)
    assert report['cross_split_groups_quarantined'] == 0
    assert report['windows_with_neighbors'] > 0  # vehicle/pedestrian co-recording pairs visible


def test_select_public_files_respects_location_split_map(tmp_path):
    for loc in ('DR_USA_Roundabout_SR', 'DR_USA_Intersection_GL'):
        for kind in ('vehicle_tracks_000.csv', 'pedestrian_tracks_000.csv'):
            p = tmp_path / 'INTERACTION' / 'recorded_trackfiles' / loc / kind
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
    picked = select_public_files(tmp_path, 'interaction', 4, include_pedestrians=True,
                                 per_location=4, location_splits=LOCATION_SPLITS_V5)
    names = {p.parent.name for p in picked}
    assert names == {'DR_USA_Roundabout_SR', 'DR_USA_Intersection_GL'}
    val_bucket = [p for p in picked if p.parent.name == 'DR_USA_Roundabout_SR']
    assert len(val_bucket) == 2  # both kinds selected under the per-location cap
