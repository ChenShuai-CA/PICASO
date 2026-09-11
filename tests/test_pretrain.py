from pathlib import Path
import numpy as np
import pytest
from scenario_lab.pretrain import prepare_public, select_public_files, pretrain


def fake_tracks(path, limit=None):
    source = 'waymo' if '.tfrecord' in path.name else 'interaction'
    split = 'val' if 'val' in str(path.parent) else 'train'
    for index in range(20):
        t = np.arange(33) * .1
        kind = 'pedestrian' if index % 2 else 'vehicle'
        velocity = np.tile([1., 0.], (len(t), 1))
        yield dict(source=source, group_id=source+'-'+split, track_id=str(index),
                   kind=kind, split=split, t=t, xy=np.c_[t, t*0],
                   velocity=velocity, heading=np.zeros(len(t)), provenance={})


def test_each_source_and_holdout_gets_budget(tmp_path, monkeypatch):
    for source, suffix in [('INTERACTION', '.csv'), ('Waymo', '.tfrecord-00000')]:
        for split in ('train', 'val'):
            path = tmp_path / source / split / (source+'_'+split+suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr('scenario_lab.data.iter_interaction_tracks', fake_tracks)
    monkeypatch.setattr('scenario_lab.waymo.iter_waymo_tracks', fake_tracks)
    report = prepare_public(tmp_path, tmp_path/'corpus', max_examples=64)
    assert report['ready_for_joint_pilot']
    # Quotas moved from per-file 3:1 to per-group 3:2 (v5 neighbor corpus, CHANGELOG
    # P1.2 R2): each fake group caps at 9 vehicle + 6 pedestrian = 15 examples.
    assert report['sources'] == {'interaction': 30, 'waymo': 30}
    assert report['splits']['train'] == report['splits']['val'] == 30
    assert report['cross_split_groups_quarantined'] == 0
    with pytest.raises(FileExistsError):
        prepare_public(tmp_path, tmp_path/'corpus', max_examples=64)


def test_original_interaction_not_mixed_with_recut_versions(tmp_path):
    original = tmp_path/'INTERACTION'/'recorded_trackfiles'/'LOC_A'/'vehicle_tracks_000.csv'
    recut = tmp_path/'INTERACTION'/'challenge'/'train'/'LOC_A_train.csv'
    ambiguous = original.with_name('pedestrian_tracks_000.csv')
    for p in (original, recut, ambiguous):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    assert select_public_files(tmp_path, 'interaction', 2) == [original]


def test_pretrain_rejects_cross_split_group(tmp_path):
    corpus = tmp_path/'bad.npz'
    np.savez(corpus, source=np.array(['waymo','waymo']), group_id=np.array(['same','same']),
             split=np.array(['train','val']))
    with pytest.raises(ValueError, match='cross-split'):
        pretrain(corpus, tmp_path/'out', epochs=1)
