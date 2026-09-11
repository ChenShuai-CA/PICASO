import json

import torch

from scenario_lab.schema import ScenarioSpec
from scenario_lab.teacher import build_teacher_corpus, pretrain_teacher


def write_conditions(path, purpose, version, crossing_values=(9.0, 9.5)):
    conditions = []
    for branch in ('single', 'dual'):
        for index, crossing_x in enumerate(crossing_values):
            spec = ScenarioSpec(
                branch=branch, scenario_id=f'{branch}-{index:05d}', ego_speed=10,
                crossing_x=crossing_x, pedestrian_y=-1, pedestrian_speed=1.5,
                pedestrian_delay=0, occluder_x=0, occluder_y=-2.8,
                occluder_speed=3, horizon=3, physics_version=2,
                sampler_version=2, condition_role='reference')
            conditions.append({'branch': branch, 'index': index, 'spec': spec.to_dict()})
    manifest = {
        'condition_set_version': version, 'purpose': purpose,
        'sampler_version': 2, 'physics_version': 2,
        'count_per_branch': len(crossing_values), 'conditions': conditions,
    }
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def write_search(path, manifest, branch):
    path.mkdir()
    size = 3 if branch == 'single' else 6
    summary = {
        'kind': 'parameters', 'seed': 3, 'role_action_mode': 'lane_locked',
        'condition_set_version': manifest['condition_set_version'],
        'branches': [branch],
        'per_condition': [
            {'condition_index': entry['index'],
             'scenario_id': entry['spec']['scenario_id'],
             'best_valid': True, 'best_dangerous': True,
             'best_parameters': [0.] * size}
            for entry in manifest['conditions'] if entry['branch'] == branch
        ],
    }
    (path / 'conditions_search.json').write_text(json.dumps(summary), encoding='utf-8')
    (path / 'attempts.jsonl').write_text('', encoding='utf-8')


def test_teacher_corpus_is_disjoint_replayable_and_trainable(tmp_path):
    training_path, dev_path = tmp_path / 'training.json', tmp_path / 'dev.json'
    training = write_conditions(training_path, 'training', 'training_test')
    write_conditions(dev_path, 'development', 'development_test', (30., 31.))
    searches = []
    for branch in ('single', 'dual'):
        directory = tmp_path / f'search_{branch}'
        write_search(directory, training, branch)
        searches.append(directory)
    report = build_teacher_corpus(training_path, dev_path, searches, tmp_path / 'corpus')
    assert report['examples'] == 4
    assert report['condition_fingerprint_overlap'] == 0
    assert report['counts'] == {
        'single_train': 1, 'single_val': 1, 'dual_train': 1, 'dual_val': 1}
    history = pretrain_teacher(tmp_path / 'corpus' / 'teacher.npz', tmp_path / 'bc',
                               epochs=1, hidden=32, seed=4, device='cpu', batch_size=2)
    assert len(history) == 1 and history[0]['val_mse'] >= 0
    bundle = torch.load(tmp_path / 'bc' / 'prior.pt', map_location='cpu', weights_only=True)
    assert bundle['config']['role_action_mode'] == 'lane_locked'
    assert bundle['extra']['scope'] == 'training-only parameter-CEM behavior cloning'


def test_teacher_corpus_rejects_physical_condition_overlap(tmp_path):
    training_path, dev_path = tmp_path / 'training.json', tmp_path / 'dev.json'
    write_conditions(training_path, 'training', 'training_test')
    write_conditions(dev_path, 'development', 'development_test')
    try:
        build_teacher_corpus(training_path, dev_path, [], tmp_path / 'corpus')
    except ValueError as error:
        assert 'condition leakage' in str(error)
    else:
        raise AssertionError('physical teacher/development overlap was accepted')
