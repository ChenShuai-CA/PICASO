import json
import numpy as np
import pytest
from scenario_lab.evaluate import ScriptPolicy, evaluate, run_episode
from scenario_lab.sampling import (sample_spec, sample_spec_v1, sample_spec_v2,
                                   reference_feasible, export_conditions, load_conditions)

# Frozen from runs/20260911_gpu_pilot/eval_script/episodes.jsonl: the v1 draw
# sequence behind every pilot evaluation. Editing sample_spec_v1 breaks this.
PILOT_ANCHOR = {
    'dual-00000': dict(ego_speed=13.73679213154749, crossing_x=25.24441793351483,
                       pedestrian_y=-5.601141524097506, pedestrian_speed=1.25515797870579,
                       pedestrian_delay=0.43543467483798504, occluder_x=15.146650760108896,
                       occluder_speed=3.329060631815236),
    'single-00000': dict(ego_speed=11.623051082037978, crossing_x=33.38494328355113,
                         pedestrian_y=-5.058116405355492, pedestrian_speed=1.182923148290211,
                         pedestrian_delay=0.6345108307440631, occluder_x=20.030965970130676,
                         occluder_speed=2.5731088402423663),
}


def test_v1_sampler_snapshot_matches_pilot():
    rng = np.random.default_rng(1000)
    specs = [sample_spec_v1(rng, branch, i) for branch in ('single', 'dual') for i in range(20)]
    assert len({s.scenario_id for s in specs}) == 40
    by_id = {s.scenario_id: s for s in specs}
    for sid, anchor in PILOT_ANCHOR.items():
        for key, value in anchor.items():
            assert getattr(by_id[sid], key) == value, f'{sid}.{key} drifted'
        assert by_id[sid].physics_version == 1 and by_id[sid].sampler_version == 1


def test_v2_reference_feasible_and_script_rollout_clean():
    rng = np.random.default_rng(2026)
    infeasible = 0
    for i in range(200):
        spec, tries = sample_spec_v2(rng, 'dual', i, return_tries=True)
        if spec.condition_role == 'reference_infeasible':
            infeasible += 1
            continue
        assert reference_feasible(spec)
    assert infeasible <= 10, f'resampling failed to resolve {infeasible}/200 conflicts'
    rng = np.random.default_rng(1000)
    for i in range(20):
        spec = sample_spec_v2(rng, 'dual', i)
        row = run_episode(ScriptPolicy(), spec, 1000 + i * 100)
        assert 'target_target_collision' not in row['invalid_reasons'], f'dual-{i:05d}'


def test_v2_stress_unconstrained():
    rng = np.random.default_rng(7)
    specs = [sample_spec_v2(rng, 'dual', i, role='stress') for i in range(50)]
    assert all(s.condition_role == 'stress' for s in specs)
    assert any(not reference_feasible(s) for s in specs)  # stress deliberately keeps hard cases


def test_sample_spec_dispatch():
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    assert sample_spec(rng1, 'dual', 0, version=1).to_dict() == sample_spec_v1(rng2, 'dual', 0).to_dict()
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    assert sample_spec(rng1, 'dual', 0).to_dict() == sample_spec_v2(rng2, 'dual', 0).to_dict()
    with pytest.raises(ValueError, match='unknown sampler version'):
        sample_spec(np.random.default_rng(0), 'dual', 0, version=9)


def test_condition_manifest_roundtrip_and_rng_decoupling(tmp_path):
    path = tmp_path / 'conds.json'
    export_conditions(path, seed=11, count=3, sampler_version=2, purpose='development')
    specs, manifest = load_conditions(path)
    assert len(specs) == 6 and manifest['purpose'] == 'development'
    assert {s.branch for s in specs} == {'single', 'dual'}
    # Evaluating a branch subset from the same manifest must use identical dual specs,
    # which the old rng-order sampling could not guarantee.
    out_full, out_dual = tmp_path / 'full', tmp_path / 'dual_only'
    evaluate(ScriptPolicy(), out_full, conditions=specs,
             condition_set_version=manifest['condition_set_version'])
    evaluate(ScriptPolicy(), out_dual, conditions=[s for s in specs if s.branch == 'dual'],
             condition_set_version=manifest['condition_set_version'])
    rows = lambda p: {json.loads(l)['scenario_id']: json.loads(l)['spec']
                      for l in (p / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()}
    assert all(sid in rows(out_full) for sid in rows(out_dual))
    assert rows(out_full) == {**rows(out_full), **rows(out_dual)}


def test_training_condition_manifest_is_explicit(tmp_path):
    path = tmp_path / 'training.json'
    export_conditions(path, seed=51, count=2, sampler_version=2, purpose='training')
    specs, manifest = load_conditions(path)
    assert manifest['purpose'] == 'training'
    assert manifest['condition_set_version'] == 'training_seed51_samplerv2'
    assert len(specs) == 4


def test_load_conditions_rejects_version_mismatch(tmp_path):
    path = tmp_path / 'bad.json'
    manifest = export_conditions(path, seed=1, count=2, sampler_version=2)
    manifest['physics_version'] = 1  # header now disagrees with the specs it contains
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='refusing silent mismatch'):
        load_conditions(path)


def test_load_conditions_rejects_missing_header(tmp_path):
    manifest = export_conditions(tmp_path / 'ok.json', seed=1, count=2, sampler_version=2)
    del manifest['condition_set_version']
    path = tmp_path / 'stripped.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='missing fields'):
        load_conditions(path)
