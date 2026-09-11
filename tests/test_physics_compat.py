import json
import math
import numpy as np
import pytest
from scenario_lab.env import ScenarioEnv
from scenario_lab.evaluate import ScriptPolicy, run_episode, replay
from scenario_lab.schema import ScenarioSpec


def test_pedestrian_frozen_during_delay():
    env = ScenarioEnv()
    spec = ScenarioSpec(branch='single', pedestrian_delay=1.0)
    obs = env.reset(spec, 3)
    start = (env.bodies[1].x, env.bodies[1].y, env.bodies[1].speed, env.bodies[1].heading)
    assert env.bodies[1].speed == 0.  # waiting means at rest, not walking in place
    for _ in range(5):  # 0.5 s of full throttle + steering attempts
        env.step(np.array([[1., 1.], [0., 0.]]))
    ped = env.bodies[1]
    assert (ped.x, ped.y) == start[:2]
    assert ped.speed == 0. and ped.heading == start[3]
    track = env.tracks[0][1]
    assert track['vx'] == 0. and track['vy'] == 0.  # reported motion matches frozen position
    while not env.done and env.time < 1.2:
        env.step(np.zeros((2, 2)))
    assert env.bodies[1].speed == pytest.approx(spec.pedestrian_speed)
    assert env.bodies[1].y > start[1]
    assert obs['tokens'][0, 1, 3] == 0.  # observer saw zero pedestrian vy all along


def test_delay_ignores_steering_and_keeps_role():
    env = ScenarioEnv()
    spec = ScenarioSpec(branch='dual', pedestrian_delay=.6)
    env.reset(spec, 3)
    while env.time < spec.pedestrian_delay - 1e-9:
        env.step(np.array([[1., 1.], [0., 0.]]))
    assert env.bodies[1].heading == pytest.approx(math.pi / 2)
    assert not env.invalid_reasons  # no fake pedestrian_role from waiting-period steering


def test_v1_legacy_semantics_preserved():
    env = ScenarioEnv()
    spec = ScenarioSpec(branch='single', pedestrian_delay=1.0, physics_version=1, sampler_version=1)
    env.reset(spec, 5)
    assert env.bodies[1].speed == pytest.approx(spec.pedestrian_speed)  # the legacy quirk itself
    for _ in range(5):
        env.step(np.zeros((2, 2)))
    assert env.bodies[1].y == pytest.approx(spec.pedestrian_y)     # position frozen
    assert env.tracks[0][1]['vy'] != 0.                            # while speed was reported


def test_replay_roundtrip_both_physics_versions(tmp_path):
    for version in (1, 2):
        spec = ScenarioSpec(branch='dual', pedestrian_delay=.5,
                            physics_version=version, sampler_version=1)
        path = tmp_path / f'trace_v{version}.json'
        run_episode(ScriptPolicy(), spec, 42, path)
        result = replay(path)
        assert result['matched'] and result['physics_version'] == version


def test_replay_legacy_trace_defaults_to_v1(tmp_path, capsys):
    spec = ScenarioSpec(branch='dual', pedestrian_delay=.5, physics_version=1, sampler_version=1)
    path = tmp_path / 'legacy.json'
    run_episode(ScriptPolicy(), spec, 7, path)
    record = json.loads(path.read_text(encoding='utf-8'))
    del record['scenario']['physics_version']  # 20260911 traces carry no version field
    path.write_text(json.dumps(record), encoding='utf-8')
    result = replay(path)
    assert result['matched'] and result['physics_version'] == 1
    assert 'legacy physics v1' in capsys.readouterr().out


def test_v1_trace_replay_fails_under_v2_semantics(tmp_path):
    # Guards the compatibility branch: forcing v1 trajectories through v2 physics
    # must fail loudly, never silently "match" under the wrong semantics.
    spec = ScenarioSpec(branch='single', pedestrian_delay=.5, physics_version=1, sampler_version=1)
    path = tmp_path / 'legacy.json'
    run_episode(ScriptPolicy(), spec, 7, path)
    record = json.loads(path.read_text(encoding='utf-8'))
    record['scenario']['physics_version'] = 2
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(AssertionError, match='replay mismatch'):
        replay(path)
