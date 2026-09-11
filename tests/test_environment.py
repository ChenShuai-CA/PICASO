from copy import deepcopy
import math
import numpy as np
import pytest
from scenario_lab.env import ScenarioEnv
from scenario_lab.schema import ScenarioSpec, Body
from scenario_lab.geometry import clearance, segment_blocked
from scenario_lab.evaluate import ScriptPolicy, run_episode, replay


def test_presence_mask_and_absent_actions():
    env, other = ScenarioEnv(), ScenarioEnv()
    obs = env.reset(seed=1)
    other.reset(seed=1)
    assert obs['actor_mask'].tolist() == [1, 0]
    assert not obs['token_mask'][1].any()
    for _ in range(5):
        a = np.array([[0., 0.], [1., -1.]])
        env.step(a)
        other.step(np.zeros((2, 2)))
    assert env.bodies == other.bodies


def test_occlusion_changes_ego_track_not_truth_filled():
    env = ScenarioEnv()
    env.reset(ScenarioSpec(branch='dual'))
    assert not env._visible(0, 1)
    assert env._estimate(0, 1) is None
    env.bodies[2].y = -9
    env._refresh_tracks()
    assert env._estimate(0, 1) is not None
    old = deepcopy(env.tracks[0][1])
    env.bodies[2].y = -2.8
    env.bodies[1].speed = 2.5
    env.time = .2
    env._refresh_tracks()
    assert not env.tracks[0][1]['visible']
    estimate = env._estimate(0, 1)
    assert estimate['vy'] == old['vy']
    assert estimate['y'] == pytest.approx(old['y'] + .2 * old['vy'])


def test_same_observation_ego_controller_does_not_read_hidden_target():
    a, b = ScenarioEnv(), ScenarioEnv()
    spec = ScenarioSpec(branch='dual')
    a.reset(spec)
    b.reset(spec)
    b.bodies[1].x += 2
    b.bodies[1].speed = 2.9
    # Observations intentionally unchanged; no global state consulted by controller.
    assert a._ego_request() == b._ego_request()


def test_geometry_oriented_clearance():
    car = Body(0, 0, 0, 0, 'vehicle')
    ped = Body(0, 2, 0, 0, 'pedestrian', .6, .6)
    assert clearance(car, ped) == pytest.approx(.8)
    ped.y = 1.1
    assert clearance(car, ped) == 0
    assert segment_blocked((-5, 0), (5, 0), car)
    assert not segment_blocked((-5, 2), (5, 2), car)


def test_initial_overlap_rejected():
    with pytest.raises(ValueError, match='overlap'):
        ScenarioEnv().reset(ScenarioSpec(branch='dual', occluder_x=.1, occluder_y=0))


def test_invalid_and_nonfinite_actions():
    env = ScenarioEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.full((2, 2), np.nan))
    env.bodies[1].x += 2
    _, reward, done, info = env.step(np.zeros((2, 2)))
    assert done and not info['valid'] and reward < 0 and not info['dangerous']


def test_replay_and_timing(tmp_path):
    path = tmp_path / 'trace.json'
    info = run_episode(ScriptPolicy(), ScenarioSpec(branch='dual'), 42, path)
    assert replay(path)['matched']
    assert info['elapsed'] <= 8.001
    assert info['decision_steps'] <= 80


def test_braking_reference_for_visible_stationary_object():
    env = ScenarioEnv()
    env.reset(ScenarioSpec(crossing_x=25, pedestrian_y=0, pedestrian_speed=0))
    while not env.done:
        env.step(np.zeros((2, 2)))
    assert env.bodies[0].speed == 0
    assert not env.collision
    assert env.first_brake_time is not None
