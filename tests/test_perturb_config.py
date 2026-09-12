"""P3.2: versioned perturbation config consumption."""
import json
from pathlib import Path

import numpy as np
import pytest

from scenario_lab.env import sample_spec
from scenario_lab.schema import ScenarioSpec
from scenario_lab.train import load_perturb_config, perturb_spec


def make_config(tmp_path, **overrides):
    params = {
        'brake_deceleration': {'dist': 'uniform', 'low': 5.455, 'high': 8.174},
        'response_delay': {'dist': 'uniform', 'low': 0.1, 'high': 0.4},
        'action_delay_steps': {'dist': 'integers', 'low': 0, 'high': 2},
        'target_accel_scale': {'dist': 'uniform', 'low': 0.85, 'high': 1.15},
    }
    params.update(overrides)
    path = tmp_path / 'perturb.json'
    path.write_text(json.dumps({'version': 'abd_supported_v1',
                                'parameters': params}), encoding='utf-8')
    return path


def test_versioned_draws_respect_bounds(tmp_path):
    config = load_perturb_config(make_config(tmp_path))
    rng = np.random.default_rng(0)
    spec = sample_spec(np.random.default_rng(1), 'dual', 0)
    for _ in range(500):
        s = perturb_spec(spec, rng, calibrated=config)
        assert 5.455 <= s.brake_deceleration <= 8.174
        assert 0.1 <= s.response_delay <= 0.4
        assert s.action_delay_steps in (0, 1, 2)
        assert 0.85 <= s.target_accel_scale <= 1.15
        assert s.perturbation_source == 'abd_supported_v1_partial'
        # base spec untouched
        assert spec.perturbation_source == 'assumed_sensitivity_not_abd_calibrated'


def test_default_path_unchanged():
    rng = np.random.default_rng(0)
    spec = sample_spec(rng, 'dual', 0)
    for _ in range(100):
        s = perturb_spec(spec, rng)
        assert 5.5 <= s.brake_deceleration <= 8.0
        assert 0.1 <= s.response_delay <= 0.4
        assert s.perturbation_source == 'assumed_sensitivity_not_abd_calibrated'


def test_load_rejects_missing_bounds(tmp_path):
    path = make_config(tmp_path, brake_deceleration={'dist': 'uniform'})
    with pytest.raises(ValueError, match='brake_deceleration'):
        load_perturb_config(path)


@pytest.mark.parametrize('override', [
    {'response_delay': {'dist': 'uniform', 'low': 0.4, 'high': 0.1}},
    {'brake_deceleration': {'dist': 'normal', 'low': 5.0, 'high': 8.0}},
    {'action_delay_steps': {'dist': 'integers', 'low': 0.0, 'high': 2}},
])
def test_load_rejects_invalid_distributions_and_bounds(tmp_path, override):
    with pytest.raises(ValueError):
        load_perturb_config(make_config(tmp_path, **override))


def test_shipped_v1_config_consumable():
    """The reviewed config produced by scripts/calibrate_abd_v1.py."""
    path = Path(__file__).resolve().parents[1] / (
        'runs/20260912_abd_calibration/abd_supported_v1.json')
    if not path.exists():
        pytest.skip('calibration output not generated yet')
    config = load_perturb_config(path)
    rng = np.random.default_rng(7)
    spec = ScenarioSpec()
    s = perturb_spec(spec, rng, calibrated=config)
    p = config['parameters']
    assert p['brake_deceleration']['low'] <= s.brake_deceleration <= p['brake_deceleration']['high']
    assert p['response_delay']['low'] <= s.response_delay <= p['response_delay']['high']
    assert p['brake_deceleration']['status'].startswith('abd_supported_')
    assert p['brake_deceleration']['low'] == pytest.approx(5.455)
    assert p['brake_deceleration']['high'] == pytest.approx(8.174)
    assert p['response_delay']['status'].startswith('retained_assumed_')
    assert (p['response_delay']['low'], p['response_delay']['high']) == (0.1, 0.4)
    assert s.perturbation_source == 'abd_supported_v1_partial'


def test_split_delay_config_does_not_overwrite_legacy_delay(tmp_path):
    params = {
        'brake_deceleration': {'dist': 'uniform', 'low': 6.0, 'high': 8.0},
        'aeb_actuation_delay': {'dist': 'uniform', 'low': 0.15, 'high': 0.35},
        'controller_preview_delay': {'dist': 'uniform', 'low': 0.25, 'high': 0.25},
        'action_delay_steps': {'dist': 'integers', 'low': 0, 'high': 2},
        'target_accel_scale': {'dist': 'uniform', 'low': 0.85, 'high': 1.15},
    }
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'version': 'abd_split_v2', 'parameters': params}),
                    encoding='utf-8')
    config = load_perturb_config(path)
    spec = ScenarioSpec(response_delay=0.4)
    sampled = perturb_spec(spec, np.random.default_rng(9), calibrated=config)
    assert 0.15 <= sampled.aeb_actuation_delay <= 0.35
    assert sampled.controller_preview_delay == pytest.approx(0.25)
    assert sampled.response_delay == pytest.approx(0.4)


def test_split_delay_config_requires_controller_preview(tmp_path):
    path = make_config(tmp_path)
    config = json.loads(path.read_text(encoding='utf-8'))
    config['parameters']['aeb_actuation_delay'] = config['parameters'].pop(
        'response_delay')
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='controller_preview_delay'):
        load_perturb_config(path)
