"""P3.2: calibrated perturbation config consumption (abd_calibrated_v1)."""
import json
from pathlib import Path

import numpy as np
import pytest

from scenario_lab.env import sample_spec
from scenario_lab.schema import ScenarioSpec
from scenario_lab.train import load_perturb_config, perturb_spec


def make_config(tmp_path, **overrides):
    params = {
        'brake_deceleration': {'dist': 'uniform', 'low': 9.128, 'high': 11.419},
        'response_delay': {'dist': 'uniform', 'low': 0.055, 'high': 0.386},
        'action_delay_steps': {'dist': 'integers', 'low': 0, 'high': 2},
        'target_accel_scale': {'dist': 'uniform', 'low': 0.85, 'high': 1.15},
    }
    params.update(overrides)
    path = tmp_path / 'perturb.json'
    path.write_text(json.dumps({'version': 'abd_calibrated_v1',
                                'parameters': params}), encoding='utf-8')
    return path


def test_calibrated_draws_respect_measured_bounds(tmp_path):
    config = load_perturb_config(make_config(tmp_path))
    rng = np.random.default_rng(0)
    spec = sample_spec(np.random.default_rng(1), 'dual', 0)
    for _ in range(500):
        s = perturb_spec(spec, rng, calibrated=config)
        assert 9.128 <= s.brake_deceleration <= 11.419
        assert 0.055 <= s.response_delay <= 0.386
        assert s.action_delay_steps in (0, 1, 2)
        assert 0.85 <= s.target_accel_scale <= 1.15
        assert s.perturbation_source == 'abd_calibrated_v1_partial'
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


def test_shipped_v1_config_consumable():
    """The real abd_calibrated_v1.json produced by scripts/calibrate_abd_v1.py."""
    path = Path(__file__).resolve().parents[1] / (
        'runs/20260912_abd_calibration/abd_calibrated_v1.json')
    if not path.exists():
        pytest.skip('calibration output not generated yet')
    config = load_perturb_config(path)
    rng = np.random.default_rng(7)
    spec = ScenarioSpec()
    s = perturb_spec(spec, rng, calibrated=config)
    p = config['parameters']
    assert p['brake_deceleration']['low'] <= s.brake_deceleration <= p['brake_deceleration']['high']
    assert p['response_delay']['low'] <= s.response_delay <= p['response_delay']['high']
    # measured AEB decel domain lies entirely above the assumed U(5.5, 8.0)
    assert p['brake_deceleration']['low'] > 8.0
    assert s.perturbation_source == 'abd_calibrated_v1_partial'
