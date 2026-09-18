"""Tests for the P3.3.4 attribution metrics (per-step rates, quantiles, continuity)."""
import numpy as np
import pytest

from scenario_lab.p33_metrics import kinematic_attribution_metrics


def _constant_velocity_trajectory(n: int = 2, frames: int = 50, speed: float = 2.0):
    """Clean straight-line trajectories at ``speed`` m/s (10 Hz)."""
    t = np.arange(frames)[None, :, None] / 10.0
    return np.concatenate([t * speed, np.zeros_like(t)], axis=-1).repeat(n, axis=0)


def test_clean_trajectory_has_zero_violations_all_rates():
    trajectories = _constant_velocity_trajectory()
    valid = np.ones_like(trajectories[:, :, 0], dtype=bool)
    report = kinematic_attribution_metrics(trajectories, valid)
    for kind in ("speed", "accel", "jerk"):
        assert report[kind]["traj_rate"] == 0.0
        assert report[kind]["step_rate"] == 0.0
        assert report[kind]["exceedance_quantiles"] is None


def test_per_step_rate_counts_fractions_not_trajectories():
    # one agent: 49 speed samples, exactly one violating step (38 m/s for one step)
    trajectories = _constant_velocity_trajectory(n=1)
    trajectories[0, 3, 0] += 3.6  # x3 jumps 0.4 -> 4.2: v2=+38, v3=-34 m/s
    valid = np.ones((1, 50), dtype=bool)
    report = kinematic_attribution_metrics(trajectories, valid)
    assert report["speed"]["traj_rate"] == 1.0
    assert report["speed"]["violating_steps"] == 1
    assert report["speed"]["step_rate"] == 1 / 49
    # the single velocity spike +38/-34/... spans three accel steps (360/-720/360)
    assert report["accel"]["traj_rate"] == 1.0
    assert report["accel"]["violating_steps"] == 3
    assert report["accel"]["step_rate"] == 3 / 48


def test_exceedance_quantiles_and_limits():
    trajectories = _constant_velocity_trajectory(n=1)
    trajectories[0, 3, 0] += 3.6   # v2 = 38  -> exceedance 3.0
    trajectories[0, 9, 0] += 4.6   # v8 = 48 -> 13.0; v9 = 44 -> 9.0
    valid = np.ones((1, 50), dtype=bool)
    quantiles = kinematic_attribution_metrics(trajectories, valid)["speed"]["exceedance_quantiles"]
    # exceedances {3.0, 9.0, 13.0}: median 9, max 13, interpolated p95 in between
    assert quantiles["max"] == pytest.approx(13.0)
    assert quantiles["p50"] == pytest.approx(9.0)
    assert 9.0 < quantiles["p95"] < 13.0


def test_step_validity_masks_spanning_derivatives():
    # a spike hidden behind an invalid step must not be counted at any order
    trajectories = _constant_velocity_trajectory(n=1)
    trajectories[0, 3, 0] += 3.6
    valid = np.ones((1, 50), dtype=bool)
    valid[0, 3] = False
    report = kinematic_attribution_metrics(trajectories, valid)
    for kind in ("speed", "accel", "jerk"):
        assert report[kind]["violating_steps"] == 0
        assert report[kind]["traj_rate"] == 0.0


def test_continuity_block_against_history_velocity():
    trajectories = _constant_velocity_trajectory(n=2, speed=2.0)
    valid = np.ones((2, 50), dtype=bool)
    # agent 0: history velocity matches first step (2 m/s) -> no jump
    # agent 1: history velocity 10 m/s opposed -> implied accel (12 * 10) >> 10 m/s^2
    history = np.array([[2.0, 0.0], [-10.0, 0.0]])
    continuity = kinematic_attribution_metrics(trajectories, valid, history)["continuity"]
    assert continuity["agents_checked"] == 2
    assert continuity["implied_accel_over_limit_rate"] == 0.5
    jumps = continuity["jump_mps_quantiles"]
    assert jumps["max"] == 12.0


def test_nan_history_rows_excluded_from_continuity_only():
    trajectories = _constant_velocity_trajectory(n=2)
    valid = np.ones((2, 50), dtype=bool)
    history = np.array([[np.nan, np.nan], [2.0, 0.0]])
    continuity = kinematic_attribution_metrics(trajectories, valid, history)["continuity"]
    assert continuity["agents_checked"] == 1


def test_invalid_step_row_agent_excluded_from_traj_rate_denominator():
    trajectories = _constant_velocity_trajectory(n=2)
    valid = np.ones((2, 50), dtype=bool)
    valid[1] = False  # agent 1 has no valid steps at all
    trajectories[0, 3, 0] += 3.6
    report = kinematic_attribution_metrics(trajectories, valid)
    assert report["speed"]["agents_with_steps"] == 1
    assert report["speed"]["traj_rate"] == 1.0
