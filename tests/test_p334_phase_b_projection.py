"""Tests for the P3.3.4 Phase B constrained projection optimizer (CPU, tiny)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research_tasks"))
from p334_phase_b import project_trajectories  # noqa: E402


def _constant_velocity(frames: int = 50, speed: float = 2.0) -> np.ndarray:
    t = np.arange(frames)[:, None] / 10.0
    return np.concatenate([t * speed, np.zeros_like(t)], axis=1)[None]  # [1, T, 2]


def _run(pred: np.ndarray, valid: np.ndarray | None = None, v0=None, **kwargs):
    valid = (np.ones((pred.shape[0], pred.shape[1]), dtype=bool)
             if valid is None else valid)
    v0 = (np.full((pred.shape[0], 2), np.nan) if v0 is None else v0)
    defaults = dict(iters=60)
    defaults.update(kwargs)
    return project_trajectories(torch.tensor(pred, dtype=torch.float32),
                                torch.tensor(valid),
                                torch.tensor(np.asarray(v0, dtype=np.float32)),
                                **defaults)


def test_warm_start_is_feasible_by_construction():
    import torch as torch_
    from p334_phase_b import _feasible_warm_start
    pred = _constant_velocity()
    pred[0, 3, 0] += 0.5                       # oscillatory junk for the filter
    pred[0, 20, 0] += 0.3
    warm = _feasible_warm_start(
        torch_.tensor(pred, dtype=torch_.float32), torch_.zeros(1, 2),
        torch_.tensor([False]),
        {"speed_limit_mps": 35.0, "accel_limit_mps2": 10.0, "jerk_limit_mps3": 20.0}
    ).numpy()
    v = np.diff(warm, axis=1) * 10
    a = np.diff(v, axis=1) * 10
    j = np.diff(a, axis=1) * 10
    assert np.linalg.norm(v, axis=-1).max() <= 35.0
    assert np.linalg.norm(a, axis=-1).max() <= 10.0
    assert np.linalg.norm(j, axis=-1).max() <= 20.0
    # first velocity equals the history-tail velocity where finite
    warm2 = _feasible_warm_start(
        torch_.tensor(pred, dtype=torch_.float32),
        torch_.tensor([[-3.0, 1.0]]), torch_.tensor([True]),
        {"speed_limit_mps": 35.0, "accel_limit_mps2": 10.0, "jerk_limit_mps3": 20.0}
    ).numpy()
    assert warm2[0, 1] - warm2[0, 0] == pytest.approx(np.array([-3.0, 1.0]) / 10.0,
                                                      abs=1e-6)


def test_spike_is_removed_and_start_pinned():
    pred = _constant_velocity()
    pred[0, 3, 0] += 0.5  # one-frame 5 m/s velocity impulse
    result = _run(pred)
    assert not any(result["post_violation_rows"].values())
    # start point is a hard equality
    assert result["trajectories"][0, 0] == pytest.approx(pred[0, 0], abs=1e-6)
    # repairing a 5 m/s impulse under jerk<=20 (|da|<=2 m/s^2 per step) forces a
    # ~10-frame S-curve: the displacement floor is ~1 m, NOT the spike size
    displacement = np.linalg.norm(result["trajectories"] - pred, axis=-1)
    assert displacement.max() < 1.5
    assert displacement[0, 3] > 1e-3


def test_clean_trajectory_barely_moves():
    pred = _constant_velocity()
    result = _run(pred, iters=200)
    displacement = np.linalg.norm(result["trajectories"] - pred, axis=-1)
    assert displacement.max() < 1e-2
    assert not any(result["post_violation_rows"].values())


def test_continuity_constraint_pulls_first_step_toward_history():
    pred = _constant_velocity()                     # first step +2 m/s
    v0 = [[-6.0, 0.0]]                              # implied accel 80 m/s^2
    result = _run(pred, v0=v0)
    first_v = (result["trajectories"][0, 1] - result["trajectories"][0, 0]) * 10.0
    implied = np.linalg.norm(first_v - np.array([-6.0, 0.0])) * 10.0
    assert implied < 11.0  # within 10% of the 10 m/s^2 limit


def test_contradictory_limits_are_counted_not_hidden():
    # history demands |v_first - 50| <= 1 m/s but speed cap 35 forbids it
    pred = _constant_velocity()
    v0 = [[50.0, 0.0]]
    result = _run(pred, v0=v0)
    failures = np.zeros(1, dtype=bool)
    for mask in result["post_violation_rows"].values():
        failures |= mask
    assert failures[0]  # solver_failure accounting, never silently dropped


def test_invalid_steps_are_exempt_from_data_and_penalties():
    pred = _constant_velocity()
    pred[0, 3, 0] += 0.5
    valid = np.ones((1, 50), dtype=bool)
    valid[0, 3] = False  # spike hidden behind an invalid frame
    result = _run(pred, valid=valid)
    # no penalty sees the hidden spike (it would violate accel/jerk if visible);
    # the data term does not pull frame 3, though the warm start may still move
    # it as part of global re-integration -- invalid frames carry no metrics
    assert not any(result["post_violation_rows"].values())
    assert result["trajectories"][0, 0] == pytest.approx(pred[0, 0], abs=1e-6)
