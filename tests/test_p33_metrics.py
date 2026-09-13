"""P3.3.2 metric contract tests: CV hand calculations, min@K, group aggregation."""
import numpy as np
import pytest

from scenario_lab.p33_metrics import (
    constant_velocity_prediction,
    displacement_errors,
    evaluate_agent_mask,
    min_k_displacement_errors,
    summarize_records,
    AgentErrorRecord,
    group_level_table,
    kinematic_diagnostics,
    FUTURE_TIMES,
)


def _history(position, velocity, valid_last=True):
    """[1 agent, 11 steps, 8 features]; ``position`` is the t = 0 state."""
    history = np.zeros((1, 11, 8), dtype=np.float32)
    times = (np.arange(11) - 10) * 0.1  # -1.0 ... 0.0 s
    history[0, :, :2] = position + velocity * times[:, None]
    history[0, :, 2:4] = velocity
    history[0, :, 4] = 1.0
    history[0, :, 6:8] = (4.5, 1.8)
    valid = np.ones((1, 11), dtype=bool)
    if not valid_last:
        valid[0, -1] = False
        history[0, -1] = 0.0
    return history, valid


def test_cv_uses_last_valid_state_and_extrapolates():
    # Accelerating history makes the last-valid timestamp observable in the
    # extrapolation: CV must run forward from t_last = -0.1 s, not from t = 0.
    position, velocity, accel = np.array([1.0, -2.0]), np.array([2.0, 0.5]), np.array([0.6, 0.0])
    times = (np.arange(11) - 10) * 0.1
    history = np.zeros((1, 11, 8), dtype=np.float32)
    history[0, :, :2] = position + velocity * times[:, None] + 0.5 * accel * times[:, None] ** 2
    history[0, :, 2:4] = velocity + accel * times[:, None]
    history[0, :, 4] = 1.0
    valid = np.ones((1, 11), dtype=bool)
    valid[0, -1] = False  # t = 0 unobserved; last valid sits at t = -0.1 s
    history[0, -1] = 0.0
    prediction = constant_velocity_prediction(history, valid)
    assert prediction.shape == (1, 50, 2)
    t_last = -0.1
    p_last = position + velocity * t_last + 0.5 * accel * t_last ** 2
    v_last = velocity + accel * t_last
    expected = p_last + v_last * (FUTURE_TIMES - t_last)[:, None]
    np.testing.assert_allclose(prediction[0], expected, atol=1e-5)
    # and it genuinely differs from a (wrong) t = 0 extrapolation:
    wrong = position + velocity * FUTURE_TIMES[:, None]
    assert np.abs(prediction[0] - wrong).max() > 1e-3


def test_cv_ade_fde_hand_computed():
    position, velocity = np.array([0.0, 0.0]), np.array([1.0, 2.0])
    history, valid = _history(position, velocity)
    prediction = constant_velocity_prediction(history, valid)
    # Ground truth drifts 0.1*t in y away from the exact CV line.
    truth = position + velocity * FUTURE_TIMES[:, None] + np.c_[np.zeros(50), 0.1 * FUTURE_TIMES]
    errors = displacement_errors(prediction, truth[None], np.ones((1, 50), dtype=bool))
    assert errors["ade"][0] == pytest.approx(np.mean(0.1 * FUTURE_TIMES), abs=1e-6)
    assert errors["fde"][0] == pytest.approx(0.5, abs=1e-6)
    assert errors["valid_steps"][0] == 50
    assert errors["endpoint_valid"][0]


def test_fde_undefined_when_endpoint_invalid():
    truth = np.zeros((1, 50, 2))
    valid = np.ones((1, 50), dtype=bool)
    valid[0, -3:] = False
    errors = displacement_errors(np.zeros((1, 50, 2)), truth, valid)
    assert not errors["endpoint_valid"][0]
    assert np.isnan(errors["fde"][0])
    assert errors["valid_steps"][0] == 47


def test_min_at_k_with_identical_samples_equals_single_sample():
    position, velocity = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    history, valid = _history(position, velocity)
    prediction = constant_velocity_prediction(history, valid)
    truth = truth = position + velocity * FUTURE_TIMES[:, None] + 0.05
    mask = np.ones((1, 50), dtype=bool)
    single = displacement_errors(prediction, truth[None], mask)
    stacked = np.stack([prediction] * 6)  # [K=6, 1 agent, 50, 2]
    best = min_k_displacement_errors(stacked, truth[None], mask)
    assert best["min_ade"][0] == pytest.approx(single["ade"][0], abs=1e-9)
    assert best["min_fde"][0] == pytest.approx(single["fde"][0], abs=1e-9)
    # One exact sample among six drives min metrics to zero.
    stacked[3, 0] = truth
    best = min_k_displacement_errors(stacked, truth[None], mask)
    assert best["min_ade"][0] == pytest.approx(0.0, abs=1e-9)
    assert best["min_fde"][0] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_agent_mask_follows_role_and_future():
    present = np.array([True, True, True, True, False])
    roles = np.array([1, 2, 3, 1, 0])
    future_valid = np.zeros((5, 50), dtype=bool)
    future_valid[0] = True
    future_valid[2] = True  # context agent with future: still excluded (role 3)
    mask = evaluate_agent_mask(present, roles, future_valid)
    np.testing.assert_array_equal(mask, [True, False, False, False, False])


def _record(source, group, ade, fde=1.0, agent_type=1, agent_role=1,
            agents_truncated=False, map_truncated=False, endpoint_valid=True):
    return AgentErrorRecord(
        source=source, group_id=group, sample_id=f"{source}-{group}-{ade}", agent_slot=0,
        agent_type=agent_type, agent_role=agent_role, agents_truncated=agents_truncated,
        map_truncated=map_truncated, ade=ade, fde=fde if endpoint_valid else float("nan"),
        min_ade=ade, min_fde=fde if endpoint_valid else float("nan"),
        valid_steps=50, endpoint_valid=endpoint_valid)


def test_group_level_aggregation_uses_group_means():
    records = [
        _record("interaction", "LOC::001", 1.0),
        _record("interaction", "LOC::001", 3.0),  # overlapping window, same case
        _record("interaction", "LOC::002", 5.0),
    ]
    table = group_level_table(records)
    interaction = table["interaction"]
    assert interaction["groups"] == 2
    assert interaction["ade"] == pytest.approx(((1.0 + 3.0) / 2 + 5.0) / 2)  # mean of group means
    assert interaction["pooled_ade"] == pytest.approx(3.0)  # agent-weighted


def test_summarize_stratifies_by_source_type_role_truncation():
    records = [
        _record("waymo", "s1", 1.0, agent_type=1, agent_role=1),
        _record("waymo", "s2", 2.0, agent_type=2, agent_role=2, agents_truncated=True),
        _record("interaction", "LOC::001", 4.0, map_truncated=True, endpoint_valid=False),
    ]
    summary = summarize_records(records)
    assert summary["group_level"]["waymo"]["groups"] == 2
    assert summary["group_level"]["interaction"]["groups"] == 1
    assert summary["strata"]["source"]["waymo"]["ade"] == pytest.approx(1.5)
    assert summary["strata"]["agent_type"]["2"]["ade"] == pytest.approx(2.0)
    assert summary["strata"]["agent_role"]["2"]["ade"] == pytest.approx(2.0)
    assert summary["strata"]["agents_truncated"]["True"]["count"] == 1
    assert summary["strata"]["map_truncated"]["True"]["count"] == 1
    assert summary["counts"]["agents"] == 3
    assert summary["counts"]["endpoint_missing"] == 1


def test_kinematic_diagnostics_flags_thresholds():
    times = FUTURE_TIMES
    straight = np.c_[2.0 * times, np.zeros(50)][None, None]  # [B=1,A=1,50,2]
    insane = np.c_[np.cumsum(np.linspace(0.0, 30.0, 50)), np.zeros(50)][None, None]
    report = kinematic_diagnostics(np.concatenate([straight, insane], axis=1))
    assert report["speed_limit_mps"] == 35.0
    assert report["accel_limit_mps2"] == 10.0
    assert report["jerk_limit_mps3"] == 20.0
    assert report["agent_count"] == 2
    assert report["speed_violations"] >= 1
    assert report["accel_violations"] >= 1
