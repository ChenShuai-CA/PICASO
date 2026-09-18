"""Tests for the P3.3.4 Phase C conditional kinematic decode head (CPU, tiny)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenario_lab.p33_model import (ARSceneV1, ARSceneV1K, KINEMATIC_C_LIMITS,  # noqa: E402
                                    ar_scene_loss_c, to_torch_batch)

LIMITS = {type_id: limits for type_id, limits in KINEMATIC_C_LIMITS.items()}


def _tiny_config():
    return {
        "model": {"d_model": 16, "num_layers": 1, "num_attention_heads": 2,
                  "ffn_dimension": 8, "dropout": 0.0},
        "data": {
            "max_agents": 3, "history_steps": 4, "future_steps": 10,
            "sample_rate_hz": 10,
            "agent_type_vocabulary": {"pad": 0, "vehicle": 1, "pedestrian": 2,
                                      "cyclist": 3, "other": 4},
            "agent_role_vocabulary": {"pad": 0, "anchor": 1, "prediction_target": 2,
                                      "context": 3},
            "map_type_vocabulary": {f"map_{i}": i for i in range(8)},
            "normalization": {"position_m": 80.0, "velocity_mps": 30.0, "length_m": 10.0,
                              "width_m": 4.0, "curvature_inv_m": 0.2,
                              "speed_limit_mps": 40.0, "clip_absolute_value": 5.0},
        },
        "motion_tokens": {"chunk_steps": 5, "vocabulary_size": 128, "ignore_index": 255},
        "training": {"loss": {"motion_token_cross_entropy": 1.0,
                              "decoded_trajectory_huber": 0.5,
                              "endpoint_huber": 0.2}},
    }


def _tiny_batch(batch_size=2, seed=0):
    rng = np.random.default_rng(seed)
    agents, history_steps, future_steps = 3, 4, 10
    batch = {
        "agent_history": rng.normal(0, 1, (batch_size, agents, history_steps, 8)).astype(np.float32),
        "state_valid_mask": np.ones((batch_size, agents, history_steps), dtype=bool),
        "pairwise_visibility_mask": np.ones((batch_size, agents, history_steps, agents), dtype=bool),
        "agent_present_mask": np.ones((batch_size, agents), dtype=bool),
        "agent_type": np.tile(np.array([[1, 2, 3]]), (batch_size, 1)),
        "agent_role": np.tile(np.array([[1, 2, 3]]), (batch_size, 1)),
        "source_id": np.zeros(batch_size, dtype=np.int64),
        "map_polylines": rng.normal(0, 1, (batch_size, 4, 5, 6)).astype(np.float32),
        "map_point_mask": np.ones((batch_size, 4, 5), dtype=bool),
        "map_type": np.zeros((batch_size, 4), dtype=np.int64),
        "motion_token_target": rng.integers(0, 128, (batch_size, agents, 2)).astype(np.int64),
        "motion_token_valid_mask": np.ones((batch_size, agents, 2), dtype=bool),
        "future_xy": rng.normal(0, 1, (batch_size, agents, future_steps, 2)).astype(np.float32),
        "future_valid_mask": np.ones((batch_size, agents, future_steps), dtype=bool),
    }
    return to_torch_batch(batch)


def _model(c_config=None, seed=0):
    torch.manual_seed(seed)
    codebook = np.random.default_rng(1).normal(0, 1, (128, 10)).astype(np.float32)
    return ARSceneV1K(_tiny_config(), codebook, c_config)


def _implied(trajectories: np.ndarray):
    """Metric-convention finite differences on the output steps (anchor excluded)."""
    velocity = np.diff(trajectories, axis=-2) * 10.0
    accel = np.diff(velocity, axis=-2) * 10.0
    jerk = np.diff(accel, axis=-2) * 10.0
    return (np.linalg.norm(velocity, axis=-1), np.linalg.norm(accel, axis=-1),
            np.linalg.norm(jerk, axis=-1))


def test_construction_bounds_hold_per_agent_type():
    model, batch = _model(seed=3), _tiny_batch(seed=4)
    with torch.no_grad():
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = outputs["kinematic"]["trajectories"].numpy()
    types = batch["agent_type"].numpy().astype(int)
    _, accel, jerk = _implied(trajectories)
    for agent_type in np.unique(types):
        jerk_limit, accel_limit = LIMITS[int(agent_type)]
        mask = types == agent_type
        assert accel[mask].max() <= accel_limit + 0.01
        assert jerk[mask].max() <= jerk_limit + 0.1
    # global thresholds hold with real margin (10 m/s^2, 20 m/s^3)
    assert accel.max() < 9.6
    assert jerk.max() < 19.2


def test_zero_heads_give_exact_constant_velocity_continuation():
    model, batch = _model(seed=5), _tiny_batch(seed=6)
    with torch.no_grad():
        model.jerk_head.weight.zero_(); model.jerk_head.bias.zero_()
        model.init_head.weight.zero_(); model.init_head.bias.zero_()
        v0, _ = model._history_tail_velocity(batch)
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = outputs["kinematic"]["trajectories"].numpy()      # [B,A,10,2]
    anchor = batch["agent_history"][:, :, -1, :2].numpy()
    steps = np.arange(1, trajectories.shape[2] + 1)[:, None] * 0.1   # (i+1)*h
    expected = anchor[:, :, None, :] + steps[None, None] * v0.numpy()[:, :, None, :]
    assert np.allclose(trajectories, expected, atol=1e-4)


def test_history_tail_velocity_matches_metric_definition():
    model, batch = _model(), _tiny_batch(seed=7)
    history = np.zeros((1, 1, 4, 8), dtype=np.float32)
    history[0, 0, :, 0] = [0.0, 1.0, 2.0, 3.5]                        # x positions
    batch = {**batch, "agent_history": torch.tensor(history)}
    v0, usable = model._history_tail_velocity(batch)
    assert usable[0, 0]
    assert v0[0, 0].numpy() == pytest.approx(np.array([15.0, 0.0]), abs=1e-5)
    invalid = {**batch, "state_valid_mask": batch["state_valid_mask"].clone()}
    invalid["state_valid_mask"][0, 0, -1] = False
    v0b, usable_b = model._history_tail_velocity(invalid)
    assert not usable_b[0, 0]
    assert np.allclose(v0b[0, 0].numpy(), 0.0)


def test_start_jump_is_capped_below_the_accel_limit():
    model, batch = _model(seed=8), _tiny_batch(seed=9)
    with torch.no_grad():
        v0, _ = model._history_tail_velocity(batch)
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = outputs["kinematic"]["trajectories"].numpy()
    first_v = (trajectories[:, :, 1] - trajectories[:, :, 0]) * 10.0
    implied = np.linalg.norm(first_v - v0.numpy(), axis=-1) * 10.0
    assert implied.max() < 9.6  # construction cap 9.5 < the 10 m/s^2 metric limit


def test_rollout_matches_parallel_token_decode():
    model, batch = _model(seed=10), _tiny_batch(seed=11)
    samples = 2
    with torch.no_grad():
        rollout = model.rollout(batch, num_samples=samples, temperature=1.0,
                                top_p=0.95, seed=7)
        batch_rep = {key: (value.repeat_interleave(samples, dim=0)
                           if torch.is_tensor(value) else value)
                     for key, value in batch.items()}
        tokens = rollout["tokens"].reshape(-1, *rollout["tokens"].shape[2:])
        parallel = model.trajectory_from_tokens(tokens, batch_rep)
    incremental = rollout["trajectories"].reshape(-1, *rollout["trajectories"].shape[2:])
    assert torch.allclose(incremental, parallel, atol=1e-5)


def test_loss_terms_are_present_and_finite():
    model, batch = _model(seed=12), _tiny_batch(seed=13)
    outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    losses = ar_scene_loss_c(outputs, batch, model.codebook, _tiny_config())
    for key in ("total", "token_cross_entropy", "trajectory_huber", "endpoint_huber",
                "token_accuracy", "intent_huber", "speed_exceedance"):
        assert key in losses, key
        assert torch.isfinite(losses[key]), key
    assert losses["total"].requires_grad
    assert float(losses["total"]) > 0.0


def test_penalty_control_variant_is_unbounded_and_penalized():
    c_config = {"construction": "penalty"}
    model, batch = _model(c_config=c_config, seed=14), _tiny_batch(seed=15)
    with torch.no_grad():
        model.jerk_head.weight.mul_(50.0)          # force large raw jerk outputs
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    jerk_norm = np.linalg.norm(outputs["kinematic"]["jerk"].numpy(), axis=-1)
    assert jerk_norm.max() > 19.0                  # no construction bound
    losses = ar_scene_loss_c(outputs, batch, model.codebook, _tiny_config(), c_config)
    assert "jerk_exceedance" in losses and "accel_exceedance" in losses
    assert float(losses["jerk_exceedance"]) > 0.0
    assert torch.isfinite(losses["total"])


def test_parameter_count_stays_in_the_frozen_range():
    from scenario_lab.p33_spec import load_p33_config
    config = load_p33_config()
    codebook = np.zeros((config["motion_tokens"]["vocabulary_size"], 10), dtype=np.float32)
    torch.manual_seed(0)
    base = sum(parameter.numel() for parameter in ARSceneV1(config, codebook).parameters())
    kinematic = sum(parameter.numel() for parameter in ARSceneV1K(config, codebook).parameters())
    low, high = config["model"]["parameter_count_range"]
    assert low <= kinematic <= high
    assert 0 < kinematic - base < 300_000  # head swap only, frozen scale untouched
