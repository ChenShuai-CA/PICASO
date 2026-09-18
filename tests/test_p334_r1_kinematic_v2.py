"""P3.3.4-R1 regressions for the C-v2 kinematic head (CPU, tiny).

Every test here encodes a defect the CODEX review demonstrated in v1 or a
guarantee SPEC_R1 (runs/20260916_p334_r1/SPEC_R1.md section 2) claims:
current-token conditioning, the joint start-pair construction, and the
recorded-v0 policy.  The v1 configurations are asserted where they must stay
defective or unchanged, so the flags' gating is itself under test.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenario_lab.p33_model import (ARSceneV1K, KINEMATIC_C_LIMITS,  # noqa: E402
                                    ar_scene_loss_c, to_torch_batch)

LIMITS = {type_id: limits for type_id, limits in KINEMATIC_C_LIMITS.items()}
V2_CONFIG = {"construction": "bounded", "token_conditioning": True,
             "start_pair_joint": True, "v0_source": "recorded",
             "v0_speed_limit_mps": 35.0}


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


def _saturate_start(model):
    """init -> (A, 0) exactly, jerk -> 0, so a_1 = a_2 = (A, 0): the CODEX
    saturated-start counterexample family."""
    with torch.no_grad():
        model.init_head.weight.zero_()
        model.init_head.bias.copy_(torch.tensor([1000.0, 0.0]))
        model.jerk_head.weight.zero_()
        model.jerk_head.bias.zero_()


def _implied_start_accel(trajectories: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Frozen-metric start term: |(P2-P1)*rate - v_hist| * rate = |a1 + a2|."""
    second_v = (trajectories[:, :, 1] - trajectories[:, :, 0]) * 10.0
    return np.linalg.norm(second_v - v0, axis=-1) * 10.0


# --------------------------------------------------------------- start pair
def test_joint_start_pair_bounds_the_sum_under_saturation():
    """CODEX P1-1: v1 scaled a2 only, so a1=9.5 + a2=4.75 -> implied 14.25."""
    batch = _tiny_batch(seed=4)
    v0 = batch["agent_history"][:, :, -1, 2:4].numpy()      # recorded source

    model_v2 = _model(V2_CONFIG, seed=3)
    _saturate_start(model_v2)
    with torch.no_grad():
        out = model_v2(batch, teacher_tokens=batch["motion_token_target"])
    implied = _implied_start_accel(out["kinematic"]["trajectories"].numpy(), v0)
    assert implied.max() <= 9.5 + 1e-4                       # joint construction holds

    model_v1 = _model(None, seed=3)                          # sealed v1 behavior
    _saturate_start(model_v1)
    with torch.no_grad():
        out_v1 = model_v1(batch, teacher_tokens=batch["motion_token_target"])
    v1_v0 = model_v1._history_tail_velocity(batch)[0].numpy()
    implied_v1 = _implied_start_accel(out_v1["kinematic"]["trajectories"].numpy(), v1_v0)
    assert implied_v1[:, 0].max() > 14.0                     # vehicle slot: 14.25 defect
    assert implied_v1[:, 0].max() > implied[:, 0].max() + 4.0


def test_joint_start_pair_handles_opposed_directions_and_types():
    """a1 and a2 opposed / different agent types keep every term in its disk."""
    batch = _tiny_batch(seed=5)
    model = _model(V2_CONFIG, seed=6)
    with torch.no_grad():
        model.init_head.weight.zero_()
        model.init_head.bias.copy_(torch.tensor([-1000.0, 1000.0]))   # saturated, oblique
        model.jerk_head.weight.zero_()
        model.jerk_head.bias.zero_()
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = out["kinematic"]["trajectories"].numpy()
    v0 = batch["agent_history"][:, :, -1, 2:4].numpy()
    assert (_implied_start_accel(trajectories, v0).max() <= 9.5 + 1e-4)
    velocity = np.diff(trajectories, axis=-2) * 10.0
    accel = np.diff(velocity, axis=-2) * 10.0
    jerk = np.diff(accel, axis=-2) * 10.0
    accel_norm, jerk_norm = np.linalg.norm(accel, axis=-1), np.linalg.norm(jerk, axis=-1)
    for agent_type in np.unique(batch["agent_type"].numpy()):
        jerk_limit, accel_limit = LIMITS[int(agent_type)]
        mask = batch["agent_type"].numpy() == agent_type
        assert accel_norm[mask].max() <= accel_limit + 0.01
        assert jerk_norm[mask].max() <= jerk_limit + 0.1


# ------------------------------------------------------ token conditioning
def test_current_token_flips_its_own_chunk():
    """CODEX P1-2a: with token_conditioning the CURRENT chunk responds."""
    model = _model(V2_CONFIG, seed=10)
    batch = _tiny_batch(seed=11)
    tokens = batch["motion_token_target"].clone()
    with torch.no_grad():
        base = model.trajectory_from_tokens(tokens, batch)
        flipped = tokens.clone()
        flipped[:, :, 0] = (flipped[:, :, 0] + 1) % 128
        other = model.trajectory_from_tokens(flipped, batch)
    assert (base[:, :, 0:5] - other[:, :, 0:5]).abs().max().item() > 1e-6

    model_v1 = _model(None, seed=10)                        # sealed v1: no path
    with torch.no_grad():
        base_v1 = model_v1.trajectory_from_tokens(tokens, batch)
        other_v1 = model_v1.trajectory_from_tokens(flipped, batch)
    assert torch.equal(base_v1[:, :, 0:5], other_v1[:, :, 0:5])   # delta exactly 0


def test_last_token_has_a_live_compute_path():
    """CODEX P1-2b: v1's final-chunk intervention delta was 0.0."""
    model = _model(V2_CONFIG, seed=12)
    batch = _tiny_batch(seed=13)
    tokens = batch["motion_token_target"].clone()
    with torch.no_grad():
        base = model.trajectory_from_tokens(tokens, batch)
        flipped = tokens.clone()
        flipped[:, :, -1] = (flipped[:, :, -1] + 1) % 128
        other = model.trajectory_from_tokens(flipped, batch)
    assert (base[:, :, 5:10] - other[:, :, 5:10]).abs().max().item() > 1e-6

    model_v1 = _model(None, seed=12)
    with torch.no_grad():
        base_v1 = model_v1.trajectory_from_tokens(tokens, batch)
        other_v1 = model_v1.trajectory_from_tokens(flipped, batch)
    assert torch.equal(base_v1[:, :, 5:10], other_v1[:, :, 5:10])


def test_rollout_matches_parallel_token_decode_v2():
    """The unified interface: rollout feeds its sampled current token through
    exactly the path trajectory_from_tokens uses for given tokens."""
    model = _model(V2_CONFIG, seed=14)
    batch = _tiny_batch(seed=15)
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


def test_ignore_token_maps_to_the_zero_centroid():
    """Teacher 255 (invalid chunk) conditions on the zero BOS row, not an OOB
    index, and stays finite."""
    model = _model(V2_CONFIG, seed=16)
    batch = _tiny_batch(seed=17)
    tokens = batch["motion_token_target"].clone()
    tokens[:, :, 0] = 255
    with torch.no_grad():
        trajectories = model.trajectory_from_tokens(tokens, batch)
    assert torch.isfinite(trajectories).all()


# -------------------------------------------------------------- v0 policy
def test_extreme_recorded_v0_is_disk_clamped_and_counted():
    """The 613 m/s family: recorded channels sane -> clamp to 35 + count."""
    model = _model(V2_CONFIG, seed=18)
    batch = _tiny_batch(seed=19)
    batch["agent_history"][0, 0, -1, 2:4] = torch.tensor([613.0, 0.0])
    v0, usable, over_limit = model._history_tail_velocity_recorded(batch)
    assert usable[0, 0] and over_limit[0, 0]
    assert torch.allclose(v0[0, 0], torch.tensor([35.0, 0.0]), atol=1e-5)
    assert not over_limit[0, 1:].any() and not over_limit[1].any()
    with torch.no_grad():
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    assert torch.isfinite(out["kinematic"]["trajectories"]).all()


def test_missing_history_tail_is_unusable_in_recorded_mode():
    model = _model(V2_CONFIG, seed=20)
    batch = _tiny_batch(seed=21)
    batch["state_valid_mask"][0, 0, -1] = False
    v0, usable, over_limit = model._history_tail_velocity_recorded(batch)
    assert not usable[0, 0] and not over_limit[0, 0]
    assert torch.allclose(v0[0, 0], torch.zeros(2))
    assert usable[0, 1:].all() and usable[1].all()


def test_zero_heads_give_cv_continuation_at_the_recorded_v0():
    model = _model(V2_CONFIG, seed=22)
    batch = _tiny_batch(seed=23)
    recorded = torch.tensor([30.0, 0.0])
    batch["agent_history"][..., -1, 2:4] = recorded            # under the 35 clamp
    with torch.no_grad():
        model.jerk_head.weight.zero_(); model.jerk_head.bias.zero_()
        model.init_head.weight.zero_(); model.init_head.bias.zero_()
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = out["kinematic"]["trajectories"].numpy()
    anchor = batch["agent_history"][:, :, -1, :2].numpy()
    steps = np.arange(1, trajectories.shape[2] + 1)[:, None] * 0.1
    expected = anchor[:, :, None, :] + steps[None, None] * recorded.numpy()[None, None]
    assert np.allclose(trajectories, expected, atol=1e-4)


# ------------------------------------------------- construction + v1 sealing
def test_construction_bounds_hold_per_agent_type_v2():
    model = _model(V2_CONFIG, seed=24)
    batch = _tiny_batch(seed=25)
    with torch.no_grad():
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = outputs["kinematic"]["trajectories"].numpy()
    types = batch["agent_type"].numpy().astype(int)
    velocity = np.diff(trajectories, axis=-2) * 10.0
    accel = np.diff(velocity, axis=-2) * 10.0
    jerk = np.diff(accel, axis=-2) * 10.0
    accel_norm, jerk_norm = np.linalg.norm(accel, axis=-1), np.linalg.norm(jerk, axis=-1)
    for agent_type in np.unique(types):
        jerk_limit, accel_limit = LIMITS[int(agent_type)]
        mask = types == agent_type
        assert accel_norm[mask].max() <= accel_limit + 0.01
        assert jerk_norm[mask].max() <= jerk_limit + 0.1
    assert np.linalg.norm(accel, axis=-1).max() < 9.6
    assert np.linalg.norm(jerk, axis=-1).max() < 19.2


def test_v1_tail_velocity_signature_and_values_are_unchanged():
    """Flags absent -> the sealed 2-tuple diff-based v1 helper, same numbers."""
    model = _model(None, seed=26)
    batch = _tiny_batch(seed=27)
    history = np.zeros((1, 1, 4, 8), dtype=np.float32)
    history[0, 0, :, 0] = [0.0, 1.0, 2.0, 3.5]
    batch = {**batch, "agent_history": torch.tensor(history)}
    result = model._history_tail_velocity(batch)
    assert len(result) == 2
    v0, usable = result
    assert usable[0, 0]
    assert v0[0, 0].numpy() == pytest.approx(np.array([15.0, 0.0]), abs=1e-5)
    model_v2 = _model(V2_CONFIG, seed=26)
    assert len(model_v2._history_tail_velocity_recorded(batch)) == 3


# ----------------------------------------------------------- penalty branch
def test_penalty_variant_masks_exceedance_by_validity():
    """Plan section 2.4: pad/invalid steps must not pollute the penalty mean."""
    c_config = {"construction": "penalty", "token_conditioning": True,
                "start_pair_joint": True, "v0_source": "recorded"}
    model = _model(c_config=c_config, seed=28)
    batch = _tiny_batch(seed=29)
    with torch.no_grad():
        model.jerk_head.weight.mul_(50.0)
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    active = ar_scene_loss_c(outputs, batch, model.codebook, _tiny_config(), c_config)
    assert float(active["jerk_exceedance"].detach()) > 0.0   # amplifier works
    invalid = {**batch, "future_valid_mask": torch.zeros_like(batch["future_valid_mask"])}
    outputs_invalid = model(invalid, teacher_tokens=invalid["motion_token_target"])
    masked = ar_scene_loss_c(outputs_invalid, invalid, model.codebook,
                             _tiny_config(), c_config)
    assert float(masked["jerk_exceedance"].detach()) == 0.0  # all-invalid -> 0
    assert torch.isfinite(masked["total"])


def test_penalty_variant_keeps_the_joint_start_construction():
    c_config = {"construction": "penalty", "start_pair_joint": True}
    model = _model(c_config=c_config, seed=30)
    _saturate_start(model)
    batch = _tiny_batch(seed=31)
    with torch.no_grad():
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    v0 = model._history_tail_velocity(batch)[0].numpy()
    implied = _implied_start_accel(out["kinematic"]["trajectories"].numpy(), v0)
    assert implied[:, 0].max() <= 9.5 + 1e-4                  # sum capped, penalty mode too
