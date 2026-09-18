"""P3.3.4-R2 regressions: continuous start recurrence (SPEC_R2.md section 3).

Every kinematic quantity here is measured from the OUTPUT POSITIONS with P0
(anchor) prepended and the reference v0 as the first velocity entry -- the
anchored convention of EVAL_PROTOCOL_V3 section 2.  The R1 tests derived
accel/jerk from P1 onward only, which is exactly how the 38 m/s^3 start-jerk
defect escaped them (R1_ERRATA 2.2).  Version 1 (absent flag) behavior is
asserted where it must stay defective for sealed replay.
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
V3_CONFIG = {**V2_CONFIG, "start_recurrence_version": 2}


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


def _anchored_kinematics(trajectories, anchor, v0, rate=10.0):
    """Protocol V3 section 2 chain: P0 prepended, v0 prepended to velocity."""
    full = np.concatenate([anchor[:, :, None, :], trajectories], axis=-2)
    velocity = np.concatenate([v0[:, :, None, :],
                               np.diff(full, axis=-2) * rate], axis=-2)
    accel = np.diff(velocity, axis=-2) * rate
    jerk = np.diff(accel, axis=-2) * rate
    return full, velocity, accel, jerk


def _rest_batch(seed):
    """Stationary history with zero recorded velocity: v0 is exactly 0 under
    BOTH references, so no recorded-vs-diff caliber dispute can arise."""
    batch = _tiny_batch(seed=seed)
    history = batch["agent_history"].clone()
    history[..., :, 2:4] = 0.0
    history[..., 1:, :2] = history[..., :1, :2]
    batch["agent_history"] = history
    return batch


def _force_commanded(model, jerk_vectors, init_bias=None):
    """state path zeroed so jerk_head bias commands saturated jerks: chunk-step
    k gets tanh(|20*v|) ~ 1 -> exactly J(type)*unit(v). init_bias saturates a0."""
    with torch.no_grad():
        for layer in model.state_mlp:
            if hasattr(layer, "weight"):
                layer.weight.zero_(); layer.bias.zero_()
        model.jerk_head.weight.zero_()
        bias = torch.zeros(model.jerk_head.out_features)
        for k, vec in enumerate(jerk_vectors):
            bias[2 * k:2 * k + 2] = torch.tensor(vec, dtype=bias.dtype) * 20.0
        model.jerk_head.bias.copy_(bias)
        model.init_head.weight.zero_()
        model.init_head.bias.copy_(torch.tensor(init_bias or (0.0, 0.0),
                                                dtype=torch.float32))


def _implied_start_accel(trajectories: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Frozen-metric start term: |(P2-P1)*rate - v_hist| * rate = |a1 + a2|."""
    second_v = (trajectories[:, :, 1] - trajectories[:, :, 0]) * 10.0
    return np.linalg.norm(second_v - v0, axis=-1) * 10.0


# ---------------------------------------------------- the 38 m/s^3 counterexample
def test_counterexample_38_red_green_from_output_positions():
    """a0=0, j=(+J e_x, -J e_x, 0, ...): version 1 realizes 2J (38 for
    vehicles), version 2 realizes <= J -- both measured from positions with
    P0 and the exact-zero v0 (audit probe_start_jerk.py, R1_ERRATA 2.2)."""
    batch = _rest_batch(seed=101)
    anchor = batch["agent_history"][:, :, -1, :2].numpy()
    v0 = np.zeros_like(anchor)
    jerks = [(1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

    model_v1 = _model(V2_CONFIG, seed=3)          # sealed R1 path (red, fixed)
    _force_commanded(model_v1, jerks)
    with torch.no_grad():
        out_v1 = model_v1(batch, teacher_tokens=batch["motion_token_target"])
    _, _, _, jerk_v1 = _anchored_kinematics(
        out_v1["kinematic"]["trajectories"].numpy(), anchor, v0)
    jerk_v1 = np.linalg.norm(jerk_v1, axis=-1)
    for agent_type in np.unique(batch["agent_type"].numpy()):
        j_limit, _ = LIMITS[int(agent_type)]
        mask = batch["agent_type"].numpy() == agent_type
        assert jerk_v1[mask].max() >= 2.0 * j_limit - 0.1   # the defect, sealed

    model_v2 = _model(V3_CONFIG, seed=3)          # R2 recurrence (green)
    _force_commanded(model_v2, jerks)
    with torch.no_grad():
        out_v2 = model_v2(batch, teacher_tokens=batch["motion_token_target"])
    _, _, accel_v2, jerk_v2 = _anchored_kinematics(
        out_v2["kinematic"]["trajectories"].numpy(), anchor, v0)
    jerk_v2 = np.linalg.norm(jerk_v2, axis=-1)
    accel_v2 = np.linalg.norm(accel_v2, axis=-1)
    for agent_type in np.unique(batch["agent_type"].numpy()):
        j_limit, a_limit = LIMITS[int(agent_type)]
        mask = batch["agent_type"].numpy() == agent_type
        assert jerk_v2[mask].max() <= j_limit + 1e-3
        assert accel_v2[mask].max() <= a_limit + 1e-3
    assert jerk_v2[0, 0, 0] == pytest.approx(19.0, abs=1e-3)  # first jerk = J


def test_recurrence_bounds_under_saturation_and_directions():
    """Saturated oblique a0 with aligned/anti-aligned/oblique jerks, per type:
    every anchored accel/jerk term and the start-pair sum stay in bounds."""
    batch = _rest_batch(seed=102)
    anchor = batch["agent_history"][:, :, -1, :2].numpy()
    v0 = np.zeros_like(anchor)
    oblique = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    patterns = [
        [(1.0, 0.0)] * 5,                       # axis-aligned, same direction
        [(1.0, 1.0)] * 5,                       # oblique vs the oblique a0
        [(-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (0.0, 1.0)],
    ]
    for jerks in patterns:
        model = _model(V3_CONFIG, seed=4)
        _force_commanded(model, jerks, init_bias=tuple(oblique * 1000.0))
        with torch.no_grad():
            out = model(batch, teacher_tokens=batch["motion_token_target"])
        trajectories = out["kinematic"]["trajectories"].numpy()
        _, _, accel, jerk = _anchored_kinematics(trajectories, anchor, v0)
        accel = np.linalg.norm(accel, axis=-1)
        jerk = np.linalg.norm(jerk, axis=-1)
        for agent_type in np.unique(batch["agent_type"].numpy()):
            j_limit, a_limit = LIMITS[int(agent_type)]
            mask = batch["agent_type"].numpy() == agent_type
            assert accel[mask].max() <= a_limit + 1e-3
            assert jerk[mask].max() <= j_limit + 1e-3
        assert _implied_start_accel(trajectories, v0).max() <= 9.5 + 1e-4


def test_full_chain_bounds_with_random_weights_and_chunk_boundary():
    """Unforced (random) weights, version 2: the bound holds on every anchored
    term including a_1/j_1 and the chunk-1/ chunk-2 boundary (carry state)."""
    model = _model(V3_CONFIG, seed=5)
    batch = _tiny_batch(seed=103)
    v0 = model._history_tail_velocity_recorded(batch)[0].numpy()
    anchor = batch["agent_history"][:, :, -1, :2].numpy()
    with torch.no_grad():
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    _, _, accel, jerk = _anchored_kinematics(
        out["kinematic"]["trajectories"].numpy(), anchor, v0)
    accel = np.linalg.norm(accel, axis=-1)
    jerk = np.linalg.norm(jerk, axis=-1)
    for agent_type in np.unique(batch["agent_type"].numpy()):
        j_limit, a_limit = LIMITS[int(agent_type)]
        mask = batch["agent_type"].numpy() == agent_type
        assert accel[mask].max() <= a_limit + 1e-3
        assert jerk[mask].max() <= j_limit + 1e-3


# ----------------------------------------------- full_anchored judgment coverage
def _full_anchored_validity(full, v0, rate=10.0, speed_limit=35.0,
                            accel_limit=10.0, jerk_limit=20.0):
    """Reference implementation of EVAL_PROTOCOL_V3 section 2 (unit-level)."""
    reasons = set()
    velocity = np.concatenate([v0[:, :, None, :],
                               np.diff(full, axis=-2) * rate], axis=-2)
    accel = np.diff(velocity, axis=-2) * rate
    jerk = np.diff(accel, axis=-2) * rate
    if not (np.isfinite(full).all() and np.isfinite(v0).all()):
        reasons.add("nonfinite")
        return False, reasons
    if np.linalg.norm(velocity, axis=-1).max() > speed_limit + 1e-4:
        reasons.add("speed")
    if np.linalg.norm(accel, axis=-1).max() > accel_limit + 1e-3:
        reasons.add("accel")
    if np.linalg.norm(jerk, axis=-1).max() > jerk_limit + 1e-3:
        reasons.add("jerk")
    return not reasons, reasons


def test_single_point_violations_are_each_caught():
    """speed-only / accel-only / jerk-only injections must each flip the
    verdict with the right reason; a clean constant-velocity pass is valid.
    Position chains are built WITH P0 so the v_1 term (where the 38 m/s^3
    signature lives) is actually measured."""
    def positions_from_velocity(anchor, v_seq):
        """full = [P0, P1..PT] with (P_t - P_{t-1})*rate = v_t for t >= 1."""
        return np.concatenate(
            [anchor[:, :, None, :],
             anchor[:, :, None, :] + np.cumsum(v_seq, axis=-2) * 0.1], axis=-2)

    anchor = np.zeros((1, 1, 2))
    v0 = np.zeros((1, 1, 2))

    clean_v = np.full((1, 1, 10, 2), 5.0)                # cruise at 5 m/s
    clean_v0 = np.full_like(v0, 5.0)
    valid, reasons = _full_anchored_validity(
        positions_from_velocity(anchor, clean_v), clean_v0)
    assert valid and reasons == set()

    fast_v = np.full((1, 1, 10, 2), 36.0)                # speed only
    fast_v0 = np.full_like(v0, 36.0)
    valid, reasons = _full_anchored_validity(
        positions_from_velocity(anchor, fast_v), fast_v0)
    assert not valid and reasons == {"speed"}

    ramp_v = np.zeros((1, 1, 10, 2))                     # accel only: v_t = 1.2t
    ramp_v[..., 0] = 1.2 * np.arange(1, 11)
    valid, reasons = _full_anchored_validity(
        positions_from_velocity(anchor, ramp_v), v0)
    assert not valid and reasons == {"accel"}

    jerk_v = np.zeros((1, 1, 10, 2))                     # jerk only: v1 = 0.19
    jerk_v[:, :, 0, 0] = 0.19                            # a = (1.9, -1.9) -> 38 m/s^3
    valid, reasons = _full_anchored_validity(
        positions_from_velocity(anchor, jerk_v), v0)
    assert not valid and reasons == {"jerk"}


# ------------------------------------------------------ flag semantics / replay
def test_flag_defaults_and_version1_bitwise_replay():
    assert _model(None).start_recurrence_version == 1
    assert _model(V2_CONFIG).start_recurrence_version == 1
    assert _model(V3_CONFIG).start_recurrence_version == 2

    batch = _tiny_batch(seed=104)
    model = _model(V2_CONFIG, seed=6)                    # sealed path, twice
    with torch.no_grad():
        first = model(batch, teacher_tokens=batch["motion_token_target"])
        second = model(batch, teacher_tokens=batch["motion_token_target"])
    assert torch.equal(first["kinematic"]["trajectories"],
                       second["kinematic"]["trajectories"])

    model_v3 = _model(V3_CONFIG, seed=6)                 # same weights, new path
    with torch.no_grad():
        other = model_v3(batch, teacher_tokens=batch["motion_token_target"])
    delta = (first["kinematic"]["trajectories"]
             - other["kinematic"]["trajectories"]).abs().max().item()
    assert delta > 1e-6                                   # the fix changes outputs


def test_token_intervention_alive_under_version2():
    model = _model(V3_CONFIG, seed=10)
    batch = _tiny_batch(seed=105)
    tokens = batch["motion_token_target"].clone()
    with torch.no_grad():
        base = model.trajectory_from_tokens(tokens, batch)
        flipped = tokens.clone()
        flipped[:, :, 0] = (flipped[:, :, 0] + 1) % 128
        other = model.trajectory_from_tokens(flipped, batch)
    assert (base[:, :, 0:5] - other[:, :, 0:5]).abs().max().item() > 1e-6
    with torch.no_grad():
        flipped_last = tokens.clone()
        flipped_last[:, :, -1] = (flipped_last[:, :, -1] + 1) % 128
        last = model.trajectory_from_tokens(flipped_last, batch)
    assert (base[:, :, 5:10] - last[:, :, 5:10]).abs().max().item() > 1e-6


def test_rollout_matches_parallel_token_decode_v3():
    model = _model(V3_CONFIG, seed=12)
    batch = _tiny_batch(seed=106)
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


def test_hidden_agent_perturbation_leaves_visible_outputs_bitwise():
    """Frozen visibility boundary under the R2 decoder: source 2 invisible to
    query 0 at every history frame -> perturbing agent 2's history cannot
    change agent 0's trajectories (frame weights are exactly zero)."""
    model = _model(V3_CONFIG, seed=14)
    batch = _tiny_batch(seed=107)
    batch["pairwise_visibility_mask"][:, 0, :, 2] = False
    tokens = batch["motion_token_target"].clone()
    with torch.no_grad():
        base = model.trajectory_from_tokens(tokens, batch)
        perturbed = {key: (value.clone() if torch.is_tensor(value) else value)
                     for key, value in batch.items()}
        perturbed["agent_history"][:, 2, :, :2] += 0.37
        perturbed["agent_history"][:, 2, :, 2:4] *= 3.0
        other = model.trajectory_from_tokens(tokens, perturbed)
    assert torch.equal(base[:, 0], other[:, 0])


# ------------------------------------------------------------ devices and AMP
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_counterexample_bounds_hold_on_gpu_fp32():
    batch = _rest_batch(seed=108)
    model = _model(V3_CONFIG, seed=3).cuda()
    batch = {key: (value.cuda() if torch.is_tensor(value) else value)
             for key, value in batch.items()}
    _force_commanded(model, [(1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    with torch.no_grad():
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = out["kinematic"]["trajectories"].cpu().numpy()
    anchor = batch["agent_history"][:, :, -1, :2].cpu().numpy()
    v0 = np.zeros_like(anchor)
    _, _, _, jerk = _anchored_kinematics(trajectories, anchor, v0)
    jerk = np.linalg.norm(jerk, axis=-1)
    for agent_type in np.unique(batch["agent_type"].cpu().numpy()):
        j_limit, _ = LIMITS[int(agent_type)]
        mask = batch["agent_type"].cpu().numpy() == agent_type
        assert jerk[mask].max() <= j_limit + 1e-3


def test_amp_training_graduates_remain_finite():
    model = _model(V3_CONFIG, seed=16)
    batch = _tiny_batch(seed=109)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
        losses = ar_scene_loss_c(outputs, batch, model.codebook,
                                 _tiny_config(), V3_CONFIG)
    losses["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name


# ------------------------------------------------------------- dirty data edges
def test_nan_tail_velocity_masked_frames_and_pad_agents():
    """A raw NaN recorded tail makes v0 UNUSABLE (skip accounting, counted
    separately from failures).  The encoder zeroes invalid frames by
    multiplication, so the data pipeline marks missing frames invalid with
    finite values; raw-NaN features propagate and are handled at eval as
    numeric_valid=false -- the anchored judgment must report, not accept,
    a nonfinite trajectory."""
    model = _model(V3_CONFIG, seed=18)
    batch = _tiny_batch(seed=110)
    batch["agent_history"][0, 0, -1, 2:4] = torch.tensor([float("nan"), 0.0])
    v0, usable, over_limit = model._history_tail_velocity_recorded(batch)
    assert not usable[0, 0] and not over_limit[0, 0]        # skip, not a failure
    batch["state_valid_mask"][0, 0, -1] = False            # pipeline marks it
    batch["agent_history"][0, 0, -1, 2:4] = 0.0            # finite placeholder
    batch["state_valid_mask"][0, 1, 1] = False              # interrupted history
    batch["agent_present_mask"][0, 2] = False               # pad agent
    batch["agent_history"][0, 2] *= 1e3                     # finite garbage
    with torch.no_grad():
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    assert torch.isfinite(out["kinematic"]["trajectories"]).all()

    poisoned = out["kinematic"]["trajectories"].clone()     # judgment side
    poisoned[0, 0, 0, 0] = float("nan")
    valid, reasons = _full_anchored_validity(
        poisoned.numpy(), v0.detach().cpu().numpy())
    assert not valid and reasons == {"nonfinite"}
