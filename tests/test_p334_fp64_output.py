"""R2-E: output-precision revision arm ``C-v3-fp64out`` (user adjudication
2026-09-19).

Same weights, no retraining: ``output_position_precision: "float64"`` in the
c-config re-integrates the SAME v_seq with a float64 anchor and float64
cumsum, removing the fp32 output quantization residual behind the 26 accel /
2351 jerk type-bound exceedances (FULL_DEV_REEVAL.json type_checks: fp64
reintegration -> 0/0).  Default (absent key / "float32") must keep the sealed
C-v3 output path bit-exact -- same idiom as ``start_recurrence_version``.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_p334_r2_start_recurrence import (  # noqa: E402
    _anchored_kinematics, _force_commanded, _model, _rest_batch, _tiny_batch)

V2_CONFIG = {"construction": "bounded", "token_conditioning": True,
             "start_pair_joint": True, "v0_source": "recorded",
             "v0_speed_limit_mps": 35.0, "start_recurrence_version": 2}
FP64_CONFIG = {**V2_CONFIG, "output_position_precision": "float64"}


def test_default_and_explicit_float32_are_bit_exact():
    batch = _tiny_batch(seed=3)
    out_default = _model(V2_CONFIG, seed=0).rollout(batch, seed=7)
    out_explicit = _model({**V2_CONFIG, "output_position_precision": "float32"},
                          seed=0).rollout(batch, seed=7)
    assert out_default["trajectories"].dtype == torch.float32
    assert torch.equal(out_default["trajectories"], out_explicit["trajectories"])
    assert torch.equal(out_default["v_seq"], out_explicit["v_seq"])


def test_fp64_output_matches_numpy_reintegration():
    batch = _tiny_batch(seed=3)
    roll = _model(FP64_CONFIG, seed=0).rollout(batch, num_samples=2, seed=7)
    traj = roll["trajectories"].cpu().numpy()          # [B, K, A, T, 2]
    v_seq = roll["v_seq"].cpu().numpy()                # [B, K, A, T, 2]
    anchor = (batch["agent_history"][:, :, -1, :2]
              .repeat_interleave(2, dim=0).view(traj.shape[0], traj.shape[1],
                                                traj.shape[2], 1, 2)
              .cpu().numpy().astype(np.float64))
    reintegrated = anchor + 0.1 * np.cumsum(v_seq.astype(np.float64), axis=-2)
    assert traj.dtype == np.float64
    assert np.abs(traj - reintegrated).max() == 0.0    # bitwise (CPU cumsum)


def test_fp64_implied_kinematics_within_type_bounds_at_frozen_tolerance():
    # Saturated commanded jerks (exactly J per agent type, the established
    # _force_commanded pattern): the analytical recursion is bounded by
    # construction, and the fp64 output arm must keep the implied anchored
    # accel/jerk within bound + frozen 1e-3.
    batch = _rest_batch(seed=5)
    model = _model(FP64_CONFIG, seed=0)
    _force_commanded(model, [(1.0, 0.0)] * 5)
    with torch.no_grad():
        out = model(batch, teacher_tokens=batch["motion_token_target"])
    trajectories = out["kinematic"]["trajectories"].cpu().numpy()
    assert trajectories.dtype == np.float64
    anchor = batch["agent_history"][:, :, -1, :2].cpu().numpy().astype(np.float64)
    v0 = np.zeros_like(anchor)
    _, _, accel, jerk = _anchored_kinematics(trajectories, anchor, v0)
    accel = np.linalg.norm(accel, axis=-1)
    jerk = np.linalg.norm(jerk, axis=-1)
    limits = {1: (19.0, 9.5), 2: (5.0, 3.0), 3: (10.0, 5.0), 4: (10.0, 5.0)}
    for a in range(batch["agent_type"].shape[1]):
        j_lim, a_lim = limits[int(batch["agent_type"][0, a])]
        assert accel[:, a].max() <= a_lim + 1e-3
        assert jerk[:, a].max() <= j_lim + 1e-3


def test_fp64_flag_changes_no_sampling_path():
    # tokens / v_seq / commanded jerk must be IDENTICAL between precision
    # arms (the flag only re-integrates positions); only trajectories differ.
    batch = _tiny_batch(seed=3)
    out32 = _model(V2_CONFIG, seed=0).rollout(batch, num_samples=2, seed=7)
    out64 = _model(FP64_CONFIG, seed=0).rollout(batch, num_samples=2, seed=7)
    assert torch.equal(out32["tokens"], out64["tokens"])
    assert torch.equal(out32["v_seq"], out64["v_seq"])
    assert torch.equal(out32["jerk"], out64["jerk"])
    assert out64["trajectories"].dtype == torch.float64
