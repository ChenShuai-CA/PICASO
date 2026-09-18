"""R0 metric-layer regression tests (EVAL_PROTOCOL_V2.md sections 1-3).

Locks in the behaviours the CODEX review flagged:
  * diversity is a TIME-MEAN per agent (v1 time-sum kept only as legacy),
  * nonfinite values are explicit failures, never ``NaN > limit == False``,
  * the anchored convention exposes start terms (v_first, first accel) that
    the frozen P1-start convention cannot see.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research_tasks"))

from p334_phase_b import candidate_diversity, candidate_diversity_legacy  # noqa: E402
from p334_review_eval import (_block, anchored_kinematic_metrics,  # noqa: E402
                              candidate_validity)


def _offset_candidates(offset: float, steps: int = 50) -> np.ndarray:
    """[K=2, A=1, T, 2]: two candidates a constant ``offset`` m apart, 1 m/s."""
    base = np.cumsum(np.full(steps, 0.1))
    return np.stack([np.c_[base, np.zeros_like(base)],
                     np.c_[base + offset, np.zeros_like(base)]])[:, None]


# ------------------------------------------------------------------ diversity
def test_diversity_v2_is_time_mean():
    trajs = _offset_candidates(1.0, steps=50)
    valid = np.ones((1, 50), dtype=bool)
    assert candidate_diversity(trajs, valid) == 1.0


def test_diversity_legacy_keeps_v1_time_sum():
    trajs = _offset_candidates(1.0, steps=50)
    valid = np.ones((1, 50), dtype=bool)
    assert candidate_diversity_legacy(trajs, valid) == 50.0


def test_diversity_v2_unequal_valid_lengths_average_agents():
    # agent 0: 50 valid steps 2 m apart; agent 1: only 10 valid steps 4 m apart
    trajs = np.zeros((2, 2, 50, 2))                 # [K=2, A=2, T, 2]
    base = np.cumsum(np.full(50, 0.1))
    trajs[0, 0, :, 0] = base                        # candidate 0: agent tracks
    trajs[1, 0, :, 0] = base + 2.0
    trajs[0, 1, :, 0] = base
    trajs[1, 1, :, 0] = base + 4.0
    valid = np.zeros((2, 50), dtype=bool)
    valid[0] = True
    valid[1, :10] = True
    # per agent time-mean: 2.0 and 4.0 -> mean over the 2 usable agents = 3.0
    assert np.isclose(candidate_diversity(trajs, valid), 3.0)


def test_diversity_all_invalid_returns_zero():
    trajs = _offset_candidates(1.0)
    valid = np.zeros((1, 50), dtype=bool)
    assert candidate_diversity(trajs, valid) == 0.0
    assert candidate_diversity_legacy(trajs, valid) == 0.0


# ------------------------------------------------------- nonfinite accounting
def test_block_counts_nonfinite_as_violation():
    values = np.array([[1.0, np.nan, 3.0]])
    mask = np.ones_like(values, dtype=bool)
    block = _block(values, mask, limit=35.0)
    assert block["violating_steps"] == 1
    assert block["nonfinite_steps"] == 1
    assert block["step_rate"] == 1.0 / 3.0


def test_block_masks_excluded_steps():
    values = np.array([[1.0, np.nan, 3.0]])
    mask = np.array([[True, False, True]])
    block = _block(values, mask, limit=35.0)
    assert block["violating_steps"] == 0
    assert block["nonfinite_steps"] == 0


# ------------------------------------------------------- candidate validity
def test_candidate_validity_layers_nan_and_overspeed():
    good = np.stack([np.full(6, 0.1), np.zeros(6)], axis=-1)          # 1 m/s
    overspeed = np.stack([np.full(6, 4.0), np.zeros(6)], axis=-1)     # 40 m/s
    broken = np.full((6, 2), np.nan)
    trajs = np.stack([good, overspeed, broken])
    valid = np.ones(6, dtype=bool)
    # anchor sits one step before the good candidate: v_first = 1 m/s = v0
    verdict = candidate_validity(trajs, valid, anchor=np.zeros(2),
                                 v0=np.array([1.0, 0.0]))
    assert verdict["attempted"] == 3
    assert verdict["numeric_valid"] == 2
    assert verdict["nonfinite"] == 1
    assert verdict["kinematic_valid"] == 1        # only the 1 m/s candidate
    assert not verdict["no_history_velocity"]


def test_candidate_validity_nan_history_skips_start_term():
    good = np.stack([np.full(6, 0.1), np.zeros(6)], axis=-1)[None]
    verdict = candidate_validity(good, np.ones(6, dtype=bool),
                                 anchor=good[0, 0], v0=np.array([np.nan, np.nan]))
    assert verdict["no_history_velocity"]
    assert verdict["kinematic_valid"] == 1


# ---------------------------------------------------------- anchored vs frozen
def _teleport_trajectory():
    """Anchor at origin, v_hist 1 m/s, then a 5 m start jump and constant
    1 m/s afterwards -- every FROZEN check passes (inner diffs 1 m/s, the
    frozen continuity compares (P2-P1)*10 = 1 m/s against v_hist = 1 m/s),
    but the anchored (P1-P0)*10 = 50 m/s start term violates speed."""
    anchor = np.array([0.0, 0.0])
    v0 = np.array([1.0, 0.0])
    positions = [np.array([5.0, 0.0])]
    for _ in range(5):
        positions.append(positions[-1] + np.array([0.1, 0.0]))
    return anchor, v0, np.stack(positions)[None]                  # [1, 6, 2]


def test_anchored_exposes_start_jump_frozen_misses():
    from scenario_lab.p33_metrics import kinematic_attribution_metrics
    anchor, v0, trajs = _teleport_trajectory()
    valid = np.ones((1, 6), dtype=bool)

    frozen = kinematic_attribution_metrics(trajs, valid, v0[None])  # v1 convention
    assert frozen["speed"]["traj_rate"] == 0.0
    assert frozen["accel"]["traj_rate"] == 0.0
    assert frozen["jerk"]["traj_rate"] == 0.0
    assert frozen["continuity"]["implied_accel_over_limit_rate"] == 0.0

    anchored = anchored_kinematic_metrics(trajs, valid, anchor[None], v0[None])
    assert anchored["speed"]["traj_rate"] == 1.0                  # (P1-P0)*10 = 50
    assert anchored["accel"]["traj_rate"] == 1.0                  # (50-1)*10 >> 9.5

    verdict = candidate_validity(trajs, valid[0], anchor, v0)
    assert verdict["kinematic_valid"] == 0                       # via start term


def test_anchored_clean_start_passes_all_layers():
    anchor = np.array([3.0, 0.0])
    v0 = np.array([3.0, 0.0])          # the trajectory CONTINUES at v0 = 3 m/s
    # P1 = anchor + v0/10, then constant 3 m/s: perfect continuity
    first = anchor + v0 / 10.0
    positions = [first]
    for _ in range(5):
        positions.append(positions[-1] + np.array([0.3, 0.0]))
    trajs = np.stack(positions)[None]
    valid = np.ones((1, 6), dtype=bool)

    anchored = anchored_kinematic_metrics(trajs, valid, anchor[None], v0[None])
    assert anchored["speed"]["traj_rate"] == 0.0
    assert anchored["accel"]["traj_rate"] == 0.0
    assert anchored["jerk"]["traj_rate"] == 0.0
    verdict = candidate_validity(trajs, valid[0], anchor, v0)
    assert verdict["kinematic_valid"] == 1
