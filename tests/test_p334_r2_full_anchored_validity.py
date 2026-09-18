"""R2-E D: direct unit tests on the FORMAL evaluator (audit finding section 5).

The 242-test suite passed while the two audit counterexamples
(``runs/20260918_p334_r2_codex_audit/EVALUATOR_EDGE_PROBES.json``) held, because
the single-point-poison / nonfinite tests ran against a locally re-implemented
``_full_anchored_validity`` copy (tests/test_p334_r2_start_recurrence.py:211-228),
never against ``research_tasks/p334_r2_diagnosis.py::full_anchored_validity`` --
the function EVAL_PROTOCOL_V3 section 2 actually freezes.  These tests import
the formal function directly.

Versioning (user adjudication 2026-09-19: double-end propagation):
* ``validity_version=1`` (default) -- byte-for-byte frozen V3 behaviour, kept
  for sealed replay (probe_evaluator.py:13 sentinel, p334_r2_diagnosis.py:357
  R1 replay anchors, p334_v3_eval.py verbatim import).  Do NOT "fix" v1 tests.
* ``validity_version=2`` -- (a) non-finite anchor is an explicit failure
  (reason ``nonfinite_anchor``), never a silent NaN>limit==False pass;
  (b) mask semantics = position-unavailable propagation: a difference-based
  quantity is judged only when EVERY frame it spans is valid (double-end,
  matching p334_review_eval.py:137, p33_metrics.py:413-421, p334_phase_b.py:132
  and the training-side transition_valid convention).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research_tasks"))

from p334_r2_diagnosis import full_anchored_validity  # noqa: E402


def _trajs(K: int = 1, T: int = 5) -> np.ndarray:
    return np.zeros((K, T, 2), dtype=np.float64)


def test_v1_nan_anchor_still_passes_frozen_behavior():
    # Frozen V3 behaviour pinned by the audit probe (probe_evaluator.py:6-13):
    # NaN anchor, finite futures -> valid=true, no reasons.  Kept defective on
    # purpose for sealed replay; v2 is the corrected reading.
    out = full_anchored_validity(_trajs(), np.ones(5, bool),
                                 np.array([np.nan, 0.0]), np.zeros(2))
    assert bool(out["valid"][0]) and out["reasons"][0] == []


def test_v2_nan_anchor_is_invalid():
    out = full_anchored_validity(_trajs(), np.ones(5, bool),
                                 np.array([np.nan, 0.0]), np.zeros(2),
                                 validity_version=2)
    assert not out["valid"].all()
    assert "nonfinite_anchor" in out["reasons"][0]


def test_v2_mask_gap_not_judged_across_unavailable_frame():
    # Audit probe second counterexample: the only large jump (100 m) sits on a
    # masked frame (mask=[T,F,T,T,T]).  v1 counts the P3-P2 differential
    # (right-end-only gating) as a speed violation; v2 double-end propagation
    # does not judge a differential that spans an unavailable frame.
    trajs = np.zeros((1, 5, 2))
    trajs[0, 1, 0] = 100.0
    mask = np.array([True, False, True, True, True])
    v1 = full_anchored_validity(trajs, mask, np.zeros(2), np.zeros(2))
    v2 = full_anchored_validity(trajs, mask, np.zeros(2), np.zeros(2),
                                validity_version=2)
    assert not v1["valid"][0] and "speed" in v1["reasons"][0]
    assert v2["valid"][0] and v2["reasons"][0] == []


def test_v2_still_catches_violation_between_two_valid_frames():
    # Double-end propagation must not become a general amnesty: a violation
    # between two VALID frames is still caught in both versions.
    trajs = np.zeros((1, 5, 2))
    trajs[0, 2, 0] = 50.0  # P3-P2 differential with both frames valid
    mask = np.ones(5, bool)
    for v in (1, 2):
        out = full_anchored_validity(trajs, mask, np.zeros(2), np.zeros(2),
                                     validity_version=v)
        assert not out["valid"][0] and "speed" in out["reasons"][0]


def test_v0_nan_skip_separate_from_anchor_rule_both_versions():
    # v0 unusable -> earliest v0-dependent accel/jerk terms are SKIPPED and
    # counted (v0_usable=False), never a failure; independent of the anchor
    # rule in both versions.
    for v in (1, 2):
        out = full_anchored_validity(_trajs(), np.ones(5, bool), np.zeros(2),
                                     np.array([np.nan, np.nan]),
                                     validity_version=v)
        assert not out["v0_usable"]
        assert out["valid"][0] and out["reasons"][0] == []


def test_v2_nan_anchor_and_v0_nan_are_reported_separately():
    out = full_anchored_validity(_trajs(), np.ones(5, bool),
                                 np.array([np.nan, np.nan]),
                                 np.array([np.nan, np.nan]),
                                 validity_version=2)
    assert not out["valid"].all()
    assert "nonfinite_anchor" in out["reasons"][0]
    assert not out["v0_usable"]


def test_synthetic_fully_valid_agrees_across_versions():
    # constant 7.5 m/s along x, v0 matching: speed 7.5, accel/jerk 0
    trajs = np.tile(np.arange(1, 6, dtype=np.float64)[:, None] * 0.75,
                    (2, 1, 2))[:, :, :1] * np.array([1.0, 0.0])
    for v in (1, 2):
        out = full_anchored_validity(trajs, np.ones(5, bool), np.zeros(2),
                                     np.array([7.5, 0.0]), validity_version=v)
        assert out["valid"].all() and out["reasons"][0] == []
        assert out["v0_usable"]
