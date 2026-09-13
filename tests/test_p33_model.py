"""P3.3.2 AR-Scene-v1 contract tests: shapes, causality, hidden-state invariance."""
import math

import numpy as np
import pytest
import torch

from scenario_lab.p33_model import (
    ARSceneV1,
    ar_scene_loss,
    decode_tokens_to_trajectory,
    to_torch_batch,
)
from scenario_lab.p33_spec import load_p33_config

CONFIG = load_p33_config()
AGENTS, HISTORY, FUTURE = 16, 11, 50
CHUNKS = FUTURE // CONFIG["motion_tokens"]["chunk_steps"]


def _make_batch(batch_size=2, hidden_pair=(0, 5), seed=11):
    rng = np.random.default_rng(seed)
    batch = {}
    present = np.zeros((batch_size, AGENTS), dtype=bool)
    present[:, :10] = True
    history = np.zeros((batch_size, AGENTS, HISTORY, 8), dtype=np.float32)
    times = (np.arange(HISTORY) - HISTORY + 1) * 0.1
    for slot in range(AGENTS):
        if not present[0, slot]:
            continue
        velocity = rng.normal(0, 3.0, 2)
        position = rng.normal(0, 20.0, 2)
        history[:, slot, :, :2] = position + velocity * times[:, None]
        history[:, slot, :, 2:4] = velocity
        history[:, slot, :, 4] = 1.0
        history[:, slot, :, 6:8] = (4.4, 1.9)
    batch["agent_history"] = history
    batch["state_valid_mask"] = np.repeat(present[:, :, None], HISTORY, axis=2)
    visibility = np.zeros((batch_size, AGENTS, HISTORY, AGENTS), dtype=bool)
    for q in range(AGENTS):
        for s in range(AGENTS):
            visibility[:, q, :, s] = np.broadcast_to((present[:, q] & present[:, s])[:, None],
                                                     (batch_size, HISTORY))
    visibility[:, hidden_pair[0], :, hidden_pair[1]] = False  # agent 5 hidden from query 0
    batch["pairwise_visibility_mask"] = visibility
    batch["agent_present_mask"] = present
    batch["agent_type"] = np.where(present, 1, 0).astype(np.uint8)
    roles = np.where(present, 3, 0).astype(np.uint8)
    roles[:, 0] = 1
    roles[:, 1:4] = 2
    batch["agent_role"] = roles
    polylines = np.zeros((batch_size, 64, 20, 6), dtype=np.float32)
    point_mask = np.zeros((batch_size, 64, 20), dtype=bool)
    for line in range(3):
        points = np.c_[np.linspace(-30, 30, 20), np.full(20, 5.0 * line - 5.0)]
        polylines[:, line, :, :2] = points
        polylines[:, line, :, 2] = 1.0
        point_mask[:, line, :] = True
    batch["map_polylines"] = polylines
    batch["map_point_mask"] = point_mask
    batch["map_type"] = np.zeros((batch_size, 64), dtype=np.uint8)
    batch["map_type"][:, :3] = 1
    future = np.zeros((batch_size, AGENTS, FUTURE, 2), dtype=np.float32)
    future_valid = np.repeat(present[:, :, None], FUTURE, axis=2)
    future_times = np.arange(1, FUTURE + 1) * 0.1
    for slot in range(AGENTS):
        if not present[0, slot]:
            continue
        position = history[:, slot, -1, :2]
        velocity = history[:, slot, -1, 2:4]
        future[:, slot] = position[:, None, :] + velocity[:, None, :] * future_times[:, None]
    batch["future_xy"] = future
    batch["future_valid_mask"] = future_valid
    tokens = np.full((batch_size, AGENTS, CHUNKS), 255, dtype=np.uint8)
    tokens[:, present[0]] = 3
    batch["motion_token_target"] = tokens
    token_valid = np.zeros((batch_size, AGENTS, CHUNKS), dtype=bool)
    token_valid[:, present[0]] = True
    batch["motion_token_valid_mask"] = token_valid
    batch["source_id"] = np.zeros(batch_size, dtype=np.int64)
    batch["sample_source"] = ["waymo"] * batch_size
    batch["sample_id"] = [f"s{i}" for i in range(batch_size)]
    batch["group_id"] = [f"g{i}" for i in range(batch_size)]
    batch["split"] = ["dev"] * batch_size
    batch["agents_truncated"] = np.zeros(batch_size, dtype=bool)
    batch["map_truncated"] = np.zeros(batch_size, dtype=bool)
    return batch


def _model(codebook=None, seed=3):
    torch.manual_seed(seed)
    if codebook is None:
        codebook = np.zeros((CONFIG["motion_tokens"]["vocabulary_size"], 10), dtype=np.float32)
        codebook[3] = np.linspace(0.05, 0.5, 10)
    return ARSceneV1(CONFIG, codebook)


def test_parameter_count_inside_frozen_range():
    model = _model()
    total = sum(parameter.numel() for parameter in model.parameters())
    low, high = CONFIG["model"]["parameter_count_range"]
    assert low <= total <= high, f"parameter count {total} outside [{low}, {high}]"


def test_forward_shapes_finite_and_backward():
    model = _model()
    batch = to_torch_batch(_make_batch())
    outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    assert outputs["motion_token_logits"].shape == (2, AGENTS, CHUNKS, 128)
    assert outputs["delta_xy_residual"].shape == (2, AGENTS, CHUNKS, 10)
    for name, value in outputs.items():
        assert torch.isfinite(value).all(), name
    losses = ar_scene_loss(outputs, batch, model.codebook, CONFIG)
    assert math.isfinite(losses["total"].item())
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert losses["token_accuracy"].item() >= 0.0


def test_hidden_source_counterfactual_invariance():
    model = _model().eval()
    batch = _make_batch()
    outputs = model(to_torch_batch(batch), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    perturbed = {key: value.copy() for key, value in batch.items() if isinstance(value, np.ndarray)}
    perturbed["agent_history"][:, 5] *= 17.0  # agent 5 is hidden from query 0
    perturbed["future_xy"][:, 5] *= 3.0
    outputs_b = model(to_torch_batch(perturbed), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    torch.testing.assert_close(outputs["motion_token_logits"][:, 0], outputs_b["motion_token_logits"][:, 0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(outputs["delta_xy_residual"][:, 0], outputs_b["delta_xy_residual"][:, 0], atol=1e-4, rtol=1e-4)
    # a visible perturbation must change the query output
    perturbed2 = {key: value.copy() for key, value in batch.items() if isinstance(value, np.ndarray)}
    perturbed2["agent_history"][:, 1] *= 17.0
    outputs_c = model(to_torch_batch(perturbed2), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    assert not torch.allclose(outputs["motion_token_logits"][:, 0], outputs_c["motion_token_logits"][:, 0])


def test_partial_history_counterfactual_bitwise_invariance():
    """P3.3.2a blocking case: source visible at t=0 but hidden at one past frame.

    The t=0-visible gate is not sufficient — the hidden frame's state must have
    exactly zero effect on the query logits (spec 4.5 per-time masking), so the
    assertion is bitwise equality, not a tolerance.
    """
    model = _model().eval()
    batch = _make_batch()
    batch["pairwise_visibility_mask"][:, 0, 3, 6] = False  # frame 3 of source 6 hidden from query 0
    base = model(to_torch_batch(batch), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    perturbed = {key: value.copy() for key, value in batch.items() if isinstance(value, np.ndarray)}
    perturbed["agent_history"][:, 6, 3, :2] += 50.0   # huge state change at the hidden frame
    perturbed["agent_history"][:, 6, 3, 2:4] += 20.0
    outputs = model(to_torch_batch(perturbed), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    assert torch.equal(base["motion_token_logits"][:, 0], outputs["motion_token_logits"][:, 0])
    assert torch.equal(base["delta_xy_residual"][:, 0], outputs["delta_xy_residual"][:, 0])
    # other queries see agent 6 fully at frame 3, so their logits must move
    assert not torch.equal(base["motion_token_logits"][:, 1], outputs["motion_token_logits"][:, 1])


def test_past_visible_frames_flow_under_t0_occlusion():
    """Any-frame visibility gate: a source hidden at t=0 but visible in the past
    still informs the query through its visible frames (and only those)."""
    model = _model().eval()
    batch = _make_batch()
    # source 6 visible to query 0 only at frames 0-4, occluded from frame 5 on
    batch["pairwise_visibility_mask"][:, 0, 5:, 6] = False
    base = model(to_torch_batch(batch), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    hidden_frames = {key: value.copy() for key, value in batch.items() if isinstance(value, np.ndarray)}
    hidden_frames["agent_history"][:, 6, 5:, :2] += 50.0  # occluded frames: no effect
    out_hidden = model(to_torch_batch(hidden_frames), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    assert torch.equal(base["motion_token_logits"][:, 0], out_hidden["motion_token_logits"][:, 0])
    visible_frame = {key: value.copy() for key, value in batch.items() if isinstance(value, np.ndarray)}
    visible_frame["agent_history"][:, 6, 2, :2] += 50.0   # visible frame: must flow
    out_visible = model(to_torch_batch(visible_frame), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    assert not torch.equal(base["motion_token_logits"][:, 0], out_visible["motion_token_logits"][:, 0])


def test_fully_occluded_query_stays_finite():
    """A query that sees no other agent at any frame (only its own key remains
    attendable) must still produce finite logits."""
    model = _model().eval()
    batch = _make_batch()
    batch["pairwise_visibility_mask"][:, 0, :, 1:] = False  # query 0 sees nobody
    outputs = model(to_torch_batch(batch), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    assert torch.isfinite(outputs["motion_token_logits"]).all()
    assert torch.isfinite(outputs["delta_xy_residual"]).all()


def test_padded_agents_never_affect_valid_queries():
    model = _model().eval()
    batch = _make_batch()
    outputs = model(to_torch_batch(batch), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    dirty = {key: value.copy() for key, value in batch.items() if isinstance(value, np.ndarray)}
    # slot 15 is padded; stuff it with garbage despite the zero contract
    dirty["agent_history"][:, 15] = 123.0
    dirty["agent_type"][:, 15] = 4
    dirty["agent_role"][:, 15] = 3
    dirty["map_polylines"][:, 40] = 99.0  # beyond the first padded polyline set
    outputs_b = model(to_torch_batch(dirty), teacher_tokens=torch.from_numpy(batch["motion_token_target"].astype(np.int64)))
    torch.testing.assert_close(outputs["motion_token_logits"][:, :10], outputs_b["motion_token_logits"][:, :10], atol=1e-4, rtol=1e-4)


def test_decoder_causality_no_current_or_future_teacher_leakage():
    model = _model().eval()
    batch = _make_batch()
    teacher = torch.from_numpy(batch["motion_token_target"].astype(np.int64)).clone()
    outputs = model(to_torch_batch(batch), teacher_tokens=teacher)
    split = 7
    shuffled = teacher.clone()
    shuffled[:, :, split:] = torch.randint(0, 128, shuffled[:, :, split:].shape)
    outputs_b = model(to_torch_batch(batch), teacher_tokens=shuffled)
    torch.testing.assert_close(outputs["motion_token_logits"][:, :, :split],
                               outputs_b["motion_token_logits"][:, :, :split], atol=1e-4, rtol=1e-4)


def test_decode_tokens_trajectory_hand_computed():
    codebook = np.zeros((128, 10), dtype=np.float32)
    codebook[3] = np.linspace(0.1, 1.0, 10)
    tokens = np.full((1, AGENTS, CHUNKS), 3, dtype=np.int64)
    tokens[:, 0, :] = 5
    codebook[5] = -np.linspace(0.1, 1.0, 10)
    residuals = np.full((1, AGENTS, CHUNKS, 10), 0.05, dtype=np.float32)
    current = np.zeros((1, AGENTS, 2), dtype=np.float32)
    current[:, 0] = (7.0, -3.0)
    trajectory = decode_tokens_to_trajectory(tokens, residuals, codebook, current)
    assert trajectory.shape == (1, AGENTS, FUTURE, 2)
    for slot in range(AGENTS):
        expected = [current[0, slot].astype(np.float64)]
        for chunk in range(CHUNKS):
            for step in range(5):
                offset = step * 2  # 5 delta_xy points per chunk vector
                expected.append(expected[-1]
                                + codebook[tokens[0, slot, chunk], offset:offset + 2].astype(np.float64)
                                + residuals[0, slot, chunk, offset:offset + 2].astype(np.float64))
        np.testing.assert_allclose(trajectory[0, slot], np.asarray(expected)[1:], atol=1e-6)


def test_loss_ce_uniform_and_mask_normalization():
    batch = _make_batch()
    torch_batch = to_torch_batch(batch)
    logits = torch.zeros(2, AGENTS, CHUNKS, 128)
    model = _model()
    residuals = torch.zeros(2, AGENTS, CHUNKS, 10)
    losses = ar_scene_loss({"motion_token_logits": logits, "delta_xy_residual": residuals},
                           torch_batch, model.codebook, CONFIG)
    assert losses["token_cross_entropy"].item() == pytest.approx(math.log(128.0), rel=1e-5)
    # ignore-index agents contribute nothing: masking them out must not change CE
    masked = dict(torch_batch)
    valid = masked["motion_token_valid_mask"].clone()
    valid[:, 6:] = False
    masked["motion_token_valid_mask"] = valid
    losses_b = ar_scene_loss({"motion_token_logits": logits, "delta_xy_residual": residuals},
                             masked, model.codebook, CONFIG)
    assert losses_b["token_cross_entropy"].item() == pytest.approx(math.log(128.0), rel=1e-5)


def test_loss_huber_zero_when_residual_exact():
    from scenario_lab.p33_pipeline import batched_motion_vectors
    batch = _make_batch()
    torch_batch = to_torch_batch(batch)
    codebook = np.zeros((128, 10), dtype=np.float32)
    codebook[3] = 0.0
    model = _model(codebook)
    vectors, valid = batched_motion_vectors(
        {key: batch[key] for key in ("future_xy", "future_valid_mask", "agent_history", "state_valid_mask")},
        CONFIG)
    residuals = torch.from_numpy(
        vectors - codebook[np.clip(batch["motion_token_target"], 0, 127)]).float()
    logits = torch.zeros(2, AGENTS, CHUNKS, 128)
    losses = ar_scene_loss({"motion_token_logits": logits, "delta_xy_residual": residuals},
                           torch_batch, model.codebook, CONFIG)
    assert losses["trajectory_huber"].item() == pytest.approx(0.0, abs=1e-5)
    assert losses["endpoint_huber"].item() == pytest.approx(0.0, abs=1e-5)


def test_rollout_reproducible_and_shapes():
    model = _model().eval()
    batch = to_torch_batch(_make_batch())
    first = model.rollout(batch, num_samples=6, seed=123)
    second = model.rollout(batch, num_samples=6, seed=123)
    assert first["trajectories"].shape == (2, 6, AGENTS, FUTURE, 2)
    assert first["tokens"].shape == (2, 6, AGENTS, CHUNKS)
    assert first["token_log_prob"].shape == (2, 6, AGENTS, CHUNKS)
    assert first["tokens"].min() >= 0 and first["tokens"].max() < 128
    assert torch.isfinite(first["trajectories"]).all()
    torch.testing.assert_close(first["trajectories"], second["trajectories"])
    other = model.rollout(batch, num_samples=6, seed=999)
    assert not torch.equal(first["tokens"], other["tokens"])


def test_state_dict_roundtrip_preserves_outputs():
    model = _model().eval()
    batch = to_torch_batch(_make_batch())
    outputs = model(batch, teacher_tokens=batch["motion_token_target"])
    clone = _model(seed=99)  # different initialization
    clone.load_state_dict(model.state_dict())
    clone.eval()
    outputs_b = clone(batch, teacher_tokens=batch["motion_token_target"])
    torch.testing.assert_close(outputs["motion_token_logits"], outputs_b["motion_token_logits"])
