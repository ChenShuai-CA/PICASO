import struct

import numpy as np
import pytest

from scenario_lab.p33_pipeline import (
    AgentTrack,
    BalancedMotionReservoir,
    MapPolyline,
    Scene,
    _segment_box_intersection,
    _utm_zone31_xy,
    assign_motion_tokens,
    batched_motion_vectors,
    build_scene_arrays,
    fit_balanced_minibatch_kmeans,
    geometric_visibility,
    group_interaction_case_files,
    motion_vectors,
    parse_waymo_scene,
)
from scenario_lab.p33_spec import load_p33_config, validate_scene_arrays


def _varint(value):
    result = bytearray()
    while value > 0x7f:
        result.append((value & 0x7f) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _vfield(key, value):
    return _varint(key << 3) + _varint(value)


def _bfield(key, value):
    return _varint((key << 3) | 2) + _varint(len(value)) + value


def _dfield(key, value):
    return _varint((key << 3) | 1) + struct.pack("<d", value)


def _ffield(key, value):
    return _varint((key << 3) | 5) + struct.pack("<f", value)


def _point(x, y):
    return _dfield(1, x) + _dfield(2, y)


def _state(x, y):
    return (_dfield(2, x) + _dfield(3, y) + _ffield(5, 4.5) + _ffield(6, 1.8)
            + _ffield(8, 0.0) + _ffield(9, 1.0) + _ffield(10, 0.0) + _vfield(11, 1))


def test_parse_waymo_scene_decodes_sdc_roles_dimensions_and_map():
    track = _vfield(1, 42) + _vfield(2, 1)
    for index in range(61):
        track += _bfield(3, _state(float(index), 0.0))
    lane = _dfield(1, 30.0) + _bfield(8, _point(0.0, 0.0)) + _bfield(8, _point(10.0, 0.0))
    map_feature = _vfield(1, 7) + _bfield(3, lane)
    scenario = b"".join(_dfield(1, index / 10) for index in range(61))
    scenario += _bfield(2, track) + _bfield(5, b"scenario-a")
    scenario += _vfield(6, 0) + _bfield(8, map_feature) + _vfield(10, 10)
    scenario += _bfield(11, _vfield(1, 0) + _vfield(2, 1))

    parsed = parse_waymo_scene(scenario, "fixture", "0" * 64)

    assert parsed.group_id == "scenario-a"
    assert parsed.anchor_indices == [0]
    assert parsed.prediction_indices == [0]
    assert parsed.tracks[0].states[10, 5:].tolist() == pytest.approx([4.5, 1.8])
    assert parsed.map_features[0].kind == "lane_center"
    assert parsed.map_features[0].speed_limit_mps == pytest.approx(30 * 0.44704)


def test_interaction_case_grouping_keeps_agent_file_families_together(tmp_path):
    location = tmp_path / "LOC"
    location.mkdir()
    paths = [location / "vehicle_tracks_001.csv", location / "pedestrian_tracks_001.csv",
             location / "vehicle_tracks_002.csv"]
    groups = group_interaction_case_files(paths)
    assert [[path.name for path in group] for group in groups] == [
        ["pedestrian_tracks_001.csv", "vehicle_tracks_001.csv"],
        ["vehicle_tracks_002.csv"],
    ]


def test_interaction_utm_projection_matches_dataset_local_scale():
    x, y = _utm_zone31_xy(0.00863338433, 0.00902873996)
    assert x == pytest.approx(1006.06, abs=0.1)
    assert y == pytest.approx(955.56, abs=0.1)


def test_geometric_visibility_hides_actor_behind_third_box():
    history = np.zeros((3, 1, 7), dtype=np.float64)
    history[:, 0, 5:7] = [4.0, 2.0]
    history[0, 0, :2] = [0.0, 0.0]
    history[1, 0, :2] = [10.0, 0.0]
    history[2, 0, :2] = [5.0, 0.0]
    valid = np.ones((3, 1), dtype=bool)
    visible = geometric_visibility(history, valid, 80.0, 0.2)
    assert not visible[0, 0, 1]
    assert visible[0, 0, 0]
    assert visible[0, 0, 2]


def test_vectorized_visibility_matches_scalar_box_intersections():
    rng = np.random.default_rng(12)
    history = np.zeros((6, 3, 7), dtype=np.float64)
    history[:, :, :2] = rng.normal(0, 10, size=(6, 3, 2))
    history[:, :, 4] = rng.uniform(-np.pi, np.pi, size=(6, 3))
    history[:, :, 5:7] = rng.uniform(0.5, 5.0, size=(6, 3, 2))
    valid = rng.random((6, 3)) > 0.15
    actual = geometric_visibility(history, valid, 80.0, 0.2)
    expected = np.zeros_like(actual)
    for time in range(3):
        for query in np.flatnonzero(valid[:, time]):
            for source in np.flatnonzero(valid[:, time]):
                if query == source:
                    expected[query, time, source] = True
                    continue
                start, end = history[query, time, :2], history[source, time, :2]
                if np.linalg.norm(end - start) > 80.0:
                    continue
                expected[query, time, source] = not any(
                    third not in (query, source)
                    and valid[third, time]
                    and _segment_box_intersection(
                        start, end, history[third, time, :2], history[third, time, 4],
                        history[third, time, 5] / 2 + 0.2,
                        history[third, time, 6] / 2 + 0.2,
                    )
                    for third in range(6)
                )
    np.testing.assert_array_equal(actual, expected)


def _linear_scene():
    states = np.zeros((61, 7), dtype=np.float64)
    states[:, 0] = np.arange(61) * 0.1
    states[:, 2] = 1.0
    states[:, 5:7] = [4.5, 1.8]
    track = AgentTrack("ego", "vehicle", states, np.ones(61, dtype=bool))
    return Scene("waymo", "linear", np.arange(61) / 10, 10, [track], [0], [], [0],
                 [MapPolyline("lane", "lane_center", np.c_[np.arange(20), np.zeros(20)])],
                 "fixture", "0" * 64)


def test_scene_arrays_motion_codebook_and_contract_are_consistent():
    config = load_p33_config()
    arrays, _metadata = build_scene_arrays(_linear_scene(), 0, config)
    vectors, valid = motion_vectors(arrays, config)
    assert valid[0].all()
    assert vectors[0, 0].reshape(5, 2)[:, 0] == pytest.approx(np.full(5, 0.1))
    codebook = np.zeros((128, 10), dtype=np.float32)
    codebook[3] = vectors[0, 0]
    labelled, _metadata = build_scene_arrays(_linear_scene(), 0, config, codebook)
    validate_scene_arrays(labelled, config)
    assert np.all(labelled["motion_token_target"][0] == 3)


def test_motion_vector_chunk_after_gap_can_recover():
    config = load_p33_config()
    arrays, _metadata = build_scene_arrays(_linear_scene(), 0, config)
    arrays["future_valid_mask"][0, 4] = False
    arrays["future_xy"][0, 4] = 0
    _vectors, valid = motion_vectors(arrays, config)
    assert not valid[0, 0]
    assert not valid[0, 1]  # transition from invalid step 4 to step 5 is missing
    assert valid[0, 2]


def test_batched_motion_vectors_match_individual_computation():
    config = load_p33_config()
    arrays, _metadata = build_scene_arrays(_linear_scene(), 0, config)
    arrays["future_valid_mask"][0, 4] = False
    arrays["future_xy"][0, 4] = 0
    expected_vectors, expected_valid = motion_vectors(arrays, config)
    needed = ("future_xy", "future_valid_mask", "agent_history", "state_valid_mask")
    batched = {key: np.stack([arrays[key], arrays[key]]) for key in needed}
    actual_vectors, actual_valid = batched_motion_vectors(batched, config)
    np.testing.assert_array_equal(actual_vectors[0], expected_vectors)
    np.testing.assert_array_equal(actual_vectors[1], expected_vectors)
    np.testing.assert_array_equal(actual_valid[0], expected_valid)
    np.testing.assert_array_equal(actual_valid[1], expected_valid)


def test_balanced_minibatch_codebook_is_deterministic():
    strata = {
        "interaction:type_1": np.zeros((200, 10), dtype=np.float32),
        "waymo:type_1": np.ones((200, 10), dtype=np.float32),
    }
    first, manifest = fit_balanced_minibatch_kmeans(strata, 8, 64, 20, 0.0, 7)
    second, _ = fit_balanced_minibatch_kmeans(strata, 8, 64, 20, 0.0, 7)
    assert np.array_equal(first, second)
    assert 10 <= manifest["steps"] <= 20
    assert manifest["cluster_update_count_min"] > 0
    labels = assign_motion_tokens(strata["waymo:type_1"][:20].reshape(2, 10, 10),
                                  np.ones((2, 10), dtype=bool), first)
    assert labels.dtype == np.uint8


def test_balanced_reservoir_stays_bounded():
    reservoir = BalancedMotionReservoir(5, 7)
    reservoir.add("waymo:type_1", np.arange(200, dtype=np.float32).reshape(20, 10))
    assert reservoir.seen["waymo:type_1"] == 20
    assert len(reservoir.arrays()["waymo:type_1"]) == 5
