import copy
import json

import numpy as np
import pytest

from scenario_lab.p33_spec import (
    CONFIG_PATH,
    SCHEMA_PATH,
    build_actor_visible_history,
    canonical_sha256,
    expected_array_contract,
    load_p33_config,
    validate_json_schema_contract,
    validate_p33_inventory,
    validate_scene_arrays,
    waymo_training_split,
)


def test_shipped_p33_config_and_json_schema_are_synchronized():
    config = load_p33_config(CONFIG_PATH)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_json_schema_contract(schema, config)
    assert config["task"]["formal_branches"] == ["single", "dual"]
    assert config["training"]["safety_guidance"]["ppo_enabled"] is False


def test_waymo_scenario_split_is_deterministic_and_near_90_10():
    ids = [f"scenario-{index}" for index in range(2000)]
    first = [waymo_training_split(sid) for sid in ids]
    second = [waymo_training_split(sid) for sid in ids]
    assert first == second
    assert 1750 <= first.count("train") <= 1850
    assert 150 <= first.count("dev") <= 250


def _valid_arrays(config):
    arrays = {name: np.zeros(shape, dtype=dtype)
              for name, (shape, dtype) in expected_array_contract(config).items()}
    arrays["agent_present_mask"][:2] = True
    arrays["agent_type"][:2] = [1, 2]
    arrays["agent_role"][:2] = [1, 2]
    arrays["state_valid_mask"][:2] = True
    arrays["future_valid_mask"][:2] = True
    arrays["agent_history"][0, :, 0] = np.linspace(-1, 0, 11)
    arrays["agent_history"][1, :, 1] = 5.0
    arrays["future_xy"][0, :, 0] = np.linspace(0.1, 5.0, 50)
    arrays["future_xy"][1, :, 1] = np.linspace(4.9, 0.0, 50)
    arrays["pairwise_visibility_mask"][0, :, 0] = True
    arrays["pairwise_visibility_mask"][1, :, 1] = True
    arrays["pairwise_visibility_mask"][1, :, 0] = True
    arrays["map_point_mask"][0, :3] = True
    arrays["map_polylines"][0, :3, 0] = [0.0, 1.0, 2.0]
    arrays["map_type"][0] = 1
    arrays["motion_token_target"].fill(config["motion_tokens"]["ignore_index"])
    arrays["motion_token_valid_mask"][:2] = True
    arrays["motion_token_target"][:2] = 3
    return arrays


def test_scene_tensor_contract_and_actor_visibility_zero_hidden_truth():
    config = load_p33_config(CONFIG_PATH)
    arrays = _valid_arrays(config)
    validate_scene_arrays(arrays, config)
    visible = build_actor_visible_history(
        arrays["agent_history"], arrays["pairwise_visibility_mask"])
    assert visible.shape == (16, 11, 16, 8)
    assert np.all(visible[0, :, 1] == 0)  # anchor cannot see the pedestrian
    np.testing.assert_array_equal(visible[1, :, 0], arrays["agent_history"][0])


def test_scene_tensor_contract_rejects_truth_in_invalid_state():
    config = load_p33_config(CONFIG_PATH)
    arrays = _valid_arrays(config)
    arrays["state_valid_mask"][1, 0] = False
    with pytest.raises(ValueError, match="invalid history states"):
        validate_scene_arrays(arrays, config)


def test_inventory_validator_rejects_heldout_development_leak():
    config = load_p33_config(CONFIG_PATH)
    files = []
    for index in range(338):
        files.append({
            "source": "interaction", "relative_path": f"dev/{index}.csv",
            "size_bytes": 1, "split_role": "train", "location": "dev",
            "shard_index": None, "stages": ["full"], "content_sha256": None,
            "content_sha256_status": "deferred_to_streaming_conversion"
        })
    for index in range(35):
        files.append({
            "source": "interaction", "relative_path": f"final/{index}.csv",
            "size_bytes": 1, "split_role": "final_confirmation", "location": "final",
            "shard_index": None, "stages": ["final_confirmation"],
            "content_sha256": None,
            "content_sha256_status": "deferred_until_final_confirmation"
        })
    inventory = {
        "inventory_version": "p33-file-inventory-v1",
        "spec_version": config["spec_version"],
        "metadata_only": True,
        "dataset_content_bytes_read": 0,
        "files": files,
        "inventory_fingerprint_sha256": canonical_sha256(files)
    }
    validate_p33_inventory(inventory, config)
    leaked = copy.deepcopy(inventory)
    leaked["files"][-1]["stages"].append("smoke")
    leaked["inventory_fingerprint_sha256"] = canonical_sha256(leaked["files"])
    with pytest.raises(ValueError, match="heldout file leaked"):
        validate_p33_inventory(leaked, config)
