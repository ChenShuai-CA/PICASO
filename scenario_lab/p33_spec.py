"""Frozen P3.3.0 task/model contract and metadata-only data inventory.

This module deliberately does not parse TFRecord or CSV contents.  It freezes
configuration invariants, assigns files to development/final roles from names
and locations, and validates the tensor boundary that the P3.3.1 streaming
converter must implement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import re

import numpy as np


SPEC_VERSION = "p33.0-v1"
INVENTORY_VERSION = "p33-file-inventory-v1"
WAYMO_SPLIT_SALT = "p33-waymo-v1"
CONFIG_PATH = Path("configs/p33/ar_scene_v1.json")
SCHEMA_PATH = Path("schemas/p33_scene_v1.schema.json")


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(value) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_p33_config(path: Path | str = CONFIG_PATH) -> dict:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_p33_config(config)
    return config


def validate_p33_config(config: dict) -> None:
    if config.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"spec_version must be {SPEC_VERSION!r}")
    if config.get("status") != "design_frozen_not_trained":
        raise ValueError("P3.3.0 config must not claim a trained model")

    task, data = config["task"], config["data"]
    model, tokens = config["model"], config["motion_tokens"]
    training, split = config["training"], config["split_policy"]
    evaluation = config["evaluation"]

    if task["formal_branches"] != ["single", "dual"]:
        raise ValueError("single and dual must remain ordered formal branches")
    if task["role_action_mode"] != "lane_locked":
        raise ValueError("P3.3 is frozen to lane_locked execution")
    if max(task["rollout_budgets"]) > task["candidate_pool_size"]:
        raise ValueError("rollout budget exceeds candidate pool")
    required_forbidden = {"future_ground_truth", "candidate_outcome",
                          "hidden_actor_ground_truth"}
    if not required_forbidden <= set(task["forbidden_online_inputs"]):
        raise ValueError("online information boundary is incomplete")

    hz = data["sample_rate_hz"]
    if not math.isclose((data["history_steps"] - 1) / hz,
                        data["history_seconds"]):
        raise ValueError("history steps must include both -1 s and current time")
    if not math.isclose(data["future_steps"] / hz, data["future_seconds"]):
        raise ValueError("future step count and duration disagree")
    if data["future_steps"] % tokens["chunk_steps"]:
        raise ValueError("motion chunks must divide the future horizon")
    if not math.isclose(tokens["chunk_steps"] / hz, tokens["chunk_seconds"]):
        raise ValueError("motion token duration disagrees with sample rate")
    if tokens["vocabulary_size"] > tokens["ignore_index"]:
        raise ValueError("uint8 ignore index must lie outside the token vocabulary")
    if tokens["fit_split"] != "train_only":
        raise ValueError("motion codebook may only be fit on training data")
    if not data["storage"]["streaming_required"]:
        raise ValueError("full public-data conversion must be streaming")
    if not data["storage"]["global_example_list_forbidden"]:
        raise ValueError("global example accumulation must remain forbidden")

    if model["d_model"] % model["num_attention_heads"]:
        raise ValueError("d_model must be divisible by attention heads")
    if model["autoregressive_axis"] != "time_chunk":
        raise ValueError("AR-Scene-v1 is autoregressive over time chunks")
    if training["target_effective_batch_size"] % training["micro_batch_size"]:
        raise ValueError("effective batch must be divisible by micro batch")
    if training["safety_guidance"]["ppo_enabled"]:
        raise ValueError("PPO is outside the P3.3 minimum model")
    if len(set(training["formal_seeds"])) != 3:
        raise ValueError("formal training requires three distinct seeds")

    waymo, interaction = split["waymo"], split["interaction"]
    previous = set(waymo["previously_read_validation_indices"])
    final_lo, final_hi = waymo["final_confirmation_validation_indices"]
    final = set(range(final_lo, final_hi + 1))
    if previous & final or final_hi >= waymo["expected_validation_shards"]:
        raise ValueError("Waymo development and final validation shards overlap")
    location_sets = [set(interaction[key]) for key in
                     ("train_locations", "dev_locations", "location_heldout")]
    if any(location_sets[i] & location_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("INTERACTION location roles overlap")
    if sum((interaction["expected_development_files"],
            interaction["expected_heldout_files"])) != interaction["expected_csv_files"]:
        raise ValueError("INTERACTION expected file counts do not add up")

    ladder = config["scale_ladder"]
    if [row["name"] for row in ladder] != ["smoke", "architecture", "scale", "full"]:
        raise ValueError("scale ladder names or order changed")
    if [row["waymo_training_shards"] for row in ladder] != [10, 100, 500, 1000]:
        raise ValueError("Waymo scale ladder changed")
    required_gates = {"g0", "g1", "g2", "g3", "g4"}
    if set(evaluation["gates"]) != required_gates:
        raise ValueError("G0-G4 must all be specified")
    if evaluation["t1_primary_metric"] != "minADE_at_6":
        raise ValueError("T1 primary metric must be frozen before model development")
    if not config["confirmation"]["p212_seed77000_forbidden_for_model_selection"]:
        raise ValueError("P2.12 heldout cannot be reused for P3.3 model selection")


def waymo_training_split(scenario_id: str) -> str:
    if not scenario_id:
        raise ValueError("scenario_id is required")
    digest = sha256(f"{WAYMO_SPLIT_SALT}\x1f{scenario_id}".encode("utf-8")).digest()
    bucket = (int.from_bytes(digest[:8], "big") * 100) // (1 << 64)
    return "train" if bucket < 90 else "dev"


def _stable_file_order(paths: list[Path], data_root: Path, salt: str) -> list[Path]:
    def key(path: Path):
        relative = path.relative_to(data_root).as_posix()
        return sha256(f"{salt}\x1f{relative}".encode("utf-8")).hexdigest(), relative
    return sorted(paths, key=key)


def _waymo_index(path: Path) -> int:
    match = re.search(r"-(\d{5})-of-\d{5}$", path.name)
    if not match:
        raise ValueError(f"cannot parse Waymo shard index: {path}")
    return int(match.group(1))


def _file_record(path: Path, data_root: Path, source: str, split_role: str,
                 stages: list[str], location: str | None = None,
                 shard_index: int | None = None) -> dict:
    stat = path.stat()
    return {
        "source": source,
        "relative_path": path.relative_to(data_root).as_posix(),
        "size_bytes": stat.st_size,
        "split_role": split_role,
        "location": location,
        "shard_index": shard_index,
        "stages": stages,
        "content_sha256": None,
        "content_sha256_status": "deferred_to_streaming_conversion"
        if split_role != "final_confirmation" else "deferred_until_final_confirmation"
    }


def build_p33_inventory(data_root: Path | str, config: dict) -> dict:
    """Inventory paths and sizes without opening any dataset file content."""
    validate_p33_config(config)
    data_root = Path(data_root)
    policy = config["split_policy"]
    salt = policy["selection_salt"]
    records = []

    waymo = policy["waymo"]
    train_paths = sorted(data_root.glob(waymo["training_glob"]))
    val_paths = sorted(data_root.glob(waymo["validation_glob"]))
    if len(train_paths) != waymo["expected_training_shards"]:
        raise ValueError(f"expected {waymo['expected_training_shards']} Waymo training shards, "
                         f"found {len(train_paths)}")
    if len(val_paths) != waymo["expected_validation_shards"]:
        raise ValueError(f"expected {waymo['expected_validation_shards']} Waymo validation shards, "
                         f"found {len(val_paths)}")
    ordered_train = _stable_file_order(train_paths, data_root, salt)
    stage_limits = {row["name"]: row["waymo_training_shards"]
                    for row in config["scale_ladder"]}
    rank = {path: index for index, path in enumerate(ordered_train)}
    for path in train_paths:
        stages = [name for name, limit in stage_limits.items() if rank[path] < limit]
        records.append(_file_record(path, data_root, "waymo", "scenario_hash_train_dev",
                                    stages, shard_index=_waymo_index(path)))

    previous = set(waymo["previously_read_validation_indices"])
    final_lo, final_hi = waymo["final_confirmation_validation_indices"]
    for path in val_paths:
        index = _waymo_index(path)
        if index in previous:
            role, stages = "development_only_previously_read", ["architecture", "scale", "full"]
        elif final_lo <= index <= final_hi:
            role, stages = "final_confirmation", ["final_confirmation"]
        else:
            raise ValueError(f"validation shard {index} has no frozen role")
        records.append(_file_record(path, data_root, "waymo", role, stages,
                                    shard_index=index))

    interaction = policy["interaction"]
    interaction_root = data_root / interaction["root"]
    csv_paths = sorted(interaction_root.rglob("*.csv"))
    if len(csv_paths) != interaction["expected_csv_files"]:
        raise ValueError(f"expected {interaction['expected_csv_files']} INTERACTION CSV files, "
                         f"found {len(csv_paths)}")
    train_locations = set(interaction["train_locations"])
    dev_locations = set(interaction["dev_locations"])
    heldout_locations = set(interaction["location_heldout"])
    by_location = defaultdict(list)
    for path in csv_paths:
        by_location[path.parent.name].append(path)
    known = train_locations | dev_locations | heldout_locations
    if set(by_location) != known:
        raise ValueError(f"unexpected INTERACTION locations: {sorted(set(by_location) ^ known)}")

    smoke_selected, architecture_selected = set(), set()
    for location in sorted(train_locations | dev_locations):
        ordered = _stable_file_order(by_location[location], data_root, salt)
        smoke_selected.update(ordered[:2])
        architecture_count = max(1, math.ceil(len(ordered) * 0.25))
        architecture_selected.update(ordered[:architecture_count])

    for path in csv_paths:
        location = path.parent.name
        if location in heldout_locations:
            role, stages = "final_confirmation", ["final_confirmation"]
        else:
            role = "train" if location in train_locations else "dev"
            stages = []
            if path in smoke_selected:
                stages.append("smoke")
            if path in architecture_selected:
                stages.append("architecture")
            stages.extend(("scale", "full"))
        records.append(_file_record(path, data_root, "interaction", role, stages,
                                    location=location))

    records.sort(key=lambda row: (row["source"], row["relative_path"]))
    counts_by_source = Counter(row["source"] for row in records)
    counts_by_role = Counter(f"{row['source']}:{row['split_role']}" for row in records)
    stage_counts = {stage: Counter(row["source"] for row in records if stage in row["stages"])
                    for stage in ("smoke", "architecture", "scale", "full",
                                  "final_confirmation")}
    payload = {
        "inventory_version": INVENTORY_VERSION,
        "spec_version": config["spec_version"],
        "metadata_only": True,
        "dataset_content_bytes_read": 0,
        "content_hash_policy": "computed by P3.3.1 while streaming development files; final files only at final confirmation",
        "counts_by_source": dict(sorted(counts_by_source.items())),
        "counts_by_role": dict(sorted(counts_by_role.items())),
        "stage_counts": {stage: dict(sorted(counts.items())) for stage, counts in stage_counts.items()},
        "total_size_bytes": sum(row["size_bytes"] for row in records),
        "files": records
    }
    payload["inventory_fingerprint_sha256"] = canonical_sha256(records)
    validate_p33_inventory(payload, config)
    return payload


def validate_p33_inventory(inventory: dict, config: dict) -> None:
    if inventory.get("inventory_version") != INVENTORY_VERSION:
        raise ValueError("unknown P3.3 inventory version")
    if inventory.get("spec_version") != config["spec_version"]:
        raise ValueError("inventory/config version mismatch")
    if not inventory.get("metadata_only") or inventory.get("dataset_content_bytes_read") != 0:
        raise ValueError("P3.3.0 inventory must be metadata-only")
    files = inventory["files"]
    paths = [row["relative_path"] for row in files]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate paths in P3.3 inventory")
    if canonical_sha256(files) != inventory["inventory_fingerprint_sha256"]:
        raise ValueError("inventory fingerprint mismatch")

    interaction = config["split_policy"]["interaction"]
    interaction_dev = [row for row in files if row["source"] == "interaction"
                       and row["split_role"] in ("train", "dev")]
    interaction_final = [row for row in files if row["source"] == "interaction"
                         and row["split_role"] == "final_confirmation"]
    if len(interaction_dev) != interaction["expected_development_files"]:
        raise ValueError("INTERACTION development count drifted")
    if len(interaction_final) != interaction["expected_heldout_files"]:
        raise ValueError("INTERACTION heldout count drifted")

    forbidden_development_stages = {"smoke", "architecture", "scale", "full"}
    for row in files:
        stages = set(row["stages"])
        if row["split_role"] == "final_confirmation":
            if stages != {"final_confirmation"}:
                raise ValueError(f"heldout file leaked into development: {row['relative_path']}")
            if row["content_sha256"] is not None:
                raise ValueError("heldout content hash must remain unread at P3.3.0")
        elif stages & {"final_confirmation"}:
            raise ValueError(f"development file assigned to final: {row['relative_path']}")
        if row["split_role"] == "final_confirmation" and stages & forbidden_development_stages:
            raise ValueError(f"heldout stage overlap: {row['relative_path']}")


def expected_array_contract(config: dict) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    data, tokens = config["data"], config["motion_tokens"]
    agents, history, future = data["max_agents"], data["history_steps"], data["future_steps"]
    polylines, points = data["max_map_polylines"], data["max_points_per_polyline"]
    chunks = future // tokens["chunk_steps"]
    return {
        "agent_history": ((agents, history, len(data["agent_feature_order"])), np.dtype("float32")),
        "state_valid_mask": ((agents, history), np.dtype("bool")),
        "pairwise_visibility_mask": ((agents, history, agents), np.dtype("bool")),
        "agent_present_mask": ((agents,), np.dtype("bool")),
        "agent_type": ((agents,), np.dtype("uint8")),
        "agent_role": ((agents,), np.dtype("uint8")),
        "map_polylines": ((polylines, points, len(data["map_feature_order"])), np.dtype("float32")),
        "map_point_mask": ((polylines, points), np.dtype("bool")),
        "map_type": ((polylines,), np.dtype("uint8")),
        "future_xy": ((agents, future, 2), np.dtype("float32")),
        "future_valid_mask": ((agents, future), np.dtype("bool")),
        "motion_token_target": ((agents, chunks), np.dtype("uint8")),
        "motion_token_valid_mask": ((agents, chunks), np.dtype("bool"))
    }


def validate_json_schema_contract(schema: dict, config: dict) -> None:
    """Keep JSON index annotations synchronized with the executable tensor contract."""
    if schema.get("properties", {}).get("schema_version", {}).get("const") != data_schema_version(config):
        raise ValueError("JSON schema/config scene version mismatch")
    array_schema = schema.get("properties", {}).get("arrays", {})
    properties = array_schema.get("properties", {})
    required = set(array_schema.get("required", ()))
    contract = expected_array_contract(config)
    if required != set(contract) or set(properties) != set(contract):
        raise ValueError("JSON schema array keys differ from executable contract")
    for name, (shape, dtype) in contract.items():
        declared = properties[name]
        if tuple(declared.get("x-shape", ())) != shape:
            raise ValueError(f"JSON schema shape drift for {name}")
        if declared.get("x-dtype") != dtype.name:
            raise ValueError(f"JSON schema dtype drift for {name}")


def data_schema_version(config: dict) -> str:
    return str(config["data"]["schema_version"])


def validate_scene_arrays(arrays: dict[str, np.ndarray], config: dict) -> None:
    """Validate one unbatched scene-shard-v1 tensor sample."""
    contract = expected_array_contract(config)
    missing = set(contract) - set(arrays)
    extra = set(arrays) - set(contract)
    if missing or extra:
        raise ValueError(f"array keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
    for name, (shape, dtype) in contract.items():
        array = np.asarray(arrays[name])
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{name}: expected {shape}/{dtype}, got {array.shape}/{array.dtype}")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/Inf")

    present = arrays["agent_present_mask"]
    state_valid = arrays["state_valid_mask"]
    visibility = arrays["pairwise_visibility_mask"]
    if not present[0] or arrays["agent_role"][0] != 1:
        raise ValueError("slot 0 must be the present anchor")
    if np.any(state_valid[~present]) or np.any(arrays["future_valid_mask"][~present]):
        raise ValueError("padded agents cannot have valid history/future")
    if np.any(arrays["agent_history"][~state_valid] != 0):
        raise ValueError("invalid history states must be zero-filled")
    if np.any(arrays["future_xy"][~arrays["future_valid_mask"]] != 0):
        raise ValueError("invalid future states must be zero-filled")
    if np.any(arrays["agent_type"][~present] != 0) or np.any(arrays["agent_role"][~present] != 0):
        raise ValueError("padded agents must use pad type and role")

    source_valid = state_valid.T[None, :, :]
    query_valid = state_valid[:, :, None]
    if np.any(visibility & ~(source_valid & query_valid)):
        raise ValueError("visibility references an invalid query or source state")
    diagonal = np.stack([visibility[i, :, i] for i in range(len(present))])
    if not np.array_equal(diagonal, state_valid):
        raise ValueError("each valid agent must see itself and invalid states must not")

    map_mask = arrays["map_point_mask"]
    if np.any(arrays["map_polylines"][~map_mask] != 0):
        raise ValueError("invalid map points must be zero-filled")
    map_present = map_mask.any(axis=1)
    if np.any(arrays["map_type"][~map_present] != 0):
        raise ValueError("padded map polylines must use pad type")

    token_valid = arrays["motion_token_valid_mask"]
    token_target = arrays["motion_token_target"]
    vocabulary = config["motion_tokens"]["vocabulary_size"]
    ignore = config["motion_tokens"]["ignore_index"]
    if np.any(token_target[token_valid] >= vocabulary):
        raise ValueError("valid motion token is outside the vocabulary")
    if np.any(token_target[~token_valid] != ignore):
        raise ValueError("invalid motion tokens must use the ignore index")


def build_actor_visible_history(agent_history: np.ndarray,
                                pairwise_visibility_mask: np.ndarray) -> np.ndarray:
    """Materialize [query,time,source,feature] input with hidden states zeroed."""
    history = np.asarray(agent_history)
    visibility = np.asarray(pairwise_visibility_mask, dtype=bool)
    if history.ndim != 3 or visibility.shape != (history.shape[0], history.shape[1], history.shape[0]):
        raise ValueError("history/visibility shapes do not match the actor-view contract")
    source_history = history.transpose(1, 0, 2)[None, :, :, :]
    visible = np.broadcast_to(source_history,
                              (history.shape[0], history.shape[1], history.shape[0], history.shape[2])).copy()
    visible[~visibility] = 0
    return visible
