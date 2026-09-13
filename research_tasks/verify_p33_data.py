#!/usr/bin/env python3
"""Independent, bounded verification for a completed P3.3.1 dataset run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import log
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_lab.p33_spec import expected_array_contract, file_sha256  # noqa: E402


REQUIRED_METADATA = {
    "schema_version", "sample_id", "source", "split", "group_id",
    "anchor_track_id", "source_file", "source_content_sha256",
    "coordinate_transform", "visibility_source", "truncation", "arrays",
}


def verify_run(output: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    content = json.loads((output / "CONTENT_MANIFEST.json").read_text(encoding="utf-8"))
    vocabulary = int(config["motion_tokens"]["vocabulary_size"])
    ignore = int(config["motion_tokens"]["ignore_index"])
    array_keys = set(expected_array_contract(config))

    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    group_splits: dict[tuple[str, str], str] = {}
    cross_split_groups: list[str] = []
    counts = Counter()
    token_counts = np.zeros(vocabulary, dtype=np.int64)
    shard_hash_mismatches = []
    shard_example_mismatches = []
    metadata_errors = []
    token_contract_errors = []

    for shard in manifest["validation"]["shards"]:
        npz_path = output / shard["path"]
        index_path = output / shard["index"]
        actual_hash = file_sha256(npz_path)
        if actual_hash != shard["sha256"]:
            shard_hash_mismatches.append(shard["path"])
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        with np.load(npz_path, allow_pickle=False) as data:
            tokens = data["motion_token_target"]
            valid = data["motion_token_valid_mask"]
            if len(tokens) != len(rows) or len(rows) != int(shard["examples"]):
                shard_example_mismatches.append(shard["path"])
            if np.any(tokens[~valid] != ignore) or np.any(tokens[valid] >= vocabulary):
                token_contract_errors.append(shard["path"])
            token_counts += np.bincount(tokens[valid], minlength=vocabulary)[:vocabulary]

        for row in rows:
            counts["examples"] += 1
            counts[f"source:{row.get('source')}"] += 1
            counts[f"split:{row.get('split')}"] += 1
            counts[f"source_split:{row.get('source')}:{row.get('split')}"] += 1
            sample_id = row.get("sample_id", "")
            if sample_id in sample_ids:
                duplicate_ids.append(sample_id)
            sample_ids.add(sample_id)
            group_key = row.get("source", ""), row.get("group_id", "")
            previous = group_splits.setdefault(group_key, row.get("split", ""))
            if previous != row.get("split"):
                cross_split_groups.append(":".join(group_key))
            if set(row) != REQUIRED_METADATA:
                metadata_errors.append(f"{sample_id}:metadata_keys")
            if (len(sample_id) != 64 or len(row.get("source_content_sha256", "")) != 64
                    or row.get("source") not in {"waymo", "interaction"}
                    or row.get("split") not in {"train", "dev"}
                    or set(row.get("arrays", {})) != array_keys):
                metadata_errors.append(f"{sample_id}:metadata_values")

    expected_counts = manifest["validation"]["counts"]
    observed_core_counts = {key: value for key, value in counts.items()
                            if key == "examples" or key.startswith(("source:", "split:", "source_split:"))}
    expected_core_counts = {key: value for key, value in expected_counts.items()
                            if key == "examples" or key.startswith(("source:", "split:", "source_split:"))}
    total_tokens = int(token_counts.sum())
    probabilities = token_counts[token_counts > 0] / max(total_tokens, 1)
    normalized_entropy = float(-(probabilities * np.log(probabilities)).sum() / log(vocabulary))
    checks = {
        "all_shard_hashes_match": not shard_hash_mismatches,
        "all_shard_example_counts_match": not shard_example_mismatches,
        "sample_ids_unique": not duplicate_ids,
        "metadata_contract_passed": not metadata_errors,
        "motion_token_contract_passed": not token_contract_errors,
        "all_motion_tokens_occupied": int(np.count_nonzero(token_counts)) == vocabulary,
        "no_cross_split_group_leakage": not cross_split_groups,
        "manifest_counts_match": observed_core_counts == expected_core_counts,
        "heldout_trajectory_content_unread": content.get("heldout_trajectory_content_read") is False,
    }
    return {
        "phase": "P3.3.1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": dict(sorted(observed_core_counts.items())),
        "independent_groups": len(group_splits),
        "motion_tokens": {
            "valid_targets": total_tokens,
            "occupied": int(np.count_nonzero(token_counts)),
            "vocabulary_size": vocabulary,
            "minimum_count": int(token_counts.min()),
            "maximum_count": int(token_counts.max()),
            "maximum_share": float(token_counts.max() / max(total_tokens, 1)),
            "normalized_entropy": normalized_entropy,
            "counts": token_counts.tolist(),
        },
        "failures": {
            "shard_hash_mismatches": shard_hash_mismatches,
            "shard_example_mismatches": shard_example_mismatches,
            "duplicate_sample_ids": duplicate_ids[:20],
            "metadata_errors": metadata_errors[:20],
            "token_contract_errors": token_contract_errors,
            "cross_split_groups": sorted(set(cross_split_groups))[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/p33/ar_scene_v1.json"))
    args = parser.parse_args()
    result = verify_run(args.output, args.config)
    target = args.output / "VERIFICATION.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
