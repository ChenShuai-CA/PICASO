"""Create the metadata-only P3.3.0 frozen specification evidence pack."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_lab.p33_spec import (
    SCHEMA_PATH,
    build_p33_inventory,
    canonical_sha256,
    file_sha256,
    load_p33_config,
    validate_json_schema_contract,
)


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def freeze_spec(data_root: Path, config_path: Path, schema_path: Path,
                output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "data_inventory.json", output / "SPEC_VALIDATION.json",
               output / "FROZEN.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("P3.3.0 output already exists; use a fresh output directory")

    config = load_p33_config(config_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_json_schema_contract(schema, config)
    inventory = build_p33_inventory(data_root, config)
    inventory_path = output / "data_inventory.json"
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")

    validation = {
        "phase": "P3.3.0",
        "status": "complete",
        "spec_version": config["spec_version"],
        "design_status": config["status"],
        "checks": {
            "config_invariants": "pass",
            "schema_matches_executable_array_contract": "pass",
            "expected_file_counts": "pass",
            "file_roles_are_disjoint": "pass",
            "heldout_has_no_development_stage": "pass",
            "metadata_only_inventory": "pass"
        },
        "heldout_trajectory_content_read": False,
        "data_counts": inventory["counts_by_source"],
        "role_counts": inventory["counts_by_role"],
        "stage_counts": inventory["stage_counts"],
        "total_size_bytes": inventory["total_size_bytes"],
        "inventory_fingerprint_sha256": inventory["inventory_fingerprint_sha256"]
    }
    validation_path = output / "SPEC_VALIDATION.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    frozen = {
        "phase": "P3.3.0",
        "status": "design_frozen_not_trained",
        "spec_version": config["spec_version"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": _git_head(),
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path)
        },
        "scene_schema": {
            "path": schema_path.as_posix(),
            "sha256": file_sha256(schema_path)
        },
        "inventory": {
            "path": inventory_path.as_posix(),
            "file_sha256": file_sha256(inventory_path),
            "metadata_fingerprint_sha256": inventory["inventory_fingerprint_sha256"]
        },
        "validation": {
            "path": validation_path.as_posix(),
            "sha256": file_sha256(validation_path)
        },
        "freeze_fingerprint_sha256": canonical_sha256({
            "spec_version": config["spec_version"],
            "config_sha256": file_sha256(config_path),
            "schema_sha256": file_sha256(schema_path),
            "inventory_fingerprint_sha256": inventory["inventory_fingerprint_sha256"]
        }),
        "next_phase": "P3.3.1_streaming_data_pipeline"
    }
    (output / "FROZEN.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/p33/ar_scene_v1.json"))
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_spec(args.data_root, args.config, args.schema, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
