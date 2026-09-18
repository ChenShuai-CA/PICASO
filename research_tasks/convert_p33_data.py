"""Run the resumable P3.3.1 public-data converter."""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_lab.p33_pipeline import (  # noqa: E402
    BalancedMotionReservoir,
    batched_motion_vectors,
    assign_motion_tokens,
    atomic_savez,
    build_scene_arrays,
    fit_balanced_minibatch_kmeans,
    group_interaction_case_files,
    interaction_location,
    iter_interaction_scenes,
    iter_waymo_scenes,
    resolve_interaction_map,
    write_json,
    write_jsonl,
)
from scenario_lab.p33_spec import (  # noqa: E402
    file_sha256,
    load_p33_config,
    validate_p33_inventory,
    validate_scene_arrays,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unit_id(source: str, paths: list[Path], data_root: Path) -> str:
    relative = [path.relative_to(data_root).as_posix() for path in paths]
    digest = sha256((source + "\n" + "\n".join(relative)).encode()).hexdigest()[:16]
    label = paths[0].stem.replace(".tfrecord", "")[:40]
    return f"{source}-{label}-{digest}"


class UnitWriter:
    def __init__(self, unit_dir: Path, maximum: int):
        self.unit_dir = unit_dir
        self.maximum = maximum
        self.buffer: list[tuple[dict[str, np.ndarray], dict]] = []
        self.shards = []
        self.count = 0

    def add(self, arrays: dict[str, np.ndarray], metadata: dict) -> None:
        self.buffer.append((arrays, metadata))
        if len(self.buffer) >= self.maximum:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_index = len(self.shards)
        stem = f"part-{shard_index:05d}"
        npz_path = self.unit_dir / f"{stem}.npz"
        index_path = self.unit_dir / f"{stem}.jsonl"
        keys = list(self.buffer[0][0])
        batched = {key: np.stack([row[0][key] for row in self.buffer]) for key in keys}
        rows = []
        for sample_index, (_arrays, metadata) in enumerate(self.buffer):
            metadata = dict(metadata)
            metadata["arrays"] = {
                key: f"{npz_path.name}::{key}[{sample_index}]" for key in keys
            }
            rows.append(metadata)
        atomic_savez(npz_path, batched)
        write_jsonl(index_path, rows)
        self.shards.append({
            "npz": npz_path.name,
            "index": index_path.name,
            "examples": len(self.buffer),
            "npz_sha256_before_token_labeling": file_sha256(npz_path),
        })
        self.count += len(self.buffer)
        self.buffer.clear()


def split_for_interaction(location: str, config: dict) -> str:
    policy = config["split_policy"]["interaction"]
    if location in policy["train_locations"]:
        return "train"
    if location in policy["dev_locations"]:
        return "dev"
    raise ValueError(f"P3.3.1 attempted to read heldout/unknown INTERACTION location {location}")


def convert_unit(source: str, paths: list[Path], data_root: Path, output: Path,
                 config: dict, pipeline: dict, content_hashes: dict[str, str]) -> dict:
    started = time.time()
    identifier = unit_id(source, paths, data_root)
    unit_dir = output / "units" / identifier
    done = unit_dir / "UNIT.json"
    if done.exists():
        result = load_json(done)
        if all((unit_dir / shard["npz"]).exists() and (unit_dir / shard["index"]).exists()
               for shard in result["shards"]):
            result["resumed"] = True
            return result
        raise ValueError(f"incomplete unit has a stale marker: {identifier}")
    # a leftover directory without a UNIT.json marker is a partially written
    # unit from an interrupted run (observed: OOM-killed process left an empty
    # dir, and mkdir(exist_ok=False) crashed the resume); wipe and reconvert
    if unit_dir.exists():
        shutil.rmtree(unit_dir)
    unit_dir.mkdir(parents=True, exist_ok=False)
    writer = UnitWriter(unit_dir, config["data"]["storage"]["examples_per_shard"])
    errors = []
    scene_count = 0
    if source == "waymo":
        iterator = iter_waymo_scenes(
            paths[0], content_hashes[str(paths[0])],
            pipeline["waymo"]["records_per_file"],
        )
    else:
        iterator = iter_interaction_scenes(
            paths, data_root, content_hashes,
            pipeline["interaction"]["window_stride_frames"],
            pipeline["interaction"]["anchors_per_window"],
        )
    for scene in iterator:
        scene_count += 1
        for anchor_index in scene.anchor_indices:
            try:
                arrays, metadata = build_scene_arrays(scene, anchor_index, config)
                if source == "interaction":
                    metadata["split"] = split_for_interaction(interaction_location(paths[0]), config)
                stamp = int(round(scene.timestamps[scene.current_index] * 10))
                raw_id = f"{source}\x1f{scene.group_id}\x1f{stamp}\x1f{metadata['anchor_track_id']}"
                metadata["sample_id"] = sha256(raw_id.encode()).hexdigest()
                writer.add(arrays, metadata)
            except (ValueError, IndexError, FloatingPointError) as exc:
                errors.append({
                    "group_id": scene.group_id,
                    "anchor_index": anchor_index,
                    "error": str(exc),
                })
    writer.flush()
    result = {
        "unit_id": identifier,
        "source": source,
        "source_files": [path.relative_to(data_root).as_posix() for path in paths],
        "source_hashes": {path.relative_to(data_root).as_posix(): content_hashes[str(path)]
                          for path in paths},
        "scenes": scene_count,
        "examples": writer.count,
        "errors": errors,
        "shards": writer.shards,
        "resumed": False,
        "token_status": "pending_codebook",
        "elapsed_seconds": time.time() - started,
    }
    if source == "interaction":
        map_path = resolve_interaction_map(data_root, interaction_location(paths[0]))
        result["map_dependency"] = {
            "path": map_path.relative_to(data_root).as_posix(),
            "sha256": content_hashes[str(map_path)],
        }
    write_json(done, result)
    return result


def iter_shards(output: Path, units: list[dict]):
    for unit in units:
        unit_dir = output / "units" / unit["unit_id"]
        for shard in unit["shards"]:
            yield unit, unit_dir / shard["npz"], unit_dir / shard["index"]


def _hash_file(path: Path) -> tuple[str, str]:
    return str(path), file_sha256(path)


def hash_files(paths: list[Path], workers: int) -> dict[str, str]:
    """SHA-256 every path.  workers>1 fans files out across forked processes;
    the returned mapping is identical to the serial loop (dict content is
    order-free)."""
    if workers <= 1:
        return {str(path): file_sha256(path) for path in paths}
    import multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("--units-parallel >1 requires a fork-capable platform "
                           "(Linux/WSL); run with --units-parallel 1 otherwise")
    with mp.get_context("fork").Pool(max(1, min(workers, len(paths)))) as pool:
        return dict(pool.map(_hash_file, paths, chunksize=1))


def _convert_unit_job(unit, data_root, output, config, pipeline, content_hashes) -> dict:
    source, paths = unit
    return convert_unit(source, paths, data_root, output, config, pipeline, content_hashes)


def convert_units(units, data_root: Path, output: Path, config: dict, pipeline: dict,
                  content_hashes: dict[str, str], workers: int = 1) -> list[dict]:
    """Convert every unit, preserving input unit order.

    workers=1 is the original serial loop (unchanged behaviour).  workers>1
    runs convert_unit in a fork Pool; results are collected with imap so the
    returned list (and therefore DATASET_MANIFEST unit ordering) matches the
    serial order exactly.  Each unit writes only to its own units/<id>/
    directory, so workers never share mutable state.  The endgame phases
    (codebook, label_and_validate_shards, previews, manifests) stay serial
    and are not part of this helper.
    """
    def progress(number: int, unit_result: dict) -> None:
        print(json.dumps({
            "progress": f"{number}/{len(units)}", "source": unit_result["source"],
            "unit_id": unit_result["unit_id"], "scenes": unit_result["scenes"],
            "examples": unit_result["examples"], "resumed": unit_result["resumed"],
            "elapsed_seconds": unit_result.get("elapsed_seconds"),
        }), flush=True)

    if workers <= 1:
        converted = []
        for number, (source, paths) in enumerate(units, start=1):
            unit_result = convert_unit(source, paths, data_root, output, config,
                                       pipeline, content_hashes)
            converted.append(unit_result)
            progress(number, unit_result)
        return converted
    import multiprocessing as mp
    from functools import partial
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("--units-parallel >1 requires a fork-capable platform "
                           "(Linux/WSL); run with --units-parallel 1 otherwise")
    job = partial(_convert_unit_job, data_root=data_root, output=output, config=config,
                  pipeline=pipeline, content_hashes=content_hashes)
    converted = []
    with mp.get_context("fork").Pool(max(1, min(workers, len(units)))) as pool:
        for number, unit_result in enumerate(pool.imap(job, units, chunksize=1), start=1):
            converted.append(unit_result)
            progress(number, unit_result)
    return converted


def fit_codebook(output: Path, units: list[dict], config: dict, pipeline: dict) -> tuple[np.ndarray, dict]:
    settings = pipeline["motion_codebook"]
    reservoir = BalancedMotionReservoir(
        settings["capacity_per_source_agent_type"], settings["random_seed"])
    for shard_number, (_unit, npz_path, index_path) in enumerate(iter_shards(output, units), 1):
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        with np.load(npz_path, allow_pickle=False) as data:
            batched = {key: data[key] for key in (
                "future_xy", "future_valid_mask", "agent_history", "state_valid_mask",
                "agent_type")}
        vectors, valid = batched_motion_vectors(batched, config)
        train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
        types = batched["agent_type"]
        source = rows[0]["source"]
        for type_id in np.unique(types[train]):
            if not type_id:
                continue
            mask = valid & train[:, None, None] & (types[:, :, None] == type_id)
            reservoir.add(f"{source}:type_{int(type_id)}", vectors[mask])
        print(json.dumps({
            "phase": "codebook_reservoir",
            "shard": shard_number,
            "path": str(npz_path.relative_to(output)),
        }), flush=True)
    strata = reservoir.arrays()
    codebook, fit = fit_balanced_minibatch_kmeans(
        strata,
        config["motion_tokens"]["vocabulary_size"],
        settings["batch_size"],
        settings["max_steps"],
        settings["convergence_tolerance"],
        settings["random_seed"],
    )
    codebook_path = output / "motion_codebook_v1.npz"
    atomic_savez(codebook_path, {"centroids": codebook})
    manifest = {
        "version": config["motion_tokens"]["version"],
        "fit_split": "train_only",
        "fit_weighting": "uniform_source_agent_type_strata_with_replacement",
        "settings": settings,
        "strata_seen": reservoir.seen,
        "strata_retained": {key: len(value) for key, value in strata.items()},
        "fit": fit,
        "codebook_path": codebook_path.name,
        "codebook_sha256": file_sha256(codebook_path),
    }
    write_json(output / "CODEBOOK_MANIFEST.json", manifest)
    return codebook, manifest


def reuse_codebook(output: Path, frozen_path: Path,
                   config: dict) -> tuple[np.ndarray, dict]:
    """Copy a frozen motion codebook instead of refitting (P3.3.5 scale run).

    Keeps the 100->500 shard comparison isolated to data scale: the copied
    file is byte-identical to the frozen source, so downstream readers
    (train/eval load ``motion_codebook_v1.npz`` next to DATASET_MANIFEST)
    are unchanged.  Raises ValueError on shape or content mismatch.
    """
    expected = (config["motion_tokens"]["vocabulary_size"],
                config["motion_tokens"]["vector_dimension"])
    with np.load(frozen_path, allow_pickle=False) as data:
        if "centroids" not in data.files:
            raise ValueError(f"{frozen_path}: no 'centroids' array")
        codebook = np.array(data["centroids"])
    if codebook.shape != expected:
        raise ValueError(f"{frozen_path}: centroids shape {codebook.shape} "
                         f"!= config vocabulary/dimension {expected}")
    target = output / "motion_codebook_v1.npz"
    shutil.copyfile(frozen_path, target)
    manifest = {
        "version": config["motion_tokens"]["version"],
        "reused_frozen": True,
        "source_path": str(frozen_path),
        "source_sha256": file_sha256(frozen_path),
        "shape": list(codebook.shape),
        "codebook_path": target.name,
        "codebook_sha256": file_sha256(target),
    }
    write_json(output / "CODEBOOK_MANIFEST.json", manifest)
    return codebook, manifest


def label_and_validate_shards(output: Path, units: list[dict], config: dict,
                              codebook: np.ndarray) -> dict:
    counts = Counter()
    max_roundtrip = 0.0
    seen_group_splits: dict[tuple[str, str], str] = {}
    shard_records = []
    for shard_number, (unit, npz_path, index_path) in enumerate(iter_shards(output, units), 1):
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        with np.load(npz_path, allow_pickle=False) as loaded:
            batched = {key: loaded[key] for key in loaded.files}
        vectors, valid = batched_motion_vectors(batched, config)
        token_targets = np.full(valid.shape,
                                config["motion_tokens"]["ignore_index"], dtype=np.uint8)
        for start in range(0, len(rows), 128):
            stop = min(start + 128, len(rows))
            token_targets[start:stop] = assign_motion_tokens(
                vectors[start:stop], valid[start:stop], codebook,
                config["motion_tokens"]["ignore_index"])
        batched["motion_token_target"] = token_targets
        batched["motion_token_valid_mask"] = valid
        for sample_index, metadata in enumerate(rows):
            arrays = {key: batched[key][sample_index] for key in batched}
            validate_scene_arrays(arrays, config)
            origin = np.asarray([
                metadata["coordinate_transform"]["origin_x_m"],
                metadata["coordinate_transform"]["origin_y_m"],
            ])
            yaw = metadata["coordinate_transform"]["yaw_rad"]
            c, s = np.cos(yaw), np.sin(yaw)
            inverse = np.asarray([[c, -s], [s, c]])
            local = arrays["future_xy"][arrays["future_valid_mask"]]
            if len(local):
                world = (inverse @ local.T).T + origin
                recovered = (inverse.T @ (world - origin).T).T
                max_roundtrip = max(max_roundtrip, float(np.max(np.abs(recovered - local))))
            key = metadata["source"], metadata["group_id"]
            old = seen_group_splits.setdefault(key, metadata["split"])
            if old != metadata["split"]:
                raise ValueError(f"cross-split group leakage: {key}")
            counts[f"source:{metadata['source']}"] += 1
            counts[f"split:{metadata['split']}"] += 1
            counts[f"source_split:{metadata['source']}:{metadata['split']}"] += 1
            counts["examples"] += 1
            counts["agent_truncations"] += int(
                metadata["truncation"]["agents_available"] > metadata["truncation"]["agents_kept"])
            counts["map_truncations"] += int(
                metadata["truncation"]["map_polylines_available"]
                > metadata["truncation"]["map_polylines_kept"])
        atomic_savez(npz_path, batched)
        shard_hash = file_sha256(npz_path)
        shard_records.append({
            "unit_id": unit["unit_id"], "path": str(npz_path.relative_to(output)),
            "index": str(index_path.relative_to(output)), "examples": len(rows),
            "sha256": shard_hash,
        })
        print(json.dumps({
            "phase": "token_label_and_validation",
            "shard": shard_number,
            "path": str(npz_path.relative_to(output)),
            "examples": len(rows),
        }), flush=True)
    by_unit = {}
    for shard in shard_records:
        by_unit.setdefault(shard["unit_id"], []).append(shard)
    for unit in units:
        unit["token_status"] = "labelled_and_validated"
        unit["final_shards"] = by_unit.get(unit["unit_id"], [])
        stored = dict(unit)
        stored["resumed"] = False
        write_json(output / "units" / unit["unit_id"] / "UNIT.json", stored)
    return {
        "counts": dict(sorted(counts.items())),
        "independent_groups": len(seen_group_splits),
        "coordinate_roundtrip_max_abs_error_m": max_roundtrip,
        "shards": shard_records,
    }


def render_previews(output: Path, validation: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = []
    by_source = {}
    for shard in validation["shards"]:
        index_path = output / shard["index"]
        rows = index_path.read_text(encoding="utf-8").splitlines()
        if not rows:
            continue
        metadata = json.loads(rows[0])
        if metadata["source"] in by_source:
            continue
        with np.load(output / shard["path"], allow_pickle=False) as data:
            arrays = {key: data[key][0] for key in data.files}
        by_source[metadata["source"]] = (metadata, arrays)
    for source, (metadata, arrays) in by_source.items():
        figure, axes = plt.subplots(figsize=(7, 7))
        for points, mask in zip(arrays["map_polylines"], arrays["map_point_mask"]):
            if mask.any():
                axes.plot(points[mask, 0], points[mask, 1], color="#b9bec7", linewidth=0.7)
        for slot in np.flatnonzero(arrays["agent_present_mask"]):
            hist = arrays["agent_history"][slot]
            hist_mask = arrays["state_valid_mask"][slot]
            future = arrays["future_xy"][slot]
            future_mask = arrays["future_valid_mask"][slot]
            color = "#d62728" if slot == 0 else "#1f77b4"
            axes.plot(hist[hist_mask, 0], hist[hist_mask, 1], color=color, linewidth=1.5)
            axes.plot(future[future_mask, 0], future[future_mask, 1], color=color,
                      linewidth=1.0, linestyle="--")
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlim(-85, 85)
        axes.set_ylim(-85, 85)
        axes.set_title(f"P3.3.1 {source} smoke sample\n{metadata['group_id']}")
        axes.set_xlabel("anchor-local x [m]")
        axes.set_ylabel("anchor-local y [m]")
        axes.grid(alpha=0.15)
        path = output / f"preview_{source}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        created.append(path.name)
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--inventory", type=Path,
                        default=Path("runs/20260913_p330_spec_v2/data_inventory.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/p33/ar_scene_v1.json"))
    parser.add_argument("--pipeline-config", type=Path,
                        default=Path("configs/p33/data_pipeline_v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/20260913_p331_data_pipeline/smoke"))
    parser.add_argument("--waymo-records-per-file", type=int)
    parser.add_argument("--codebook-reuse", type=Path, default=None,
                        help="frozen motion codebook .npz to reuse instead of "
                             "refitting (P3.3.5 scale isolation; default refits)")
    parser.add_argument("--units-parallel", type=int, default=1,
                        help="worker processes for per-file SHA-256 hashing and "
                             "per-unit conversion (P3.3.5 HPC run; 1 = original "
                             "serial behaviour; unit order and all endgame "
                             "phases are unaffected)")
    args = parser.parse_args()
    started = time.time()
    config = load_p33_config(args.config)
    pipeline = load_json(args.pipeline_config)
    if args.waymo_records_per_file is not None:
        pipeline["waymo"]["records_per_file"] = args.waymo_records_per_file
    inventory = load_json(args.inventory)
    validate_p33_inventory(inventory, config)
    if pipeline["scene_spec_version"] != config["spec_version"]:
        raise ValueError("pipeline/config version mismatch")
    stage = pipeline["stage"]
    selected = [row for row in inventory["files"] if stage in row["stages"]]
    if any(row["split_role"] == "final_confirmation" for row in selected):
        raise ValueError("heldout file selected for P3.3.1")
    waymo_paths = [args.data_root / row["relative_path"] for row in selected
                   if row["source"] == "waymo"]
    interaction_paths = [args.data_root / row["relative_path"] for row in selected
                         if row["source"] == "interaction"]
    units = [("waymo", [path]) for path in waymo_paths]
    units.extend(("interaction", paths) for paths in group_interaction_case_files(interaction_paths))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "RUN_CONFIG.json", {
        "config": str(args.config), "config_sha256": file_sha256(args.config),
        "pipeline_config": str(args.pipeline_config),
        "pipeline_config_sha256": file_sha256(args.pipeline_config),
        "inventory": str(args.inventory), "inventory_sha256": file_sha256(args.inventory),
        "effective_pipeline": pipeline,
    })
    interaction_maps = sorted({
        resolve_interaction_map(args.data_root, interaction_location(paths[0]))
        for source, paths in units if source == "interaction"
    })
    workers = max(1, int(args.units_parallel))
    hash_targets = [path for _source, paths in units for path in paths]
    hash_targets.extend(interaction_maps)
    content_hashes = hash_files(hash_targets, workers)
    converted = convert_units(units, args.data_root, args.output, config, pipeline,
                              content_hashes, workers)
    if any(unit["errors"] for unit in converted):
        write_json(args.output / "CONVERSION_ERRORS.json", converted)
        raise RuntimeError("conversion produced sample errors; inspect CONVERSION_ERRORS.json")
    if args.codebook_reuse is not None:
        codebook, codebook_manifest = reuse_codebook(args.output, args.codebook_reuse,
                                                     config)
    else:
        codebook, codebook_manifest = fit_codebook(args.output, converted, config,
                                                   pipeline)
    validation = label_and_validate_shards(args.output, converted, config, codebook)
    tolerance = pipeline["validation"]["coordinate_roundtrip_tolerance_m"]
    checks = {
        "heldout_trajectory_content_unread": True,
        "both_sources_present": all(validation["counts"].get(f"source:{source}", 0) > 0
                                    for source in ("waymo", "interaction")),
        "train_and_dev_present": all(validation["counts"].get(f"split:{split}", 0) > 0
                                     for split in ("train", "dev")),
        "coordinate_roundtrip_within_tolerance":
            validation["coordinate_roundtrip_max_abs_error_m"] <= tolerance,
        "cross_split_groups": 0,
        "finite_and_mask_contract": "all_samples_passed",
        "resumable_units": len(converted),
        "maximum_resident_examples": config["data"]["storage"]["examples_per_shard"],
    }
    if not all(value is True or isinstance(value, (int, str)) for value in checks.values()):
        raise AssertionError("unexpected G0 check type")
    if not checks["both_sources_present"] or not checks["train_and_dev_present"] \
            or not checks["coordinate_roundtrip_within_tolerance"]:
        raise RuntimeError(f"G0 smoke checks failed: {checks}")
    previews = render_previews(args.output, validation)
    content_manifest = {
        "trajectory_files_read": [
            {"path": path.relative_to(args.data_root).as_posix(),
             "sha256": content_hashes[str(path)], "split_role": "development"}
            for _source, paths in units for path in paths
        ],
        "map_files_read": [
            {"path": path.relative_to(args.data_root).as_posix(),
             "sha256": content_hashes[str(path)], "split_role": "development"}
            for path in interaction_maps
        ],
        "heldout_trajectory_content_read": False,
    }
    write_json(args.output / "CONTENT_MANIFEST.json", content_manifest)
    manifest = {
        "phase": "P3.3.1",
        "status": "smoke_data_pipeline_complete",
        "stage": stage,
        "diagnostic_waymo_record_limit": pipeline["waymo"]["records_per_file"],
        "units": converted,
        "codebook": codebook_manifest,
        "validation": validation,
        "checks": checks,
        "previews": previews,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "DATASET_MANIFEST.json", manifest)
    write_json(args.output / "G0_VALIDATION.json", checks)
    print(json.dumps({
        "status": manifest["status"],
        "counts": validation["counts"],
        "shards": len(validation["shards"]),
        "elapsed_seconds": manifest["elapsed_seconds"],
        "heldout_trajectory_content_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
