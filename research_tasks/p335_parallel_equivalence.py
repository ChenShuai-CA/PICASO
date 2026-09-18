"""P3.3.5 HPC gate: prove --units-parallel conversion is equivalent to serial.

Run this on the rented CPU node BEFORE the full 674-unit scale conversion:

    python research_tasks/p335_parallel_equivalence.py \
        --data-root Data \
        --codebook-reuse runs/20260913_p333_architecture/data/motion_codebook_v1.npz \
        --output-root <scratch>/p335_equivalence_gate --workers 16

It converts a small unit subset twice into two output directories -- once with
workers=1 (the original serial path) and once with workers=N -- through the
exact functions the real run uses (convert_units / reuse_codebook /
label_and_validate_shards), then compares:

* every .jsonl                       -> byte-identical
* every .npz                         -> array-level exact equality (immune to
                                        zip member timestamps)
* every UNIT.json / CODEBOOK_MANIFEST-> deep-equal after stripping timing
                                        fields (elapsed_seconds)
* npz sha256 equality                -> reported informationally (probes
                                        byte-level determinism of the stack)

Exit 0 only if the parallel run is equivalent.  Timing fields legitimately
differ between ANY two runs (serial included), so they are stripped; every
content-bearing field must match exactly.  The unit subset here is a
mechanism gate, NOT the scale selection: waymo files are the first N sorted
training tfrecords with a record limit, interaction units are the first
groups of one train and one dev location (exercises both split labels).
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_tasks.convert_p33_data import (  # noqa: E402
    convert_units,
    hash_files,
    label_and_validate_shards,
    reuse_codebook,
)
from scenario_lab.p33_pipeline import (  # noqa: E402
    group_interaction_case_files,
    interaction_location,
    resolve_interaction_map,
)
from scenario_lab.p33_spec import file_sha256, load_p33_config  # noqa: E402

TIMING_KEYS = {"elapsed_seconds"}


def strip_timing(value):
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items()
                if key not in TIMING_KEYS}
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


def compare_npz(a: Path, b: Path) -> list[str]:
    problems = []
    with np.load(a, allow_pickle=False) as da, np.load(b, allow_pickle=False) as db:
        if sorted(da.files) != sorted(db.files):
            return [f"key sets differ: {sorted(da.files)} vs {sorted(db.files)}"]
        for key in sorted(da.files):
            xa, xb = da[key], db[key]
            if xa.shape != xb.shape or xa.dtype != xb.dtype:
                problems.append(f"{key}: {xa.shape}/{xa.dtype} vs {xb.shape}/{xb.dtype}")
            elif not np.array_equal(xa, xb):
                delta = np.max(np.abs(xa.astype(np.float64) - xb.astype(np.float64)))
                problems.append(f"{key}: values differ (max |delta| = {delta})")
    return problems


def compare_tree(a: Path, b: Path) -> dict:
    report = {"files_compared": 0, "npz_sha256_equal": 0, "npz_sha256_total": 0,
              "problems": []}
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    if files_a != files_b:
        only_a = set(files_a) - set(files_b)
        only_b = set(files_b) - set(files_a)
        report["problems"].append(f"file sets differ; only serial: {sorted(only_a)}, "
                                  f"only parallel: {sorted(only_b)}")
        return report
    for relative in files_a:
        pa, pb = a / relative, b / relative
        report["files_compared"] += 1
        if relative.suffix == ".npz":
            report["npz_sha256_total"] += 1
            if file_sha256(pa) == file_sha256(pb):
                report["npz_sha256_equal"] += 1
            report["problems"].extend(f"{relative}: {problem}"
                                      for problem in compare_npz(pa, pb))
        elif relative.suffix == ".jsonl":
            if not filecmp.cmp(pa, pb, shallow=False):
                report["problems"].append(f"{relative}: jsonl bytes differ")
        elif relative.suffix == ".json":
            ja = json.loads(pa.read_text(encoding="utf-8"))
            jb = json.loads(pb.read_text(encoding="utf-8"))
            if strip_timing(ja) != strip_timing(jb):
                report["problems"].append(f"{relative}: json differs beyond timing fields")
        else:
            if not filecmp.cmp(pa, pb, shallow=False):
                report["problems"].append(f"{relative}: bytes differ")
    return report


def build_units(data_root: Path, config: dict, waymo_count: int) -> list:
    split = config["split_policy"]["interaction"]
    waymo_paths = sorted((data_root / "Waymo" / "training").glob("*.tfrecord-*"))
    if len(waymo_paths) < waymo_count:
        raise SystemExit(f"need {waymo_count} waymo files, found {len(waymo_paths)}")
    units = [("waymo", [path]) for path in waymo_paths[:waymo_count]]
    recorded = data_root / split["root"]
    for location in (split["train_locations"][0], split["dev_locations"][0]):
        location_paths = [path for path in sorted(recorded.rglob("*.csv"))
                          if interaction_location(path) == location]
        groups = group_interaction_case_files(location_paths)
        if not groups:
            raise SystemExit(f"no interaction case files for location {location}")
        units.extend(("interaction", group) for group in groups[:1])
    return units


def run_once(label: str, units, data_root: Path, output: Path, config: dict,
             pipeline: dict, content_hashes: dict, codebook: Path, workers: int) -> dict:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = time.time()
    converted = convert_units(units, data_root, output, config, pipeline,
                              content_hashes, workers)
    codebook_array, _manifest = reuse_codebook(output, codebook, config)
    validation = label_and_validate_shards(output, converted, config, codebook_array)
    print(json.dumps({"gate_phase": label, "workers": workers,
                      "units": len(converted), "elapsed_seconds": time.time() - started,
                      "counts": validation["counts"]}), flush=True)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/p33/ar_scene_v1.json"))
    parser.add_argument("--pipeline-config", type=Path,
                        default=Path("configs/p33/data_pipeline_scale_v1.json"))
    parser.add_argument("--codebook-reuse", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel workers for the B run (A is always serial)")
    parser.add_argument("--waymo-units", type=int, default=2)
    parser.add_argument("--records-per-file", type=int, default=40,
                        help="waymo diagnostic record limit, keeps the gate fast")
    args = parser.parse_args()

    config = load_p33_config(args.config)
    pipeline = json.loads(args.pipeline_config.read_text(encoding="utf-8"))
    pipeline["waymo"]["records_per_file"] = args.records_per_file
    units = build_units(args.data_root, config, args.waymo_units)
    maps = sorted({resolve_interaction_map(args.data_root, interaction_location(paths[0]))
                   for source, paths in units if source == "interaction"})
    targets = [path for _source, paths in units for path in paths] + maps
    content_hashes = hash_files(targets, 1)
    print(json.dumps({"gate_phase": "units", "total": len(units),
                      "sources": {source: sum(1 for s, _ in units if s == source)
                                  for source, _ in units}}), flush=True)

    serial_dir = args.output_root / "serial"
    parallel_dir = args.output_root / "parallel"
    validation = run_once("serial", units, args.data_root, serial_dir, config,
                          pipeline, content_hashes, args.codebook_reuse, 1)
    counts = validation["counts"]
    if counts.get("split:train", 0) == 0 or counts.get("split:dev", 0) == 0 \
            or counts.get("source:waymo", 0) == 0 \
            or counts.get("source:interaction", 0) == 0:
        raise SystemExit("gate subset lost a source/split; widen --waymo-units")
    run_once("parallel", units, args.data_root, parallel_dir, config,
             pipeline, content_hashes, args.codebook_reuse, max(2, args.workers))

    report = compare_tree(serial_dir, parallel_dir)
    verdict = {
        "equivalent": not report["problems"],
        "files_compared": report["files_compared"],
        "npz_sha256_equal": report["npz_sha256_equal"],
        "npz_sha256_total": report["npz_sha256_total"],
        "note": "sha256 equality is informational (zip timestamps may differ); "
                "array-level equality is the gate",
        "problems": report["problems"][:20],
    }
    print(json.dumps({"gate_verdict": verdict}, indent=2), flush=True)
    (args.output_root / "EQUIVALENCE_VERDICT.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    sys.exit(0 if verdict["equivalent"] else 1)


if __name__ == "__main__":
    main()
