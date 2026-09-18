"""P3.3.5 --validation-manifest interface (plan Task 7.4, TDD).

The scale run trains on the 500-shard manifest but must early-stop and
pair against the FIXED old dev (11,586 samples, architecture manifest).
``resolve_validation_catalog`` decides which catalog validation uses:
default None keeps the current behaviour (training manifest's dev split,
same object, byte-identical readings); an explicit path builds a separate
catalog whose dev split is the fixed one, independent of what the
training manifest is.
"""
from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_tasks.train_p33_nominal import resolve_validation_catalog  # noqa: E402
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_pipeline import (  # noqa: E402
    AgentTrack, MapPolyline, Scene, atomic_savez, build_scene_arrays)
from scenario_lab.p33_spec import file_sha256, load_p33_config  # noqa: E402

CONFIG = load_p33_config()
CODEBOOK = np.zeros((CONFIG["motion_tokens"]["vocabulary_size"], 10),
                    dtype=np.float32)


def _scene(source: str, group_id: str, offset: float) -> Scene:
    def track(track_id, lateral):
        states = np.zeros((61, 7), dtype=np.float64)
        states[:, 0] = np.arange(61) * 0.1 + offset
        states[:, 1] = lateral
        states[:, 2] = 1.0
        states[:, 5:7] = (4.5, 1.8)
        return AgentTrack(track_id, "vehicle", states, np.ones(61, dtype=bool))

    return Scene(source, group_id, np.arange(61) / 10.0, 10,
                 [track("0", 0.0), track("1", 5.0)], [0], [], [0, 1],
                 [MapPolyline("lane", "lane_center",
                              np.c_[np.arange(20), np.zeros(20)])],
                 "fixture.csv", "0" * 64)


def _write_shard(unit_dir: Path, shard_name: str, source: str, split: str,
                 group_id: str, count: int):
    unit_dir.mkdir(parents=True, exist_ok=True)
    keys = None
    rows = []
    stacked: dict[str, np.ndarray] = {}
    samples = []
    for index in range(count):
        arrays, metadata = build_scene_arrays(
            _scene(source, group_id, index * 0.01), 0, CONFIG, CODEBOOK)
        metadata["source"] = source
        metadata["split"] = split
        metadata["sample_id"] = sha256(
            f"{source}|{split}|{group_id}|{index}".encode()).hexdigest()
        samples.append((arrays, metadata))
    keys = list(samples[0][0].keys())
    stacked = {key: np.stack([row[0][key] for row in samples]) for key in keys}
    npz_path = unit_dir / f"{shard_name}.npz"
    atomic_savez(npz_path, stacked)
    for sample_index, (_arrays, metadata) in enumerate(samples):
        row = dict(metadata)
        row["arrays"] = {key: f"{shard_name}.npz::{key}[{sample_index}]"
                         for key in keys}
        rows.append(row)
    (unit_dir / f"{shard_name}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {
        "unit_id": unit_dir.name,
        "path": npz_path.relative_to(unit_dir.parents[1]).as_posix(),
        "index": (unit_dir / f"{shard_name}.jsonl")
            .relative_to(unit_dir.parents[1]).as_posix(),
        "examples": count,
        "sha256": file_sha256(npz_path),
    }


def _mini_dataset(base: Path, shards: list[tuple[str, str, str, int]]) -> Path:
    manifest_shards = []
    for number, (source, split, group_id, count) in enumerate(shards):
        manifest_shards.append(_write_shard(
            base / "units" / f"unit-{number}", "part-00000", source, split,
            group_id, count))
    (base / "DATASET_MANIFEST.json").write_text(
        json.dumps({"validation": {"shards": manifest_shards}}), encoding="utf-8")
    return base / "DATASET_MANIFEST.json"


@pytest.fixture
def training_manifest(tmp_path) -> Path:
    return _mini_dataset(tmp_path / "train_root", [
        ("waymo", "train", "w1", 3),
        ("interaction", "train", "LOC::001", 2),
        ("waymo", "dev", "w9", 2),
    ])


@pytest.fixture
def fixed_dev_manifest(tmp_path) -> Path:
    return _mini_dataset(tmp_path / "fixed_root", [
        ("waymo", "dev", "fixed-1", 4),
        ("interaction", "dev", "LOC::fixed", 3),
    ])


def test_default_none_returns_training_catalog_unchanged(training_manifest):
    catalog = ShardCatalog.from_manifest(training_manifest, CONFIG)
    resolved, provenance, fixed = resolve_validation_catalog(
        None, training_manifest, catalog, CONFIG)
    assert resolved is catalog
    assert provenance == str(training_manifest)
    assert fixed is False


def test_explicit_path_builds_separate_catalog(training_manifest,
                                               fixed_dev_manifest):
    catalog = ShardCatalog.from_manifest(training_manifest, CONFIG)
    resolved, provenance, fixed = resolve_validation_catalog(
        fixed_dev_manifest, training_manifest, catalog, CONFIG)
    assert resolved is not catalog
    assert provenance == str(fixed_dev_manifest)
    assert fixed is True
    assert len(resolved.entries("dev")) == 2
    assert len(resolved.entries("train")) == 0
    # training catalog untouched
    assert len(catalog.entries("train")) == 2


def test_rejects_manifest_without_dev_split(tmp_path, training_manifest):
    empty = _mini_dataset(tmp_path / "empty_root", [
        ("waymo", "train", "w1", 2)])
    catalog = ShardCatalog.from_manifest(training_manifest, CONFIG)
    with pytest.raises(ValueError):
        resolve_validation_catalog(empty, training_manifest, catalog, CONFIG)


def test_fixed_dev_batches_independent_of_training_manifest(
        tmp_path, fixed_dev_manifest):
    """Isolation property (plan Task 7.4): with --validation-manifest set,
    the dev batches depend only on the fixed manifest -- two different
    training manifests yield identical validation streams."""
    catalog_a = ShardCatalog.from_manifest(_mini_dataset(tmp_path / "ta", [
        ("waymo", "train", "wa", 3), ("waymo", "dev", "w9", 2)]), CONFIG)
    catalog_b = ShardCatalog.from_manifest(_mini_dataset(tmp_path / "tb", [
        ("interaction", "train", "LB::2", 5), ("waymo", "dev", "w9", 2)]), CONFIG)
    ids = []
    for catalog in (catalog_a, catalog_b):
        resolved, _, fixed = resolve_validation_catalog(
            fixed_dev_manifest, None, catalog, CONFIG)
        assert fixed is True
        batches = list(iter_split_batches(resolved, split="dev", batch_size=4))
        ids.append([sample for batch in batches for sample in batch["sample_id"]])
    assert ids[0] == ids[1]
    assert len(ids[0]) == 7
