"""P3.3.2 loader contract tests: bounded reads, hashes, balance, determinism."""
import gc
import json
import weakref
from hashlib import sha256

import numpy as np
import pytest

from scenario_lab.p33_dataset import (
    SOURCE_IDS,
    SceneLoader,
    ShardCatalog,
    ShardReader,
    audit_catalog,
    iter_split_batches,
)
from scenario_lab.p33_pipeline import AgentTrack, MapPolyline, Scene, atomic_savez, build_scene_arrays
from scenario_lab.p33_spec import expected_array_contract, file_sha256, load_p33_config


CONFIG = load_p33_config()
CODEBOOK = np.zeros((CONFIG["motion_tokens"]["vocabulary_size"], 10), dtype=np.float32)
SCHEMA_VERSION = CONFIG["data"]["schema_version"]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _scene(source: str, group_id: str, offset: float) -> Scene:
    def track(track_id, lateral):
        states = np.zeros((61, 7), dtype=np.float64)
        states[:, 0] = np.arange(61) * 0.1 + offset
        states[:, 1] = lateral
        states[:, 2] = 1.0
        states[:, 5:7] = (4.5, 1.8)
        return AgentTrack(track_id, "vehicle", states, np.ones(61, dtype=bool))

    return Scene(
        source, group_id, np.arange(61) / 10.0, 10,
        [track("0", 0.0), track("1", 5.0)], [0], [], [0, 1],
        [MapPolyline("lane", "lane_center", np.c_[np.arange(20), np.zeros(20)])],
        "fixture.csv", "0" * 64,
    )


def _sample(source: str, split: str, group_id: str, index: int):
    arrays, metadata = build_scene_arrays(_scene(source, group_id, index * 0.01), 0, CONFIG, CODEBOOK)
    metadata["source"] = source
    metadata["split"] = split
    metadata["sample_id"] = sha256(f"{source}|{split}|{group_id}|{index}".encode()).hexdigest()
    return arrays, metadata


def _write_shard(unit_dir, shard_name, source, split, group_ids, count, splits=None):
    unit_dir.mkdir(parents=True, exist_ok=True)
    row_splits = splits if splits is not None else [split] * count
    assert len(row_splits) == count
    samples = [_sample(source, row_splits[i], group_ids[i % len(group_ids)], i) for i in range(count)]
    keys = list(samples[0][0].keys())
    npz_path = unit_dir / f"{shard_name}.npz"
    atomic_savez(npz_path, {key: np.stack([row[0][key] for row in samples]) for key in keys})
    rows = []
    for sample_index, (_arrays, metadata) in enumerate(samples):
        row = dict(metadata)
        row["arrays"] = {key: f"{shard_name}.npz::{key}[{sample_index}]" for key in keys}
        rows.append(row)
    _write_jsonl(unit_dir / f"{shard_name}.jsonl", rows)
    return {
        "unit_id": unit_dir.name,
        "path": npz_path.relative_to(unit_dir.parents[1]).as_posix(),
        "index": (unit_dir / f"{shard_name}.jsonl").relative_to(unit_dir.parents[1]).as_posix(),
        "examples": count,
        "sha256": file_sha256(npz_path),
    }


@pytest.fixture
def data_root(tmp_path):
    shards = []
    base = tmp_path / "runs" / "smoke"
    shards.append(_write_shard(base / "units" / "waymo-a", "part-00000",
                                "waymo", "train", ["w1", "w2"], 4))
    shards.append(_write_shard(base / "units" / "waymo-b", "part-00000",
                                "waymo", "train", ["w3", "w3", "w4"], 3,
                                splits=["train", "train", "dev"]))
    shards.append(_write_shard(base / "units" / "waymo-c", "part-00000",
                                "waymo", "dev", ["w9"], 2))
    shards.append(_write_shard(base / "units" / "int-a", "part-00000",
                                "interaction", "train", ["LOC::001"], 5))
    shards.append(_write_shard(base / "units" / "int-b", "part-00000",
                                "interaction", "dev", ["LOC::009"], 3))
    _write_json(base / "DATASET_MANIFEST.json", {"validation": {"shards": shards}})
    return base


def _catalog(data_root):
    return ShardCatalog.from_manifest(data_root / "DATASET_MANIFEST.json", CONFIG)


def test_catalog_counts_match_manifest_jsonl_and_schema(data_root):
    catalog = _catalog(data_root)
    assert len(catalog.entries()) == 5
    train = catalog.entries(split="train")
    # waymo-b is a mixed-split shard (Waymo splits by Scenario hash, not by file).
    assert {entry.path.split("/")[1] for entry in train} == {"waymo-a", "waymo-b", "int-a"}
    assert sum(len(entry.rows_by_split["train"]) for entry in train) == 11
    audit = audit_catalog(catalog)
    assert audit["status"] == "pass"
    assert audit["counts"]["source:waymo"] == 9
    assert audit["counts"]["source:interaction"] == 8
    assert audit["counts"]["split:train"] == 11
    assert audit["counts"]["split:dev"] == 6
    assert audit["counts"]["source_split:waymo:train"] == 6
    assert audit["schema_version_mismatches"] == []


def test_hash_verification_detects_tampering(data_root):
    catalog = _catalog(data_root)
    cache = {}
    assert catalog.verify_hashes(cache) == {}
    assert len(cache) == 5
    tampered = data_root / catalog.entries(split="dev")[0].path
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    mismatches = _catalog(data_root).verify_hashes({})
    assert len(mismatches) == 1


def test_balanced_batches_have_fixed_composition_and_seed_reproducibility(data_root):
    catalog = _catalog(data_root)
    loader = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    batches = list(loader.iter_batches(limit=4))
    assert len(batches) == 3  # one epoch = one pass over the smaller (interaction, pool 5) source
    for batch in batches:
        assert set(batch["sample_source"]) == {"waymo", "interaction"}
        assert [sum(1 for s in batch["sample_source"] if s == src) for src in ("waymo", "interaction")] == [2, 2]
        assert set(batch["split"]) == {"train"}
    ids_a = [batch["sample_id"] for batch in batches]

    loader_b = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    assert [batch["sample_id"] for batch in loader_b.iter_batches(limit=4)] == ids_a

    loader_c = SceneLoader(catalog, split="train", batch_per_source=2, seed=99)
    assert [batch["sample_id"] for batch in loader_c.iter_batches(limit=4)] != ids_a


def test_collate_batch_contract(data_root):
    catalog = _catalog(data_root)
    loader = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    batch = next(iter(loader.iter_batches(limit=1)))
    size = len(batch["sample_id"])
    for name, (shape, dtype) in expected_array_contract(CONFIG).items():
        assert batch[name].shape == (size,) + shape, name
        assert batch[name].dtype == dtype, name
    assert batch["source_id"].dtype == np.int64
    assert set(batch["source_id"].tolist()) <= {0, 1}
    assert [SOURCE_IDS[source] for source in batch["sample_source"]] == batch["source_id"].tolist()
    assert batch["agents_truncated"].dtype == np.bool_
    assert batch["map_truncated"].dtype == np.bool_
    assert len(batch["group_id"]) == size
    assert batch["future_xy"].shape == (size, 16, 50, 2)


def test_epoch_exposure_reports_smaller_source_pass(data_root):
    catalog = _catalog(data_root)
    loader = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    consumed = sum(1 for _batch in loader.iter_batches())
    # interaction train pool is 5 samples (smaller) -> ceil(5/2)=3 batches; it cycles into a
    # second seeded pass to fill the third batch.  waymo (6) never finishes its pass.
    assert consumed == 3
    exposure = loader.exposure()
    assert exposure["epoch_source"] == "interaction"
    assert exposure["samples_emitted"]["interaction"] == 6
    assert exposure["passes_completed"]["interaction"] == 1
    assert exposure["samples_emitted"]["waymo"] == 6
    assert loader.max_resident_shards == 2  # one per source


def test_loader_state_restores_the_exact_stream(data_root):
    catalog = _catalog(data_root)
    loader = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    first = [batch["sample_id"] for batch in loader.iter_batches(limit=2)]
    state = loader.state()
    resumed_ids = [batch["sample_id"] for batch in loader.iter_batches(limit=2)]
    fresh = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    fresh.set_state(state)
    assert [batch["sample_id"] for batch in fresh.iter_batches(limit=2)] == resumed_ids
    assert len(first) == 2


def test_eval_iteration_is_shard_ordered_and_complete(data_root):
    catalog = _catalog(data_root)
    batches = list(iter_split_batches(catalog, split="dev", batch_size=3))
    ids = [sample_id for batch in batches for sample_id in batch["sample_id"]]
    expected = []
    for entry in catalog.entries(split="dev"):
        rows = [json.loads(line) for line in entry.index_path.read_text(encoding="utf-8").splitlines()]
        expected.extend(rows[index]["sample_id"] for index in entry.rows_by_split["dev"])
    assert len(ids) == 6
    assert ids == expected


def test_advance_epoch_continues_the_stream_deterministically(data_root):
    catalog = _catalog(data_root)
    loader = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    first_epoch = [batch["sample_id"] for batch in loader.iter_batches()]
    with pytest.raises(StopIteration):
        next(iter(loader.iter_batches()))
    summary = loader.advance_epoch()
    assert summary["epoch"] == 1
    second_epoch = [batch["sample_id"] for batch in loader.iter_batches(limit=2)]
    assert second_epoch != first_epoch[:2]  # fresh seeded pass order, not a replay
    state = loader.state()
    resumed_ids = [batch["sample_id"] for batch in loader.iter_batches(limit=2)]
    restored = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    restored.set_state(state)
    assert restored.epoch == 1
    assert [batch["sample_id"] for batch in restored.iter_batches(limit=2)] == resumed_ids


def test_reader_keeps_one_resident_shard_per_source_and_validates_contract(data_root):
    catalog = _catalog(data_root)
    reader = ShardReader(CONFIG)
    entry = catalog.entries(split="train")[0]
    reader.load(entry)
    arrays, metadata = reader.get(0)
    assert arrays["agent_history"].shape == (16, 11, 8)
    assert metadata["schema_version"] == SCHEMA_VERSION
    with pytest.raises(IndexError):
        reader.get(entry.examples)
    other = catalog.entries(split="train")[2]
    reader.load(other)
    assert reader.resident == other.path
    reader.release()
    assert reader.resident is None


def test_row_copies_do_not_pin_previous_shard_backing_arrays(data_root):
    """P3.3.2a: rows must be copies, not views.  Holding a returned row across a
    shard swap must not keep the previous shard's full backing arrays alive
    (the transient two-shards-per-source retention found in review)."""
    catalog = _catalog(data_root)
    reader = ShardReader(CONFIG)
    entries = catalog.entries(split="train")
    reader.load(entries[0])
    refs = [weakref.ref(array) for array in reader._arrays.values()]
    held_row, _ = reader.get(0)  # kept alive across the swap below
    reader.load(entries[2])
    gc.collect()
    assert all(ref() is None for ref in refs), "previous shard backing arrays still alive"
    # the held row itself is an independent copy
    assert held_row["agent_history"].shape == (16, 11, 8)
    assert held_row["agent_history"].flags["OWNDATA"]
