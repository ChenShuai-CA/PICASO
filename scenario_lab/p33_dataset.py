"""Bounded, source-balanced loader for ``scene-shard-v1`` P3.3.1 outputs.

The loader never materializes more than one shard per source: each compressed
NPZ is decompressed once into a resident dictionary that is dropped as soon as
the stream moves to another shard of that source.  Sample order is derived from
(seed, source, pass index) so that training streams, exposure counters and
checkpoint resume are all deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .p33_spec import expected_array_contract, file_sha256

SOURCE_IDS = {"waymo": 0, "interaction": 1}


@dataclass(frozen=True)
class ShardEntry:
    root: Path
    path: str  # relative to the manifest directory
    index_path: Path
    examples: int
    sha256: str
    unit_id: str
    source: str
    splits: tuple[str, ...]  # distinct row-level splits inside this shard
    rows_by_split: dict[str, tuple[int, ...]]  # ascending row indices per split

    @property
    def npz_path(self) -> Path:
        return self.root / self.path


class ShardCatalog:
    """Manifest-driven shard list with per-row source/split consistency checks."""

    def __init__(self, root: Path, config: dict, entries: list[ShardEntry]):
        self.root = root
        self.config = config
        self._entries = entries

    @classmethod
    def from_manifest(cls, manifest_path: Path, config: dict) -> "ShardCatalog":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        schema = config["data"]["schema_version"]
        entries: list[ShardEntry] = []
        for shard in manifest["validation"]["shards"]:
            index_path = root / shard["index"]
            rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
            if len(rows) != shard["examples"]:
                raise ValueError(f"{shard['path']}: jsonl rows {len(rows)} != "
                                 f"manifest examples {shard['examples']}")
            sources = {row["source"] for row in rows}
            if len(sources) != 1:
                raise ValueError(f"{shard['path']}: mixed sources within one shard: {sorted(sources)}")
            # Waymo splits by Scenario hash, so one shard legitimately mixes train and dev
            # rows; splits are therefore tracked per row, not per shard.
            rows_by_split: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                rows_by_split.setdefault(row["split"], []).append(index)
            bad = [row["sample_id"] for row in rows if row["schema_version"] != schema]
            if bad:
                raise ValueError(f"{shard['path']}: {len(bad)} rows with schema_version != {schema}")
            entries.append(ShardEntry(
                root=root, path=shard["path"], index_path=index_path,
                examples=int(shard["examples"]), sha256=shard["sha256"],
                unit_id=shard["unit_id"], source=next(iter(sources)),
                splits=tuple(sorted(rows_by_split)),
                rows_by_split={split: tuple(indices) for split, indices in rows_by_split.items()}))
        return cls(root, config, entries)

    def entries(self, split: str | None = None, source: str | None = None) -> list[ShardEntry]:
        return [entry for entry in self._entries
                if (split is None or split in entry.splits)
                and (source is None or entry.source == source)]

    def verify_hashes(self, cache: dict | None = None) -> dict[str, str]:
        """Check every shard NPZ sha256; ``cache`` memoizes verified shards."""
        cache = {} if cache is None else cache
        mismatches: dict[str, str] = {}
        for entry in self._entries:
            absolute = entry.npz_path
            digest = cache.get(str(absolute))
            if digest is None:
                digest = file_sha256(absolute)
                cache[str(absolute)] = digest
            if digest != entry.sha256:
                mismatches[entry.path] = digest
        return mismatches


def audit_catalog(catalog: ShardCatalog, verify_hashes: bool = True,
                  hash_cache: dict | None = None) -> dict:
    """D0 audit: counts, schema consistency, cross-split groups, shard hashes."""
    counts: dict[str, int] = {}
    schema_mismatches: list[str] = []
    groups: dict[tuple[str, str], set[str]] = {}
    for entry in catalog.entries():
        rows = [json.loads(line) for line in entry.index_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != entry.examples:
            raise ValueError(f"{entry.path}: row count drift ({len(rows)} != {entry.examples})")
        bad = [row["sample_id"] for row in rows if row["schema_version"]
               != catalog.config["data"]["schema_version"]]
        if bad:
            schema_mismatches.append(entry.path)
        for row in rows:
            counts["examples"] = counts.get("examples", 0) + 1
            for key, value in (("source", row["source"]), ("split", row["split"]),
                               ("source_split", f"{row['source']}:{row['split']}")):
                counts[f"{key}:{value}"] = counts.get(f"{key}:{value}", 0) + 1
            groups.setdefault((row["source"], row["group_id"]), set()).add(row["split"])
    leaks = [f"{source}:{group_id}:{sorted(splits)}"
             for (source, group_id), splits in sorted(groups.items()) if len(splits) > 1]
    hash_mismatches = catalog.verify_hashes(hash_cache) if verify_hashes else {}
    return {
        "status": "pass" if not (hash_mismatches or leaks or schema_mismatches) else "fail",
        "shards": len(catalog.entries()),
        "counts": dict(sorted(counts.items())),
        "independent_groups": len(groups),
        "cross_split_groups": leaks,
        "schema_version_mismatches": schema_mismatches,
        "shard_hash_mismatches": hash_mismatches,
        "max_resident_shards_per_source": 1,
    }


class ShardReader:
    """Single-resident shard reader with structural contract validation.

    ``config`` may be the full P3.3 config or any mapping; only the array
    contract is used.
    """

    def __init__(self, config: dict):
        self.contract = (expected_array_contract(config) if "data" in config
                         else config)
        self._arrays: dict[str, np.ndarray] | None = None
        self._rows: list[dict] | None = None
        self._entry: ShardEntry | None = None

    @property
    def resident(self) -> str | None:
        return None if self._entry is None else self._entry.path

    def load(self, entry: ShardEntry) -> None:
        if self._entry is not None and self._entry.path == entry.path:
            return
        self.release()
        with np.load(entry.npz_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        missing = set(self.contract) - set(arrays)
        extra = set(arrays) - set(self.contract)
        if missing or extra:
            raise ValueError(f"{entry.path}: array keys differ; "
                             f"missing={sorted(missing)}, extra={sorted(extra)}")
        for name, (shape, dtype) in self.contract.items():
            array = arrays[name]
            if array.shape[0] != entry.examples or array.shape[1:] != shape or array.dtype != dtype:
                raise ValueError(f"{entry.path}:{name}: expected ({entry.examples},)+{shape}/{dtype}, "
                                 f"got {array.shape}/{array.dtype}")
        rows = [json.loads(line) for line in entry.index_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != entry.examples:
            raise ValueError(f"{entry.path}: jsonl rows {len(rows)} != npz rows {entry.examples}")
        self._arrays, self._rows, self._entry = arrays, rows, entry

    def get(self, index: int) -> tuple[dict[str, np.ndarray], dict]:
        if self._arrays is None or self._entry is None:
            raise RuntimeError("no shard is resident")
        if not 0 <= index < self._entry.examples:
            raise IndexError(index)
        return {key: value[index] for key, value in self._arrays.items()}, self._rows[index]

    def release(self) -> None:
        self._arrays = None
        self._rows = None
        self._entry = None


def collate(samples: list[tuple[dict[str, np.ndarray], dict]]) -> dict:
    arrays, metadata = zip(*samples)
    keys = list(arrays[0].keys())
    batch = {key: np.stack([row[key] for row in arrays]) for key in keys}
    batch["source_id"] = np.asarray([SOURCE_IDS[row["source"]] for row in metadata], dtype=np.int64)
    batch["sample_source"] = [row["source"] for row in metadata]
    batch["sample_id"] = [row["sample_id"] for row in metadata]
    batch["group_id"] = [row["group_id"] for row in metadata]
    batch["split"] = [row["split"] for row in metadata]
    batch["agents_truncated"] = np.asarray(
        [row["truncation"]["agents_available"] > row["truncation"]["agents_kept"]
         for row in metadata], dtype=bool)
    batch["map_truncated"] = np.asarray(
        [row["truncation"]["map_polylines_available"] > row["truncation"]["map_polylines_kept"]
         for row in metadata], dtype=bool)
    return batch


class _SourceStream:
    """Deterministic shard-major sample stream with seeded pass reshuffling."""

    def __init__(self, entries: list[ShardEntry], split: str, source_id: int,
                 seed: int, shuffle: bool = True):
        self.entries = entries
        self.split = split
        self.source_id = source_id
        self.seed = seed
        self.shuffle = shuffle
        self.pool = sum(len(entry.rows_by_split[split]) for entry in entries)
        self.pass_index = 0
        self.position = 0
        self.emitted = 0
        self._order = self._build_order()

    def _build_order(self) -> list[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self.source_id, self.pass_index])
        shard_indices = list(range(len(self.entries)))
        if self.shuffle:
            rng.shuffle(shard_indices)
        order: list[tuple[int, int]] = []
        for shard_index in shard_indices:
            rows = np.asarray(self.entries[shard_index].rows_by_split[self.split])
            if self.shuffle:
                rng.shuffle(rows)
            order.extend((shard_index, int(row)) for row in rows)
        return order

    def take(self, count: int) -> list[tuple[int, int]]:
        refs: list[tuple[int, int]] = []
        while len(refs) < count:
            if self.position >= len(self._order):
                self.pass_index += 1
                self._order = self._build_order()
                self.position = 0
            refs.append(self._order[self.position])
            self.position += 1
            self.emitted += 1
        return refs

    def state(self) -> dict:
        return {"pass_index": self.pass_index, "position": self.position, "emitted": self.emitted}

    def set_state(self, state: dict) -> None:
        self.pass_index = int(state["pass_index"])
        self.position = int(state["position"])
        self.emitted = int(state["emitted"])
        self._order = self._build_order()
        if self.position > len(self._order):
            raise ValueError("stream state points past the end of its order")


class SceneLoader:
    """Yields source-balanced batches; an epoch is one pass over the smaller source.

    A batch always takes ``batch_per_source`` samples from each source.  When a
    source drains mid-batch its order cycles deterministically (a new seeded
    pass), and the exposure counters record every extra pass.
    """

    def __init__(self, catalog: ShardCatalog, split: str, batch_per_source: int = 8,
                 seed: int = 7, shuffle: bool = True, verify_hashes: bool = True,
                 hash_cache: dict | None = None):
        self.catalog = catalog
        self.split = split
        self.batch_per_source = int(batch_per_source)
        self.seed = int(seed)
        entries = catalog.entries(split=split)
        if not entries:
            raise ValueError(f"no shards for split={split!r}")
        by_source: dict[str, list[ShardEntry]] = {}
        for entry in entries:
            by_source.setdefault(entry.source, []).append(entry)
        missing_sources = set(SOURCE_IDS) - set(by_source)
        if missing_sources:
            raise ValueError(f"split {split!r} is missing sources: {sorted(missing_sources)}")
        if verify_hashes:
            mismatches = catalog.verify_hashes(hash_cache)
            if mismatches:
                raise ValueError(f"shard hash mismatches, refusing to train: {mismatches}")
        self.streams = {source: _SourceStream(source_entries, split, SOURCE_IDS[source],
                                              self.seed, shuffle)
                        for source, source_entries in sorted(by_source.items())}
        self.readers = {source: ShardReader(catalog.config) for source in self.streams}
        self._epoch_source = min(self.streams, key=lambda source: self.streams[source].pool)
        self.batches_emitted = 0
        self.epoch = 0
        self.shard_loads = {source: 0 for source in self.streams}
        self.max_resident_shards = len(self.streams)

    def _samples(self, source: str, refs: list[tuple[int, int]]):
        stream_entries = self.streams[source].entries
        reader = self.readers[source]
        for shard_index, row in refs:
            entry = stream_entries[shard_index]
            if reader.resident != entry.path:
                reader.load(entry)
                self.shard_loads[source] += 1
            yield reader.get(row)

    def iter_batches(self, limit: int | None = None):
        produced = 0
        smaller = self.streams[self._epoch_source]
        while smaller.emitted < smaller.pool:
            if limit is not None and produced >= limit:
                return
            refs = {source: stream.take(self.batch_per_source)
                    for source, stream in self.streams.items()}
            samples: list[tuple[dict, dict]] = []
            for source in sorted(refs):
                samples.extend(self._samples(source, refs[source]))
            yield collate(samples)
            produced += 1
            self.batches_emitted += 1

    def exposure(self) -> dict:
        return {
            "epoch_source": self._epoch_source,
            "epoch_pool": {source: stream.pool for source, stream in self.streams.items()},
            "batches_emitted": self.batches_emitted,
            "samples_emitted": {source: stream.emitted for source, stream in self.streams.items()},
            "passes_completed": {source: stream.pass_index for source, stream in self.streams.items()},
            "shard_loads": dict(self.shard_loads),
            "max_resident_shards": self.max_resident_shards,
            "max_resident_shards_per_source": 1,
        }

    def advance_epoch(self) -> dict:
        """Close the finished epoch and open the next one.

        ``iter_batches`` yields exactly one epoch (a pass over the smaller
        source); training loops that need more call this on StopIteration.
        Pass orders continue deterministically (``pass_index`` is untouched).
        """
        summary = self.exposure()
        for stream in self.streams.values():
            stream.emitted = 0
        self.batches_emitted = 0
        self.epoch += 1
        summary["epoch"] = self.epoch
        return summary

    def state(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch,
                "batches_emitted": self.batches_emitted,
                "streams": {source: stream.state() for source, stream in self.streams.items()}}

    def set_state(self, state: dict) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("cannot restore a loader state from a different seed")
        for source, stream_state in state["streams"].items():
            self.streams[source].set_state(stream_state)
        self.batches_emitted = int(state["batches_emitted"])
        self.epoch = int(state.get("epoch", 0))


def iter_split_batches(catalog: ShardCatalog, split: str, batch_size: int = 16):
    """Deterministic shard-ordered iteration over one split (evaluation use)."""
    reader = ShardReader(catalog.config)
    try:
        buffer: list[tuple[dict, dict]] = []
        for entry in catalog.entries(split=split):
            reader.load(entry)
            for index in entry.rows_by_split[split]:
                buffer.append(reader.get(index))
                if len(buffer) == batch_size:
                    yield collate(buffer)
                    buffer = []
        if buffer:
            yield collate(buffer)
    finally:
        reader.release()
