"""P3.3.5 --units-parallel orchestration (HPC conversion fan-out).

The scale conversion migrates to a rented CPU cluster; convert_units /
hash_files fan independent work across a fork Pool.  These tests pin the
orchestration contract only (unit parsing stays untouched code):

* workers=1 is the original serial path and results are order-preserved
* workers>1 returns the same results in the same input order even when
  completion order differs
* a worker exception propagates instead of being swallowed
* parallel hashing returns the identical mapping as the serial loop

End-to-end byte/array equivalence on real data (3-unit serial vs parallel)
is a separate gate on the HPC side (research_tasks/p335_parallel_equivalence.py)
before the full 674-unit run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research_tasks.convert_p33_data as conv  # noqa: E402
from scenario_lab.p33_spec import file_sha256  # noqa: E402


@pytest.fixture()
def units(tmp_path):
    """Fake units; paths exist but only the names are consumed by the stub."""
    paths = []
    for index in range(3):
        path = tmp_path / f"training.tfrecord-{index:05d}-of-01000"
        path.write_bytes(b"stub")
        paths.append(path)
    return [("waymo", [path]) for path in paths]


@pytest.fixture()
def stub_convert_unit(monkeypatch):
    """Pure-function payloads derived from the unit input; stall the FIRST
    input unit so its completion order differs under workers>1 (fork gives
    each worker its own copy of any closure state, so payloads must not
    depend on cross-call counters)."""
    def fake(source, paths, data_root, output, config, pipeline, content_hashes):
        first = paths[0].name.endswith("-00000")
        if first:
            time.sleep(0.3)  # first input job finishes last under workers>1
        return {
            "unit_id": f"stub-{paths[0].stem}",
            "source": source,
            "scenes": 1 if first else 2,
            "examples": 1 if first else 2,
            "errors": [],
            "resumed": False,
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(conv, "convert_unit", fake)


def test_convert_units_serial_matches_parallel_order(units, stub_convert_unit):
    serial = conv.convert_units(units, Path("data"), Path("out"), {}, {}, {}, workers=1)
    parallel = conv.convert_units(units, Path("data"), Path("out"), {}, {}, {}, workers=3)
    assert serial == parallel
    # input order is preserved even though the first job completed last
    assert [unit["unit_id"] for unit in parallel] == [
        f"stub-{path.stem}" for _source, [path] in units]


def test_convert_units_workers_floor_is_serial(units, stub_convert_unit):
    result = conv.convert_units(units, Path("data"), Path("out"), {}, {}, {}, workers=0)
    assert len(result) == 3


def test_convert_units_worker_exception_propagates(units, monkeypatch):
    def boom(source, paths, *_args):
        raise ValueError("stale marker")

    monkeypatch.setattr(conv, "convert_unit", boom)
    with pytest.raises(ValueError, match="stale marker"):
        conv.convert_units(units, Path("data"), Path("out"), {}, {}, {}, workers=2)


def test_hash_files_parallel_equals_serial(tmp_path):
    files = []
    for index in range(4):
        path = tmp_path / f"blob-{index}.bin"
        path.write_bytes(bytes([index]) * 1024)
        files.append(path)
    serial = conv.hash_files(files, workers=1)
    parallel = conv.hash_files(files, workers=4)
    assert serial == parallel
    assert serial == {str(path): file_sha256(path) for path in files}


def test_hash_files_empty(tmp_path):
    assert conv.hash_files([], workers=4) == {}
