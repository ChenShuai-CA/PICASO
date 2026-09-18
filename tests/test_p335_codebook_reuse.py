"""P3.3.5 frozen-codebook reuse (plan Task 7.2, TDD).

The scale run must reuse the frozen motion_codebook_v1.npz instead of
refitting on 500 shards, so the 100->500 comparison isolates data scale
(plan Task 7.5).  Default behaviour (no flag) refits as before and is
not touched here -- these tests cover the reuse path only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_tasks.convert_p33_data import reuse_codebook  # noqa: E402
from scenario_lab.p33_pipeline import atomic_savez  # noqa: E402
from scenario_lab.p33_spec import file_sha256, load_p33_config  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_p33_config(ROOT / "configs/p33/ar_scene_v1.json")


def _frozen_codebook(directory: Path, vocab: int, dim: int = 10) -> Path:
    path = directory / "frozen_codebook.npz"
    centroids = np.arange(vocab * dim, dtype=np.float32).reshape(vocab, dim)
    atomic_savez(path, {"centroids": centroids})
    return path


def test_reuse_copies_frozen_codebook_byte_identical(tmp_path, config):
    frozen = _frozen_codebook(tmp_path, config["motion_tokens"]["vocabulary_size"])
    output = tmp_path / "out"
    output.mkdir()
    codebook, manifest = reuse_codebook(output, frozen, config)

    copied = output / "motion_codebook_v1.npz"
    assert copied.exists()
    assert file_sha256(copied) == file_sha256(frozen)
    assert manifest["reused_frozen"] is True
    assert manifest["source_sha256"] == file_sha256(frozen)
    assert manifest["version"] == config["motion_tokens"]["version"]
    assert codebook.shape == (config["motion_tokens"]["vocabulary_size"],
                              config["motion_tokens"]["vector_dimension"])
    with np.load(copied, allow_pickle=False) as data:
        assert np.array_equal(data["centroids"], codebook)


def test_reuse_rejects_wrong_vocabulary(tmp_path, config):
    frozen = _frozen_codebook(tmp_path, config["motion_tokens"]["vocabulary_size"] + 1)
    output = tmp_path / "out_vocab"
    output.mkdir()
    with pytest.raises(ValueError):
        reuse_codebook(output, frozen, config)


def test_reuse_rejects_wrong_dimension(tmp_path, config):
    frozen = _frozen_codebook(tmp_path, config["motion_tokens"]["vocabulary_size"], dim=11)
    output = tmp_path / "out_dim"
    output.mkdir()
    with pytest.raises(ValueError):
        reuse_codebook(output, frozen, config)


def test_reuse_rejects_missing_centroids(tmp_path, config):
    path = tmp_path / "not_a_codebook.npz"
    atomic_savez(path, {"something_else": np.zeros(3, dtype=np.float32)})
    output = tmp_path / "out_missing"
    output.mkdir()
    with pytest.raises(ValueError):
        reuse_codebook(output, path, config)


def test_reuse_manifest_on_disk_matches(tmp_path, config):
    frozen = _frozen_codebook(tmp_path, config["motion_tokens"]["vocabulary_size"])
    output = tmp_path / "out_manifest"
    output.mkdir()
    _, manifest = reuse_codebook(output, frozen, config)
    on_disk = json.loads((output / "CODEBOOK_MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["codebook_sha256"] == file_sha256(output / "motion_codebook_v1.npz")
