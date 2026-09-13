"""checkpoint snapshot semantics for P3.3.2 M2 resume (CPU, real manifest data).

Locks the defect class discovered during M2: ``checkpoint_payload`` must be an
immutable snapshot. The first implementation stored live references
(``optimizer.state_dict()`` tensors and the history list) that kept mutating as
phase-A training continued, so the resumed branch silently loaded wrong state
while the loss comparison (an alias of the same array) reported exact equality.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scenario_lab.p33_dataset import SceneLoader, ShardCatalog  # noqa: E402
from scenario_lab.p33_model import ar_scene_loss, to_torch_batch  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_p33_nominal", REPO / "research_tasks" / "train_p33_nominal.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

CONFIG = load_p33_config()


def _setup():
    train.seed_everything(7)
    codebook = train.load_codebook(train.MANIFEST)
    catalog = ShardCatalog.from_manifest(train.MANIFEST, CONFIG)
    loader = SceneLoader(catalog, split="train", batch_per_source=2, seed=7)
    batches = [to_torch_batch(batch, "cpu") for batch in loader.iter_batches(limit=3)]
    model, optimizer = train.build(CONFIG, codebook, torch.device("cpu"))
    return model, optimizer, loader, batches


def _step(model, optimizer, batch) -> None:
    model.train()
    losses = ar_scene_loss(model(batch, teacher_tokens=batch["motion_token_target"]),
                           batch, model.codebook, CONFIG)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def _eval_losses(model, batch) -> dict:
    model.eval()
    with torch.no_grad():
        return ar_scene_loss(model(batch, teacher_tokens=batch["motion_token_target"]),
                             batch, model.codebook, CONFIG)


def test_checkpoint_payload_is_an_immutable_snapshot():
    model, optimizer, loader, batches = _setup()
    _step(model, optimizer, batches[0])
    payload = train.checkpoint_payload(model, optimizer, loader, [{"total": 1.0}], 1)
    frozen_model = {key: value.clone() for key, value in payload["model"].items()}
    frozen_optimizer = copy.deepcopy(payload["optimizer"])
    frozen_history = list(payload["history"])

    _step(model, optimizer, batches[1])  # continued training must not touch the payload

    assert all(torch.equal(payload["model"][key], frozen_model[key]) for key in frozen_model)
    assert payload["optimizer"]["state"].keys() == frozen_optimizer["state"].keys()
    for key, state in payload["optimizer"]["state"].items():
        for name, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, frozen_optimizer["state"][key][name]), name
    assert payload["history"] == frozen_history
    # the payload is a true snapshot of the save point, not an alias of live state
    assert not torch.equal(payload["model"]["token_head.weight"],
                           model.state_dict()["token_head.weight"])


def test_resume_from_payload_reproduces_restore_point_bitwise():
    model, optimizer, loader, batches = _setup()
    _step(model, optimizer, batches[0])
    _step(model, optimizer, batches[1])
    payload = train.checkpoint_payload(model, optimizer, loader, [], 2)
    live = _eval_losses(model, batches[2])

    train.seed_everything(7)
    codebook = train.load_codebook(train.MANIFEST)
    model_b, optimizer_b = train.build(CONFIG, codebook, torch.device("cpu"))
    model_b.load_state_dict(payload["model"])
    train.restore_optimizer_isolated(optimizer_b, payload["optimizer"])
    train.set_rng_state(payload["rng"])
    twin = _eval_losses(model_b, batches[2])

    for key, value in live.items():
        assert torch.equal(value, twin[key]), f"{key}: {value} != {twin[key]}"


def test_optimizer_restore_does_not_alias_the_payload():
    """P3.3.2a regression: Optimizer.load_state_dict shares tensor storage with
    the payload when device/dtype already match, so a replica that trains after
    restoring would silently mutate the checkpoint (the bug that invalidated
    the first allocator probe)."""
    model, optimizer, loader, batches = _setup()
    _step(model, optimizer, batches[0])
    payload = train.checkpoint_payload(model, optimizer, loader, [], 1)
    frozen = copy.deepcopy(payload["optimizer"])

    model_b, optimizer_b = train.build(CONFIG, train.load_codebook(train.MANIFEST),
                                       torch.device("cpu"))
    model_b.load_state_dict(payload["model"])
    train.restore_optimizer_isolated(optimizer_b, payload["optimizer"])
    _step(model_b, optimizer_b, batches[1])  # replica trains on the restored state

    for key, state in payload["optimizer"]["state"].items():
        for name, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, frozen["state"][key][name]), name
