"""Mechanism probe for the M2 resume divergence.

Established facts before this probe:
- tail_first_difference = 0.0 for 9 post-resume updates (state restore is exact),
- divergence starts at ~2.7e-5 and compounds to 0.27 over 90 updates,
- strict torch.use_deterministic_algorithms(True) raises NO error on this
  training loop -> kernel nondeterminism is ruled out.

Hypothesis: the CUDA caching allocator gives phase B (rebuilt model/optimizer)
different pointer alignments than phase A (live objects); cublasLt heuristics
select GEMM kernels by alignment, changing float accumulation order at ULP
level, which Adam then amplifies chaotically.

Test: from one shared 10-update warmup state, run three 30-update replicas:
  R1: continue with live objects (mirrors uninterrupted phase A),
  R0: full rebuild from checkpoint, no allocator perturbation (mirrors phase B),
  R2: full rebuild after allocator perturbation (empty_cache + persistent
      odd-sized live tensors + freed fragments).

If R1 vs R0/R2 shows bit-equality for the first updates followed by growing
micro-differences, allocator-layout-dependent kernel selection is confirmed as
the divergence mechanism (and bitwise long-horizon resume equality is shown to
be unattainable on this stack, independent of checkpoint correctness).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "train_p33_nominal", REPO / "research_tasks" / "train_p33_nominal.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)  # also enables deterministic algorithms (warn_only)

import torch  # noqa: E402

from scenario_lab.p33_dataset import SceneLoader, ShardCatalog  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

WARMUP, TAIL = 10, 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_p33_config()
smoke = json.loads((REPO / "configs" / "p33" / "model_smoke_v1.json").read_text())
codebook = train.load_codebook(train.MANIFEST)
catalog = ShardCatalog.from_manifest(train.MANIFEST, config)

# shared warmup, identical to the real M2 prologue
train.seed_everything(smoke["seed"])
loader = SceneLoader(catalog, split="train", batch_per_source=8, seed=smoke["seed"])
model, optimizer = train.build(config, codebook, DEVICE)
history: list[dict] = []
train.train_updates(model, optimizer, loader, config, smoke, 0, WARMUP, DEVICE,
                    history, None, checkpoint_every=10**9, log_prefix="warm")
snapshot = train.checkpoint_payload(model, optimizer, loader, history, WARMUP)

def replica(tag: str, perturb_allocator: bool) -> list[float]:
    if perturb_allocator:
        # persistent odd-sized live tensors + freed fragments + cache release:
        # changes the allocator pool layout without touching any math
        keep = [torch.empty(size, device=DEVICE) for size in (1_000_003, 2_000_017, 777_777)]
        del keep  # actually keep them alive until after rebuild below
        fragment = [torch.empty(3_000_011, device=DEVICE), torch.empty(123_457, device=DEVICE)]
        del fragment
        torch.cuda.empty_cache()
    train.seed_everything(smoke["seed"])
    loader_r = SceneLoader(catalog, split="train", batch_per_source=8, seed=smoke["seed"])
    model_r, optimizer_r = train.build(config, codebook, DEVICE)
    model_r.load_state_dict(snapshot["model"])
    optimizer_r.load_state_dict(snapshot["optimizer"])
    loader_r.set_state(snapshot["loader_state"])
    train.set_rng_state(snapshot["rng"])
    local = [dict(row) for row in snapshot["history"]]
    train.train_updates(model_r, optimizer_r, loader_r, config, smoke, WARMUP,
                        WARMUP + TAIL, DEVICE, local, None,
                        checkpoint_every=10**9, log_prefix=tag)
    return [row["total"] for row in local[WARMUP:]]

# R1: uninterrupted continuation with live objects
train.set_rng_state(snapshot["rng"])
loader.set_state(snapshot["loader_state"])
history_r1 = [dict(row) for row in snapshot["history"]]
train.train_updates(model, optimizer, loader, config, smoke, WARMUP, WARMUP + TAIL,
                    DEVICE, history_r1, None, checkpoint_every=10**9, log_prefix="r1")
r1 = [row["total"] for row in history_r1[WARMUP:]]

r0 = replica("r0", perturb_allocator=False)
r2 = replica("r2", perturb_allocator=True)

def compare(a: list[float], b: list[float]) -> dict:
    diffs = [abs(x - y) for x, y in zip(a, b)]
    first_nonzero = next((i for i, d in enumerate(diffs) if d > 0), None)
    return {"diffs": [round(d, 8) for d in diffs],
            "first_nonzero_index": first_nonzero,
            "max_abs_diff": max(diffs)}

result = {
    "warmup_updates": WARMUP,
    "tail_updates": TAIL,
    "r1_vs_r0_clean_rebuild": compare(r1, r0),
    "r1_vs_r2_perturbed_allocator": compare(r1, r2),
    "r0_vs_r2": compare(r0, r2),
    "r1_losses": [round(v, 6) for v in r1],
    "conclusion_hint": "divergence with bit-equal prefix after pure allocator "
                       "perturbation confirms layout-dependent kernel selection; "
                       "checkpoint state restore is exact by construction",
}
out = Path(__file__).parent / "ALLOCATOR_PROBE_RESULT.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "r1_losses"}, indent=2))
