"""P3.3.2a rebuilt resume/determinism probe (supersedes the 20260913_p332 probe).

Why rebuilt: the first probe was invalid on two counts found in review
(runs/20260913_p332_model_smoke/CODEX_REVIEW.md):
1. it loaded the same ``snapshot["optimizer"]`` payload into R0 and R2
   sequentially; ``Optimizer.load_state_dict`` shares tensor storage with the
   payload when device/dtype match, so R0's training mutated the state that R2
   later restored (R0/R2 divergence proved state pollution, not allocator
   effects);
2. the run it explained used ``warn_only=True`` while the bf16 memory-efficient
   attention backward is nondeterministic (UserWarning in m2_run.log), so
   "kernel nondeterminism ruled out" was unfounded.

This probe imports the corrected train module (strict
``torch.use_deterministic_algorithms``, mem-efficient/flash SDPA disabled,
``restore_optimizer_isolated``) and asks two questions:
A. under the deterministic configuration, do independent rebuilds reproduce
   the uninterrupted continuation bitwise (R1 vs R0)?
B. does a pure CUDA allocator layout perturbation change anything (R1 vs R2),
   i.e. does the withdrawn allocator->kernel-selection attribution survive a
   clean test?

It also measures throughput with deterministic (math) vs mem-efficient SDPA so
P3.3.3 can decide the production backend with data.
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
spec.loader.exec_module(train)  # strict determinism + math SDPA are enabled here

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
    held: list[torch.Tensor] = []
    if perturb_allocator:
        # persistent odd-sized live tensors (held across the replica's updates)
        # plus freed fragments and a cache release: changes the allocator pool
        # layout without touching any math
        held = [torch.empty(size, device=DEVICE) for size in (1_000_003, 2_000_017, 777_777)]
        fragment = [torch.empty(3_000_011, device=DEVICE), torch.empty(123_457, device=DEVICE)]
        del fragment
        torch.cuda.empty_cache()
    train.seed_everything(smoke["seed"])
    loader_r = SceneLoader(catalog, split="train", batch_per_source=8, seed=smoke["seed"])
    model_r, optimizer_r = train.build(config, codebook, DEVICE)
    model_r.load_state_dict(snapshot["model"])
    train.restore_optimizer_isolated(optimizer_r, snapshot["optimizer"])
    loader_r.set_state(snapshot["loader_state"])
    train.set_rng_state(snapshot["rng"])
    local = [dict(row) for row in snapshot["history"]]
    train.train_updates(model_r, optimizer_r, loader_r, config, smoke, WARMUP,
                        WARMUP + TAIL, DEVICE, local, None,
                        checkpoint_every=10**9, log_prefix=tag)
    del held
    return [row["total"] for row in local[WARMUP:]]


# R1: uninterrupted continuation with live objects (must run first: replicas
# rebuild from the snapshot, this branch keeps using the live objects)
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
    "probe": "p332a_rebuilt_determinism_probe",
    "deterministic_config": {
        "use_deterministic_algorithms": "strict (warn_only=False)",
        "sdpa": {"mem_efficient": False, "flash": False, "math": True},
        "optimizer_restore": "restore_optimizer_isolated (no payload aliasing)",
    },
    "warmup_updates": WARMUP,
    "tail_updates": TAIL,
    "r1_vs_r0_clean_rebuild": compare(r1, r0),
    "r1_vs_r2_perturbed_allocator": compare(r1, r2),
    "r0_vs_r2": compare(r0, r2),
    "r1_losses": [round(v, 6) for v in r1],
}

# throughput: deterministic math SDPA (current) vs mem-efficient (previous)
if DEVICE.type == "cuda":
    def timed_updates(tag: str) -> float:
        train.seed_everything(smoke["seed"] + 1)
        loader_t = SceneLoader(catalog, split="train", batch_per_source=8,
                               seed=smoke["seed"] + 1)
        model_t, optimizer_t = train.build(config, codebook, DEVICE)
        import time
        started = time.time()
        local: list[dict] = []
        train.train_updates(model_t, optimizer_t, loader_t, config, smoke, 0, 20,
                            DEVICE, local, None, checkpoint_every=10**9,
                            log_prefix=tag)
        elapsed = time.time() - started
        return 20 * 16 * 16 / elapsed  # samples per second

    result["throughput_math_sdpa_strict"] = round(timed_updates("t_math"), 2)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    result["throughput_mem_efficient_warn_only"] = round(timed_updates("t_memeff"), 2)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=False)

out = Path(__file__).parent / "ALLOCATOR_PROBE_RESULT.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("r1_losses",)}, indent=2))
