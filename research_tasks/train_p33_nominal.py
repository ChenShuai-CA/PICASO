"""P3.3.2 nominal-model smoke training: M1 single-batch overfit, M2 GPU smoke + resume.

P3.3.3 adds --mode architecture: epoch-scheduled training on the 100-shard
architecture-stage dataset with per-epoch dev validation, early stopping on
source_macro_minade_at_6, best/last checkpoints and crash resume.

P3.3.4 adds --variant kinematic: the Phase C conditional kinematic decode head
(ARSceneV1K, runs/20260915_p334_kinematics/SPEC.md section 6) trained under the
identical frozen budget; --max-updates is an engineering smoke cap only.

All budgets come from the frozen ar_scene_v1.json (max_epochs, early stopping,
optimizer, loss weights, precision policy, gradient clipping); smoke-specific
overrides come from configs/p33/model_smoke_v1.json.

Usage (WSL):
  python research_tasks/train_p33_nominal.py --mode overfit
  python research_tasks/train_p33_nominal.py --mode smoke
  python research_tasks/train_p33_nominal.py --mode architecture \
      [--resume runs/20260913_p333_architecture/arch_last_checkpoint.pt]
  python research_tasks/train_p33_nominal.py --mode architecture \
      --variant kinematic --output-dir runs/20260915_p334_kinematics/phase_c_train
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# resume equality needs kernel-level determinism.  The first M2 run log showed
# that the bf16 memory-efficient attention backward is nondeterministic (the
# warn_only=True setting downgraded the strict-mode error to a UserWarning);
# disable the nondeterministic SDPA backends and use strict mode so that any
# remaining nondeterministic op raises instead of silently perturbing the
# resume comparison.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
torch.use_deterministic_algorithms(True, warn_only=False)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scenario_lab.p33_dataset import SceneLoader, ShardCatalog  # noqa: E402
from scenario_lab.p33_model import (ARSceneV1, ARSceneV1K, ar_scene_loss,  # noqa: E402
                                    ar_scene_loss_c, to_torch_batch)
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

MANIFEST = REPO / "runs/20260913_p331_data_pipeline/smoke/DATASET_MANIFEST.json"
OUTPUT_DIR = REPO / "runs/20260913_p332a_visibility_fix"
ARCH_MANIFEST = REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json"
ARCH_OUTPUT_DIR = REPO / "runs/20260913_p333_architecture"
C_CONFIG_PATH = REPO / "configs/p33/model_kinematic_c_v1.json"


def load_codebook(manifest_path: Path) -> np.ndarray:
    codebook_path = manifest_path.parent / "motion_codebook_v1.npz"
    with np.load(codebook_path, allow_pickle=False) as data:
        return np.asarray(data["centroids"], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(config: dict, codebook: np.ndarray, device: torch.device,
          variant: str = "residual", c_config: dict | None = None):
    model = (ARSceneV1K(config, codebook, c_config) if variant == "kinematic"
             else ARSceneV1(config, codebook)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    return model, optimizer


def learning_rate_scale(step: int, total: int, warmup_fraction: float) -> float:
    warmup = max(1, int(round(total * warmup_fraction)))
    return min(1.0, (step + 1) / warmup)


def rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])


def checkpoint_payload(model, optimizer, loader: SceneLoader, history: list[dict],
                       update_index: int, extra: dict | None = None) -> dict:
    # everything must be a SNAPSHOT: optimizer.state_dict() and the history list
    # hold live references that keep mutating as training continues
    payload = {
        "update_index": update_index,
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "loader_state": loader.state(),
        "sample_stream_snapshot": {source: list(stream._order[:stream.position])
                                   for source, stream in loader.streams.items()},
        "history": list(history),
        "rng": rng_state(),
    }
    if extra:
        payload.update(extra)
    return payload


def restore_optimizer_isolated(optimizer, payload: dict) -> None:
    """Restore optimizer state without storage aliasing.

    ``Optimizer.load_state_dict`` keeps the payload's tensors when device and
    dtype already match, so a later ``step`` would mutate the checkpoint
    payload in place — the aliasing bug that invalidated the first allocator
    probe (R0 training polluted the state later loaded into R2).  Clone every
    restored state tensor so optimizer and payload are fully independent.
    """
    optimizer.load_state_dict(payload)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.clone()


def run_overfit(args: argparse.Namespace) -> dict:
    """M1: one 16-sample batch, <=500 updates, last-20 mean loss <= 50% of first-20."""
    config = load_p33_config()
    smoke = json.loads((REPO / "configs/p33/model_smoke_v1.json").read_text())
    m1 = smoke["m1_single_batch_overfit"]
    seed = smoke["seed"]
    seed_everything(seed)
    device = torch.device(args.device)
    codebook = load_codebook(MANIFEST)
    catalog = ShardCatalog.from_manifest(MANIFEST, config)
    loader = SceneLoader(catalog, split="train", batch_per_source=m1["batch_per_source"],
                         seed=seed)
    batch = to_torch_batch(next(iter(loader.iter_batches(limit=1))), device)
    model, optimizer = build(config, codebook, device)
    history = []
    started = time.time()
    for update in range(m1["max_updates"]):
        model.train()
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
        losses = ar_scene_loss(outputs, batch, model.codebook, config)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       config["training"]["gradient_clip_norm"])
        optimizer.step()
        history.append({key: float(value.item()) for key, value in losses.items()})
        if update in (0, m1["max_updates"] // 2, m1["max_updates"] - 1):
            print(f"[m1] update {update + 1}/{m1['max_updates']} "
                  f"total={losses['total'].item():.4f} ce={losses['token_cross_entropy'].item():.4f} "
                  f"traj={losses['trajectory_huber'].item():.4f} "
                  f"acc={losses['token_accuracy'].item():.3f}", flush=True)
    first = float(np.mean([row["total"] for row in history[:m1["first_window"]]]))
    last = float(np.mean([row["total"] for row in history[-m1["last_window"]:]]))
    gate = last <= m1["loss_ratio_gate"] * first
    result = {
        "mode": "m1_single_batch_overfit",
        "seed": seed,
        "device": str(device),
        "updates": len(history),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "first_window_mean_total": first,
        "last_window_mean_total": last,
        "loss_ratio": last / first,
        "gate": "last20 <= 0.5 * first20",
        "gate_passed": bool(gate),
        "first_update": history[0],
        "final_update": history[-1],
        "elapsed_seconds": time.time() - started,
        "samples_per_second": len(history) * len(batch["sample_id"]) / (time.time() - started),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6
                         if device.type == "cuda" else None),
        "sample_ids": batch["sample_id"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "M1_OVERFIT_RESULT.json").write_text(json.dumps(result, indent=2))
    np.savetxt(OUTPUT_DIR / "M1_overfit_loss_curve.csv",
               np.array([[row["total"], row["token_cross_entropy"],
                          row["trajectory_huber"], row["endpoint_huber"],
                          row["token_accuracy"]] for row in history]),
               delimiter=",",
               header="total,token_cross_entropy,trajectory_huber,endpoint_huber,token_accuracy",
               comments="")
    if not gate:
        print(f"[m1] GATE FAILED: last20={last:.4f} > 0.5*first20={0.5 * first:.4f}")
    return result


class BatchStream:
    """Endless batch feed across epoch boundaries (StopIteration -> advance_epoch)."""

    def __init__(self, loader: SceneLoader):
        self.loader = loader
        self._iterator = iter(loader.iter_batches())

    def take(self) -> dict:
        try:
            return next(self._iterator)
        except StopIteration:
            summary = self.loader.advance_epoch()
            print(f"[loader] epoch {summary['epoch']} started "
                  f"(previous: {summary['batches_emitted']} batches)", flush=True)
            self._iterator = iter(self.loader.iter_batches())
            return next(self._iterator)


def train_updates(model, optimizer, loader, config, smoke, start_update: int,
                  total_updates: int, device: torch.device, history: list[dict],
                  checkpoints: list[dict] | None = None, checkpoint_every: int = 100,
                  log_prefix: str = "m2"):
    """Source-balanced micro-batches, gradient accumulation to the effective batch."""
    training = config["training"]
    micro_per_step = smoke["m2_gpu_smoke"]["micro_batches_per_step"]
    clip = training["gradient_clip_norm"]
    use_bf16 = smoke["m2_gpu_smoke"]["precision"] == "bf16" and device.type == "cuda"
    stream = BatchStream(loader)
    for update in range(start_update, total_updates):
        scale = learning_rate_scale(update, total_updates, training["warmup_fraction"])
        for group in optimizer.param_groups:
            group["lr"] = training["learning_rate"] * scale
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_losses = {}
        for _ in range(micro_per_step):
            batch = to_torch_batch(stream.take(), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_bf16):
                outputs = model(batch, teacher_tokens=batch["motion_token_target"])
                losses = ar_scene_loss(outputs, batch, model.codebook, config)
            (losses["total"] / micro_per_step).backward()
            for key, value in losses.items():
                step_losses[key] = step_losses.get(key, 0.0) + float(value.item()) / micro_per_step
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        history.append(step_losses)
        if (update + 1) % 25 == 0 or update == start_update:
            print(f"[{log_prefix}] update {update + 1}/{total_updates} "
                  f"total={step_losses['total']:.4f} ce={step_losses['token_cross_entropy']:.4f} "
                  f"acc={step_losses['token_accuracy']:.3f}", flush=True)
        if checkpoints is not None and (update + 1) % checkpoint_every == 0:
            checkpoints.append(checkpoint_payload(model, optimizer, loader, history, update + 1))
    return history


def run_smoke(args: argparse.Namespace) -> dict:
    """M2: 200 bf16 updates; resume-from-100 must reproduce the uninterrupted run."""
    config = load_p33_config()
    smoke = json.loads((REPO / "configs/p33/model_smoke_v1.json").read_text())
    m2 = smoke["m2_gpu_smoke"]
    total = args.updates or m2["updates"]
    every = args.checkpoint_every or m2["checkpoint_every"]
    seed = smoke["seed"]
    device = torch.device(args.device)
    codebook = load_codebook(MANIFEST)

    seed_everything(seed)
    catalog = ShardCatalog.from_manifest(MANIFEST, config)
    loader = SceneLoader(catalog, split="train", batch_per_source=8, seed=seed)
    model, optimizer = build(config, codebook, device)
    started = time.time()
    history, checkpoints = [], []
    train_updates(model, optimizer, loader, config, smoke, 0, total, device,
                  history, checkpoints, every, "m2a")
    uninterrupted = [row["total"] for row in history]
    mid = checkpoints[0]
    final = checkpoints[-1]
    resume_from = mid["update_index"]

    # resumed branch: rebuild everything, restore from the mid checkpoint
    seed_everything(seed)
    loader_b = SceneLoader(catalog, split="train", batch_per_source=8, seed=seed)
    model_b, optimizer_b = build(config, codebook, device)
    model_b.load_state_dict({key: value.to(device) for key, value in mid["model"].items()})
    restore_optimizer_isolated(optimizer_b, mid["optimizer"])
    loader_b.set_state(mid["loader_state"])
    set_rng_state(mid["rng"])
    resumed_history = [dict(row) for row in mid["history"]]
    train_updates(model_b, optimizer_b, loader_b, config, smoke, resume_from,
                  total, device, resumed_history, None, every, "m2b")
    resumed = [row["total"] for row in resumed_history]
    assert len(resumed) == total, f"resumed history has {len(resumed)} rows, expected {total}"
    stream_match = all(
        loader_b.streams[source]._order[:loader_b.streams[source].position]
        == final["sample_stream_snapshot"][source]
        for source in loader_b.streams)
    tail_uninterrupted = uninterrupted[resume_from:total]
    tail_resumed = resumed[resume_from:total]
    tail_diffs = [abs(a - b) for a, b in zip(tail_uninterrupted, tail_resumed)]
    max_diff = max(tail_diffs)
    gate = m2["resume_gate"]
    # exact-restore proof: the FIRST post-resume loss must be bitwise equal to
    # the uninterrupted run. It is computed by one forward pass from the
    # restored weights over the restored batch with the restored dropout RNG,
    # so any wrong checkpoint content (weights, Adam moments, loader position,
    # RNG) diverges here.
    restore_point_bitwise = tail_diffs[0] == 0.0
    # short-horizon trajectory equality: a safety net against sub-ULP effects
    # that strict determinism might still allow (e.g. allocator-dependent
    # kernel selection); any state error diverges macroscopically within the
    # first updates. Beyond the window, residual drift (if any remains under
    # the deterministic configuration) is recorded as a diagnostic, never
    # gated.
    window = gate["compare_window"]
    window_diffs = tail_diffs[:window]
    window_mean = float(np.mean(tail_uninterrupted[:window]))
    window_bound = gate["window_max_fraction_of_window_mean"] * abs(window_mean)
    window_bounded = max(window_diffs) <= window_bound
    # the resumed branch must still be training: finite losses and end-of-run
    # improvement against the restore-point level.
    resumed_finite = all(np.isfinite(resumed))
    resumed_progress = (float(np.mean(resumed[-gate["progress_window"]:]))
                        < gate["progress_factor"] * window_mean)
    # diagnostic only (not gated): chaotic weight drift after the compare window
    per_key_diff = {key: float((value.detach().float().cpu()
                                - final["model"][key].float().cpu()).abs().max())
                    for key, value in model_b.state_dict().items()}
    worst = sorted(per_key_diff.items(), key=lambda item: -item[1])[:5]

    result = {
        "mode": "m2_gpu_smoke_and_resume",
        "seed": seed,
        "device": str(device),
        "updates": total,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "effective_batch_size": 16 * m2["micro_batches_per_step"],
        "precision": m2["precision"] if device.type == "cuda" else "fp32_cpu_fallback",
        "deterministic_algorithms": "strict (warn_only=False)",
        "sdpa_backends": ({"mem_efficient": False, "flash": False, "math": True}
                          if device.type == "cuda" else "cpu_default"),
        "loss_first": uninterrupted[0],
        f"loss_update{resume_from}": uninterrupted[resume_from - 1],
        "loss_final": uninterrupted[-1],
        "elapsed_seconds": time.time() - started,
        "samples_per_second": ((len(history) + len(resumed_history) - len(mid["history"]))
                               * m2["micro_batches_per_step"] * 16
                               / (time.time() - started)),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6
                         if device.type == "cuda" else None),
        "resume_check": {
            "restored_from_update": resume_from,
            "gate_semantics": "restore-point bitwise equality + short-window "
                              "trajectory equality + stream match + resumed-branch "
                              "training sanity; deterministic SDPA (math backend) "
                              "under strict use_deterministic_algorithms; "
                              "long-horizon drift diagnostic only",
            "restore_point_bitwise_equal": bool(restore_point_bitwise),
            "compare_window": window,
            "window_max_abs_difference": max(window_diffs),
            "window_bound": window_bound,
            "window_bounded": bool(window_bounded),
            "resumed_losses_finite": bool(resumed_finite),
            "resumed_progress": bool(resumed_progress),
            "tail_first_difference": tail_diffs[0],
            "tail_max_abs_difference_diagnostic_only": max_diff,
            "tail_diff_trend": [round(value, 6) for value in tail_diffs[:10]],
            "sample_stream_matches_checkpoint": bool(stream_match),
            "final_weight_drift_diagnostic_only": {key: diff for key, diff in worst},
        },
        "resume_gate_passed": bool(restore_point_bitwise and window_bounded
                                   and stream_match and resumed_finite
                                   and resumed_progress),
        "exposure": loader.exposure(),
    }
    suffix = "_probe" if args.updates else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"M2_SMOKE_RESULT{suffix}.json").write_text(json.dumps(result, indent=2))
    np.savetxt(OUTPUT_DIR / f"M2_smoke_loss_curve{suffix}.csv",
               np.array([[row["total"], row["token_cross_entropy"],
                          row["trajectory_huber"], row["endpoint_huber"],
                          row["token_accuracy"]] for row in history]),
               delimiter=",",
               header="total,token_cross_entropy,trajectory_huber,endpoint_huber,token_accuracy",
               comments="")
    if not suffix:
        torch.save(final, OUTPUT_DIR / "m2_final_checkpoint.pt")
    print(f"[m2] resume gate: {'PASS' if result['resume_gate_passed'] else 'FAIL'} "
          f"(tail max diff {max_diff:.2e})")
    return result


def derive_architecture_budget(config: dict) -> dict:
    """Derive the architecture-stage training budget from the frozen config.

    Pure function (unit-tested): micro-batches per update, batches per epoch
    are resolved at runtime against the catalog; everything here depends only
    on ar_scene_v1.json so the budget is checkable without data.
    """
    training = config["training"]
    micro_batch = int(training["micro_batch_size"])
    effective = int(training["target_effective_batch_size"])
    if effective % micro_batch or (micro_batch % 2):
        raise ValueError("effective batch must be a multiple of the 2-source micro batch")
    return {
        "batch_per_source": micro_batch // 2,
        "micro_batches_per_step": effective // micro_batch,
        "max_epochs": int(training["max_epochs"]),
        "early_stopping_metric": training["early_stopping_metric"],
        "patience": int(training["early_stopping_patience"]),
        "seed": int(training["formal_seeds"][0]),
        "t1_sampling": dict(training["t1_sampling"]),
    }


def update_early_stopping(best: float, stagnant: int, metric: float,
                          patience: int) -> tuple[float, int, bool]:
    """Early-stopping state transition; returns (best, stagnant, improved)."""
    improved = metric < best
    return (metric, 0, True) if improved else (best, stagnant + 1, False)


def validate_architecture(model, catalog, config: dict, device: torch.device,
                          budget: dict, loss_fn=ar_scene_loss) -> dict:
    """Per-epoch dev validation (fp32, deterministic rollout seed).

    Reports the early-stopping metric (source_macro_minade_at_6, group-level),
    per-source group metrics including joint-scene best-of-6, teacher-forced
    NLL/accuracy, and the teacher-argmax vs rollout guardrail (P3.3.2 REPORT
    §8: ratio monitored as an autoregressive-drift early warning).
    """
    from scenario_lab.p33_dataset import iter_split_batches
    from scenario_lab.p33_metrics import (evaluate_agent_mask, group_level_table,
                                          rollout_records_for_batch)
    from scenario_lab.p33_model import decode_tokens_to_trajectory

    sampling = budget["t1_sampling"]
    rollout_seed = budget["seed"]
    codebook_numpy = model.codebook.cpu().numpy()
    records: list = []
    argmax_records: list = []
    ce_weighted, acc_weighted, token_count = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch_raw in iter_split_batches(catalog, split="dev", batch_size=16):
            batch = to_torch_batch(batch_raw, device)
            outputs = model(batch, teacher_tokens=batch["motion_token_target"])
            losses = loss_fn(outputs, batch, model.codebook, config)
            valid = (batch["motion_token_valid_mask"]
                     & (batch["motion_token_target"] != 255))
            tokens_in_batch = int(valid.sum())
            ce_weighted += float(losses["token_cross_entropy"]) * tokens_in_batch
            acc_weighted += float(losses["token_accuracy"]) * tokens_in_batch
            token_count += tokens_in_batch
            rollout = model.rollout(batch, num_samples=sampling["num_samples"],
                                    temperature=sampling["temperature"],
                                    top_p=sampling["top_p"], seed=rollout_seed)
            trajectories = rollout["trajectories"].cpu().numpy().astype(np.float64)
            records.extend(rollout_records_for_batch(trajectories, batch_raw))
            # teacher-argmax decode (guardrail): greedy token error only, no
            # sampling and no autoregressive drift beyond teacher prefixes
            if hasattr(model, "trajectory_from_tokens"):
                decoded = model.trajectory_from_tokens(
                    outputs["motion_token_logits"].argmax(-1), batch).cpu().numpy()
            else:
                tokens = outputs["motion_token_logits"].argmax(-1).cpu().numpy()
                residuals = outputs["delta_xy_residual"].cpu().numpy()
                current = batch["agent_history"][:, :, -1, :2].cpu().numpy()
                decoded = np.stack([decode_tokens_to_trajectory(
                    tokens[i].astype(np.int64), residuals[i], codebook_numpy, current[i])
                    for i in range(len(batch["sample_id"]))])
            argmax_records.extend(rollout_records_for_batch(decoded[:, None], batch_raw))
    table = group_level_table(records)
    argmax_table = group_level_table(argmax_records)
    sources = sorted(table)
    macro = float(np.mean([table[source]["min_ade"] for source in sources])) \
        if sources else float("nan")
    guardrail = {source: (float(table[source]["min_ade"]) / argmax_table[source]["ade"]
                          if argmax_table[source]["ade"] > 0 else float("inf"))
                 for source in sources}
    return {
        "source_macro_minade_at_6": macro,
        "group_level": table,
        "teacher_argmax_group_level": argmax_table,
        "guardrail_rollout_over_teacher_argmax": guardrail,
        "teacher_forced_token_nll": ce_weighted / max(1, token_count),
        "token_accuracy": acc_weighted / max(1, token_count),
        "evaluated_tokens": token_count,
    }


def run_architecture(args: argparse.Namespace) -> dict:
    """P3.3.3 architecture-stage training.

    Epoch-scheduled (loader epoch = one pass of the smaller source), dev
    validation at every epoch boundary, early stopping on
    source_macro_minade_at_6 with the frozen patience, best/last checkpoints,
    and full crash resume (weights/optimizer/loader/RNG + schedule state).
    """
    config = load_p33_config()
    training = config["training"]
    budget = derive_architecture_budget(config)
    manifest = Path(args.manifest) if args.manifest else ARCH_MANIFEST
    out_dir = Path(args.output_dir) if args.output_dir else ARCH_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    codebook = load_codebook(manifest)
    catalog = ShardCatalog.from_manifest(manifest, config)
    c_config = (json.loads(Path(args.c_config).read_text(encoding="utf-8"))
                if args.variant == "kinematic" else None)

    seed_everything(budget["seed"])
    loader = SceneLoader(catalog, split="train",
                         batch_per_source=budget["batch_per_source"],
                         seed=budget["seed"])
    model, optimizer = build(config, codebook, device, args.variant, c_config)

    def loss_fn(outputs, batch, codebook, config_arg=None):
        if args.variant == "kinematic":
            return ar_scene_loss_c(outputs, batch, codebook, config, c_config)
        return ar_scene_loss(outputs, batch, codebook, config)

    micro_per_step = budget["micro_batches_per_step"]
    clip = training["gradient_clip_norm"]
    use_bf16 = device.type == "cuda"  # frozen precision policy: amp bf16

    # LR warmup horizon: the full planned budget (max_epochs), independent of
    # early stopping so the schedule stays deterministic.
    batches_per_epoch = -(-loader.streams[loader._epoch_source].pool
                          // budget["batch_per_source"])  # ceil
    updates_per_epoch = -(-batches_per_epoch // micro_per_step)
    planned_total_updates = updates_per_epoch * budget["max_epochs"]

    state = {"update_index": 0, "epoch": 0, "history": [], "val_history": [],
             "best": float("inf"), "stagnant": 0, "best_epoch": None}
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict({key: value.to(device) for key, value in payload["model"].items()})
        restore_optimizer_isolated(optimizer, payload["optimizer"])
        loader.set_state(payload["loader_state"])
        set_rng_state(payload["rng"])
        state = {key: payload["arch"][key] for key in state}
        print(f"[arch] resumed from {args.resume}: epoch {state['epoch']}, "
              f"update {state['update_index']}, best {state['best']:.4f}", flush=True)

    stream = BatchStream(loader)
    started = time.time()
    log_path = out_dir / "ARCH_TRAIN_LOG.jsonl"
    curve_path = out_dir / "ARCH_loss_curve.csv"
    if not args.resume:
        log_path.write_text("")
        curve_path.write_text("epoch,update,total,token_cross_entropy,trajectory_huber,"
                              "endpoint_huber,token_accuracy\n")
    max_epochs = args.max_epochs if args.max_epochs else budget["max_epochs"]
    update_cap = args.max_updates if args.max_updates else float("inf")
    while state["epoch"] < max_epochs and state["update_index"] < update_cap:
        epoch_start = state["epoch"]
        epoch_losses = []
        # run updates until the loader crosses the epoch boundary (the final
        # micro-batch of an epoch triggers advance_epoch inside BatchStream)
        while loader.epoch == epoch_start and state["update_index"] < update_cap:
            update = state["update_index"]
            scale = learning_rate_scale(update, planned_total_updates,
                                        training["warmup_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = training["learning_rate"] * scale
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_losses = {}
            for _ in range(micro_per_step):
                batch = to_torch_batch(stream.take(), device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    outputs = model(batch, teacher_tokens=batch["motion_token_target"])
                    losses = loss_fn(outputs, batch, model.codebook, config)
                (losses["total"] / micro_per_step).backward()
                for key, value in losses.items():
                    step_losses[key] = (step_losses.get(key, 0.0)
                                        + float(value.item()) / micro_per_step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            state["history"].append(step_losses)
            epoch_losses.append(step_losses["total"])
            state["update_index"] += 1
            if update % 50 == 0:
                print(f"[arch] epoch {epoch_start + 1} update {update + 1} "
                      f"total={step_losses['total']:.4f} "
                      f"ce={step_losses['token_cross_entropy']:.4f} "
                      f"acc={step_losses['token_accuracy']:.3f}", flush=True)
        state["epoch"] = loader.epoch
        with curve_path.open("a", encoding="utf-8") as curve:
            for row in state["history"][-len(epoch_losses):]:
                curve.write(f"{state['epoch']},{len(state['history'])},"
                            f"{row['total']:.6f},{row['token_cross_entropy']:.6f},"
                            f"{row['trajectory_huber']:.6f},{row['endpoint_huber']:.6f},"
                            f"{row['token_accuracy']:.6f}\n")

        if (state["epoch"] % args.validate_every == 0
                or state["epoch"] == max_epochs
                or state["update_index"] >= update_cap):
            validation = validate_architecture(model, catalog, config, device,
                                               budget, loss_fn)
            state["best"], state["stagnant"], improved = update_early_stopping(
                state["best"], state["stagnant"],
                validation["source_macro_minade_at_6"], budget["patience"])
            if improved:
                state["best_epoch"] = state["epoch"]
            entry = {"epoch": state["epoch"], "update_index": state["update_index"],
                     "train_loss_epoch_mean": float(np.mean(epoch_losses)),
                     "improved": improved, "stagnant": state["stagnant"],
                     "elapsed_seconds": time.time() - started,
                     **validation}
            state["val_history"].append(entry)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, default=str) + "\n")
            print(f"[arch] epoch {state['epoch']} val: macro_minADE@6="
                  f"{validation['source_macro_minade_at_6']:.4f} "
                  f"(best {state['best']:.4f} @ep{state['best_epoch']}, "
                  f"stagnant {state['stagnant']}/{budget['patience']}) "
                  f"nll={validation['teacher_forced_token_nll']:.4f} "
                  f"guardrail={validation['guardrail_rollout_over_teacher_argmax']}",
                  flush=True)

        payload = checkpoint_payload(model, optimizer, loader, state["history"],
                                     state["update_index"],
                                     {"arch": {key: state[key] for key in state},
                                      "variant": args.variant, "c_config": c_config})
        torch.save(payload, out_dir / "arch_last_checkpoint.pt")
        if state["best_epoch"] == state["epoch"]:
            torch.save(payload, out_dir / "arch_best_checkpoint.pt")
        if state["stagnant"] >= budget["patience"]:
            print(f"[arch] early stopping: {state['stagnant']} epochs without "
                  f"improvement (best {state['best']:.4f} @ epoch {state['best_epoch']})",
                  flush=True)
            break

    result = {
        "mode": "p333_architecture_training",
        "variant": args.variant,
        "seed": budget["seed"],
        "device": str(device),
        "manifest": str(manifest),
        "budget": budget,
        "epochs_completed": state["epoch"],
        "updates_completed": state["update_index"],
        "planned_total_updates": planned_total_updates,
        "updates_per_epoch": updates_per_epoch,
        "best_source_macro_minade_at_6": state["best"],
        "best_epoch": state["best_epoch"],
        "early_stopped": state["stagnant"] >= budget["patience"],
        "deterministic_algorithms": "strict (warn_only=False)",
        "sdpa_backends": ({"mem_efficient": False, "flash": False, "math": True}
                          if device.type == "cuda" else "cpu_default"),
        "elapsed_seconds": time.time() - started,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6
                         if device.type == "cuda" else None),
        "exposure": loader.exposure(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (out_dir / "ARCH_TRAIN_RESULT.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["overfit", "smoke", "architecture"],
                        required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variant", choices=["residual", "kinematic"], default="residual",
                        help="architecture mode decode head: base residual or "
                             "Phase C kinematic (P3.3.4 SPEC 6)")
    parser.add_argument("--c-config", default=str(C_CONFIG_PATH),
                        help="Phase C head/loss config (kinematic variant only)")
    parser.add_argument("--max-updates", type=int, default=None,
                        help="engineering smoke cap on total updates (never set "
                             "for formal runs)")
    parser.add_argument("--updates", type=int, default=None,
                        help="override m2 update count (probe runs)")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="override m2 checkpoint cadence (probe runs)")
    parser.add_argument("--manifest", default=None,
                        help="architecture mode: dataset manifest path")
    parser.add_argument("--output-dir", default=None,
                        help="architecture mode: output directory")
    parser.add_argument("--resume", default=None,
                        help="architecture mode: checkpoint to resume from")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="engineering sanity override of the frozen epoch cap")
    parser.add_argument("--validate-every", type=int, default=1,
                        help="validate every N epochs (early stopping still honors it)")
    args = parser.parse_args()
    if args.mode == "overfit":
        result = run_overfit(args)
    elif args.mode == "architecture":
        result = run_architecture(args)
    else:
        result = run_smoke(args)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("sample_ids",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
