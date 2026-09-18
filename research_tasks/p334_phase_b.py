"""P3.3.4 Phase B: frozen-model constrained projection repair (no retraining).

Per SPEC §2 (runs/20260915_p334_kinematics/SPEC.md): every rollout candidate is
independently projected onto the kinematic-feasible set by batched GPU penalty
optimization -- minimize valid-step position deviation to the original
prediction subject to speed/accel/jerk limits, fixed start point, and
prediction-start continuity (first-step implied accel from the history-tail
velocity).  No future ground truth is touched; rows still violating after
optimization are COUNTED as solver_failure, never dropped.

One rollout pass produces three trajectory variants under identical sampling:
  orig     rollout trajectories (= P3.3.3 final evaluation)
  zero_res sampled tokens decoded with zero residuals (Phase A pred_token arm,
           here additionally scored for minADE@6 as the no-retrain fallback)
  proj     orig after constrained projection (this phase's repair)

Writes PHASE_B_METRICS.json with, per variant: group-level minADE@6/minFDE@6/
joint tables, per-(source,type) kinematic attribution, candidate diversity; and
for proj additionally repair displacement quantiles, solver_failure counts,
latency.  Gate evaluation happens in the report, not here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p334_kinematics_attribution import (ArmAccumulator, SAMPLING,  # noqa: E402
                                         history_tail_velocity, load_model, sha256_of)
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_metrics import (KINEMATIC_LIMITS, evaluate_agent_mask,  # noqa: E402
                                      group_level_table, kinematic_attribution_metrics,
                                      rollout_records_for_batch)
from scenario_lab.p33_model import decode_tokens_to_trajectory, to_torch_batch  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

FRAME_RATE = 10.0


def _clamp_disk(values: torch.Tensor, limit: float) -> torch.Tensor:
    """Scale each 2-vector to the limit disk (no-op inside it)."""
    norm = torch.linalg.norm(values, dim=-1, keepdim=True)
    return values * torch.clamp(limit / (norm + 1e-9), max=1.0)


def _feasible_warm_start(pred: torch.Tensor, v0_safe: torch.Tensor,
                         finite_v0: torch.Tensor, limits: dict,
                         window: int = 7) -> torch.Tensor:
    """Feasible-by-construction init: smooth, then a clamped second-order filter.

    Raw predictions violate persistently (residual oscillation at jerk ~600
    m/s^3), so clamping derivatives magnitude-wise and re-integrating
    accumulates the surviving sign bias and diverges (smoke run: 60-118 m
    displacement).  Instead: (1) a box moving-average of positions kills the
    oscillation; (2) the first velocity is replaced by the history-tail
    velocity; (3) a clamped second-order filter tracks the smoothed velocity
    profile with |a| <= 9.5 m/s^2 and |delta-a| <= 1.9 m/s^2 per step, so
    jerk <= 19 m/s^3 holds BY CONSTRUCTION (a first-order pursuit cap does
    not bound jerk when unsaturated); (4) positions re-integrate from the
    pinned start.  The polish then only re-merges toward the prediction
    INSIDE the feasible set.
    """
    flat = pred.permute(0, 2, 1).reshape(-1, 1, pred.shape[1])          # [N*2, 1, T]
    padded = torch.nn.functional.pad(flat, (window // 2, window // 2),
                                     mode="replicate")
    smoothed = (torch.nn.functional.avg_pool1d(padded, kernel_size=window, stride=1)
                .reshape(pred.shape[0], 2, -1).permute(0, 2, 1))
    target = _clamp_disk(torch.diff(smoothed, dim=1) * FRAME_RATE,
                         limits["speed_limit_mps"])
    velocity = torch.empty_like(target)
    velocity[:, 0] = torch.where(finite_v0[:, None], v0_safe, target[:, 0])
    # clamped second-order filter: accel chases the target-reaching accel with
    # |delta-a| <= 1.9 m/s^2 per step (jerk <= 19 m/s^3 BY CONSTRUCTION, a
    # first-order pursuit cap does NOT bound jerk when unsaturated) and |a| <= 9.5
    steps = target.shape[1]
    accel = torch.zeros_like(velocity[:, 0])
    for t in range(steps - 1):
        desired = (target[:, t + 1] - velocity[:, t]) * FRAME_RATE
        accel = _clamp_disk(accel + _clamp_disk(desired - accel, 1.9), 9.5)
        velocity[:, t + 1] = velocity[:, t] + accel / FRAME_RATE
    return torch.cat([pred[:, :1],
                      pred[:, :1] + torch.cumsum(velocity, dim=1) / FRAME_RATE],
                     dim=1)


def project_trajectories(pred: torch.Tensor, valid: torch.Tensor, v0: torch.Tensor,
                         *, iters: int = 50, lr: float = 0.5,
                         limits: dict | None = None) -> dict:
    """Batched projection of [N, T, 2] positions onto the feasible set.

    Feasible-interior method: the warm start (:func:`_feasible_warm_start`)
    satisfies speed/accel/jerk and the start-continuity by construction, then
    plain gradient descent on the PURE position deviation loss -- the data term
    is a separable quadratic, so lr=0.5 proposes ``x -> pred`` exactly and the
    backtracking line search accepts the largest feasible step fraction along
    that direction (a well-defined approximate projection).  Penalty polishing
    and Adam both failed here: constraint gradients 1000x the data gradients
    either stall Adam's second moment or trip the jerk limit on every step
    because per-coordinate normalization distorts the descent direction away
    from the feasible blend path.

    A difference quantity is checked only when every frame it spans is valid
    (same convention as :func:`kinematic_attribution_metrics`).  ``v0`` [N, 2]
    history-tail velocities; NaN rows keep the predicted first velocity.  The
    post-check runs at the FULL limits against the ORIGINAL ``v0`` -- rows the
    warm start could not repair (e.g. |v0| above the speed limit) stay
    violating and are counted, never silently dropped.
    """
    limits = limits or KINEMATIC_LIMITS
    finite_v0 = torch.isfinite(v0).all(dim=1)
    v0_safe = torch.where(finite_v0[:, None], v0, torch.zeros_like(v0))
    # the warm start integrates v0 as first velocity: clamp it into the speed
    # disk so warm-start speed stays feasible; the post-check still compares
    # against the UNCLAMPED v0, leaving genuinely over-speed histories as
    # counted solver failures
    v0_clamped = _clamp_disk(v0_safe, limits["speed_limit_mps"])
    x = _feasible_warm_start(pred.detach(), v0_clamped, finite_v0,
                             limits).requires_grad_(True)
    data_weight = valid.float()

    def derivatives(positions: torch.Tensor) -> tuple:
        velocity = torch.diff(positions, dim=1) * FRAME_RATE
        v_valid = valid[:, :-1] & valid[:, 1:]
        accel = torch.diff(velocity, dim=1) * FRAME_RATE
        a_valid = v_valid[:, :-1] & v_valid[:, 1:]
        jerk = torch.diff(accel, dim=1) * FRAME_RATE
        j_valid = a_valid[:, :-1] & a_valid[:, 1:]
        return (velocity, v_valid), (accel, a_valid), (jerk, j_valid)

    def violation_mask(positions: torch.Tensor) -> dict[str, torch.Tensor]:
        (velocity, v_valid), (accel, a_valid), (jerk, j_valid) = derivatives(positions)
        speed = torch.linalg.norm(velocity, dim=-1)
        cont = (torch.linalg.norm(velocity[:, 0] - v0_safe, dim=-1) * FRAME_RATE
                if finite_v0.any() else None)
        return {
            "speed": (speed > limits["speed_limit_mps"]) & v_valid,
            "accel": (torch.linalg.norm(accel, dim=-1) > limits["accel_limit_mps2"]) & a_valid,
            "jerk": (torch.linalg.norm(jerk, dim=-1) > limits["jerk_limit_mps3"]) & j_valid,
            "continuity": ((cont > limits["accel_limit_mps2"]) & finite_v0
                           if cont is not None else torch.zeros_like(finite_v0)),
        }

    def infeasible_rows(positions: torch.Tensor) -> torch.Tensor:
        masks = violation_mask(positions)
        bad = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
        for mask in masks.values():
            bad |= (mask.any(dim=1) if mask.ndim == 2 else mask)
        return bad

    for _ in range(iters):
        with torch.no_grad():
            previous = x.detach().clone()  # last accepted (feasible) iterate
        x.grad = None
        data = ((x - pred) ** 2 * data_weight[:, :, None]).sum()
        data.backward()
        with torch.no_grad():
            # plain GD: lr 0.5 on the separable quadratic lands exactly on pred
            x -= lr * x.grad
            x[:, 0] = pred[:, 0]  # hard equality: start point pinned to the anchor
            delta = x.detach() - previous
            alpha = 1.0
            while alpha >= 1.0 / 64:
                bad = infeasible_rows(x)
                if not bool(bad.any()):
                    break
                alpha /= 2.0
                x.copy_(previous + alpha * delta)
                x[:, 0] = pred[:, 0]
            if alpha < 1.0 / 64:
                # rows that cannot stay feasible along this direction freeze at
                # the previous (feasible) iterate; they are NOT failures
                bad = infeasible_rows(x)
                x[bad] = previous[bad]
            if (x - previous).abs().max() < 1e-4:
                break  # no row moved: converged

    with torch.no_grad():
        projected = x.detach().clone()
        violations = {}
        for name, mask in violation_mask(projected).items():
            # per-step masks collapse over the time axis; continuity is already [N]
            violations[name] = mask.any(dim=1) if mask.ndim == 2 else mask
        step_displacement = (torch.linalg.norm(projected - pred, dim=-1) * valid)
    return {"trajectories": projected.cpu().numpy(),
            "post_violation_rows": {name: mask.cpu().numpy() for name, mask in violations.items()},
            "step_displacement": step_displacement.cpu().numpy(),
            "iters": iters, "lr": lr, "limits": limits}


def candidate_diversity(trajectories: np.ndarray, valid: np.ndarray) -> float:
    """Mean pairwise L2 between candidates, averaged over valid steps then agents.

    ``trajectories`` [K, A, T, 2]; ``valid`` [A, T].  R0 fix (CODEX review P1):
    the v1 implementation summed over time instead of averaging, so a pair 1 m
    apart for 50 valid steps scored 50 instead of 1 -- the docstring's claimed
    time-mean never existed.  This function now returns the documented
    time-mean per agent, then the mean over agents with >= 1 valid step.
    """
    samples, _, _, _ = trajectories.shape
    pair_distance = 0.0
    pairs = 0
    for k in range(samples):
        for l in range(k + 1, samples):
            pair_distance += _pair_time_mean(trajectories[k], trajectories[l], valid)
            pairs += 1
    return pair_distance / max(pairs, 1)


def _pair_time_mean(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    per_step = np.linalg.norm(a - b, axis=-1) * valid          # [A, T]
    steps = valid.sum(axis=1)
    usable = steps > 0
    return float((per_step.sum(axis=1)[usable] / steps[usable]).mean()) if usable.any() else 0.0


def candidate_diversity_legacy(trajectories: np.ndarray, valid: np.ndarray) -> float:
    """The v1 (time-SUM) diversity CODEX flagged -- kept ONLY for cross-checking.

    Two trajectories a constant 1 m apart over 50 valid steps score 50.0 here
    and 1.0 in :func:`candidate_diversity`; ratios between arms computed with
    this function must not be interpreted as mean-distance ratios.
    """
    samples, _, _, _ = trajectories.shape
    pair_distance = 0.0
    pairs = 0
    for k in range(samples):
        for l in range(k + 1, samples):
            per_step = np.linalg.norm(trajectories[k] - trajectories[l], axis=-1)
            steps = valid.sum(axis=1)
            usable = steps > 0
            pair_distance += float((per_step * valid)[usable].sum(axis=1).mean()
                                   if usable.any() else 0.0)
            pairs += 1
    return pair_distance / max(pairs, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/arch_best_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "runs/20260915_p334_kinematics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke-test cap; None = full dev split")
    args = parser.parse_args()

    config = load_p33_config()
    device = torch.device(args.device)
    model = load_model(args.manifest, args.checkpoint, config, device)
    codebook = model.codebook.cpu().numpy().astype(np.float64)
    catalog = ShardCatalog.from_manifest(args.manifest, config)

    records = {v: [] for v in ("orig", "zero_res", "proj")}
    accumulators = {v: ArmAccumulator() for v in records}
    diversity = {v: [] for v in records}
    displacement_steps: list[np.ndarray] = []
    solver_rows = 0
    total_rows = 0
    projection_seconds = 0.0
    samples_seen = 0
    started = time.time()

    for batch_number, batch_raw in enumerate(
            iter_split_batches(catalog, split=SAMPLING["split"],
                               batch_size=args.batch_size)):
        if args.max_batches is not None and batch_number >= args.max_batches:
            break
        batch_torch = to_torch_batch(batch_raw, device)
        rollout = model.rollout(batch_torch, num_samples=SAMPLING["samples"],
                                temperature=SAMPLING["temperature"],
                                top_p=SAMPLING["top_p"], seed=SAMPLING["rollout_seed"])
        size = len(batch_raw["sample_id"])
        samples_seen += size
        eval_masks = np.stack([
            evaluate_agent_mask(batch_raw["agent_present_mask"][index],
                                batch_raw["agent_role"][index],
                                batch_raw["future_valid_mask"][index])
            for index in range(size)])
        history = history_tail_velocity(batch_raw)
        originals = rollout["trajectories"].cpu().numpy().astype(np.float64)
        variants: dict[str, np.ndarray] = {"orig": originals.copy()}
        zero_res = np.zeros_like(originals)
        projected = np.zeros_like(originals)

        # zero-residual decode for evaluated agents only (others never read)
        tokens_all = rollout["tokens"].cpu().numpy()
        for index in range(size):
            mask = eval_masks[index]
            if not mask.any():
                continue
            current = batch_raw["agent_history"][index][mask][:, -1, :2].astype(np.float64)
            for slot, agent in enumerate(np.flatnonzero(mask)):
                block = decode_tokens_to_trajectory(
                    tokens_all[index][:, agent],
                    np.zeros(tokens_all[index][:, agent].shape + (10,)),
                    codebook, current[slot])
                zero_res[index][:, agent] = block

        # constrained projection: flatten (sample, candidate, agent) rows
        rows_pred, rows_valid, rows_v0, rows_index = [], [], [], []
        for index in range(size):
            mask = eval_masks[index]
            for agent in np.flatnonzero(mask):
                for k in range(originals.shape[1]):
                    rows_pred.append(originals[index, k, agent])
                    rows_valid.append(batch_raw["future_valid_mask"][index][agent])
                    rows_v0.append(history[index][agent])
                    rows_index.append((index, k, int(agent)))
        if rows_pred:
            torch.manual_seed(7)
            tick = time.time()
            result = project_trajectories(
                torch.tensor(np.stack(rows_pred), dtype=torch.float32, device=device),
                torch.tensor(np.stack(rows_valid), dtype=torch.bool, device=device),
                torch.tensor(np.stack(rows_v0), dtype=torch.float32, device=device),
                iters=args.iters)
            projection_seconds += time.time() - tick
            for n, (index, k, agent) in enumerate(rows_index):
                projected[index, k, agent] = result["trajectories"][n]
            displacement_steps.append(result["step_displacement"][result["step_displacement"] > 0])
            failure = np.zeros(len(rows_index), dtype=bool)
            for mask_rows in result["post_violation_rows"].values():
                failure |= mask_rows
            solver_rows += int(failure.sum())
            total_rows += len(rows_index)
        variants["zero_res"] = zero_res
        variants["proj"] = projected

        for name, trajectories in variants.items():
            records[name].extend(rollout_records_for_batch(trajectories, batch_raw, eval_masks))
            for index in range(size):
                mask = eval_masks[index]
                if mask.any():
                    diversity[name].append(candidate_diversity(
                        trajectories[index][:, mask],
                        batch_raw["future_valid_mask"][index][mask]))
            for index in range(size):
                mask = eval_masks[index]
                for agent in np.flatnonzero(mask):
                    count = trajectories.shape[1]
                    block_valid = batch_raw["future_valid_mask"][index][agent]
                    accumulators[name].add(
                        batch_raw["sample_source"][index],
                        batch_raw["agent_type"][index][agent],
                        trajectories[index][:, agent],
                        np.repeat(block_valid[None], count, axis=0),
                        np.repeat(history[index][agent][None], count, axis=0))

    displacement = (np.concatenate(displacement_steps) if displacement_steps
                    else np.array([]))
    displacement_stats = ({
        "count": int(displacement.size),
        "mean": float(displacement.mean()),
        "p50": float(np.percentile(displacement, 50)),
        "p95": float(np.percentile(displacement, 95)),
        "p99": float(np.percentile(displacement, 99)),
        "max": float(displacement.max()),
    } if displacement.size else None)

    output = {
        "phase": "B_constrained_projection",
        "spec": "runs/20260915_p334_kinematics/SPEC.md#2",
        "checkpoint": {"path": str(args.checkpoint.relative_to(REPO)),
                       "sha256": sha256_of(args.checkpoint)},
        "sampling": {key: SAMPLING[key] for key in ("samples", "temperature", "top_p",
                                                    "rollout_seed")},
        "samples": samples_seen,
        "elapsed_seconds": time.time() - started,
        "projection": {"iters": args.iters, "seconds": projection_seconds,
                       "ms_per_sample": 1000.0 * projection_seconds / max(samples_seen, 1),
                       "rows": total_rows, "solver_failure_rows": solver_rows,
                       "solver_failure_rate": solver_rows / max(total_rows, 1),
                       "repair_displacement_m": displacement_stats},
        "variants": {},
    }
    for name in ("orig", "zero_res", "proj"):
        output["variants"][name] = {
            "group_level": group_level_table(records[name]),
            "kinematics": accumulators[name].table(),
            "diversity_mean_pairwise_m": float(np.mean(diversity[name])),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "PHASE_B_METRICS.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(out_path),
                      "elapsed_seconds": round(output["elapsed_seconds"], 1),
                      "solver_failure_rate": output["projection"]["solver_failure_rate"]}))


if __name__ == "__main__":
    main()
