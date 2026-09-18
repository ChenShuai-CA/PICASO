"""P3.3.4-R0 shared re-evaluation (EVAL_PROTOCOL_V2.md): orig / proj(B) / C-v1
under one identical dev pass, with the metrics the CODEX review flagged.

Adds over the v1 evaluations:
  * diversity as a true time-mean (legacy time-sum kept side by side),
  * explicit candidate validity layers (numeric / kinematic / nonfinite counts
    at candidate, agent and scene level) -- never inferred from finite minADE,
  * a second, anchored kinematic convention that prepends the history anchor
    P0 and the history-tail velocity so start-of-prediction terms are visible,
  * input-outlier scan of history-tail speed by BOTH position differencing and
    the recorded vx/vy channels; flagged rows stay in every denominator,
  * phased latency measurement (warmup + cuda-synced, generation separated
    from projection and from CPU metric work).

Writes METRICS_V2.json / INPUT_OUTLIERS.jsonl / ARTIFACT_MANIFEST.json into the
output directory.  Nothing in the v1 artifacts is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p334_kinematics_attribution import (AGENT_TYPE_NAMES, ArmAccumulator,  # noqa: E402
                                         SAMPLING, history_tail_velocity,
                                         load_model, sha256_of)
from p334_phase_b import (candidate_diversity, candidate_diversity_legacy,  # noqa: E402
                          project_trajectories)
from p334_phase_c_eval import load_kinematic_model  # noqa: E402
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_metrics import (KINEMATIC_LIMITS, evaluate_agent_mask,  # noqa: E402
                                      group_level_table,
                                      kinematic_attribution_metrics,
                                      rollout_records_for_batch)
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

FRAME_RATE = 10.0
SPEED_LIMIT = KINEMATIC_LIMITS["speed_limit_mps"]
ACCEL_LIMIT = KINEMATIC_LIMITS["accel_limit_mps2"]
JERK_LIMIT = KINEMATIC_LIMITS["jerk_limit_mps3"]
OUTLIER_SPEED_MPS = SPEED_LIMIT          # protocol section 4: |v| > 35 flags
V0_RECORD_LIMIT = SPEED_LIMIT            # SPEC_R1 section 2.3 recorded-v0 clamp


def recorded_tail_velocity(batch: dict) -> tuple[np.ndarray, np.ndarray]:
    """SPEC_R1 section 2.3 diagnostic convention: last-frame RECORDED vx/vy,
    disk-clamped to V0_RECORD_LIMIT, NaN where the tail is invalid/non-finite
    (so candidate_validity's no-history path applies).  Returns (v0, over)."""
    recorded = batch["agent_history"][:, :, -1, 2:4].astype(np.float64)
    valid = batch["state_valid_mask"][:, :, -1]
    finite = np.isfinite(recorded).all(axis=-1)
    norm = np.linalg.norm(recorded, axis=-1)
    over = valid & finite & (norm > V0_RECORD_LIMIT)
    clamped = recorded * np.minimum(1.0, V0_RECORD_LIMIT / np.maximum(norm, 1e-9))[..., None]
    clamped[~(valid & finite)] = np.nan
    return clamped, over


# --------------------------------------------------------------------- blocks
def _block(values: np.ndarray, mask: np.ndarray, limit: float) -> dict:
    """Violation block with EXPLICIT nonfinite accounting (protocol section 2)."""
    finite = np.isfinite(values)
    violating = ((values > limit) | ~finite) & mask
    agents_with_steps = mask.any(axis=1)
    agent_count = int(agents_with_steps.sum())
    total_steps = int(mask.sum())
    traj_rate = (float(violating.any(axis=1)[agents_with_steps].mean())
                 if agent_count else None)
    step_rate = float(violating.sum() / total_steps) if total_steps else None
    good = finite & mask
    exceedance = (values[good] - limit)[values[good] > limit] if good.any() else np.array([])
    quantiles = ({q: float(np.percentile(exceedance, int(q[1:])))
                  for q in ("p50", "p95", "p99")} | {"max": float(exceedance.max())}
                 if exceedance.size else None)
    return {"limit": limit, "agents_with_steps": agent_count,
            "traj_rate": traj_rate, "step_rate": step_rate,
            "exceedance_quantiles": quantiles,
            "violating_steps": int(violating.sum()), "total_steps": total_steps,
            "nonfinite_steps": int((~finite & mask).sum())}


def anchored_kinematic_metrics(trajectories: np.ndarray, step_valid: np.ndarray,
                               anchors: np.ndarray, v0: np.ndarray,
                               frame_rate: float = FRAME_RATE) -> dict:
    """Anchored convention (protocol section 3): prepend P0 and the history
    velocity so v_first = (P1-P0)*rate and the first accel (v_first - v0)*rate
    are inside the tables.  ``v0`` rows may be NaN (invalid history tail) --
    the earliest accel/jerk terms that would use them are masked out, and the
    count of rows without a usable v0 is reported."""
    flat = np.asarray(trajectories, dtype=np.float64)
    valid = np.asarray(step_valid, dtype=bool)
    extended = np.concatenate([anchors[:, None, :], flat], axis=1)      # [N, T+1, 2]
    ext_valid = np.concatenate([np.ones((valid.shape[0], 1), dtype=bool), valid], axis=1)
    velocity = np.diff(extended, axis=1) * frame_rate                   # [N, T, 2]
    v_valid = ext_valid[:, :-1] & ext_valid[:, 1:]
    speed = np.linalg.norm(velocity, axis=-1)

    v0_ok = np.isfinite(v0).all(axis=1)                                 # [N]
    velocity_ext = np.concatenate(
        [np.where(v0_ok[:, None], v0, np.nan)[:, None, :], velocity], axis=1)
    ve_finite = np.isfinite(velocity_ext).all(axis=-1)
    ve_valid = np.concatenate([v0_ok[:, None], v_valid], axis=1)
    accel = np.diff(velocity_ext, axis=1) * frame_rate                  # [N, T, 2]
    a_valid = ve_valid[:, :-1] & ve_valid[:, 1:] & ve_finite[:, :-1] & ve_finite[:, 1:]
    accel_norm = np.linalg.norm(accel, axis=-1)
    jerk = np.diff(accel, axis=1) * frame_rate
    j_valid = a_valid[:, :-1] & a_valid[:, 1:]
    jerk_norm = np.linalg.norm(jerk, axis=-1)

    return {"speed": _block(speed, v_valid, SPEED_LIMIT),
            "accel": _block(accel_norm, a_valid, ACCEL_LIMIT),
            "jerk": _block(jerk_norm, j_valid, JERK_LIMIT),
            "rows_without_history_velocity": int((~v0_ok).sum())}


def candidate_validity(trajs: np.ndarray, step_valid: np.ndarray,
                       anchor: np.ndarray, v0: np.ndarray) -> dict:
    """Per-candidate validity layers (protocol section 2).

    ``trajs`` [K, T, 2] one agent's candidates; ``step_valid`` [T];
    ``anchor`` [2]; ``v0`` [2] (may be NaN)."""
    numeric = np.isfinite(trajs).all(axis=(1, 2))                       # [K]

    velocity = np.diff(trajs, axis=1) * FRAME_RATE                      # [K, T-1, 2]
    v_valid = step_valid[:-1] & step_valid[1:]                          # [T-1]
    speed_ok = np.isfinite(velocity).all(axis=-1) & (
        np.linalg.norm(velocity, axis=-1) <= SPEED_LIMIT)
    accel = np.diff(velocity, axis=1) * FRAME_RATE
    a_valid = v_valid[:-1] & v_valid[1:]
    accel_ok = np.isfinite(accel).all(axis=-1) & (
        np.linalg.norm(accel, axis=-1) <= ACCEL_LIMIT)
    jerk = np.diff(accel, axis=1) * FRAME_RATE
    j_valid = a_valid[:-1] & a_valid[1:]
    jerk_ok = np.isfinite(jerk).all(axis=-1) & (
        np.linalg.norm(jerk, axis=-1) <= JERK_LIMIT)
    frozen_ok = ((speed_ok | ~v_valid).all(axis=1) & (accel_ok | ~a_valid).all(axis=1)
                 & (jerk_ok | ~j_valid).all(axis=1))

    v_first = (trajs[:, 0] - anchor[None, :]) * FRAME_RATE             # [K, 2]
    if np.isfinite(v0).all():
        start_ok = np.isfinite(v_first).all(axis=-1) & (
            np.linalg.norm(v_first - v0[None, :], axis=-1) * FRAME_RATE <= ACCEL_LIMIT)
        no_history = False
    else:
        start_ok = np.ones(trajs.shape[0], dtype=bool)
        no_history = True
    kinematic = numeric & frozen_ok & start_ok
    return {"attempted": int(trajs.shape[0]),
            "numeric_valid": int(numeric.sum()),
            "kinematic_valid": int(kinematic.sum()),
            "nonfinite": int((~numeric).sum()),
            "no_history_velocity": bool(no_history)}


# ---------------------------------------------------------------- accumulators
class AnchoredAccumulator:
    """Streams (trajs, valid, anchor, v0) blocks into per-(source, type) tables."""

    def __init__(self) -> None:
        self.groups: dict[tuple, list] = defaultdict(list)

    def add(self, source: str, agent_type: int, trajectories: np.ndarray,
            step_valid: np.ndarray, anchor: np.ndarray, v0: np.ndarray) -> None:
        self.groups[(source, AGENT_TYPE_NAMES.get(int(agent_type), "other"))].append(
            (trajectories, step_valid, anchor, v0))

    def table(self) -> dict:
        out: dict = {}
        for (source, type_name), chunks in sorted(self.groups.items()):
            out[f"{source}/{type_name}"] = anchored_kinematic_metrics(
                np.concatenate([c[0] for c in chunks]),
                np.concatenate([c[1] for c in chunks]),
                np.concatenate([c[2] for c in chunks]),
                np.concatenate([c[3] for c in chunks]))
        return out


def index_rows(catalog: ShardCatalog) -> dict[str, dict]:
    """sample_id -> shard index row (source_file, group_id, transform, ...)."""
    rows: dict[str, dict] = {}
    for entry in catalog.entries():
        for line in entry.index_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["sample_id"]] = row | {"_shard": entry.path}
    return rows


def artifact_manifest() -> dict:
    files = [
        "runs/20260915_p334_kinematics/SPEC.md",
        "runs/20260916_p334_review_closure/EVAL_PROTOCOL_V2.md",
        "runs/20260916_p334_r1/SPEC_R1.md",
        "configs/p33/ar_scene_v1.json", "configs/p33/model_smoke_v1.json",
        "configs/p33/model_kinematic_c_v1.json",
        "configs/p33/model_kinematic_c_v2.json",
        "scenario_lab/p33_model.py", "scenario_lab/p33_metrics.py",
        "scenario_lab/p33_dataset.py",
        "research_tasks/p334_phase_b.py", "research_tasks/p334_phase_c_eval.py",
        "research_tasks/p334_kinematics_attribution.py",
        "research_tasks/p334_review_eval.py",
        "runs/20260913_p333_architecture/arch_best_checkpoint.pt",
        "runs/20260915_p334_kinematics/phase_c_train/arch_best_checkpoint.pt",
        "runs/20260915_p334_kinematics/PHASE_B_METRICS.json",
        "runs/20260915_p334_kinematics/PHASE_C_METRICS.json",
        "runs/20260915_p334_kinematics/codex_review/EVIDENCE.json",
    ]
    c2 = Path("runs/20260916_p334_r1/c_v2_train/arch_best_checkpoint.pt")
    if c2.exists():
        files.append(str(c2))
    hashes = {}
    for rel in files:
        path = REPO / rel
        hashes[rel] = sha256_of(path) if path.exists() else "MISSING"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--stat"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": head,
            "git_diff_stat": diff, "files_sha256": hashes}


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--orig-checkpoint", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/arch_best_checkpoint.pt")
    parser.add_argument("--c-checkpoint", type=Path,
                        default=REPO / "runs/20260915_p334_kinematics/phase_c_train/arch_best_checkpoint.pt")
    parser.add_argument("--c2-checkpoint", type=Path, default=None,
                        help="R1 C-v2 checkpoint; adds the C-v2 arm (SPEC_R1 section 6)")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "runs/20260916_p334_review_closure")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke cap; None = full dev split")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-batches", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_p33_config()
    device = torch.device(args.device)
    model_orig = load_model(args.manifest, args.orig_checkpoint, config, device)
    model_c, _ = load_kinematic_model(args.c_checkpoint, args.manifest, config, device)
    model_c2 = (load_kinematic_model(args.c2_checkpoint, args.manifest, config, device)[0]
                if args.c2_checkpoint else None)
    catalog = ShardCatalog.from_manifest(args.manifest, config)
    rows_by_id = index_rows(catalog)

    arms = ("orig", "proj", "C") + (("C-v2",) if model_c2 else ())
    records = {a: [] for a in arms}
    # clean_stratum (protocol section 0): same tables with the flagged input
    # outlier SAMPLES removed -- diagnostic only, never a gate reading
    clean_records = {a: [] for a in arms}
    outlier_ids: set[str] = set()
    frozen_acc = {a: ArmAccumulator() for a in arms}
    anchored_acc = {a: AnchoredAccumulator() for a in arms}
    diversity_v2 = {a: [] for a in arms}
    diversity_legacy = {a: [] for a in arms}
    validity = {a: defaultdict(lambda: {"attempted": 0, "numeric_valid": 0,
                                        "kinematic_valid": 0, "nonfinite": 0,
                                        "agents": 0, "agents_all6": 0,
                                        "agents_any": 0, "agents_none": 0,
                                        "no_history_velocity": 0})
                for a in arms}
    # SPEC_R1 section 2.3: same layers with the start term checked against the
    # recorded-clamped v0 (diagnostic column; the diff convention above stays
    # the primary, continuous with R0)
    validity_rec = {a: defaultdict(lambda: {"attempted": 0, "numeric_valid": 0,
                                            "kinematic_valid": 0, "nonfinite": 0,
                                            "agents": 0, "agents_all6": 0,
                                            "agents_any": 0, "agents_none": 0,
                                            "no_history_velocity": 0})
                    for a in arms}
    v0_over_limit = {"agents": 0, "over_limit": 0}
    scene_counts = {a: {"scenes": 0, "scenes_no_valid_agent": 0} for a in arms}
    solver = {"rows": 0, "failure_rows": 0}
    outliers: list[dict] = []
    sample_ids: list[str] = []
    projection_seconds = 0.0
    samples_seen = 0
    started = time.time()

    for batch_number, batch_raw in enumerate(
            iter_split_batches(catalog, split=SAMPLING["split"],
                               batch_size=args.batch_size)):
        if args.max_batches is not None and batch_number >= args.max_batches:
            break
        batch_torch = to_torch_batch(batch_raw, device)
        size = len(batch_raw["sample_id"])
        samples_seen += size
        sample_ids.extend(batch_raw["sample_id"])
        eval_masks = np.stack([
            evaluate_agent_mask(batch_raw["agent_present_mask"][i],
                                batch_raw["agent_role"][i],
                                batch_raw["future_valid_mask"][i])
            for i in range(size)])
        history = history_tail_velocity(batch_raw)
        recorded_v = batch_raw["agent_history"][:, :, -1, 2:4].astype(np.float64)
        history_valid = batch_raw["state_valid_mask"][:, :, -1]

        rollout = model_orig.rollout(
            batch_torch, num_samples=SAMPLING["samples"],
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            seed=SAMPLING["rollout_seed"])
        originals = rollout["trajectories"].cpu().numpy().astype(np.float64)
        projected = originals.copy()

        rows_pred, rows_valid, rows_v0, rows_index = [], [], [], []
        for i in range(size):
            mask = eval_masks[i]
            anchor_all = batch_raw["agent_history"][i][:, -1, :2].astype(np.float64)
            for agent in np.flatnonzero(mask):
                # ---- input outlier scan (protocol section 4) ----
                v_diff = history[i, agent]
                v_rec = recorded_v[i, agent] if history_valid[i, agent] else np.full(2, np.nan)
                if (np.isfinite(v_diff).any() and np.linalg.norm(v_diff) > OUTLIER_SPEED_MPS) or \
                   (np.isfinite(v_rec).any() and np.linalg.norm(v_rec) > OUTLIER_SPEED_MPS):
                    row = rows_by_id.get(batch_raw["sample_id"][i], {})
                    outlier_ids.add(batch_raw["sample_id"][i])
                    outliers.append({
                        "sample_id": batch_raw["sample_id"][i],
                        "source": batch_raw["sample_source"][i],
                        "group_id": row.get("group_id"),
                        "agent_slot": int(agent),
                        "agent_type": int(batch_raw["agent_type"][i][agent]),
                        "shard": (str(row["_shard"]) if row.get("_shard") else None),
                        "source_file": row.get("source_file"),
                        "coordinate_transform": row.get("coordinate_transform"),
                        "history_xy_last2": batch_raw["agent_history"][i][agent, -2:, :2].tolist(),
                        "state_valid_last2": batch_raw["state_valid_mask"][i][agent, -2:].tolist(),
                        "diff_speed_mps": float(np.linalg.norm(v_diff)) if np.isfinite(v_diff).all() else None,
                        "recorded_speed_mps": float(np.linalg.norm(v_rec)) if np.isfinite(v_rec).all() else None,
                        "recorded_vxy": (v_rec.tolist() if np.isfinite(v_rec).all() else None),
                        "in_denominator": True,
                    })
                for k in range(originals.shape[1]):
                    rows_pred.append(originals[i, k, agent])
                    rows_valid.append(batch_raw["future_valid_mask"][i][agent])
                    rows_v0.append(history[i, agent])
                    rows_index.append((i, k, int(agent)))
        if rows_pred:
            torch.manual_seed(7)
            tick = time.perf_counter()
            result = project_trajectories(
                torch.tensor(np.stack(rows_pred), dtype=torch.float32, device=device),
                torch.tensor(np.stack(rows_valid), dtype=torch.bool, device=device),
                torch.tensor(np.stack(rows_v0), dtype=torch.float32, device=device),
                iters=args.iters)
            projection_seconds += time.perf_counter() - tick
            for n, (i, k, agent) in enumerate(rows_index):
                projected[i, k, agent] = result["trajectories"][n]
            failure = np.zeros(len(rows_index), dtype=bool)
            for mask_rows in result["post_violation_rows"].values():
                failure |= mask_rows
            solver["failure_rows"] += int(failure.sum())
            solver["rows"] += len(rows_index)

        rollout_c = model_c.rollout(
            batch_torch, num_samples=SAMPLING["samples"],
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            seed=SAMPLING["rollout_seed"])
        c_trajs = rollout_c["trajectories"].cpu().numpy().astype(np.float64)
        arm_trajs = {"orig": originals, "proj": projected, "C": c_trajs}
        if model_c2 is not None:
            arm_trajs["C-v2"] = model_c2.rollout(
                batch_torch, num_samples=SAMPLING["samples"],
                temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                seed=SAMPLING["rollout_seed"])["trajectories"].cpu().numpy().astype(np.float64)
        v0_recorded, over_rec = recorded_tail_velocity(batch_raw)
        v0_over_limit["agents"] += int(eval_masks.sum())
        v0_over_limit["over_limit"] += int(over_rec[eval_masks].sum())

        for name, trajectories in arm_trajs.items():
            batch_records = rollout_records_for_batch(trajectories, batch_raw, eval_masks)
            records[name].extend(batch_records)
            clean_records[name].extend(r for r in batch_records
                                       if r.sample_id not in outlier_ids)
            for i in range(size):
                mask = eval_masks[i]
                if not mask.any():
                    continue
                scene_counts[name]["scenes"] += 1
                agent_level = []
                anchor_all = batch_raw["agent_history"][i][:, -1, :2].astype(np.float64)
                future_valid = batch_raw["future_valid_mask"][i]
                diversity_v2[name].append(candidate_diversity(
                    trajectories[i][:, mask], future_valid[mask]))
                diversity_legacy[name].append(candidate_diversity_legacy(
                    trajectories[i][:, mask], future_valid[mask]))
                for agent in np.flatnonzero(mask):
                    key = (batch_raw["sample_source"][i],
                           int(batch_raw["agent_type"][i][agent]))

                    def accumulate(table, v0_row, count):
                        stats = table[key]
                        verdict = candidate_validity(
                            trajectories[i][:, agent], future_valid[agent],
                            anchor_all[agent], v0_row)
                        for field in ("attempted", "numeric_valid", "kinematic_valid",
                                      "nonfinite"):
                            stats[field] += verdict[field]
                        stats["agents"] += 1
                        stats["agents_all6"] += verdict["kinematic_valid"] == verdict["attempted"] == count
                        stats["agents_any"] += verdict["kinematic_valid"] > 0
                        stats["agents_none"] += verdict["kinematic_valid"] == 0
                        stats["no_history_velocity"] += int(verdict["no_history_velocity"])
                        return verdict

                    verdict = accumulate(validity[name], history[i, agent],
                                         trajectories.shape[1])
                    accumulate(validity_rec[name], v0_recorded[i, agent],
                               trajectories.shape[1])
                    agent_level.append(verdict["kinematic_valid"] > 0)
                    count = trajectories.shape[1]
                    frozen_acc[name].add(
                        batch_raw["sample_source"][i],
                        int(batch_raw["agent_type"][i][agent]),
                        trajectories[i][:, agent],
                        np.repeat(future_valid[agent][None], count, axis=0),
                        np.repeat(history[i][agent][None], count, axis=0))
                    anchored_acc[name].add(
                        batch_raw["sample_source"][i],
                        int(batch_raw["agent_type"][i][agent]),
                        trajectories[i][:, agent],
                        np.repeat(future_valid[agent][None], count, axis=0),
                        np.repeat(anchor_all[agent][None], count, axis=0),
                        np.repeat(history[i][agent][None], count, axis=0))
                scene_counts[name]["scenes_no_valid_agent"] += not any(agent_level)

    id_digest = hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest()
    if args.max_batches is None and samples_seen != 11586:
        print(f"WARNING: expected 11586 dev samples, saw {samples_seen}", file=sys.stderr)

    latency = None
    if not args.skip_latency:
        latency = measure_latency(args, catalog, model_orig, model_c, device,
                                  model_c2=model_c2)

    output = {
        "phase": "R0_shared_reeval_v2",
        "protocol": "runs/20260916_p334_review_closure/EVAL_PROTOCOL_V2.md",
        "r1": ({"spec": "runs/20260916_p334_r1/SPEC_R1.md",
                "c2_checkpoint": (str(args.c2_checkpoint)
                                  if args.c2_checkpoint else None)}
               if model_c2 is not None else None),
        "sampling": {k: SAMPLING[k] for k in ("samples", "temperature", "top_p",
                                              "rollout_seed")},
        "samples": samples_seen,
        "sample_id_sorted_sha256": id_digest,
        "outlier_samples_excluded_in_clean_stratum": sorted(outlier_ids),
        "v0_recorded_over_limit": v0_over_limit,
        "elapsed_seconds": time.time() - started,
        "arms": {},
        "solver_status": solver | {"failure_rate": solver["failure_rows"]
                                   / max(solver["rows"], 1)},
    }
    for name in arms:
        output["arms"][name] = {
            "group_level": group_level_table(records[name]),
            "group_level_clean_stratum": group_level_table(clean_records[name]),
            "kinematics_frozen": frozen_acc[name].table(),
            "kinematics_anchored": anchored_acc[name].table(),
            "diversity_v2_mean": float(np.mean(diversity_v2[name])) if diversity_v2[name] else None,
            "diversity_legacy_mean": (float(np.mean(diversity_legacy[name]))
                                      if diversity_legacy[name] else None),
            "validity": {f"{src}/{AGENT_TYPE_NAMES.get(typ, typ)}": stats
                         for (src, typ), stats in sorted(validity[name].items())},
            "validity_recorded_v0": {f"{src}/{AGENT_TYPE_NAMES.get(typ, typ)}": stats
                                     for (src, typ), stats
                                     in sorted(validity_rec[name].items())},
            "validity_totals": scene_counts[name],
        }
    if latency is not None:
        output["latency"] = latency

    metrics_path = args.output_dir / "METRICS_V2.json"
    metrics_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    with (args.output_dir / "INPUT_OUTLIERS.jsonl").open("w", encoding="utf-8") as fh:
        for row in outliers:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(artifact_manifest(), indent=1), encoding="utf-8")
    print(json.dumps({"written": str(metrics_path), "samples": samples_seen,
                      "outliers": len(outliers),
                      "elapsed_seconds": output["elapsed_seconds"]}))


def measure_latency(args, catalog, model_orig, model_c, device,
                    model_c2=None) -> dict:
    """Phased latency (protocol section 5): warmup, cuda-synced, per phase."""
    def timed(fn) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - tick

    batches = []
    for n, batch_raw in enumerate(iter_split_batches(
            catalog, split=SAMPLING["split"], batch_size=args.batch_size)):
        batches.append(batch_raw)
        if len(batches) >= args.latency_batches + 2:
            break
    timers = {"orig_generate": [], "c_generate": [], "c2_generate": [], "b_projection": [],
              "cpu_metrics": []}
    for n, batch_raw in enumerate(batches):
        batch_torch = to_torch_batch(batch_raw, device)
        size = len(batch_raw["sample_id"])
        originals = model_orig.rollout(
            batch_torch, num_samples=SAMPLING["samples"],
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            seed=SAMPLING["rollout_seed"])["trajectories"].cpu().numpy().astype(np.float64)
        if n < 2:
            continue  # warmup batch, not timed
        timers["orig_generate"].append(timed(lambda: model_orig.rollout(
            batch_torch, num_samples=SAMPLING["samples"],
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            seed=SAMPLING["rollout_seed"])) / size)
        timers["c_generate"].append(timed(lambda: model_c.rollout(
            batch_torch, num_samples=SAMPLING["samples"],
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            seed=SAMPLING["rollout_seed"])) / size)
        if model_c2 is not None:
            timers["c2_generate"].append(timed(lambda: model_c2.rollout(
                batch_torch, num_samples=SAMPLING["samples"],
                temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                seed=SAMPLING["rollout_seed"])) / size)
        # projection phase: all agent rows of the batch (throughput measurement,
        # not a correctness arm -- v0=0 / all-valid rows are representative load)
        rows = [originals[i, k, agent] for i in range(size)
                for agent in range(originals.shape[2]) for k in range(originals.shape[1])]
        valid = [np.ones(r.shape[0], dtype=bool) for r in rows]
        v0 = [np.zeros(2, dtype=np.float32) for _ in rows]
        timers["b_projection"].append(timed(lambda: project_trajectories(
            torch.tensor(np.stack(rows), dtype=torch.float32, device=device),
            torch.tensor(np.stack(valid), dtype=torch.bool, device=device),
            torch.tensor(np.stack(v0), dtype=torch.float32, device=device),
            iters=args.iters)) / size)
        eval_masks = np.ones((size, originals.shape[2]), dtype=bool)
        candidates = originals.shape[1]
        timers["cpu_metrics"].append(timed(lambda: [
            rollout_records_for_batch(originals, batch_raw, eval_masks),
            kinematic_attribution_metrics(
                originals[0][:, 0],
                np.repeat(batch_raw["future_valid_mask"][0][0][None], candidates, axis=0),
                np.repeat(np.zeros(2)[None], candidates, axis=0))]) / size)
    def stats(values: list[float]) -> dict:
        arr = np.asarray(values)
        return {"mean_ms": float(arr.mean() * 1000), "median_ms": float(np.median(arr) * 1000),
                "batches": int(arr.size)}
    return {"batch_size": args.batch_size, "warmup_batches": 2,
            "phases_ms_per_sample": {k: stats(v) for k, v in timers.items()}}


from scenario_lab.p33_model import to_torch_batch  # noqa: E402  (import kept last: heavy)


if __name__ == "__main__":
    main()
