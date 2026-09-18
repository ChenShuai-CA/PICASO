"""R2-C step 1 (SPEC_R2 section 5): frozen C-v2 ep29 weights through the R2
decoder -- a no-retrain paired diagnosis, NOT a trained C-v3.

Two arms, one pass, identical sampling (seed 7, batch 16, K=6, T=1, top-p
0.95, original dev order):

* ``C-v2``          -- start_recurrence_version 1, the sealed R1 path.  Its
  group-level error table, diversity and legacy validity counts must EXACTLY
  match the sealed runs/20260917_p334_r1_eval/METRICS_V2.json C-v2 arm
  (bitwise replay anchor; any mismatch aborts with a non-zero exit).
* ``C-v2-R2decoder``-- same state_dict, ``start_recurrence_version: 2``.

Metrics per EVAL_PROTOCOL_V3: three velocity references (diff_raw /
recorded_raw / recorded_clamped, MAIN = recorded_clamped), full_anchored_valid
with frozen tolerances (speed 1e-4, accel 1e-3, jerk 1e-3), per-candidate
ledger at the MAIN reference, legacy columns for continuity, frozen/anchored
step tables, group-level errors, diversity, scene criteria (a)/(b)/(c).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "research_tasks")]

from p334_review_eval import (SAMPLING, AGENT_TYPE_NAMES, AnchoredAccumulator,  # noqa: E402
                              ArmAccumulator, FRAME_RATE, SPEED_LIMIT, ACCEL_LIMIT,
                              JERK_LIMIT, V0_RECORD_LIMIT, candidate_validity,
                              history_tail_velocity, load_p33_config, index_rows)
from scenario_lab.p33_metrics import (evaluate_agent_mask, group_level_table,  # noqa: E402
                                      rollout_records_for_batch)
from p334_phase_b import candidate_diversity  # noqa: E402
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_model import ARSceneV1K, to_torch_batch  # noqa: E402
from p334_phase_c_eval import load_kinematic_model  # noqa: E402

# EVAL_PROTOCOL_V3 section 2: frozen tolerances, never widened post hoc.
SPEED_TOL, ACCEL_TOL, JERK_TOL = 1e-4, 1e-3, 1e-3
REFERENCES = ("recorded_clamped", "diff_raw", "recorded_raw")
MAIN_REFERENCE = "recorded_clamped"
ARMS = ("C-v2", "C-v2-R2decoder")
SEALED_R1_METRICS = REPO / "runs/20260917_p334_r1_eval/METRICS_V2.json"


def clamp_disk(vectors: np.ndarray, limit: float = V0_RECORD_LIMIT) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, limit / (norm + 1e-9))


# R2-E adjudication (user, 2026-09-19): corrected semantics live behind
# validity_version=2; the default MUST stay 1 so sealed replays keep holding
# (audit probe sentinel runs/20260918_p334_r2_codex_audit/probe_evaluator.py:13,
# R1 replay anchors below in this file, p334_v3_eval.py verbatim import).
FULL_VALIDITY_VERSION = 1


def full_anchored_validity(trajs: np.ndarray, step_valid: np.ndarray,
                           anchor: np.ndarray, v0: np.ndarray, *,
                           validity_version: int = FULL_VALIDITY_VERSION) -> dict:
    """Per-candidate verdicts under one velocity reference.

    ``trajs`` [K, T, 2]; ``step_valid`` [T]; ``anchor`` [2]; ``v0`` [2] (NaN
    when the reference is unusable -- the v0-dependent earliest accel/jerk
    terms are then SKIPPED, counted separately, never silently passed).

    ``validity_version`` (R2-E, EVAL_PROTOCOL_V3 section 2 semantics):
    * 1 (default) -- frozen V3 behaviour, byte-for-byte, kept for sealed
      replay.  Known gaps retained deliberately: a non-finite anchor passes
      silently (``numeric`` never inspects it) and differentials are gated
      right-end-only, so a masked frame can leak into judgments.
    * 2 -- corrected semantics: (a) a non-finite anchor is an explicit
      failure (reason ``nonfinite_anchor``); (b) double-end propagation -- a
      difference-based quantity is judged only when EVERY frame it spans is
      valid, matching p334_review_eval.candidate_validity,
      p33_metrics.kinematic_attribution_metrics, p334_phase_b.derivatives and
      the training-side ``transition_valid`` convention.
    """
    K, T, _ = trajs.shape
    numeric = np.isfinite(trajs).all(axis=(1, 2))                              # [K]
    anchor_ok = bool(np.isfinite(anchor).all())
    ext = np.concatenate([np.broadcast_to(anchor, (K, 1, 2)), trajs], axis=1)
    velocity = np.diff(ext, axis=1) * FRAME_RATE                               # [K, T]
    speed = np.linalg.norm(velocity, axis=-1)

    v0_ok = bool(np.isfinite(v0).all())
    v0_row = v0 if v0_ok else np.zeros(2)
    vext = np.concatenate([np.broadcast_to(v0_row, (K, 1, 2)), velocity], axis=1)
    accel = np.diff(vext, axis=1) * FRAME_RATE                                 # [K, T]

    if validity_version == 2:
        # Frame validity in ext coordinates (ext[0] = anchor, ext[k+1] =
        # trajs[k]).  AND-chaining consecutive span masks yields the exact
        # frame-union spans: v_ok[j] covers ext[j..j+1]; a_ok[j>=1] =
        # v_ok[j-1] & v_ok[j] covers ext[j-1..j+1]; j_valid covers the union
        # of two accel spans (they overlap, so AND == union).
        ext_valid = np.concatenate([[anchor_ok],
                                    np.asarray(step_valid, dtype=bool)])      # [T+1]
        v_ok = ext_valid[:-1] & ext_valid[1:]                                  # [T]
        v_valid = v_ok[None, :]                                                # [1, T]
        a_valid = np.concatenate([[v0_ok and bool(v_ok[0])],
                                  v_ok[:-1] & v_ok[1:]])[None, :]              # [1, T]
        verdict_numeric = numeric & anchor_ok
    elif validity_version == 1:
        v_valid = step_valid[None, :]                                          # [1, T]
        a_valid = np.concatenate([[v0_ok and step_valid[0]],
                                  step_valid[:-1] & step_valid[1:]])[None, :]  # [1, T]
        verdict_numeric = numeric
    else:
        raise ValueError(
            f"validity_version must be 1 or 2, got {validity_version!r}")

    speed_bad = v_valid & (speed > SPEED_LIMIT + SPEED_TOL)
    accel_norm = np.linalg.norm(accel, axis=-1)
    accel_bad = a_valid & (accel_norm > ACCEL_LIMIT + ACCEL_TOL)

    jerk = np.diff(accel, axis=1) * FRAME_RATE                                 # [K, T-1]
    j_valid = a_valid[:, :-1] & a_valid[:, 1:]
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    jerk_bad = j_valid & (jerk_norm > JERK_LIMIT + JERK_TOL)

    def _max(where, values):
        masked = np.where(where, values, np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmax(masked, axis=1) if masked.size else np.full(K, np.nan)
        return np.where(np.isfinite(masked).any(axis=1), out, 0.0)

    verdicts = verdict_numeric & ~(speed_bad | accel_bad[:, :T] | np.pad(
        jerk_bad, ((0, 0), (0, 1)), constant_values=False)).any(axis=1)
    reasons: list[list[str]] = []
    for k in range(K):
        why = []
        if not numeric[k]:
            why.append("nonfinite")
        if validity_version == 2 and not anchor_ok:
            why.append("nonfinite_anchor")
        if speed_bad[k].any():
            why.append("speed")
        if accel_bad[k].any():
            why.append("accel")
        if jerk_bad[k].any():
            why.append("jerk")
        reasons.append(why)

    first_step = np.full(K, -1)
    for k in range(K):
        hits = [np.flatnonzero(speed_bad[k] | accel_bad[k] | np.pad(
            jerk_bad[k], (0, 1), constant_values=False))]
        if hits[0].size:
            first_step[k] = int(hits[0][0])

    return {"valid": verdicts,
            "reasons": reasons,
            "first_violation_step": first_step,
            "v0_usable": v0_ok,
            "speed_max": _max(v_valid, speed),
            "accel_max": _max(a_valid, accel_norm),
            "jerk_max": _max(j_valid, jerk_norm),
            "speed_bad_steps": speed_bad.sum(axis=1),
            "accel_bad_steps": accel_bad.sum(axis=1),
            "jerk_bad_steps": jerk_bad.sum(axis=1)}


def _group(source: str, agent_type: int) -> str:
    return f"{source}/{AGENT_TYPE_NAMES.get(int(agent_type), agent_type)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--c2-checkpoint", type=Path,
                        default=REPO / "runs/20260916_p334_r1/c_v2_train/arch_best_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "runs/20260917_p334_r2/diagnosis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke cap; None = full dev split")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sealed = json.loads(SEALED_R1_METRICS.read_text())
    config = load_p33_config()
    device = torch.device(args.device)
    model_replay, payload = load_kinematic_model(
        args.c2_checkpoint, args.manifest, config, device)
    c_config = dict(payload["c_config"])
    assert c_config.get("start_pair_joint") and c_config.get("v0_source") == "recorded"

    codebook_path = Path(args.manifest).parent / "motion_codebook_v1.npz"
    with np.load(codebook_path, allow_pickle=False) as data:
        codebook = np.asarray(data["centroids"], dtype=np.float32)
    torch.manual_seed(7)
    model_r2 = ARSceneV1K(config, codebook,
                          {**c_config, "start_recurrence_version": 2}).to(device)
    model_r2.load_state_dict({key: value.to(device)
                              for key, value in payload["model"].items()})
    model_r2.eval()
    assert model_r2.start_recurrence_version == 2
    assert model_replay.start_recurrence_version == 1
    for (name_a, param_a), (name_b, param_b) in zip(
            model_replay.named_parameters(), model_r2.named_parameters()):
        assert name_a == name_b and torch.equal(param_a, param_b), name_a

    catalog = ShardCatalog.from_manifest(args.manifest, config)
    rows_by_id = index_rows(catalog)

    models = {"C-v2": model_replay, "C-v2-R2decoder": model_r2}
    records = {a: [] for a in ARMS}
    frozen_acc = {a: ArmAccumulator() for a in ARMS}
    anchored_acc = {a: AnchoredAccumulator() for a in ARMS}
    diversity = {a: [] for a in ARMS}
    legacy = {a: defaultdict(lambda: {"attempted": 0, "kinematic_valid": 0,
                                      "agents": 0, "agents_none": 0})
              for a in ARMS}
    legacy_rec = {a: defaultdict(lambda: {"attempted": 0, "kinematic_valid": 0,
                                          "agents": 0, "agents_none": 0})
                  for a in ARMS}
    full = {a: {ref: defaultdict(lambda: {"attempted": 0, "numeric_valid": 0,
                                          "full_anchored_valid": 0, "agents": 0,
                                          "agents_none_full": 0, "v0_unusable": 0,
                                          "reasons": defaultdict(int),
                                          "first_step_hist": defaultdict(int),
                                          "speed_bad_steps": 0, "accel_bad_steps": 0,
                                          "jerk_bad_steps": 0})
                for ref in REFERENCES} for a in ARMS}
    scenes = {a: {"scenes": 0, "a_no_valid_agent": 0, "b_target_without_candidate": 0,
                  "c_no_common_k": 0} for a in ARMS}
    sample_ids: list[str] = []
    started = time.time()
    ledger_path = args.output_dir / "candidate_validity_v3.jsonl"
    ledger = ledger_path.open("w", encoding="utf-8")
    samples_seen = 0

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
        history = history_tail_velocity(batch_raw)                       # diff_raw
        recorded_raw = batch_raw["agent_history"][:, :, -1, 2:4].astype(np.float64)
        tail_valid = batch_raw["state_valid_mask"][:, :, -1]
        recorded_raw = np.where(tail_valid[..., None], recorded_raw, np.nan)
        recorded_clamped = np.where(np.isfinite(recorded_raw).all(axis=-1, keepdims=True),
                                    clamp_disk(np.nan_to_num(recorded_raw)), np.nan)
        references = {"diff_raw": history, "recorded_raw": recorded_raw,
                      "recorded_clamped": recorded_clamped}

        rollouts = {}
        for arm, model in models.items():
            rollout = model.rollout(
                batch_torch, num_samples=SAMPLING["samples"],
                temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                seed=SAMPLING["rollout_seed"])
            rollouts[arm] = rollout
        assert torch.equal(rollouts["C-v2"]["tokens"],
                           rollouts["C-v2-R2decoder"]["tokens"]), \
            "decoder path must be untouched by the recurrence version"

        for arm in ARMS:
            trajectories = rollouts[arm]["trajectories"].cpu().numpy().astype(np.float64)
            count = trajectories.shape[1]
            records[arm].extend(
                rollout_records_for_batch(trajectories, batch_raw, eval_masks))
            for i in range(size):
                mask = eval_masks[i]
                if not mask.any():
                    continue
                scenes[arm]["scenes"] += 1
                future_valid = batch_raw["future_valid_mask"][i]
                anchor_all = batch_raw["agent_history"][i][:, -1, :2].astype(np.float64)
                diversity[arm].append(candidate_diversity(
                    trajectories[i][:, mask], future_valid[mask]))
                valid_matrix = np.ones((int(mask.sum()), count), dtype=bool)
                for row, agent in enumerate(np.flatnonzero(mask)):
                    source = batch_raw["sample_source"][i]
                    agent_type = int(batch_raw["agent_type"][i][agent])
                    group = _group(source, agent_type)
                    trajs_agent = trajectories[i][:, agent]
                    step_valid = future_valid[agent]
                    anchor = anchor_all[agent]

                    verdict_legacy = candidate_validity(
                        trajs_agent, step_valid, anchor, history[i, agent])
                    verdict_legacy_rec = candidate_validity(
                        trajs_agent, step_valid, anchor,
                        recorded_clamped[i, agent])
                    for table, verdict in ((legacy[arm], verdict_legacy),
                                           (legacy_rec[arm], verdict_legacy_rec)):
                        stats = table[group]
                        stats["attempted"] += verdict["attempted"]
                        stats["kinematic_valid"] += verdict["kinematic_valid"]
                        stats["agents"] += 1
                        stats["agents_none"] += verdict["kinematic_valid"] == 0

                    frozen_acc[arm].add(
                        source, agent_type, trajs_agent,
                        np.repeat(step_valid[None], count, axis=0),
                        np.repeat(history[i, agent][None], count, axis=0))
                    anchored_acc[arm].add(
                        source, agent_type, trajs_agent,
                        np.repeat(step_valid[None], count, axis=0),
                        np.repeat(anchor[None], count, axis=0),
                        np.repeat(history[i, agent][None], count, axis=0))

                    full_main = None
                    for ref in REFERENCES:
                        verdict = full_anchored_validity(
                            trajs_agent, step_valid, anchor, references[ref][i, agent])
                        stats = full[arm][ref][group]
                        stats["attempted"] += verdict["valid"].shape[0]
                        stats["numeric_valid"] += int(
                            np.isfinite(trajs_agent).all(axis=(1, 2)).sum())
                        stats["full_anchored_valid"] += int(verdict["valid"].sum())
                        stats["agents"] += 1
                        stats["agents_none_full"] += not verdict["valid"].any()
                        stats["v0_unusable"] += int(not verdict["v0_usable"])
                        for why in verdict["reasons"]:
                            for reason in why:
                                stats["reasons"][reason] += 1
                        for step in verdict["first_violation_step"]:
                            if step >= 0:
                                stats["first_step_hist"][int(step)] += 1
                        stats["speed_bad_steps"] += int(verdict["speed_bad_steps"].sum())
                        stats["accel_bad_steps"] += int(verdict["accel_bad_steps"].sum())
                        stats["jerk_bad_steps"] += int(verdict["jerk_bad_steps"].sum())
                        if ref == MAIN_REFERENCE:
                            full_main = verdict
                    assert full_main is not None

                    valid_matrix[row] = full_main["valid"]
                    if args.max_batches is None:   # ledger only on the full run
                        row_meta = rows_by_id.get(batch_raw["sample_id"][i], {})
                        for k in range(count):
                            ledger.write(json.dumps({
                                "sample_id": batch_raw["sample_id"][i],
                                "group_id": row_meta.get("group_id"),
                                "source": source,
                                "agent_slot": int(agent),
                                "agent_type": agent_type,
                                "branch": "not_available",
                                "candidate_id": k,
                                "arm": arm,
                                "velocity_reference": MAIN_REFERENCE,
                                "numeric_valid": bool(
                                    np.isfinite(trajs_agent[k]).all()),
                                "legacy_valid": bool(candidate_validity(
                                    trajs_agent[k:k + 1], step_valid, anchor,
                                    history[i, agent])["kinematic_valid"] == 1),
                                "full_anchored_valid": bool(full_main["valid"][k]),
                                "history_usable": bool(full_main["v0_usable"]),
                                "v0_over_limit": bool(np.linalg.norm(
                                    recorded_raw[i, agent]) > V0_RECORD_LIMIT)
                                if np.isfinite(recorded_raw[i, agent]).all() else False,
                                "failure_reasons": full_main["reasons"][k],
                                "first_violation_step": int(
                                    full_main["first_violation_step"][k]),
                                "speed_max_mps": float(full_main["speed_max"][k]),
                                "accel_max_mps2": float(full_main["accel_max"][k]),
                                "jerk_max_mps3": float(full_main["jerk_max"][k]),
                                "speed_bad_steps": int(full_main["speed_bad_steps"][k]),
                                "accel_bad_steps": int(full_main["accel_bad_steps"][k]),
                                "jerk_bad_steps": int(full_main["jerk_bad_steps"][k]),
                            }, ensure_ascii=False) + "\n")

                scenes[arm]["a_no_valid_agent"] += int(not valid_matrix.any())
                scenes[arm]["b_target_without_candidate"] += int(
                    (~valid_matrix.any(axis=1)).any())
                scenes[arm]["c_no_common_k"] += int(not valid_matrix.all(axis=0).any())
        if batch_number % 50 == 0:
            print(f"[{time.time() - started:7.1f}s] batch {batch_number}, "
                  f"{samples_seen} samples", flush=True)
    ledger.close()

    id_digest = hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest()
    if args.max_batches is None and samples_seen != 11586:
        print(f"ERROR: expected 11586 dev samples, saw {samples_seen}",
              file=sys.stderr)
        sys.exit(1)

    # ---- replay anchor: the version-1 arm must reproduce the sealed R1
    # readings (full runs only; a --max-batches smoke cannot match them).
    if args.max_batches is None:
        sealed_arm = sealed["arms"]["C-v2"]
        mismatches = []
        if group_level_table(records["C-v2"]) != sealed_arm["group_level"]:
            mismatches.append("group_level")
        if abs(float(np.mean(diversity["C-v2"]))
               - sealed_arm["diversity_v2_mean"]) > 1e-12:
            mismatches.append("diversity_v2_mean")
        diff_valid = sum(v["kinematic_valid"] for v in legacy["C-v2"].values())
        rec_valid = sum(v["kinematic_valid"] for v in legacy_rec["C-v2"].values())
        if diff_valid != 600756:
            mismatches.append(f"legacy diff valid {diff_valid} != 600756")
        if rec_valid != 600792:
            mismatches.append(f"legacy recorded valid {rec_valid} != 600792")
        if mismatches:
            print(f"ERROR: replay anchor broken: {mismatches}", file=sys.stderr)
            sys.exit(1)

    def _total(table):
        attempted = sum(v["attempted"] for v in table.values())
        valid = sum(v["full_anchored_valid"] for v in table.values())
        agents = sum(v["agents"] for v in table.values())
        none_full = sum(v["agents_none_full"] for v in table.values())
        return {"attempted": attempted, "full_anchored_valid": valid,
                "rate": valid / attempted if attempted else None,
                "agents": agents, "agents_none_full": none_full}

    output = {
        "phase": "R2_C_step1_no_retrain_diagnosis",
        "protocol": "runs/20260917_p334_r2/EVAL_PROTOCOL_V3.md",
        "spec": "runs/20260917_p334_r2/SPEC_R2.md",
        "checkpoint": str(args.c2_checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "arms_definition": {
            "C-v2": "sealed R1 decoder path (start_recurrence_version 1)",
            "C-v2-R2decoder": "same weights, start_recurrence_version 2 "
                              "(no retraining; NOT a trained C-v3)"},
        "main_reference": MAIN_REFERENCE,
        "tolerances": {"speed": SPEED_TOL, "accel": ACCEL_TOL, "jerk": JERK_TOL},
        "sampling": {k: SAMPLING[k] for k in ("samples", "temperature", "top_p",
                                              "rollout_seed")},
        "samples": samples_seen,
        "sample_id_sorted_sha256": id_digest,
        "elapsed_seconds": time.time() - started,
        "replay_anchor": "C-v2 arm group_level/diversity/legacy validity "
                         "exactly match sealed runs/20260917_p334_r1_eval/METRICS_V2.json",
        "arms": {},
    }
    for arm in ARMS:
        output["arms"][arm] = {
            "group_level": group_level_table(records[arm]),
            "kinematics_frozen": frozen_acc[arm].table(),
            "kinematics_anchored": anchored_acc[arm].table(),
            "diversity_v2_mean": float(np.mean(diversity[arm])),
            "legacy_validity": {g: v for g, v in sorted(legacy[arm].items())},
            "legacy_validity_recorded_v0": {g: v for g, v in sorted(legacy_rec[arm].items())},
            "full_anchored": {
                ref: {"totals": _total(full[arm][ref]),
                      "groups": {g: dict(stats) | {
                          "reasons": dict(stats["reasons"]),
                          "first_step_hist": dict(stats["first_step_hist"])}
                          for g, stats in sorted(full[arm][ref].items())}}
                for ref in REFERENCES},
            "scenes": scenes[arm] | {
                "scenes_no_valid_agent": scenes[arm]["a_no_valid_agent"]},
        }
    metrics_path = args.output_dir / "METRICS_V3.json"
    metrics_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(json.dumps({"written": str(metrics_path), "ledger": str(ledger_path),
                      "samples": samples_seen,
                      "elapsed_seconds": round(output["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
