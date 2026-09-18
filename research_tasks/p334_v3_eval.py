"""SPEC_R2 section 7 acceptance eval: trained C-v3 through EVAL_PROTOCOL_V3.

Single arm ``C-v3`` -- the ep29 checkpoint from
runs/20260917_p334_r2/c_v3_train (trained from scratch with
start_recurrence_version 2).  Same sampling as every sealed eval
(seed 7, batch 16, K=6, top-p 0.95, original dev order); metric
machinery is imported verbatim from p334_r2_diagnosis (frozen V3
tolerances / three velocity references / full_anchored_validity), so
this file adds no new judgment logic -- only the arm under test and a
rollout wall-clock column for the protocol section 6 latency report.

Identity anchor: the sorted sample-id digest must equal the R2
diagnosis run's (same 11,586 dev samples in the same order).
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
                              ArmAccumulator, candidate_validity,
                              history_tail_velocity, load_p33_config, index_rows)
from scenario_lab.p33_metrics import (evaluate_agent_mask, group_level_table,  # noqa: E402
                                      rollout_records_for_batch)
from p334_phase_b import candidate_diversity  # noqa: E402
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_model import to_torch_batch  # noqa: E402
from p334_phase_c_eval import load_kinematic_model  # noqa: E402
from p334_r2_diagnosis import (REFERENCES, MAIN_REFERENCE, SPEED_TOL, ACCEL_TOL,  # noqa: E402
                               JERK_TOL, V0_RECORD_LIMIT, clamp_disk,
                               full_anchored_validity, _group)

SEALED_DIAGNOSIS = REPO / "runs/20260917_p334_r2/diagnosis/METRICS_V3.json"
EXPECTED_SAMPLES = 11586
ARM = "C-v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "runs/20260917_p334_r2/c_v3_train/arch_best_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "runs/20260917_p334_r2/c_v3_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke cap; None = full dev split")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_p33_config()
    device = torch.device(args.device)
    model, payload = load_kinematic_model(
        args.checkpoint, args.manifest, config, device)
    c_config = dict(payload["c_config"])
    assert int(c_config.get("start_recurrence_version", 1)) == 2, \
        "C-v3 checkpoint must carry start_recurrence_version 2"
    assert model.start_recurrence_version == 2
    assert c_config.get("start_pair_joint") and c_config.get("v0_source") == "recorded"

    catalog = ShardCatalog.from_manifest(args.manifest, config)
    rows_by_id = index_rows(catalog)

    records: list = []
    frozen_acc = ArmAccumulator()
    anchored_acc = AnchoredAccumulator()
    diversity: list = []
    legacy = defaultdict(lambda: {"attempted": 0, "kinematic_valid": 0,
                                  "agents": 0, "agents_none": 0})
    legacy_rec = defaultdict(lambda: {"attempted": 0, "kinematic_valid": 0,
                                      "agents": 0, "agents_none": 0})
    full = {ref: defaultdict(lambda: {"attempted": 0, "numeric_valid": 0,
                                      "full_anchored_valid": 0, "agents": 0,
                                      "agents_none_full": 0, "v0_unusable": 0,
                                      "reasons": defaultdict(int),
                                      "first_step_hist": defaultdict(int),
                                      "speed_bad_steps": 0, "accel_bad_steps": 0,
                                      "jerk_bad_steps": 0})
            for ref in REFERENCES}
    scenes = {"scenes": 0, "a_no_valid_agent": 0, "b_target_without_candidate": 0,
              "c_no_common_k": 0}
    v0_over_limit = 0
    sample_ids: list[str] = []
    started = time.time()
    rollout_seconds = 0.0
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

        t_roll = time.perf_counter()
        rollout = model.rollout(
            batch_torch, num_samples=SAMPLING["samples"],
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            seed=SAMPLING["rollout_seed"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        rollout_seconds += time.perf_counter() - t_roll

        trajectories = rollout["trajectories"].cpu().numpy().astype(np.float64)
        count = trajectories.shape[1]
        records.extend(
            rollout_records_for_batch(trajectories, batch_raw, eval_masks))
        for i in range(size):
            mask = eval_masks[i]
            if not mask.any():
                continue
            scenes["scenes"] += 1
            future_valid = batch_raw["future_valid_mask"][i]
            anchor_all = batch_raw["agent_history"][i][:, -1, :2].astype(np.float64)
            diversity.append(candidate_diversity(
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
                for table, verdict in ((legacy, verdict_legacy),
                                       (legacy_rec, verdict_legacy_rec)):
                    stats = table[group]
                    stats["attempted"] += verdict["attempted"]
                    stats["kinematic_valid"] += verdict["kinematic_valid"]
                    stats["agents"] += 1
                    stats["agents_none"] += verdict["kinematic_valid"] == 0

                frozen_acc.add(
                    source, agent_type, trajs_agent,
                    np.repeat(step_valid[None], count, axis=0),
                    np.repeat(history[i, agent][None], count, axis=0))
                anchored_acc.add(
                    source, agent_type, trajs_agent,
                    np.repeat(step_valid[None], count, axis=0),
                    np.repeat(anchor[None], count, axis=0),
                    np.repeat(history[i, agent][None], count, axis=0))

                full_main = None
                for ref in REFERENCES:
                    verdict = full_anchored_validity(
                        trajs_agent, step_valid, anchor, references[ref][i, agent])
                    stats = full[ref][group]
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
                if np.isfinite(recorded_raw[i, agent]).all() and np.linalg.norm(
                        recorded_raw[i, agent]) > V0_RECORD_LIMIT:
                    v0_over_limit += 1
                if args.max_batches is None:   # ledger only on the full run
                    row_meta = rows_by_id.get(batch_raw["sample_id"][i], {})
                    for k in range(count):
                        ledger.write(json.dumps({
                            "sample_id": batch_raw["sample_id"][i],
                            "group_id": row_meta.get("group_id"),
                            "source": source,
                            "agent_slot": int(agent),
                            "agent_type": agent_type,
                            "candidate_id": k,
                            "arm": ARM,
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

            scenes["a_no_valid_agent"] += int(not valid_matrix.any())
            scenes["b_target_without_candidate"] += int(
                (~valid_matrix.any(axis=1)).any())
            scenes["c_no_common_k"] += int(not valid_matrix.all(axis=0).any())
        if batch_number % 50 == 0:
            print(f"[{time.time() - started:7.1f}s] batch {batch_number}, "
                  f"{samples_seen} samples", flush=True)
    ledger.close()

    id_digest = hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest()
    if args.max_batches is None:
        if samples_seen != EXPECTED_SAMPLES:
            print(f"ERROR: expected {EXPECTED_SAMPLES} dev samples, saw {samples_seen}",
                  file=sys.stderr)
            sys.exit(1)
        sealed_digest = json.loads(SEALED_DIAGNOSIS.read_text())["sample_id_sorted_sha256"]
        if id_digest != sealed_digest:
            print(f"ERROR: sample identity anchor broken: {id_digest[:16]} != "
                  f"{sealed_digest[:16]}", file=sys.stderr)
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
        "phase": "R2_Cv3_acceptance_eval",
        "protocol": "runs/20260917_p334_r2/EVAL_PROTOCOL_V3.md",
        "spec": "runs/20260917_p334_r2/SPEC_R2.md",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "arm": ARM,
        "main_reference": MAIN_REFERENCE,
        "tolerances": {"speed": SPEED_TOL, "accel": ACCEL_TOL, "jerk": JERK_TOL},
        "sampling": {k: SAMPLING[k] for k in ("samples", "temperature", "top_p",
                                              "rollout_seed")},
        "samples": samples_seen,
        "sample_id_sorted_sha256": id_digest,
        "elapsed_seconds": time.time() - started,
        "latency": {
            "rollout_seconds_total": rollout_seconds,
            "rollout_ms_per_sample": 1000.0 * rollout_seconds / samples_seen,
            "note": "wall clock of model.rollout (K=6 sampling) incl. cuda sync; "
                    "single arm, same pass as all metrics",
        },
        "v0_over_limit_agents": v0_over_limit,
        "group_level": group_level_table(records),
        "kinematics_frozen": frozen_acc.table(),
        "kinematics_anchored": anchored_acc.table(),
        "diversity_v2_mean": float(np.mean(diversity)),
        "legacy_validity": {g: v for g, v in sorted(legacy.items())},
        "legacy_validity_recorded_v0": {g: v for g, v in sorted(legacy_rec.items())},
        "full_anchored": {
            ref: {"totals": _total(full[ref]),
                  "groups": {g: dict(stats) | {
                      "reasons": dict(stats["reasons"]),
                      "first_step_hist": dict(stats["first_step_hist"])}
                      for g, stats in sorted(full[ref].items())}}
            for ref in REFERENCES},
        "scenes": scenes | {"scenes_no_valid_agent": scenes["a_no_valid_agent"]},
    }
    metrics_path = args.output_dir / "METRICS_V3.json"
    metrics_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(json.dumps({"written": str(metrics_path), "ledger": str(ledger_path),
                      "samples": samples_seen,
                      "elapsed_seconds": round(output["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
