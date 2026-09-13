"""P3.3.2 evaluation: B0 constant-velocity baseline and M3 model diagnostics on smoke-dev.

B0 (no checkpoint needed):
  python research_tasks/evaluate_p33_t1.py --mode cv
M3 (after train_p33_nominal.py --mode smoke):
  python research_tasks/evaluate_p33_t1.py --mode model

All evaluation is group-aggregated (Waymo scenario_id / INTERACTION location::case);
sample-level pooling is reported separately and never substituted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_metrics import (  # noqa: E402
    cv_records_for_batch,
    evaluate_agent_mask,
    group_level_table,
    kinematic_diagnostics,
    rollout_records_for_batch,
    summarize_records,
    token_frequency_report,
)
from scenario_lab.p33_model import ARSceneV1, ar_scene_loss, to_torch_batch  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

MANIFEST = REPO / "runs/20260913_p331_data_pipeline/smoke/DATASET_MANIFEST.json"
OUTPUT_DIR = REPO / "runs/20260913_p332_model_smoke"


def _groups_csv(records: list) -> str:
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for record in records:
        grouped[(record.source, record.group_id)].append(record)
    lines = ["source,group_id,agent_records,ade,fde,min_ade,min_fde,endpoint_missing"]
    for (source, group_id), rows in sorted(grouped.items()):
        endpoints = [row for row in rows if row.endpoint_valid]
        def mean(values):
            return float(np.mean(values)) if len(values) else float("nan")
        lines.append(",".join(map(str, [
            source, group_id, len(rows),
            f"{mean([r.ade for r in rows]):.6f}",
            f"{mean([r.fde for r in endpoints]):.6f}" if endpoints else "nan",
            f"{mean([r.min_ade for r in rows]):.6f}",
            f"{mean([r.min_fde for r in endpoints]):.6f}" if endpoints else "nan",
            sum(1 for r in rows if not r.endpoint_valid)])))
    return "\n".join(lines) + "\n"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, (np.integer, int, bool, str)):
        return value
    return str(value)


def run_cv(args: argparse.Namespace) -> dict:
    """B0: constant-velocity baseline over every dev sample."""
    config = load_p33_config()
    catalog = ShardCatalog.from_manifest(MANIFEST, config)
    records = []
    batches = 0
    started = time.time()
    expected = sum(len(entry.rows_by_split["dev"]) for entry in catalog.entries(split="dev"))
    for batch in iter_split_batches(catalog, split="dev", batch_size=args.batch_size):
        records.extend(cv_records_for_batch(batch))
        batches += 1
        if args.limit and batches * args.batch_size >= args.limit:
            break
    if args.limit is None and len({(r.source, r.group_id, r.sample_id) for r in records}) != expected:
        raise RuntimeError(f"dev iteration covered {batches} batches, expected {expected} samples")
    table = group_level_table(records)
    summary = summarize_records(records)
    cv_minade_equals_ade = all(
        abs(source_table["min_ade"] - source_table["ade"]) < 1e-12
        for source_table in table.values())
    result = {
        "mode": "b0_cv_baseline",
        "split": "dev",
        "samples_covered": len({(r.source, r.group_id, r.sample_id) for r in records}),
        "batches": batches,
        "group_level": table,
        "strata": summary["strata"],
        "counts": summary["counts"],
        "determinism_check_minade_equals_ade": cv_minade_equals_ade,
        "elapsed_seconds": time.time() - started,
    }
    if args.limit is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "CV_DEV_METRICS.json").write_text(
            json.dumps(_json_safe(result), indent=2))
        (OUTPUT_DIR / "CV_DEV_GROUPS.csv").write_text(_groups_csv(records), encoding="utf-8")
    return result


def load_model(args: argparse.Namespace, config: dict, device: torch.device):
    codebook_path = MANIFEST.parent / "motion_codebook_v1.npz"
    with np.load(codebook_path, allow_pickle=False) as data:
        codebook = np.asarray(data["centroids"], dtype=np.float32)
    torch.manual_seed(7)
    model = ARSceneV1(config, codebook).to(device)
    payload = torch.load(OUTPUT_DIR / "m2_final_checkpoint.pt", map_location=device,
                         weights_only=False)
    model.load_state_dict({key: value.to(device) for key, value in payload["model"].items()})
    model.eval()
    return model, payload["update_index"]


def _overlay_plot(path: Path, batch: dict, trajectories: np.ndarray, cv: np.ndarray,
                  sample_index: int) -> None:
    """One PNG: CV vs 6-sample model rollout against map/history/truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = batch["agent_history"][sample_index]
    valid = batch["state_valid_mask"][sample_index]
    future = batch["future_xy"][sample_index]
    future_valid = batch["future_valid_mask"][sample_index]
    mask = evaluate_agent_mask(batch["agent_present_mask"][sample_index],
                               batch["agent_role"][sample_index], future_valid)
    polylines = batch["map_polylines"][sample_index]
    point_mask = batch["map_point_mask"][sample_index]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, title in ((axes[0], "CV baseline"), (axes[1], "model rollouts (6)")):
        for line in range(polylines.shape[0]):
            points = polylines[line][point_mask[line]]
            if len(points):
                axis.plot(points[:, 0], points[:, 1], color="0.85", lw=0.7, zorder=0)
        for slot in range(history.shape[0]):
            if not valid[slot].any():
                continue
            observed = history[slot][valid[slot]]
            axis.plot(observed[:, 0], observed[:, 1], color="0.4", lw=1, zorder=1)
            if future_valid[slot].any():
                truth = future[slot][future_valid[slot]]
                axis.plot(truth[:, 0], truth[:, 1], color="tab:green", lw=2, zorder=3)
        if title.startswith("CV"):
            for slot in np.flatnonzero(mask):
                axis.plot(cv[sample_index, slot, :, 0], cv[sample_index, slot, :, 1],
                          "tab:blue", lw=1.5, ls="--", zorder=2)
        else:
            for sample in range(trajectories.shape[1]):
                for slot in np.flatnonzero(mask):
                    axis.plot(trajectories[sample_index, sample, slot, :, 0],
                              trajectories[sample_index, sample, slot, :, 1],
                              color="tab:blue", lw=0.8, alpha=0.6, zorder=2)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle(f"{batch['sample_source'][sample_index]} {batch['sample_id'][sample_index]}")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run_model(args: argparse.Namespace) -> dict:
    """M3: teacher-forced NLL + 6-sample rollout diagnostics on every dev sample."""
    from scenario_lab.p33_metrics import constant_velocity_prediction
    config = load_p33_config()
    smoke = json.loads((REPO / "configs/p33/model_smoke_v1.json").read_text())
    m3 = smoke["m3_eval"]
    catalog = ShardCatalog.from_manifest(MANIFEST, config)
    device = torch.device(args.device)
    model, trained_updates = load_model(args, config, device)
    records = []
    token_true, token_valid, token_argmax, token_sampled, token_logprob = [], [], [], [], []
    ce_weighted, accuracy_weighted, ce_tokens = 0.0, 0.0, 0
    kinematics = {"speed_violations": 0, "accel_violations": 0, "jerk_violations": 0,
                  "agent_trajectories": 0}
    batches = 0
    plots_written = 0
    started = time.time()
    full_run = args.limit is None
    if full_run:
        plots_dir = OUTPUT_DIR / "overlay_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
    for batch_raw in iter_split_batches(catalog, split=m3["split"], batch_size=args.batch_size):
        batch = to_torch_batch(batch_raw, device)
        with torch.no_grad():
            outputs = model(batch, teacher_tokens=batch["motion_token_target"])
            losses = ar_scene_loss(outputs, batch, model.codebook, config)
            valid = (batch["motion_token_valid_mask"]
                     & (batch["motion_token_target"] != 255))
            ce_weighted += float(losses["token_cross_entropy"]) * int(valid.sum())
            accuracy_weighted += float(losses["token_accuracy"]) * int(valid.sum())
            ce_tokens += int(valid.sum())
            rollout = model.rollout(batch, num_samples=m3["samples"],
                                    temperature=m3["temperature"], top_p=m3["top_p"],
                                    seed=m3["rollout_seed"])
        token_true.append(batch["motion_token_target"].cpu().numpy().reshape(-1))
        token_valid.append(valid.cpu().numpy().reshape(-1))
        token_argmax.append(outputs["motion_token_logits"].argmax(-1).cpu().numpy().reshape(-1))
        token_sampled.append(rollout["tokens"][:, 0].cpu().numpy().reshape(-1))
        token_logprob.append(rollout["token_log_prob"][:, 0].cpu().numpy().reshape(-1))
        trajectories = rollout["trajectories"].cpu().numpy().astype(np.float64)
        records.extend(rollout_records_for_batch(trajectories, batch_raw))
        # kinematic diagnostics over rollout trajectories of main-eval agents
        for index in range(trajectories.shape[0]):
            mask = evaluate_agent_mask(batch_raw["agent_present_mask"][index],
                                       batch_raw["agent_role"][index],
                                       batch_raw["future_valid_mask"][index])
            if mask.any():
                report = kinematic_diagnostics(trajectories[index, :, mask])
                for key in ("speed_violations", "accel_violations", "jerk_violations"):
                    kinematics[key] += report[key]
                kinematics["agent_trajectories"] += report["agent_count"]
        if full_run and plots_written < m3["overlay_plot_count"]:
            cv = np.stack([constant_velocity_prediction(batch_raw["agent_history"][i],
                                                        batch_raw["state_valid_mask"][i])
                           for i in range(len(batch_raw["sample_id"]))])
            per_batch = 2
            for index in range(min(per_batch, trajectories.shape[0])):
                if plots_written >= m3["overlay_plot_count"]:
                    break
                _overlay_plot(plots_dir / f"overlay_{plots_written:02d}.png", batch_raw,
                              trajectories, cv, index)
                plots_written += 1
        batches += 1
        if args.limit and batches * args.batch_size >= args.limit:
            break
    truth = np.concatenate(token_true)
    validity = np.concatenate(token_valid)
    argmax = np.concatenate(token_argmax)
    sampled = np.concatenate(token_sampled)
    logprob = np.concatenate(token_logprob)
    nll_per_token = ce_weighted / max(1, ce_tokens)
    sampled_nll = -float(logprob[validity].mean())
    table = group_level_table(records)
    summary = summarize_records(records)
    kinematic_violation_rates = {
        key: (kinematics[key] / kinematics["agent_trajectories"] if
              kinematics["agent_trajectories"] else None)
        for key in ("speed_violations", "accel_violations", "jerk_violations")}
    result = {
        "mode": "m3_model_diagnostics",
        "checkpoint": "m2_final_checkpoint.pt",
        "trained_updates": int(trained_updates),
        "split": m3["split"],
        "batches": batches,
        "group_level": table,
        "strata": summary["strata"],
        "counts": summary["counts"],
        "token_nll": nll_per_token,
        "token_accuracy": accuracy_weighted / max(1, ce_tokens),
        "sampled_token_nll": sampled_nll,
        "token_frequency": token_frequency_report(truth, validity, argmax),
        "sampled_token_frequency": token_frequency_report(truth, validity, sampled),
        "kinematic_violation_rates": kinematic_violation_rates,
        "kinematic_trajectories": kinematics["agent_trajectories"],
        "overlay_plots": plots_written,
        "elapsed_seconds": time.time() - started,
    }
    if full_run:
        (OUTPUT_DIR / "MODEL_DEV_METRICS.json").write_text(json.dumps(_json_safe(result),
                                                                     indent=2))
        (OUTPUT_DIR / "MODEL_DEV_GROUPS.csv").write_text(_groups_csv(records), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cv", "model"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N samples (smoke subset)")
    parser.add_argument("--plots", action="store_true", help="write overlay plots (model mode)")
    args = parser.parse_args()
    result = run_cv(args) if args.mode == "cv" else run_model(args)
    summary = {"mode": result.get("mode"), "batches": result.get("batches"),
               "samples_covered": result.get("samples_covered")}
    for source, row in result.get("group_level", {}).items():
        summary[f"group_{source}"] = {key: row[key] for key in
                                      ("groups", "agent_records", "ade", "min_ade",
                                       "fde", "min_fde")}
    for key in ("token_nll", "token_accuracy", "sampled_token_nll"):
        if key in result:
            summary[key] = result[key]
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
