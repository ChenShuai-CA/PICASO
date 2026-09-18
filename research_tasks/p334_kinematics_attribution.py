"""P3.3.4 Phase A: kinematic violation attribution ablation (no retraining).

Five arms over the SAME dev split / ep29 checkpoint / rollout sampling as the
P3.3.3 final evaluation (see runs/20260915_p334_kinematics/SPEC.md §1):

  gt              GT future_xy, same difference calculus          (data+limit floor)
  gt_token        GT tokens decoded with ZERO residuals           (codebook reconstruction)
  pred_token      rollout-sampled tokens with ZERO residuals      (+ prediction error)
  pred_full       rollout tokens + model residuals (= final eval) (+ residual effect)
  teacher_forced  teacher-forced argmax tokens + residuals        (autoregression contrast)

Each arm writes attribution_<arm>.json with per-trajectory AND per-timestep
violation rates, exceedance quantiles, prediction-start continuity, broken down
by source and by agent type.  Run ``gt`` FIRST and freeze SPEC §4 before reading
any other arm's output (pre-registration procedure).
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
sys.path.insert(0, str(REPO))

from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_metrics import evaluate_agent_mask, kinematic_attribution_metrics  # noqa: E402
from scenario_lab.p33_model import (ARSceneV1, decode_tokens_to_trajectory,  # noqa: E402
                                    to_torch_batch)
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

AGENT_TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}
SAMPLING = json.loads((REPO / "configs/p33/model_smoke_v1.json").read_text())["m3_eval"]


def load_model(manifest: Path, checkpoint: Path, config: dict, device: torch.device):
    with np.load(manifest.parent / "motion_codebook_v1.npz", allow_pickle=False) as data:
        codebook = np.asarray(data["centroids"], dtype=np.float32)
    torch.manual_seed(7)
    model = ARSceneV1(config, codebook).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict({key: value.to(device) for key, value in payload["model"].items()})
    model.eval()
    return model


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_forced(model: ARSceneV1, batch_torch: dict) -> dict:
    """Teacher-forced forward pass; argmax path needs no gradients."""
    with torch.no_grad():
        return model(batch_torch, teacher_tokens=batch_torch["motion_token_target"])


def history_tail_velocity(batch: dict) -> np.ndarray:
    """Last history-step velocity [B, A, 2]; NaN where the last two states are invalid."""
    history = batch["agent_history"][:, :, -2:, :2].astype(np.float64)
    state_valid = batch["state_valid_mask"][:, :, -2:]
    velocity = (history[:, :, 1] - history[:, :, 0]) * 10.0
    velocity[~(state_valid[:, :, 0] & state_valid[:, :, 1])] = np.nan
    return velocity


class ArmAccumulator:
    """Streams per-agent trajectory blocks into per-(source, type) metric tables.

    ``history_velocity`` rows may contain NaN (invalid history tail); the
    metrics function masks them out of the continuity block only.
    """

    def __init__(self) -> None:
        self.groups: dict[tuple, list] = defaultdict(list)

    def add(self, source: str, agent_type: int, trajectories: np.ndarray,
            step_valid: np.ndarray, history_velocity: np.ndarray) -> None:
        self.groups[(source, AGENT_TYPE_NAMES.get(int(agent_type), "other"))].append(
            (trajectories, step_valid, history_velocity))

    def table(self) -> dict:
        out: dict = {}
        for (source, type_name), chunks in sorted(self.groups.items()):
            out[f"{source}/{type_name}"] = kinematic_attribution_metrics(
                np.concatenate([c[0] for c in chunks]),
                np.concatenate([c[1] for c in chunks]),
                np.concatenate([c[2] for c in chunks]))
        return out


def _per_agent_blocks(trajectories: np.ndarray, step_valid: np.ndarray) -> list:
    """[A, T, 2] + [A, T] -> list of per-agent ([1, T, 2], [1, T]) blocks."""
    return [(trajectories[n:n + 1], step_valid[n:n + 1]) for n in range(trajectories.shape[0])]


def run_arm(arm: str, args: argparse.Namespace, model, config: dict) -> dict:
    catalog = ShardCatalog.from_manifest(args.manifest, config)
    device = torch.device(args.device)
    accumulator = ArmAccumulator()
    codebook = (model.codebook.cpu().numpy().astype(np.float64) if model is not None else None)
    started = time.time()
    batches = 0
    for batch_raw in iter_split_batches(catalog, split=SAMPLING["split"],
                                        batch_size=args.batch_size):
        batches += 1
        batch_torch = to_torch_batch(batch_raw, device) if model is not None else None
        # per-batch heavy compute hoisted out of the sample loop
        rollout = (None if arm not in ("pred_token", "pred_full") else
                   model.rollout(batch_torch, num_samples=SAMPLING["samples"],
                                 temperature=SAMPLING["temperature"],
                                 top_p=SAMPLING["top_p"], seed=SAMPLING["rollout_seed"]))
        outputs = (None if arm != "teacher_forced" else
                   _teacher_forced(model, batch_torch))
        history_all = history_tail_velocity(batch_raw)

        for index in range(len(batch_raw["sample_id"])):
            mask = evaluate_agent_mask(batch_raw["agent_present_mask"][index],
                                       batch_raw["agent_role"][index],
                                       batch_raw["future_valid_mask"][index])
            if not mask.any():
                continue
            source = batch_raw["sample_source"][index]
            types = batch_raw["agent_type"][index][mask]
            step_valid = batch_raw["future_valid_mask"][index][mask]
            current = batch_raw["agent_history"][index][mask][:, -1, :2].astype(np.float64)

            if arm == "gt":
                per_agent = _per_agent_blocks(
                    batch_raw["future_xy"][index][mask].astype(np.float64), step_valid)
            elif arm == "gt_token":
                tokens = batch_raw["motion_token_target"][index][mask]
                decoded = decode_tokens_to_trajectory(
                    tokens, np.zeros(tokens.shape + (10,)), codebook, current)
                per_agent = _per_agent_blocks(decoded, step_valid)
            elif arm in ("pred_token", "pred_full"):
                tokens = rollout["tokens"][index][:, mask].cpu().numpy()      # [K, A, C]
                if arm == "pred_full":
                    residuals = rollout["residuals"][index][:, mask].cpu().numpy()
                else:
                    residuals = np.zeros(tokens.shape + (10,))
                # tokens/residuals are [K, A, C(,10)]; stack per-agent [K, T, 2]
                # blocks into [A, K, T, 2]: candidates act as independent trajectories
                decoded = np.stack([decode_tokens_to_trajectory(
                    tokens[:, agent, :], residuals[:, agent, :], codebook, current[agent])
                    for agent in range(tokens.shape[1])])
                per_agent = [(decoded[n], np.repeat(step_valid[n][None], decoded.shape[1], axis=0))
                             for n in range(decoded.shape[0])]
            else:  # teacher_forced
                tokens = outputs["motion_token_logits"][index][mask].argmax(-1).cpu().numpy()
                residuals = outputs["delta_xy_residual"][index][mask].cpu().numpy()
                decoded = np.stack([decode_tokens_to_trajectory(
                    tokens[n], residuals[n], codebook, current[n])
                    for n in range(tokens.shape[0])])
                per_agent = _per_agent_blocks(decoded, step_valid)

            for n, agent_type in enumerate(types):
                trajectories, valid_steps = per_agent[n]
                count = trajectories.shape[0]
                accumulator.add(source, agent_type, trajectories, valid_steps,
                                np.repeat(history_all[index][mask][n][None], count, axis=0))
    return {"batches": batches, "elapsed_seconds": time.time() - started,
            "table": accumulator.table()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        choices=["gt", "gt_token", "pred_token", "pred_full", "teacher_forced"])
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/arch_best_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "runs/20260915_p334_kinematics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = load_p33_config()
    model = (load_model(args.manifest, args.checkpoint, config, torch.device(args.device))
             if args.arm != "gt" else None)
    result = run_arm(args.arm, args, model, config)
    output = {
        "arm": args.arm,
        "spec": "runs/20260915_p334_kinematics/SPEC.md#1",
        "checkpoint": ({"path": str(args.checkpoint.relative_to(REPO)),
                        "sha256": sha256_of(args.checkpoint)}
                       if model is not None else None),
        "sampling": ({key: SAMPLING[key] for key in ("samples", "temperature", "top_p",
                                                     "rollout_seed")}
                     if args.arm in ("pred_token", "pred_full") else None),
        "batch_size": args.batch_size,
        **result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"attribution_{args.arm}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"arm": args.arm, "written": str(out_path),
                      "elapsed_seconds": round(result["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
