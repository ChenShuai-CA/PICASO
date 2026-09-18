"""P3.3.4 Phase C evaluation: trained kinematic-head model under the Phase B
protocol (same dev split, same 6-candidate sampling, same group-level and
per-(source,type) kinematic tables, same diversity measure), so the joint gate
review (SPEC section 4) can compare orig / proj(B) / C side by side.

No projection is applied here: the whole point of C is that feasibility holds
by construction at decode time.  Any residual violation (speed is the one
quantity not bounded by construction) is reported, never hidden.

Writes PHASE_C_METRICS.json into the output directory.
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
                                         history_tail_velocity, sha256_of)
from p334_phase_b import candidate_diversity  # noqa: E402
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_metrics import (evaluate_agent_mask, group_level_table,  # noqa: E402
                                      rollout_records_for_batch)
from scenario_lab.p33_model import ARSceneV1K, to_torch_batch  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402


def load_kinematic_model(checkpoint_path: Path, manifest_path: Path, config: dict,
                         device: torch.device):
    codebook_path = Path(manifest_path).parent / "motion_codebook_v1.npz"
    with np.load(codebook_path, allow_pickle=False) as data:
        codebook = np.asarray(data["centroids"], dtype=np.float32)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("variant") != "kinematic":
        raise ValueError(f"checkpoint {checkpoint_path} is variant "
                         f"{payload.get('variant')!r}, expected 'kinematic'")
    torch.manual_seed(7)
    model = ARSceneV1K(config, codebook, payload.get("c_config")).to(device)
    model.load_state_dict({key: value.to(device) for key, value in payload["model"].items()})
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "runs/20260915_p334_kinematics/phase_c_train/arch_best_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "runs/20260915_p334_kinematics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke cap; None = full dev split")
    args = parser.parse_args()

    config = load_p33_config()
    device = torch.device(args.device)
    model, payload = load_kinematic_model(args.checkpoint, args.manifest, config, device)
    catalog = ShardCatalog.from_manifest(args.manifest, config)

    records: list = []
    accumulator = ArmAccumulator()
    diversity: list[float] = []
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
        trajectories = rollout["trajectories"].cpu().numpy().astype(np.float64)
        eval_masks = np.stack([
            evaluate_agent_mask(batch_raw["agent_present_mask"][index],
                                batch_raw["agent_role"][index],
                                batch_raw["future_valid_mask"][index])
            for index in range(size)])
        history = history_tail_velocity(batch_raw)
        records.extend(rollout_records_for_batch(trajectories, batch_raw, eval_masks))
        for index in range(size):
            mask = eval_masks[index]
            if mask.any():
                diversity.append(candidate_diversity(
                    trajectories[index][:, mask],
                    batch_raw["future_valid_mask"][index][mask]))
                for agent in np.flatnonzero(mask):
                    count = trajectories.shape[1]
                    block_valid = batch_raw["future_valid_mask"][index][agent]
                    accumulator.add(
                        batch_raw["sample_source"][index],
                        batch_raw["agent_type"][index][agent],
                        trajectories[index][:, agent],
                        np.repeat(block_valid[None], count, axis=0),
                        np.repeat(history[index][agent][None], count, axis=0))

    output = {
        "phase": "C_kinematic_head",
        "spec": "runs/20260915_p334_kinematics/SPEC.md#6",
        "checkpoint": {"path": str(args.checkpoint.relative_to(REPO)),
                       "sha256": sha256_of(args.checkpoint)},
        "arch_state": {key: payload["arch"][key] for key in
                       ("epoch", "update_index", "best", "best_epoch")
                       if isinstance(payload.get("arch"), dict) and key in payload["arch"]},
        "c_config": payload.get("c_config"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sampling": {key: SAMPLING[key] for key in ("samples", "temperature", "top_p",
                                                    "rollout_seed")},
        "samples": samples_seen,
        "elapsed_seconds": time.time() - started,
        "group_level": group_level_table(records),
        "kinematics": accumulator.table(),
        "diversity_mean_pairwise_m": float(np.mean(diversity)) if diversity else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "PHASE_C_METRICS.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(out_path),
                      "elapsed_seconds": round(output["elapsed_seconds"], 1),
                      "diversity": output["diversity_mean_pairwise_m"]}))


if __name__ == "__main__":
    main()
