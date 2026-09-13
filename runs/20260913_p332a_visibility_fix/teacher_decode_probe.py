"""Teacher-forced decode probe (P3.3.2a rerun): separates decode-path error
from token drift.

M3 rollout error mixes two causes: (a) autoregressive token drift (each of the
10 chunks conditions on the model's own previous chunks) and (b) the decode
path itself (centroid codebook + residual head + cumsum). This probe decodes
the GROUND-TRUTH token sequence with the model's predicted residuals, so the
token distribution is perfect and any remaining trajectory error isolates (b)
plus the residual head's accuracy under teacher forcing.

Interpretation limits (P3.3.2a wording corrections): teacher-forced numbers
support an exposure-error diagnosis but do not uniquely rule out
representation or decoder defects under off-distribution prefixes; any
per-chunk-accuracy power calculation (e.g. 0.7^10) is a heuristic, not a
measured rollout all-correct probability.

Writes TEACHER_DECODE_PROBE.json next to this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches  # noqa: E402
from scenario_lab.p33_metrics import (  # noqa: E402
    AgentErrorRecord,
    displacement_errors,
    evaluate_agent_mask,
    group_level_table,
)
from scenario_lab.p33_model import ARSceneV1, decode_tokens_to_trajectory, to_torch_batch  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

MANIFEST = REPO / "runs/20260913_p331_data_pipeline/smoke/DATASET_MANIFEST.json"
CHECKPOINT = REPO / "runs/20260913_p332a_visibility_fix/m2_final_checkpoint.pt"

config = load_p33_config()
catalog = ShardCatalog.from_manifest(MANIFEST, config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codebook_path = MANIFEST.parent / "motion_codebook_v1.npz"
with np.load(codebook_path, allow_pickle=False) as data:
    codebook = np.asarray(data["centroids"], dtype=np.float32)
torch.manual_seed(7)
model = ARSceneV1(config, codebook).to(device)
payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict({key: value.to(device) for key, value in payload["model"].items()})
model.eval()

records: list[AgentErrorRecord] = []
argmax_records: list[AgentErrorRecord] = []
batches = 0
with torch.no_grad():
    for batch_raw in iter_split_batches(catalog, split="dev", batch_size=16):
        batch = to_torch_batch(batch_raw, device)
        outputs = model(batch, teacher_tokens=batch["motion_token_target"])
        truth_tokens = batch["motion_token_target"].cpu().numpy()
        argmax_tokens = outputs["motion_token_logits"].argmax(-1).cpu().numpy()
        residuals = outputs["delta_xy_residual"].cpu().numpy()
        current = batch["agent_history"][:, :, -1, :2].cpu().numpy()
        for tokens, sink in ((truth_tokens, records), (argmax_tokens, argmax_records)):
            decoded = np.stack([
                decode_tokens_to_trajectory(tokens[i].astype(np.int64), residuals[i],
                                            codebook, current[i])
                for i in range(len(batch["sample_id"]))])
            for index in range(len(batch["sample_id"])):
                mask = evaluate_agent_mask(batch_raw["agent_present_mask"][index],
                                           batch_raw["agent_role"][index],
                                           batch_raw["future_valid_mask"][index])
                # per-step truth validity, matching M3's rollout records (the
                # decoded trajectory does not depend on future validity)
                step_valid = batch_raw["future_valid_mask"][index] > 0
                errors = displacement_errors(decoded[index], batch_raw["future_xy"][index],
                                              step_valid)
                for slot in np.flatnonzero(mask):
                    sink.append(AgentErrorRecord(
                        source=batch_raw["sample_source"][index],
                        group_id=batch_raw["group_id"][index],
                        sample_id=batch_raw["sample_id"][index],
                        agent_slot=int(slot),
                        agent_type=int(batch_raw["agent_type"][index, slot]),
                        agent_role=int(batch_raw["agent_role"][index, slot]),
                        agents_truncated=bool(batch_raw["agents_truncated"][index]),
                        map_truncated=bool(batch_raw["map_truncated"][index]),
                        ade=float(errors["ade"][slot]),
                        fde=float(errors["fde"][slot]),
                        # K=1 semantics: min@1 == the single sample's error
                        min_ade=float(errors["ade"][slot]),
                        min_fde=float(errors["fde"][slot]),
                        valid_steps=int(errors["valid_steps"][slot]),
                        endpoint_valid=bool(errors["endpoint_valid"][slot]),
                    ))
        batches += 1

result = {
    "probe": "teacher_forced_decode_ground_truth_tokens_vs_argmax_tokens",
    "checkpoint": CHECKPOINT.name,
    "batches": batches,
    "ground_truth_tokens_group_level": group_level_table(records),
    "argmax_tokens_group_level": group_level_table(argmax_records),
    "interpretation": "ground_truth row isolates decode path + residual head "
                      "(token distribution perfect); argmax row adds greedy token "
                      "error without sampling/autoregressive drift. Teacher-forced "
                      "numbers are not rollout numbers and support exposure-error "
                      "diagnosis only; 0.7^10-style calculations are heuristics, "
                      "not measured rollout all-correct probabilities.",
}
out = Path(__file__).parent / "TEACHER_DECODE_PROBE.json"
out.write_text(json.dumps(result, indent=2))
for key in ("ground_truth_tokens_group_level", "argmax_tokens_group_level"):
    print(key, {source: {k: round(v, 4) for k, v in table.items()
                         if isinstance(v, float)} for source, table in result[key].items()})
