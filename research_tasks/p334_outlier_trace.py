"""P3.3.4-R0 input-outlier tracing (P334_CODEX_NEXT_PLAN.md section 1.4).

For every sample flagged in codex_review/EVIDENCE.json (plus extra sample ids
passed on the command line): locate the shard row, dump the flagged agent's
full history (x, y, recorded vx/vy, validity), and compare the recorded
velocity channels against position differencing.  The comparison splits the
possible causes the review listed -- raw-data jump / identity switch (recorded
velocity agrees with the diff), interpolation or time misalignment in the
conversion (recorded velocity sane while the diff is extreme), or transform/
unit errors (both velocities extreme but inconsistent with earlier frames).

Read-only: no shard, manifest, or v1 artifact is modified.  Writes
OUTLIER_TRACE.json next to the other R0 deliverables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_lab.p33_dataset import ShardCatalog, ShardReader  # noqa: E402
from scenario_lab.p33_spec import load_p33_config  # noqa: E402

FRAME_RATE = 10.0
SPEED_LIMIT = 35.0
EVIDENCE_PATH = REPO / "runs/20260915_p334_kinematics/codex_review/EVIDENCE.json"
# agent_history channels (p33_pipeline): 0 x, 1 y, 2 vx, 3 vy, 4 heading,
# 5 length, 6 width; "invalid rows are zero" so recorded v is meaningful only
# where state_valid_mask is set


def frame_table(history: np.ndarray, valid: np.ndarray) -> list[dict]:
    """Per-frame dump for one agent: positions, recorded velocity, validity,
    position-diff speed vs recorded speed side by side."""
    frames = []
    for t in range(history.shape[0]):
        row = {"t": t, "valid": bool(valid[t]),
               "x": float(history[t, 0]), "y": float(history[t, 1])}
        if valid[t]:
            row |= {"vx_rec": float(history[t, 2]), "vy_rec": float(history[t, 3]),
                    "speed_rec": float(np.hypot(history[t, 2], history[t, 3]))}
        if t > 0 and valid[t] and valid[t - 1]:
            delta = history[t, :2] - history[t - 1, :2]
            row["speed_diff"] = float(np.linalg.norm(delta) * FRAME_RATE)
        frames.append(row)
    return frames


def classify(frames: list[dict], history: np.ndarray, valid: np.ndarray) -> dict:
    """Compare the tail jump against the recorded velocity and the agent's
    earlier motion.  Returns the evidence fields for the cause assessment --
    the final label is written by the report, not here."""
    valid_idx = np.flatnonzero(valid)
    tail = frames[-1]
    prev = next((f for f in reversed(frames[:-1]) if f.get("valid")), None)
    assessment: dict = {
        "tail_speed_diff": tail.get("speed_diff"),
        "tail_speed_rec": tail.get("speed_rec"),
        "tail_speed_rec_prev": (prev or {}).get("speed_rec"),
    }
    if prev is not None:
        recorded_tail_velocity = np.array([tail["vx_rec"], tail["vy_rec"]])
        recorded_prev_velocity = np.array([prev["vx_rec"], prev["vy_rec"]])
        implied = (history[valid_idx[-1], :2] - history[valid_idx[-2], :2]) * FRAME_RATE
        assessment |= {
            # do the recorded channels agree with the position diff?
            "rec_vs_diff_angle_cos": float(
                np.dot(recorded_tail_velocity, implied)
                / (np.linalg.norm(recorded_tail_velocity) * np.linalg.norm(implied) + 1e-9)),
            # is the recorded velocity continuous with the PREVIOUS frame?
            "recorded_jump_mps": float(np.linalg.norm(
                recorded_tail_velocity - recorded_prev_velocity)),
        }
    # earlier-motion baseline: median speed over the valid history minus tail
    earlier = [f["speed_diff"] for f in frames[:-1]
               if "speed_diff" in f and f["speed_diff"] < SPEED_LIMIT]
    assessment["earlier_speed_median"] = (float(np.median(earlier))
                                          if earlier else None)
    # validity structure around the tail: a gap the conversion interpolated over
    assessment["valid_mask_tail10"] = [bool(v) for v in valid[-10:]]
    assessment["invalid_frames_in_history"] = int((~valid).sum())
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "runs/20260913_p333_architecture/data/DATASET_MANIFEST.json")
    parser.add_argument("--sample-id", action="append", default=[],
                        help="extra sample_id to trace (repeatable)")
    parser.add_argument("--output", type=Path,
                        default=REPO / "runs/20260916_p334_review_closure/OUTLIER_TRACE.json")
    args = parser.parse_args()

    targets = {item["sample_id"]: item["agent_slot"]
               for item in json.loads(EVIDENCE_PATH.read_text())["history_speed_outliers"]}
    for sample_id in args.sample_id:
        targets.setdefault(sample_id, None)      # slot unknown: dump agent 0
    config = load_p33_config()
    catalog = ShardCatalog.from_manifest(args.manifest, config)
    reader = ShardReader(config)

    results = []
    started = time.time()
    for entry in catalog.entries():
        if not targets:
            break
        rows = [json.loads(line)
                for line in entry.index_path.read_text(encoding="utf-8").splitlines()]
        reader.load(entry)
        for index, row in enumerate(rows):
            if row["sample_id"] not in targets:
                continue
            slot = targets.pop(row["sample_id"])
            arrays, meta = reader.get(index)
            history = arrays["agent_history"][slot]
            valid = arrays["state_valid_mask"][slot]
            frames = frame_table(history, valid)
            results.append({
                "sample_id": row["sample_id"],
                "agent_slot": slot,
                "shard": entry.path,
                "row_metadata": {key: meta[key] for key in sorted(meta)
                                 if key not in ("arrays",)},
                "agent_frames": frames,
                "assessment": classify(frames, history, valid),
            })
            print(f"traced {row['sample_id'][:12]} slot={slot} "
                  f"diff={results[-1]['assessment']['tail_speed_diff']} "
                  f"rec={results[-1]['assessment']['tail_speed_rec']}")
    if targets:
        print(f"WARNING: not found in catalog: {list(targets)}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evidence_path": str(EVIDENCE_PATH.relative_to(REPO)),
        "elapsed_seconds": time.time() - started,
        "traces": results,
    }, indent=1), encoding="utf-8")
    print(json.dumps({"written": str(args.output), "traced": len(results)}))


if __name__ == "__main__":
    main()
