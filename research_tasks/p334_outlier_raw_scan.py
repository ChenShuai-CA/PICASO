"""P3.3.4-R0 raw-source scan for the 613 m/s history outlier (plan section 1.4).

The shard trace (OUTLIER_TRACE.json) shows sample c1b38a6d.../slot 1 with a
61.3 m position jump into the anchor frame while the SAME box's recorded
velocity is 4.16 m/s, no validity gap, and build_scene_arrays performs no
interpolation -- so the inconsistency must already exist in
track.states of the parsed tfrecord.  This script re-reads the raw
training.tfrecord-00601-of-01000 and flags every ObjectState whose implied
position-diff speed exceeds 40 m/s while its own recorded speed is below
10 m/s (the "position vs velocity inconsistent" signature), plus any implied
speed over 60 m/s regardless, reporting scenario/track/frame so the finding
can be pinned to the raw data or to the parser.

Read-only.  Results to runs/20260916_p334_review_closure/RAW_SCAN.json.
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

from scenario_lab.p33_pipeline import iter_waymo_scenes  # noqa: E402

TARGET_TFRECORD = Path("Data/Waymo/training/training.tfrecord-00601-of-01000")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO)
    parser.add_argument("--tfrecord", type=Path, default=TARGET_TFRECORD)
    parser.add_argument("--output", type=Path,
                        default=REPO / "runs/20260916_p334_review_closure/RAW_SCAN.json")
    args = parser.parse_args()

    path = args.tfrecord if args.tfrecord.is_absolute() else args.data_root / args.tfrecord
    findings = []
    scenarios = 0
    started = time.time()
    for scene in iter_waymo_scenes(path, ""):
        scenarios += 1
        for track in scene.tracks:
            valid = np.flatnonzero(track.valid)
            if len(valid) < 2:
                continue
            positions = track.states[valid, :2]
            velocities = track.states[valid, 2:4]
            implied = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 10.0
            recorded = np.linalg.norm(velocities, axis=1)
            for k in range(len(valid) - 1):
                jump = implied[k]
                if jump > 40.0 and (jump > 60.0 or recorded[k + 1] < 10.0):
                    findings.append({
                        "scenario_id": scene.group_id,
                        "track_id": track.track_id,
                        "kind": track.kind,
                        "frame": int(valid[k + 1]),
                        "prev_frame": int(valid[k]),
                        "world_xy": positions[k + 1].tolist(),
                        "world_xy_prev": positions[k].tolist(),
                        "implied_speed_mps": float(jump),
                        "recorded_speed_mps": float(recorded[k + 1]),
                        "recorded_vxy": velocities[k + 1].tolist(),
                        "gap_frames": int(valid[k + 1] - valid[k] - 1),
                    })
    output = {
        "tfrecord": str(path.relative_to(args.data_root)),
        "scenarios_scanned": scenarios,
        "elapsed_seconds": time.time() - started,
        "signature": "implied_speed>40 and (implied>60 or recorded<10)",
        "findings": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(json.dumps({"scenarios": scenarios, "findings": len(findings),
                      "elapsed_seconds": round(output["elapsed_seconds"], 1)}))
    for item in findings[:10]:
        print(json.dumps({k: item[k] for k in ("scenario_id", "track_id", "kind",
                                               "frame", "gap_frames",
                                               "implied_speed_mps",
                                               "recorded_speed_mps")}))


if __name__ == "__main__":
    main()
