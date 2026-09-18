"""R1 report helper: C-v2 invalid-candidate and start-jump attribution."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M = json.loads((REPO / "runs/20260917_p334_r1_eval/METRICS_V2.json").read_text())

for arm in ("C", "C-v2"):
    print("===", arm)
    for group, v in M["arms"][arm]["validity"].items():
        invalid = v["attempted"] - v["kinematic_valid"]
        if invalid or v["agents_none"] or v["no_history_velocity"]:
            print(f"  {group}: attempted {v['attempted']} valid "
                  f"{v['kinematic_valid']} invalid {invalid} nonfinite "
                  f"{v['nonfinite']} agents_none {v['agents_none']} "
                  f"no_hist_vel {v['no_history_velocity']}")
    rv = M["arms"][arm]["validity_recorded_v0"]
    keys = next(iter(rv.values())).keys()
    totals = {k: sum(s.get(k, 0) for s in rv.values()) for k in keys}
    print("  recorded_v0 totals:", json.dumps(totals))

print("=== C-v2 frozen continuity (start jump) per group")
for group, meters in M["arms"]["C-v2"]["kinematics_frozen"].items():
    c = meters.get("continuity")
    if c and c["implied_accel_over_limit_rate"] > 0:
        q = c["jump_mps_quantiles"]
        print(f"  {group}: rate {c['implied_accel_over_limit_rate']:.4%} "
              f"agents {c['agents_checked']} jump_q50 {q.get('0.5')} "
              f"q95 {q.get('0.95')}")

print("=== C-v2 anchored kinematics nonzero rates")
for group, meters in M["arms"]["C-v2"]["kinematics_anchored"].items():
    for metric, p in meters.items():
        rate = p.get("step_rate", p.get("implied_accel_over_limit_rate", 0)) or 0
        if rate > 0.001:
            print(f"  {group}/{metric}: {rate:.4%}")
