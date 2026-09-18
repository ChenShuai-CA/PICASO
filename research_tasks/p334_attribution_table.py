"""P3.3.4 Phase A: cross-arm attribution comparison table.

Reads the five attribution_<arm>.json written by p334_kinematics_attribution.py
and prints one table per (source, type) group with per-step rates, exceedance
p95 and continuity for every arm, plus the arm-minus-gt deltas that answer the
SPEC §1 attribution questions (codebook vs prediction vs residual vs
autoregression).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "runs/20260915_p334_kinematics"
ARMS = ["gt", "gt_token", "pred_token", "pred_full", "teacher_forced"]


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def q95(block: dict) -> str:
    quantiles = block["exceedance_quantiles"]
    return "—" if quantiles is None else f"{quantiles['p95']:.1f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    tables = {}
    for arm in ARMS:
        path = OUT_DIR / f"attribution_{arm}.json"
        if not path.exists():
            print(f"MISSING {path.name} — run that arm first")
            continue
        tables[arm] = json.loads(path.read_text(encoding="utf-8"))["table"]
    if not tables:
        sys.exit("no attribution results found")

    groups = sorted({group for table in tables.values() for group in table})
    lines = ["# P3.3.4 Phase A 跨臂归因表（dev 全量，ep29，同采样）", ""]
    for group in groups:
        lines += [f"## {group}", "",
                  "| 臂 | speed step | accel step | accel 轨迹 | jerk step | jerk 轨迹 | jerk越限p95 | 起点jump p95 | 起点implied超限 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            if arm not in tables or group not in tables[arm]:
                continue
            table = tables[arm][group]
            continuity = table["continuity"]
            jump = continuity["jump_mps_quantiles"]
            lines.append(
                f"| {arm} | {pct(table['speed']['step_rate'])} "
                f"| {pct(table['accel']['step_rate'])} | {pct(table['accel']['traj_rate'])} "
                f"| {pct(table['jerk']['step_rate'])} | {pct(table['jerk']['traj_rate'])} "
                f"| {q95(table['jerk'])} "
                f"| {jump['p95']:.3f} | {pct(continuity['implied_accel_over_limit_rate'])} |")
        lines.append("")
    output = "\n".join(lines)
    (OUT_DIR / "ATTRIBUTION_TABLE.md").write_text(output, encoding="utf-8")
    print(output)
    print(f"\nwritten: {OUT_DIR / 'ATTRIBUTION_TABLE.md'}")


if __name__ == "__main__":
    main()
