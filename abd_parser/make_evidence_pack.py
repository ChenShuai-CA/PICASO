#!/usr/bin/env python3
"""G.5 人工校验证据包：为 20 run 抽样表逐 run 生成轨迹摘要 + 具体核对问题。

输出：inventory/manual_check_evidence.md
用法：python abd_parser/make_evidence_pack.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abd_inventory import BRANDS, classify  # noqa: E402
from abd_parser.labels import event_end, locate_t0, target_pose  # noqa: E402
from abd_parser.params import parse_params  # noqa: E402
from abd_parser.spec_reader import parse_spec  # noqa: E402
from abd_parser.txt_reader import WANTED_LABELS, read_run  # noqa: E402

QUESTIONS = {
    "collision": [
        "核对 1（最重要）：数据回放/测试台账中，目标物是否真的被 VUT 接触/推移？",
        "  - 是 -> verdict 填 confirm",
        "  - 否 -> verdict 填 reject，note 写你观察到的实际情况（如：目标被推开但未接触 / 减速度尖峰来自急打方向）",
    ],
    "near_zero_flagged": [
        "核对 2：目标物是否横向错开通过（从车前方/侧方穿过但未进入碰撞走廊）？",
        "  - 是（未接触）-> confirm（纵向 TTC 低是通道口径限制，已有记录）",
        "  - 否（看起来应该撞上）-> reject + note",
    ],
    "near_miss": [
        "核对 3：minTTC（表中 min_ttc）与 t_aeb_rel（制动起始）是否与台账/回放一致？",
        "  - 大致一致（±0.2s）-> confirm；明显不符 -> reject + note",
    ],
    "safe": [
        "核对 4：此 run 是否确实安全通过/提前减速（无险情）？是 -> confirm",
    ],
}


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    mc = pd.read_csv(ROOT / "inventory" / "manual_check_sample_v0.csv",
                     encoding="utf-8-sig", dtype=str)
    lines = ["# G.5 人工校验证据包（labels_v0.3，2026-09-10）", "",
             "每 run 三件事：标签值（自动提取）→ 轨迹摘要（T0 前后，0.25 s 采样）→ 需要你回答的问题。",
             "结论回填 `inventory/manual_check_sample_v0.csv` 的 verdict(confirm/reject) 与 note 列。",
             "列含义：v=VUT速度 m/s；a=纵向加速度 m/s²；TTC=碰撞时间通道 s；relD=纵向相对距离 m；dy=横向间距 m",
             "（dy 小于约 1.6-2.2 m 表示目标在本车碰撞走廊内，取决于车宽+目标宽）。", ""]

    for k, row in mc.iterrows():
        rel = row["rel_path"]
        txt = ROOT / "Data" / "ABD_Data" / rel
        if not txt.exists():
            lines += [f"## [{k+1}] MISSING: {rel}", ""]
            continue
        spec = parse_spec(txt.with_suffix(".spec"))
        cond = str(txt.parent.relative_to(ROOT / "Data" / "ABD_Data"))
        brand = BRANDS[rel.split("/")[0]]
        _, acro, _, _, _ = classify(rel)
        params = parse_params(cond, brand, acro, spec)
        wanted = list(WANTED_LABELS)
        if params.get("tt_number"):
            ch = f"Time tolerance {params['tt_number']}"
            if ch not in wanted:
                wanted.append(ch)
        run = read_run(txt, wanted)

        t0, meth = locate_t0(run, params, params.get("family") in ("c2c", "vru"))
        v = run.get("Forward velocity")
        n = len(v) if v is not None else 0
        e_end = event_end(run, params, t0, n - 1) if t0 is not None else n - 1
        a = run.get("Forward acceleration")
        ttc = run.get("Time to collision (longitudinal)")
        rel_d = run.get("Relative longitudinal distance")
        tx, ty, _ = target_pose(run, params)
        vy = run.get("Y position")
        dy = np.abs(ty - vy) if ty is not None and vy is not None else None

        lines.append(f"## [{k+1}] {row['brand']} {acro} — outcome={row['outcome']}")
        lines.append(f"路径：`{rel}`")
        lines.append(f"标签：T0={row['T0']}s ({meth}) | collision={row['collision']} "
                     f"| minTTC={row['min_ttc']} | t_AEB={row['t_aeb_rel']}s ({row['aeb_method']}) "
                     f"| min_relD={row['min_dist']} | dy@min={row['dy_at_min']} | E1-E4="
                     f"{row['E1']}/{row['E2']}/{row['E3']}/{row['E4']}")
        if t0 is None:
            lines.append("（T0 未定位——本 run 无 AEB 族标签，仅核对场景分类是否正确）")
        else:
            lines.append(f"事件窗：T0={run.time[t0]:.2f}s ~ T_end={run.time[min(e_end, n-1)]:.2f}s")
            lines.append("```")
            lines.append(f"{'t':>7s} {'v':>6s} {'a':>6s} {'TTC':>7s} {'relD':>8s} {'dy':>7s}")
            for i in range(max(0, t0 - 25), min(n - 1, t0 + 801), 25):
                if i > e_end + 100:
                    break
                t_v = f"{v[i]:6.2f}" if v is not None else "     -"
                t_a = f"{a[i]:6.2f}" if a is not None else "     -"
                t_ttc = f"{ttc[i]:7.2f}" if ttc is not None else "      -"
                t_rd = f"{rel_d[i]:8.2f}" if rel_d is not None else "       -"
                t_dy = f"{dy[i]:7.2f}" if dy is not None else "      -"
                lines.append(f"{run.time[i]:7.2f} {t_v} {t_a} {t_ttc} {t_rd} {t_dy}")
            lines.append("```")
        lines += QUESTIONS.get(row["outcome"], ["（无特定问题，抽查标签合理性）"])
        lines.append("")
        print(f"[{k+1}/20] {row['brand']} {acro} {row['outcome']}")

    p = ROOT / "inventory" / "manual_check_evidence.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
