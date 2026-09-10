#!/usr/bin/env python3
"""附录 G.5 人工校验抽样表（20 run 分层抽样）。

分层维度：品牌 × 类别 × 结果（碰撞/险胜/安全/旗标）。每 run 给出
关键证据（T0/方法、E1-E4、minTTC、t_AEB、minDist、dy），供对照
ABD 数据回放人工确认。校验结论回填 verdict 列后用于固定阈值。

输出：inventory/manual_check_sample_v0.csv
用法：python abd_parser/make_manual_check.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_SAMPLE = 20
# 优先覆盖主表类 + 碰撞/旗标案例；其余按类分层
MAIN_CLASSES = ["CCRs", "CPTA", "CCFT", "CSTA", "CPLA", "CCOv", "SCP"]


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    df = pd.read_csv(ROOT / "inventory" / "labels_v0.csv", encoding="utf-8-sig", dtype=str)
    df = df[df["label_family"].isin(["c2c", "vru"])].copy()
    df["mttc"] = pd.to_numeric(df["min_ttc"], errors="coerce")

    def outcome(r):
        if r["collision"] == "1":
            return "collision"
        review = str(r.get("needs_manual_review") or "")
        if "near_zero" in review:
            return "near_zero_flagged"
        if pd.notna(r["mttc"]) and r["mttc"] < 0.7:
            return "near_miss"
        return "safe"

    df["outcome"] = df.apply(outcome, axis=1)

    picked: list[pd.Series] = []
    used = set()

    def take(mask, k, label):
        """(品牌×类别) 双轮转抽样：每轮每个组各取 1 条，直至配额用尽。"""
        cand = df[mask & ~df.index.isin(used)].sort_values("rel_path")
        if cand.empty or k <= 0:
            return
        groups: dict[tuple, list] = {}
        for key, grp in cand.groupby(["brand", "scenario_acronym"], sort=True):
            groups[key] = list(grp.index)
        ptr = {g: 0 for g in groups}
        # 品牌交错排序：不同品牌轮流被取，避免字母序靠前的品牌占满配额
        by_brand: dict[str, list] = {}
        for (b, cls) in sorted(groups):
            by_brand.setdefault(b, []).append((b, cls))
        order = []
        for i in range(max(len(v) for v in by_brand.values())):
            for b in sorted(by_brand):
                if i < len(by_brand[b]):
                    order.append(by_brand[b][i])
        while k > 0 and len(picked) < N_SAMPLE:
            took = False
            for g in order:
                idxs = groups[g]
                if ptr[g] < len(idxs) and k > 0 and len(picked) < N_SAMPLE:
                    picked.append(df.loc[idxs[ptr[g]]])
                    used.add(idxs[ptr[g]])
                    ptr[g] += 1
                    k -= 1
                    took = True
            if not took:
                break

    take(df["outcome"] == "collision", 6, "collision")
    take(df["outcome"] == "near_zero_flagged", 4, "flagged")
    take((df["outcome"] == "near_miss") & df["scenario_acronym"].isin(MAIN_CLASSES), 6, "near_miss")
    take(df["scenario_acronym"].isin(MAIN_CLASSES), N_SAMPLE - len(picked), "fill_main")

    cols = ["rel_path", "brand", "scenario_acronym", "condition_id", "T0", "T0_method",
            "collision", "t_collision_rel", "E1", "E2", "E3", "E4",
            "min_ttc", "t_min_ttc_rel", "t_aeb_rel", "aeb_method",
            "min_dist", "dy_at_min", "outcome"]
    out = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in picked])
    out["verdict"] = ""          # 人工回填: confirm / reject(原因)
    out["note"] = ""
    p = ROOT / "inventory" / "manual_check_sample_v0.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"wrote {p} ({len(out)} runs)")
    print(out[["brand", "scenario_acronym", "outcome", "collision", "min_ttc",
               "t_aeb_rel", "min_dist", "dy_at_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
