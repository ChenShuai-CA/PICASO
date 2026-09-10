#!/usr/bin/env python3
"""labels_v0 全量提取管线（Stage4 附录 G.6/G.7）。

输入：inventory/all_runs_inventory.csv 的 protocol 池有效 run（497）
输出：inventory/labels_v0.csv + 类别平衡审计 + 20 run 人工校验抽样表

用法：python abd_parser/extract_labels.py [--limit N] [--only <test_id>]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abd_parser.labels import extract_labels  # noqa: E402
from abd_parser.params import parse_params  # noqa: E402
from abd_parser.spec_reader import parse_spec, trigger_number  # noqa: E402
from abd_parser.txt_reader import WANTED_LABELS, read_run  # noqa: E402

EXTRACTOR_VERSION = "labels_v0.3"

# AEB 族标签提取所需通道（缺失 -> channel_gate_fail）。
# v0.2：Relative resultant distance 实测恒 0（死通道）-> 换 Relative longitudinal distance（E1/E3 依赖）
REQUIRED_AEB = ["Time", "X position", "Y position", "Forward velocity",
                "Forward acceleration", "Time to collision (longitudinal)",
                "Relative longitudinal distance", "Relative longitudinal velocity",
                "SR path abort"]

PARAM_COLS = ["vut_speed_kph", "target_speed_kph", "overlap_pct", "lighting",
              "sub_variant", "target_type", "family", "is_crossing",
              "tt_number", "tt_source", "tt_channel", "tt_min", "tt_max",
              "use_sync", "veh_length_m", "veh_width_m", "veh_wheelbase_m", "veh_mass_kg"]

LABEL_COLS = ["T0", "T0_method", "collision", "t_collision_rel", "min_ttc", "t_min_ttc_rel",
              "t_aeb_rel", "aeb_triggered", "aeb_method", "E1", "E2", "E3", "E4",
              "equipment_abort", "min_dist", "dy_at_min", "label_family", "needs_manual_review"]


def gate_check(run, params) -> str:
    """标签关键通道存在性门（替代旧 ≥400 列规则：A66=384/E8=395 列完整 run 误杀修正）。"""
    if params.get("family") not in ("c2c", "vru"):
        return ""
    missing = [c for c in REQUIRED_AEB if not run.has(c)]
    if not (run.has("Brake force (unfiltered)") or run.has("BR Position")):
        missing.append("brake_channel")
    if not ((run.has("Object 1 actual X") and run.has("Object 1 actual Y"))
            or (run.has("Head tracker actual X") and run.has("Head tracker actual Y"))):
        missing.append("target_pose")
    if not (run.has("Object 1 forward velocity") or run.has("Head tracker forward velocity")):
        missing.append("target_velocity")
    if params.get("tt_number") and not run.has(f"Time tolerance {params['tt_number']}"):
        missing.append(f"tt_channel_{params['tt_number']}")
    return ",".join(missing)


def process_run(rel_path: str) -> dict:
    txt = ROOT / "Data" / "ABD_Data" / rel_path
    spec = parse_spec(txt.with_suffix(".spec"))
    cond = str(Path(rel_path).parent)
    # brand/scenario 由 inventory 行传入，这里从路径段重建以保持独立可测
    brand_dir = rel_path.split("/")[0]
    from abd_inventory import BRANDS
    brand = BRANDS[brand_dir]
    from abd_inventory import classify
    _, acro, _, _, _ = classify(rel_path)
    params = parse_params(cond, brand, acro, spec)

    wanted = list(WANTED_LABELS)
    if params.get("tt_number"):
        ch = f"Time tolerance {params['tt_number']}"
        if ch not in wanted:
            wanted.append(ch)
    run = read_run(txt, wanted)

    row = {"rel_path": rel_path, "brand": brand, "scenario_acronym": acro,
           "condition_id": cond, "extractor_version": EXTRACTOR_VERSION}
    row.update({k: params[k] for k in PARAM_COLS})

    gate = gate_check(run, params)
    if gate:
        row.update({"T0": "", "T0_method": "channel_gate_fail", "needs_manual_review": gate,
                    "collision": "", "min_ttc": "", "t_aeb_rel": ""})
        row.update({c: "" for c in LABEL_COLS if c not in row})
        row["T0_method"] = "channel_gate_fail"
        row["needs_manual_review"] = gate
        return row

    row.update(extract_labels(run, spec, params))
    # 物理特征（T0 时刻接近速度，surrogate 特征的实测版）
    row["closing_speed_ms"] = _closing_speed(run, row)
    return row


def _closing_speed(run, row) -> str:
    """T0 时刻纵向接近速度（m/s，物理特征 v0）。"""
    if not row.get("T0"):
        return ""
    rv = run.get("Relative longitudinal velocity")
    if rv is None:
        return ""
    t0_idx = int(round(float(row["T0"]) * 100))
    if t0_idx < len(rv) and np_finite(rv[t0_idx]):
        return f"{-rv[t0_idx]:.2f}"
    return ""


def np_finite(v) -> bool:
    try:
        return v == v and abs(v) != float("inf")
    except TypeError:
        return False


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    inv = list(csv.DictReader(open(ROOT / "inventory" / "all_runs_inventory.csv", encoding="utf-8-sig")))
    pool = [r for r in inv if r["data_role"] == "protocol" and r["validity_label"] == "pass_parser_v1"]
    if args.only:
        pool = [r for r in pool if args.only in r["test_id"]]
    if args.limit:
        pool = pool[:args.limit]
    print(f"runs to process: {len(pool)}")

    out_rows = []
    for k, r in enumerate(pool):
        try:
            out_rows.append(process_run(r["rel_path"]))
        except Exception as e:  # 单 run 失败不阻塞全量
            out_rows.append({"rel_path": r["rel_path"], "brand": r["brand"],
                             "scenario_acronym": r["scenario_acronym"],
                             "condition_id": r["condition_id"],
                             "T0_method": f"extract_error:{type(e).__name__}",
                             "needs_manual_review": str(e)[:80],
                             "extractor_version": EXTRACTOR_VERSION})
        if (k + 1) % 50 == 0:
            print(f"  ...{k + 1}/{len(pool)}")

    cols = (["rel_path", "brand", "scenario_acronym", "condition_id", "extractor_version"]
            + PARAM_COLS + ["closing_speed_ms"] + LABEL_COLS)
    cols += [c for c in out_rows[0] if c not in cols]
    out_path = ROOT / "inventory" / "labels_v0.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {out_path} ({len(out_rows)} rows)")

    audit(out_rows)


def audit(rows):
    """附录 G.7 类别平衡审计 + T0 方法分布。"""
    print("\n==== T0 method distribution ====")
    m = defaultdict(int)
    for r in rows:
        m[r.get("T0_method", "")] += 1
    for k, v in sorted(m.items(), key=lambda x: -x[1]):
        print(f"  {k:35s} {v}")

    print("\n==== brand x scenario: n / collision / minTTC(median) / aeb_trig ====")
    agg = defaultdict(lambda: {"n": 0, "col": 0, "ttc": [], "aeb": 0, "abort": 0, "gate": 0, "err": 0})
    for r in rows:
        key = (r.get("scenario_acronym") or "?", r.get("brand") or "?")
        a = agg[key]
        a["n"] += 1
        if r.get("T0_method") == "channel_gate_fail":
            a["gate"] += 1
            continue
        if r.get("T0_method", "").startswith("extract_error"):
            a["err"] += 1
            continue
        if r.get("collision") == 1:
            a["col"] += 1
        if r.get("equipment_abort") == 1:
            a["abort"] += 1
        if r.get("min_ttc"):
            a["ttc"].append(float(r["min_ttc"]))
        if r.get("aeb_triggered") == 1:
            a["aeb"] += 1
    hdr = f"{'acro':8s} {'brand':13s} {'n':>4s} {'gate':>4s} {'err':>3s} {'col':>3s} {'abort':>5s} {'aeb':>4s} {'minTTC_med':>10s}"
    print(hdr)
    for (acro, brand), a in sorted(agg.items()):
        med = f"{sorted(a['ttc'])[len(a['ttc']) // 2]:.2f}" if a["ttc"] else "-"
        print(f"{acro:8s} {brand:13s} {a['n']:4d} {a['gate']:4d} {a['err']:3d} "
              f"{a['col']:3d} {a['abort']:5d} {a['aeb']:4d} {med:>10s}")

    n_aeb = sum(1 for r in rows if r.get("label_family") in ("c2c", "vru") and not str(r.get("T0_method", "")).startswith(("channel", "extract")))
    n_col = sum(1 for r in rows if r.get("collision") == 1)
    print(f"\nAEB-family labeled runs: {n_aeb}, collisions: {n_col} "
          f"({(n_col / n_aeb * 100 if n_aeb else 0):.1f}%)")


if __name__ == "__main__":
    main()
