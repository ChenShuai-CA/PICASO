#!/usr/bin/env python3
# abd_inventory.py — M1 数据盘点管线 v1
# 扫描 Data/ABD_Data，产出 inventory/all_runs_inventory.csv 与 inventory/coverage_matrix.csv
#
# 数据分流规则（2026-09-01 修订）：
#   protocol            -> 生成/响应建模池（C-NCAP 规程类）
#   surrogate_negative  -> FalseReaction，"不应制动"负样本池（surrogate 边界建模）
#   engineering_envelope-> 调参/标定/预热等，仅用于 ABD 执行包络标定
#   unclassified        -> 待人工确认
#
# 有效性 v1 启发式：有同名 .spec 且 .txt > 100KB 且通道数 > 50 -> pass_parser_v1
# （正式口径以 Stage4 的 T0 可定位、通道完整性、规程确认等完整校验为准）

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "ABD_Data"
OUT = ROOT / "inventory"

BRANDS = {
    "Guang_Qi_A66_data": "GAC_A66",
    "Guang_Qi_E8_data": "GAC_E8",
    "Guang_Qi_S9_data": "GAC_S9",
    "XPengP7+_data": "XPENG_P7PLUS",
}

# 长名优先（SCPO 先于 SCP，CCRs/CCRH 先于 CCR）
SCENARIO_ACRONYMS = sorted(
    ["CPLA", "CPNCO", "CPFAO", "CPTA", "CSTA", "CBNAO", "CBLA", "CSFAO",
     "CCRs", "CCRH", "CCFT", "CCOv", "SCPO", "SCP", "BSD", "LKA", "ELK",
     "ACC", "ICA", "TJA", "ILC", "RCW", "ISLS", "TSR", "DMS", "SAS",
     "DOW", "RCTA", "LSS", "LDW"],
    key=len, reverse=True,
)
SYSTEM_MODES = ["AEB", "FCW"]

ENGINEERING_KEYWORDS = [
    "speed tuning", "path following", "preset", "new group", "tuning",
    "calibration", "characterisation", "characterization", "conditioning",
    "warming", "pre-test", "hmi", "steady state", "step test", "constant level",
]

CLAUSE_RE = re.compile(r"\b[LO]\.?\d+(?:\.\d+)+\b")
RUN_RE = re.compile(r"^(V\d+)_T(\d+)_R(\d+)\.txt$", re.IGNORECASE)


def read_text_head(path: Path, n_lines=5):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return [f.readline() for _ in range(n_lines)]
    except OSError:
        return []


def parse_txt_head(lines):
    """ABD .txt 结构: 0=厂商横幅, 1=Points=N, 2=通道名, 3=单位。返回 (n_cols, n_points)。"""
    n_cols, n_points = 0, None
    for line in lines:
        if line.startswith("Points="):
            try:
                n_points = int(line.strip().split("=")[1])
            except (ValueError, IndexError):
                pass
    if len(lines) >= 3:
        n_cols = len(lines[2].rstrip("\n").split("\t"))
    return n_cols, n_points


def _time_of_line(line: str):
    cols = line.split("\t")
    if len(cols) < 2:
        return None
    try:
        return float(cols[1])
    except ValueError:
        return None


def read_time_range(path: Path, head_lines):
    """首数据行（head 第 5 行）与末数据行的 Time 列（第 2 列），返回 (t_first, t_last)。"""
    t_first = _time_of_line(head_lines[4]) if len(head_lines) >= 5 else None
    t_last = None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.strip().splitlines()):
            t_last = _time_of_line(line)
            if t_last is not None:
                break
    except OSError:
        pass
    return t_first, t_last


def parse_spec(spec_path: Path):
    info = {"spec_type": "", "spec_desc": ""}
    if not spec_path or not spec_path.exists():
        return info
    try:
        text = spec_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().lower()
        if k == "type" and not info["spec_type"]:
            info["spec_type"] = v.strip()
        elif k in ("description", "name") and not info["spec_desc"]:
            info["spec_desc"] = v.strip()
    return info


def classify(rel_path: str):
    """返回 (data_role, scenario_acronym, system_mode, clause_ref, is_cncap_named)。

    acronym 取"最深路径段 × 最长缩写"的匹配；AEB/FCW 是系统模式而非场景类，单列。
    role：protocol 需有规程条款号或 C-NCAP 命名；仅缩写命中的功能测试记为 function_test。
    """
    low = rel_path.lower()
    parts = rel_path.replace("\\", "/").split("/")
    acro = ""
    for seg in reversed(parts):  # 最深段优先
        for a in SCENARIO_ACRONYMS:
            if a.lower() in seg.lower():
                acro = a
                break
        if acro:
            break
    mode = ""
    for seg in parts:
        for sm in SYSTEM_MODES:
            if sm.lower() in seg.lower():
                mode = sm
                break
        if mode:
            break
    clause = ""
    m = CLAUSE_RE.search(rel_path)
    if m:
        clause = m.group(0)
    is_cncap_named = "c-ncap" in low or "cncap" in low
    if "falsereaction" in low.replace("-", "") or "false reaction" in low:
        role = "surrogate_negative"
    elif any(k in low for k in ENGINEERING_KEYWORDS):
        role = "engineering_envelope"
    elif clause or is_cncap_named:
        role = "protocol"
    elif acro:
        role = "function_test"
    else:
        role = "unclassified"
    return role, acro, mode, clause, is_cncap_named


def main():
    try:
        sys.stdout.reconfigure(errors="replace")  # GBK 控制台容错
    except Exception:
        pass
    OUT.mkdir(exist_ok=True)
    rows = []
    for brand_dir, brand in BRANDS.items():
        bdir = DATA / brand_dir
        if not bdir.is_dir():
            print(f"[warn] missing {bdir}", file=sys.stderr)
            continue
        for txt in sorted(bdir.rglob("*.txt")):
            m = RUN_RE.match(txt.name)
            if not m:
                continue  # 跳过 CurrentTestSpec.txt / ExpInfo.txt
            rel = txt.relative_to(DATA).as_posix()
            condition_rel = str(Path(rel).parent).replace("\\", "/")
            spec = txt.with_suffix(".spec")
            head = read_text_head(txt)
            n_cols, n_points = parse_txt_head(head)
            t_first, t_last = read_time_range(txt, head)
            duration = (t_last - t_first) if (t_first is not None and t_last is not None) else None
            rate = f"{(n_points - 1) / duration:.1f}" if (n_points and duration and duration > 0) else ""
            spec_info = parse_spec(spec)
            role, acro, mode, clause, cncap_named = classify(rel)
            size_mb = txt.stat().st_size / 1e6
            has_spec = spec.exists()
            if not has_spec:
                validity, reason = "exclude", "no_spec"
            elif size_mb <= 0.1:
                validity, reason = "exclude", "tiny_txt"
            elif n_cols <= 50:
                validity, reason = "exclude", "few_channels"
            else:
                validity, reason = "pass_parser_v1", ""
            rows.append({
                "brand": brand,
                "data_role": role,
                "scenario_acronym": acro,
                "system_mode": mode,
                "clause_ref": clause,
                "cncap_named": int(cncap_named),
                "condition_id": condition_rel,
                "rel_path": rel,
                "file": txt.name,
                "test_id": m.group(2),
                "run_id": m.group(3),
                "size_mb": f"{size_mb:.2f}",
                "n_cols": n_cols,
                "n_points": n_points if n_points is not None else "",
                "duration_s": f"{duration:.2f}" if duration is not None else "",
                "sample_rate_hz": rate,
                "has_spec": int(has_spec),
                "spec_type": spec_info["spec_type"],
                "spec_desc": spec_info["spec_desc"],
                "validity_label": validity,
                "exclude_reason": reason,
            })

    inv_path = OUT / "all_runs_inventory.csv"
    with inv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # coverage matrix: scenario_acronym x brand（独立工况数 / run 数），按 data_role 分文件输出
    # coverage_matrix.csv = protocol 池（C-NCAP 规程覆盖，论文口径）
    # coverage_matrix_function_test.csv = 企标功能项池（不与规程混表；2026-09-10 评审 P5 修订）
    def write_coverage(role: str, path: Path) -> int:
        cond = defaultdict(set)
        runs = defaultdict(int)
        for r in rows:
            if r["validity_label"] != "pass_parser_v1" or r["data_role"] != role:
                continue
            key = (r["scenario_acronym"] or "(unknown)", r["brand"])
            cond[key].add(r["condition_id"])
            runs[key] += 1
        acros = sorted({k[0] for k in cond})
        brands = list(BRANDS.values())
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["scenario_acronym"] + [f"{b}_conditions" for b in brands] + [f"{b}_runs" for b in brands])
            for a in acros:
                w.writerow([a]
                           + [len(cond.get((a, b), set())) for b in brands]
                           + [runs.get((a, b), 0) for b in brands])
        return len(acros)

    n_proto_acros = write_coverage("protocol", OUT / "coverage_matrix.csv")
    write_coverage("function_test", OUT / "coverage_matrix_function_test.csv")

    # 汇总 + M1 go/no-go 门
    brands = list(BRANDS.values())
    print(f"total runs scanned: {len(rows)}")
    by_role = defaultdict(int)
    valid_by_role = defaultdict(int)
    brand_valid = defaultdict(int)
    for r in rows:
        by_role[r["data_role"]] += 1
        if r["validity_label"] == "pass_parser_v1":
            valid_by_role[r["data_role"]] += 1
            brand_valid[r["brand"]] += 1
    for role in ("protocol", "function_test", "surrogate_negative", "engineering_envelope", "unclassified"):
        print(f"  {role:22s} total={by_role[role]:5d}  valid={valid_by_role[role]:5d}")
    for b in brands:
        print(f"  brand {b:12s} valid={brand_valid[b]:5d}")
    proto_conditions = {r["condition_id"] for r in rows
                        if r["data_role"] == "protocol" and r["validity_label"] == "pass_parser_v1"}
    proto_classes = {r["scenario_acronym"] for r in rows
                     if r["data_role"] == "protocol" and r["validity_label"] == "pass_parser_v1"
                     and r["scenario_acronym"]}
    n_proto_valid = valid_by_role["protocol"]
    print(f"protocol distinct conditions: {len(proto_conditions)}")
    print(f"protocol scenario classes: {sorted(proto_classes)} ({len(proto_classes)})")
    gate_runs = n_proto_valid >= 100
    gate_cov = len(proto_classes) >= 6
    print(f"M1 gate: protocol valid runs >= 100 -> {'PASS' if gate_runs else 'FAIL'} ({n_proto_valid})")
    print(f"M1 gate: >=6 protocol classes      -> {'PASS' if gate_cov else 'FAIL'} ({len(proto_classes)})")
    print(f"wrote {inv_path}")
    print(f"wrote {OUT / 'coverage_matrix.csv'} (protocol, {n_proto_acros} classes)")
    print(f"wrote {OUT / 'coverage_matrix_function_test.csv'} (function_test)")


if __name__ == "__main__":
    main()
