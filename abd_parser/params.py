"""场景参数提取器（可执行场景空间 E 的 v0 子集，Stage3 §2.3）。

来源：condition 目录名（目录名即工况标识，含速度/重叠率/日夜）+ .spec（触发方式/车辆参数）。
实测目录名形态：
  S9/A66/P7+: "27-VUT 20 kph; CCRs -50 % AEB" / "225-VUT 20 kph; CPLA-25 5 kph AEB (Day)"
              CCFT 的 GVT 速度在父目录: "41-GVT 20 kph/260-VUT 10 kph; CCFT 20 kph AEB"
  E8:         "170-L6.1.5-CCRs-VUT-AEB-20kph-(-50%)"
"""

from __future__ import annotations

import re

# 用户提供的实际测试质量（整备+乘员200+设备35 kg，Stage4 §2.2.B）
MASS_KG = {"GAC_A66": 2535, "GAC_E8": 2410, "GAC_S9": 2600, "XPENG_P7PLUS": 2395}

# 目标类型 -> (包络长 m, 包络宽 m, 类别)。初值，经附录 G.5 人工校验后固定。
# GVT/假目标按 C-NCAP 目标规格量级；VRU 含载具/骑行人投影。
TARGET_DIMS = {
    "GVT": (4.6, 1.9), "false_target": (4.6, 1.9),
    "PT_adult": (0.6, 0.6), "PT_child": (0.5, 0.5),
    "bicycle": (1.8, 0.6), "scooter": (1.5, 0.6),
    "none": (0.0, 0.0),
}

# 缩写 -> (目标类型, 场景族, 是否交叉/横穿)
ACRO_META = {
    "CCRs": ("GVT", "c2c", 0), "CCRH": ("GVT", "c2c", 0),
    "CCOv": ("GVT", "c2c", 0),
    "SCP": ("GVT", "c2c", 1), "SCPO": ("GVT", "c2c", 1),
    "CCFT": ("false_target", "c2c", 1),
    "CPLA": ("PT_adult", "vru", 0), "CPNCO": ("PT_child", "vru", 0),
    "CPFAO": ("PT_adult", "vru", 0), "CPTA": ("PT_adult", "vru", 1),
    "CBNAO": ("bicycle", "vru", 0), "CBLA": ("bicycle", "vru", 0),
    "CSFAO": ("scooter", "vru", 0), "CSTA": ("scooter", "vru", 1),
    # 无 TTC 目标族（v0 不进 surrogate 回归，仅留账）
    "LKA": ("none", "steering", 0), "ELK": ("GVT", "steering", 0),
    "LDW": ("none", "steering", 0), "BSD": ("GVT", "warning", 0),
    "DOW": ("none", "warning", 0), "RCTA": ("GVT", "warning", 1),
    "TSR": ("none", "warning", 0), "ISLS": ("none", "warning", 0),
    "ICA": ("GVT", "comfort", 0), "SAS": ("none", "comfort", 0),
    "DMS": ("none", "comfort", 0), "LSS": ("none", "steering", 0),
}

# 响应标签是否有定义（TTC 目标存在 = AEB 族）
AEB_FAMILY = {"c2c", "vru"}

_SUB_RE = re.compile(r"\b(LN|LF|RF|RN)\b")

# VRU 转向/遮挡类规程的标准重叠率（目录名常缺省；C-NCAP 2024 附录 A）
ACRO_OVERLAP = {"CPLA": 25, "CPNCO": 25, "CPFAO": 25, "CPTA": 50,
                "CBNAO": 50, "CBLA": 25, "CSFAO": 50, "CSTA": 50}


def parse_params(condition_id: str, brand: str, acro: str, spec: dict) -> dict:
    """condition 目录名 + .spec -> 参数字典。未解析出的键为 None（审计时报告）。"""
    cond = condition_id.replace("\\", "/")
    segs = cond.split("/")
    last = segs[-1]
    text = " ".join(segs)  # CCFT 等的速度在父目录

    p = {
        "brand": brand,
        "scenario_acronym": acro,
        "sub_variant": (_SUB_RE.search(last).group(1) if _SUB_RE.search(last) else ""),
    }
    meta = ACRO_META.get(acro, ("unknown", "unknown", None))
    p["target_type"], p["family"], p["is_crossing"] = meta
    # ELK 父目录（L.6.3.5 超车避让）下的 CCOv 是转向测试：VUT 不制动、目标横
    # 向错开通过，无 AEB 响应语义 -> steering 族（2026-09-10 G.5 抽样实证）
    if re.search(r"ELK|L\.6\.3\.5", cond, re.I):
        p["family"] = "steering"

    # VUT 速度：S9/A66/P7+ "VUT 20 kph"；E8 "…-20kph-…"
    m = re.search(r"VUT\s*(\d+(?:\.\d+)?)\s*kph", text, re.I) \
        or re.search(r"(\d+(?:\.\d+)?)\s*kph", last, re.I)
    p["vut_speed_kph"] = float(m.group(1)) if m else None

    # 目标速度：VRU "CPLA-25 5 kph"（同段）；C2C 交叉 "GVT 20 kph"（父目录）；纵向静止目标=0
    m = re.search(r"(?:PT|GVT|Bicycle|Scooter|Target)\s*(\d+(?:\.\d+)?)\s*kph", text, re.I) \
        or re.search(r"(?:CPLA|CSTA|CPTA|CPFAO|CBNAO|CBLA|CSFAO|CPNCO)[^/]*?(\d+(?:\.\d+)?)\s*kph", last, re.I)
    if m:
        p["target_speed_kph"] = float(m.group(1))
    elif p["family"] == "c2c" and acro in ("CCRs", "CCRH", "CCOv"):
        p["target_speed_kph"] = 0.0
    else:
        p["target_speed_kph"] = None

    # 重叠率：CCRs "-50 %"/"+50 %"；E8 "(-50%)"；VRU 目录名常缺省 -> 规程缺省表（附录 A）
    m = re.search(r"\(?([+-]?\d+)\s*%\)?", last)
    if m and p["family"] == "c2c":
        p["overlap_pct"] = float(m.group(1))
    elif p["family"] == "vru":
        m2 = re.search(r"\b(25|50)\b", last)
        p["overlap_pct"] = float(m2.group(1)) if m2 else ACRO_OVERLAP.get(acro)
    else:
        p["overlap_pct"] = None

    # 日/夜
    low = last.lower()
    p["lighting"] = "night" if "night" in low else ("day" if "day" in low else "")

    # .spec 侧：触发方式 + 车辆几何
    from .spec_reader import trigger_number
    ttn, ttsrc = trigger_number(spec)
    tt_cfg = spec["tt_configs"].get(ttn, {}) if ttn else {}
    p["tt_number"], p["tt_source"] = ttn, ttsrc
    p["tt_channel"] = tt_cfg.get("channel1")
    p["tt_min"] = _f(tt_cfg.get("mintrigger1"))
    p["tt_max"] = _f(tt_cfg.get("maxtrigger1"))
    p["use_sync"] = spec["use_sync"]
    p["veh_length_m"] = _f(spec["vehicle"]["length"])
    p["veh_width_m"] = _f(spec["vehicle"]["width"])
    p["veh_wheelbase_m"] = _f(spec["vehicle"]["wheelbase"])
    p["veh_mass_kg"] = float(MASS_KG.get(brand, 0)) or None
    return p


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
