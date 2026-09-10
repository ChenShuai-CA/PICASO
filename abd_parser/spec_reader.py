"""ABD .spec 解析器（附录 E 规范的最小实现）。

.spec 为空行分段的扁平 key=value 文本，每段以 Type= 行标识段类型。关键段：
  - 顶层 Type/Description（测试名，与目录名一致）
  - PF Standard / PF Straight Line 段（路径跟随）
  - AR Speed Throttle Event 段（油门事件：StartTrigger 常为 Time-tolerance trigger N，
    EndTrigger=Speed；AEB 规程中机器人 T0 后 Drop throttle、不主动制动）
  - TimeToleranceN* 参数（触发通道与阈值）
  - PATH FOLLOWING 段（车辆几何/质量参数；Mass 均为默认 1300，实际质量由用户表替代）
"""

from __future__ import annotations

import re
from pathlib import Path

TT_NUM_RE = re.compile(r"Time-tolerance trigger (\d+)", re.IGNORECASE)
TT_PARAM_RE = re.compile(r"^TimeTolerance(\d+)(\w+?)\s*=", re.IGNORECASE)


def parse_spec(path) -> dict:
    """返回 {sections, top, vehicle, tt_configs, ar_start_trigger, sync...}。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                sections.append(current)
                current = {}
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            current[k.strip()] = v.strip()
        else:
            # 无 '=' 的标注行（如 'PATH FOLLOWING'）视为新段开始
            if current:
                sections.append(current)
            current = {"_label": line.strip()}
    if current:
        sections.append(current)

    top = next((s for s in sections if "Type" in s and "Description" in s
                and "PathFile" not in s and "SpeedControl" not in s), {})
    ar = next((s for s in sections if s.get("Type", "").startswith("AR Speed Throttle")), {})
    pf = next((s for s in sections if "PathFile" in s or s.get("Type", "").startswith("PF ")), {})
    veh = next((s for s in sections if "VehicleLength" in s or s.get("_label") == "PATH FOLLOWING"), pf)

    tt: dict[str, dict] = {}
    for s in sections:
        for k, v in s.items():
            m = TT_PARAM_RE.match(k)
            if m:
                num, param = m.group(1), m.group(2)
                tt.setdefault(num, {})[param.lower()] = v

    def _tt_number(val: str | None) -> str | None:
        if not val:
            return None
        m = TT_NUM_RE.search(val)
        return m.group(1) if m else None

    # R4（2026-09-10）：StartTrigger 常位于 AR 段之后的独立无名段，
    # 扫描全部非 PF 段找首个 Time-tolerance StartTrigger
    ar_start_tt = ""
    for s in sections:
        if "PathFile" in s:  # PF 段的 StartTrigger 是路径起点（Closed Loop 等），跳过
            continue
        if _tt_number(s.get("StartTrigger", "")):
            ar_start_tt = s["StartTrigger"]
            break
    if not ar_start_tt:
        ar_start_tt = ar.get("StartTrigger", "")

    return {
        "sections": sections,
        "type": top.get("Type", ""),
        "description": top.get("Description", ""),
        "ar_start_trigger": ar_start_tt or ar.get("StartTrigger", ""),
        "ar_end_trigger": ar.get("EndTrigger", ""),
        "ar_end_trigger_value": ar.get("EndTriggerValue", ""),
        "post_event_mode": ar.get("PostEventMode", ""),
        "use_brake_robot": ar.get("UseBrakeRobot", ""),
        "speed_control": ar.get("SpeedControl", pf.get("SpeedControl", "")),
        "use_sync": ar.get("UseSynchronizationMode", pf.get("UseSynchronizationMode", "")),
        "sync_end_trigger": ar.get("SyncEndTriggerType", ""),
        "pf_start_trigger": pf.get("StartTrigger", ""),
        "tt_configs": tt,
        "vehicle": {
            "length": veh.get("VehicleLength"), "width": veh.get("VehicleWidth"),
            "wheelbase": veh.get("WheelBase"), "mass": veh.get("Mass"),
            "front_overhang": veh.get("FrontOverhang"), "fwd_shift": veh.get("FwdShift"),
        },
    }


def trigger_number(spec: dict) -> tuple[str | None, str]:
    """T0 触发器号：优先 AR 油门事件 StartTrigger，其次 SyncEndTriggerType，再次 PF StartTrigger。

    返回 (trigger_number, source)。实测分布：S9/A66/P7+ 多为 TT1（Time to POI / 距离阈值），
    E8 CCRs 为 TT2（TTC 通道）；LKA 为 TT4（距离 0-999，常开）→ 由 labels 层做兜底。
    """
    n = _match(spec.get("ar_start_trigger", ""))
    if n:
        return n, "ar_start_trigger"
    n = _match(spec.get("sync_end_trigger", ""))
    if n:
        return n, "sync_end_trigger"
    n = _match(spec.get("pf_start_trigger", ""))
    if n:
        return n, "pf_start_trigger"
    return None, ""


def _match(val: str) -> str | None:
    m = TT_NUM_RE.search(val or "")
    return m.group(1) if m else None
