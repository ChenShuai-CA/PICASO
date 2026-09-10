"""结果标签提取（Stage4 附录 G 实现，v0.2 — 经 2026-09-10 样本轨迹核对修订）。

标签：
  collision   = E1 ∨ (E2 ∧ E3)；仅 E4 -> equipment_abort（剔除）
  min_ttc     = [T0, 碰撞帧/run末] 窗口内 TTC 通道最小值
                （护栏：TTC∈(0,30] 且纵向接近 rv<-0.3 且 VUT 仍在运动 v>0.5，
                 排除停车后 9999 哨兵/伪低 TTC 帧）
  t_aeb_rel   = T0 后首个系统制动响应时刻

v0.2 相对附录 G 初稿的实证修订（4 run 轨迹核对，2026-09-10）：
  R1  E1/E3 不用 Relative resultant distance（实测恒 0，死通道）与绝对矩形 SAT
      （Object 1 前轴 ≠ 目标中心，偏 ~2.3m 致假阳性）；改用 Relative longitudinal
      distance（实测 = 有效车间隙）+ 位置横向重叠判定。
  R2  t_AEB 主证据改为纵向减速度 onset（a<-2 m/s² 持续 0.1s，IMU 品牌无关）；
      Brake force 通道在三品牌 AEB 全制动中均 ~0-13N，不可用；BR Position
      负基线->0 跳变（三品牌一致）作辅助证据。
  R3  TTC 通道含 9999 哨兵与停车后伪值，护栏加 v>0.5、rv<-0.3。
  R4  E8 的 Time tolerance 1 常开（本 run 真触发器为 TT2），.spec StartTrigger
      位于独立段 -> spec_reader 跨段扫描。
  R5  事件窗终点（VUT 停车 / 目标纵向越过 / T0+15s）：G.5 抽样发现 E1/min_dist
      在通过后垃圾帧（relD 无界负值、目标返回段）误触发；E1 纵向接触改双侧界
      |rel_d|<thr。TTC≈3 校验仅适用于 TTC 通道型触发器（距离型触发器不打旗标）。

所有阈值为处理参数初值（附录 G.1），经 G.5 人工校验后固定。
时间输出统一为相对量（_rel = 绝对时间 − T0）。
"""

from __future__ import annotations

import numpy as np

from .params import AEB_FAMILY, TARGET_DIMS

# ---- 处理参数初值（附录 G，待 G.5 校验后固定） ----
TTC_VERIFY_TOL = 0.5        # T0 时刻 TTC 应 ≈3.0s（AEB 族核对）
E2_DA = 5.0                 # 帧间 |Δa| 崩溃尖峰阈值 m/s²/frame（AEB 平滑制动 ≈0.4/frame，不触发）
E2_DV = -0.3                # 0.1s 内速度下降量（同向确认）m/s
E3_FRAC = 0.10              # 纵向接触阈值 = 10% × (车长+目标长)
LAT_MARGIN = 0.30           # 横向重叠裕度 m
MIN_TTC_VALID_MAX = 30.0    # TTC 有效性上界
CLOSING_VEL = -0.3          # 纵向接近阈值 m/s（真接近，排除停车噪声）
VUT_MOVING = 0.5            # VUT 仍在运动阈值 m/s
AEB_DECEL = -2.0            # 减速度 onset 阈值 m/s²
AEB_DECEL_SUSTAIN = 10      # 持续帧（0.1s）
BR_STEP_MM = 10.0           # BR Position 台阶阈值（负基线 -> 0 方向）
BR_SUSTAIN = 5
RATE = 100.0


def rising_edge(x: np.ndarray, level: float = 0.5, start: int = 0) -> int | None:
    """首个 0->1 上升沿索引；信号在 start 处已为高电平（常开触发器，如 E8 TT1）则 None。"""
    if x is None or len(x) == 0:
        return None
    if x[start] >= level:
        return None
    idx = np.flatnonzero(x[start:] >= level)
    return int(start + idx[0]) if len(idx) else None


def locate_t0(run, params: dict, aeb_family: bool) -> tuple[int | None, str]:
    """T0 多优先级定位（Stage4 §2.6.B）。返回 (索引, 方法标记)。"""
    # S1: Time-tolerance trigger 上升沿（触发器号来自 .spec，R4）
    ch = f"Time tolerance {params['tt_number']}" if params.get("tt_number") else None
    if ch and run.has(ch):
        i = rising_edge(run.get(ch))
        if i is not None:
            verified = ""
            # TTC≈3.0 校验仅适用于 TTC 通道型触发器；距离型（POI/LCRP）触发器
            # 在 T0 处 TTC 任意值都正常，不做校验也不打 unverified 旗标
            ttc_ch = (params.get("tt_channel") or "").lower()
            if aeb_family and "collision" in ttc_ch:
                ttc = run.get("Time to collision (longitudinal)")
                if ttc is not None and i < len(ttc):
                    v = ttc[i]
                    if np.isfinite(v) and abs(v - 3.0) <= TTC_VERIFY_TOL:
                        verified = "_verified"
                    else:
                        verified = "_unverified"
            return i, f"tt{params['tt_number']}{verified}"

    # S2: TTC 下降穿越 3.0s（AEB 族）
    if aeb_family:
        ttc = run.get("Time to collision (longitudinal)")
        if ttc is not None:
            ok = np.isfinite(ttc)
            for i in range(1, min(len(ttc), int(60 * RATE))):
                if ok[i] and ok[i - 1] and ttc[i - 1] >= 3.0 > ttc[i] and ttc[i] > 0.5:
                    return i, "ttc_crossing_3s"

    # S3: 目标运动起始（交叉/VRU：目标从静止启动）
    tv = target_velocity_channel(run, params)
    if tv is not None and len(tv):
        moving = np.flatnonzero(tv > 0.1)
        if len(moving):
            return int(moving[0]), "target_motion_onset"

    # S4: Path phase 0->1（路径开始，弱标记）
    pp = run.get("Path phase")
    if pp is not None and len(pp) and pp[0] < 0.5:
        trans = np.flatnonzero((pp[:-1] < 0.5) & (pp[1:] >= 0.5))
        if len(trans):
            return int(trans[0]) + 1, "path_phase_start"
    return None, "missing"


def target_velocity_channel(run, params: dict) -> np.ndarray | None:
    """目标速度通道：C2C 用 Object 1，VRU 用 Head tracker（缺失时互相回退，D4 决策）。"""
    pref = "Object 1 forward velocity" if params.get("family") == "c2c" else "Head tracker forward velocity"
    alt = "Head tracker forward velocity" if pref.startswith("Object") else "Object 1 forward velocity"
    for name in (pref, alt):
        if run.has(name):
            return run.get(name)
    return None


def target_pose(run, params: dict):
    """目标位姿通道 (x, y, yaw_deg)：族优先 + 互回退。"""
    pairs = [("Object 1 actual X", "Object 1 actual Y", "Object 1 yaw"),
             ("Head tracker actual X", "Head tracker actual Y", "Head tracker yaw")]
    if params.get("family") == "vru":
        pairs.reverse()
    for xk, yk, hk in pairs:
        if run.has(xk) and run.has(yk):
            return run.get(xk), run.get(yk), (run.get(hk) if run.has(hk) else None)
    return None, None, None


def _lateral_gap(run, params):
    """横向间距序列：目标 Y − VUT Y（同帧全局系；缺目标位姿时 None）。"""
    tx, ty, _ = target_pose(run, params)
    vy = run.get("Y position")
    if ty is None or vy is None:
        return None
    return np.abs(ty - vy)


def event_end(run, params: dict, t0: int, n: int) -> int:
    """事件窗终点（v0.3，R5）：最早发生者——

    a) VUT 停车（v<0.05 m/s 且此后 1s 内不再超过）
    b) 目标纵向越过 VUT（relD < -(车长+目标长)：通过/返回段通道数据无意义）
    c) T0+15s（通道垃圾段防护）
    d) run 末
    G.5 抽样实证：E1 曾在 T0+9~22s 的通过后垃圾帧（relD 无界负值、目标返回
    起点 dy>20m）上误触发碰撞。
    """
    v = run.get("Forward velocity")
    rel_d = run.get("Relative longitudinal distance")
    tlen, _ = TARGET_DIMS.get(params.get("target_type") or "none", (0.0, 0.0))
    vlen = params.get("veh_length_m") or 0.0
    cap = min(n - 1, t0 + int(15 * RATE))
    end = cap
    if v is not None:
        stop = np.flatnonzero(v[t0:cap + 1] < 0.05)
        if len(stop):
            i = t0 + int(stop[0])
            seg = v[i:min(cap, i + int(RATE)) + 1]
            if len(seg) >= int(RATE) and np.nanmax(seg) < 0.05:
                end = min(end, i)
    if rel_d is not None:
        passed = np.flatnonzero(rel_d[t0:cap + 1] < -(vlen + tlen))
        if len(passed):
            end = min(end, t0 + int(passed[0]))
    return max(end, t0)


def adjudicate_collision(run, params: dict, t0: int, end: int) -> dict:
    """附录 G.2 多证据碰撞判定（v0.2：R1 相对通道几何）。窗口 [t0, end]。"""
    out = {"collision": "", "t_collision_rel": "", "E1": 0, "E2": 0, "E3": 0,
           "E4": 0, "equipment_abort": 0, "min_dist": "", "dy_at_min": ""}
    if t0 is None:
        return out

    fv = run.get("Forward velocity")
    n = len(fv) if fv is not None else 0
    end = min(end if end is not None else n - 1, n - 1)
    if end <= t0:
        return out

    v = fv
    a = run.get("Forward acceleration")
    rel_d = run.get("Relative longitudinal distance")
    dy = _lateral_gap(run, params)

    tlen, twid = TARGET_DIMS.get(params.get("target_type") or "none", (0.0, 0.0))
    vlen = params.get("veh_length_m") or 0.0
    vwid = params.get("veh_width_m") or 0.0
    contact_thr = E3_FRAC * (vlen + tlen)
    lat_lim = (vwid + twid) / 2 + LAT_MARGIN

    # E4: 中止旗标（T0 后任一帧）
    e4 = 0
    for flag in ("SR path abort", "Abort path speed"):
        f = run.get(flag)
        if f is not None and np.any(f[t0:] >= 0.5):
            e4 = 1
    out["E4"] = e4

    # E2: 崩溃尺度加速度尖峰 + 速度崩塌（AEB 平滑制动不触发）
    e2_idx = None
    if a is not None and v is not None:
        da = np.abs(np.diff(a))
        dv = np.diff(v)
        look_end = min(end, len(a) - 11)
        for i in range(max(t0, 1), look_end):
            if da[i - 1] >= E2_DA and dv[i - 1] < 0 and (v[i + 10] - v[i]) <= E2_DV:
                e2_idx = i
                break
    out["E2"] = int(e2_idx is not None)

    # E1/E3: 纵向接近（rel_d 有效）+ 横向重叠
    e1_idx = None
    if rel_d is not None:
        w = rel_d[t0:end + 1]
        ok = np.isfinite(w)
        if ok.any():
            dmin = float(w[ok].min())
            out["min_dist"] = f"{dmin:.3f}"
            j = t0 + int(np.nanargmin(np.where(ok, w, np.inf)))
            if dy is not None and np.isfinite(dy[j]):
                out["dy_at_min"] = f"{dy[j]:.3f}"
            out["E3"] = int(dmin < contact_thr)
            # E1: 纵向接触（|rel_d|<thr，双侧界：通过后无界负值不算接触）且横向重叠
            for i in range(t0, end + 1):
                if (np.isfinite(rel_d[i]) and -contact_thr < rel_d[i] < contact_thr
                        and v[i] > 0.05
                        and dy is not None and np.isfinite(dy[i]) and dy[i] < lat_lim):
                    e1_idx = i
                    break
    out["E1"] = int(e1_idx is not None)

    # 判定规则：E1 ∨ (E2 ∧ E3)
    hit = e1_idx if e1_idx is not None else (e2_idx if (e2_idx is not None and out["E3"]) else None)
    if hit is not None:
        out["collision"] = 1
        out["t_collision_rel"] = f"{run.time[hit] - run.time[t0]:.3f}"
    elif e4:
        out["equipment_abort"] = 1
    return out


def extract_min_ttc(run, t0: int, end: int) -> tuple[float | None, float | None]:
    """附录 G.3 + R3 护栏：TTC∈(0,30]、真接近(rv<-0.3)、VUT 在运动(v>0.5)。"""
    ttc = run.get("Time to collision (longitudinal)")
    rv = run.get("Relative longitudinal velocity")
    v = run.get("Forward velocity")
    if ttc is None or t0 is None:
        return None, None
    n = len(ttc)
    end = min(end if end is not None else n - 1, n - 1)
    if end <= t0:
        return None, None
    w_ttc = ttc[t0:end + 1]
    valid = np.isfinite(w_ttc) & (w_ttc > 0) & (w_ttc <= MIN_TTC_VALID_MAX)
    if rv is not None:
        valid &= (rv[t0:end + 1] < CLOSING_VEL)
    if v is not None:
        valid &= (v[t0:end + 1] > VUT_MOVING)
    if not valid.any():
        return None, None
    masked = np.where(valid, w_ttc, np.inf)
    i = int(np.argmin(masked))
    return float(masked[i]), float(i / RATE)


def extract_t_aeb(run, t0: int, end: int) -> tuple[bool, float | None, str]:
    """附录 G.4 + R2：制动响应时刻。

    主证据：纵向减速度 onset（a < -2 m/s² 持续 0.1s；AEB 规程中机器人 T0 后不制动，
    T0 后的减速度 = 车辆系统制动）。辅助：BR Position 负基线 -> 0 方向台阶。
    """
    if t0 is None:
        return False, None, ""
    fv = run.get("Forward velocity")
    n = len(fv) if fv is not None else 0
    end = min(end if end is not None else n - 1, n - 1)

    # 证据 1：减速度 onset
    a = run.get("Forward acceleration")
    if a is not None:
        for i in range(t0, min(end, len(a) - AEB_DECEL_SUSTAIN)):
            if a[i] < AEB_DECEL:
                seg = a[i:i + AEB_DECEL_SUSTAIN]
                if np.nanmedian(seg) < AEB_DECEL:
                    return True, float((i - t0) / RATE), "decel_onset"

    # 证据 2：BR Position 台阶（负基线 -> 0 方向）
    bp = run.get("BR Position")
    if bp is not None:
        b0 = max(0, t0 - int(RATE))
        base = np.nanmedian(bp[b0:t0]) if t0 > b0 else 0.0
        if np.isfinite(base):
            level = base + BR_STEP_MM  # 基线为负，向 0 方向的台阶
            for i in range(t0, end):
                if bp[i] >= level:
                    seg = bp[i:min(i + BR_SUSTAIN, len(bp))]
                    if np.nanmedian(seg) >= level:
                        return True, float((i - t0) / RATE), "br_step"
    return False, None, ""


def extract_labels(run, spec: dict, params: dict) -> dict:
    """单 run 全标签。非 AEB 族只记 T0/E4（v0 无响应标签定义）。"""
    aeb_family = params.get("family") in AEB_FAMILY
    flags = []

    fv = run.get("Forward velocity")
    n = len(fv) if fv is not None else 0
    t0_idx, t0_method = locate_t0(run, params, aeb_family)
    if t0_idx is None:
        flags.append("t0_missing")

    row = {"T0": f"{run.time[t0_idx]:.3f}" if t0_idx is not None else "",
           "T0_method": t0_method, "needs_manual_review": ""}

    if not aeb_family:
        e4 = 0
        for flagch in ("SR path abort", "Abort path speed"):
            f = run.get(flagch)
            if f is not None and np.any(f >= 0.5):
                e4 = 1
        row.update({"collision": "", "t_collision_rel": "", "min_ttc": "", "t_min_ttc_rel": "",
                    "t_aeb_rel": "", "aeb_triggered": "", "aeb_method": "", "E1": "", "E2": "",
                    "E3": "", "E4": e4, "equipment_abort": "", "min_dist": "", "dy_at_min": "",
                    "label_family": params.get("family")})
        if flags:
            row["needs_manual_review"] = ";".join(flags)
        return row

    col = adjudicate_collision(run, params, t0_idx,
                               event_end(run, params, t0_idx, n - 1) if t0_idx is not None else n - 1)
    collision = bool(col["collision"])
    end_idx = None
    if collision:
        end_idx = t0_idx + int(float(col["t_collision_rel"]) * RATE)

    e_end = event_end(run, params, t0_idx, n - 1) if t0_idx is not None else None
    min_ttc, t_min_ttc_rel = extract_min_ttc(run, t0_idx, end_idx if collision else e_end)
    triggered, t_aeb_rel, aeb_method = extract_t_aeb(run, t0_idx, end_idx if collision else e_end)

    # 复核旗标
    if t0_idx is not None and "unverified" in t0_method:
        flags.append("t0_ttc_unverified")
    if col["equipment_abort"]:
        flags.append("equipment_abort")
    if collision and col["E4"] == 0:
        flags.append("collision_without_abort_flag")
    if min_ttc is not None and min_ttc < 0.1 and not collision:
        flags.append("near_zero_ttc_no_contact")  # 横向错过或通道伪值，人工复核

    row.update({
        "collision": int(collision), "t_collision_rel": col["t_collision_rel"],
        "min_ttc": f"{min_ttc:.3f}" if min_ttc is not None else "",
        "t_min_ttc_rel": f"{t_min_ttc_rel:.3f}" if t_min_ttc_rel is not None else "",
        "t_aeb_rel": f"{t_aeb_rel:.3f}" if triggered and t_aeb_rel is not None else "",
        "aeb_triggered": int(triggered),
        "aeb_method": aeb_method,
        "E1": col["E1"], "E2": col["E2"], "E3": col["E3"], "E4": col["E4"],
        "equipment_abort": col["equipment_abort"], "min_dist": col["min_dist"],
        "dy_at_min": col["dy_at_min"],
        "label_family": params.get("family"),
    })
    row["needs_manual_review"] = ";".join(flags)
    return row
