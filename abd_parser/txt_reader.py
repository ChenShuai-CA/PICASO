"""ABD .txt 读取器。

文件结构（RC Software Manual §8.3, p.143-144）：
  行1 厂商横幅 / 行2 Points=N / 行3 通道名（Tab 分隔）/ 行4 单位 / 行5+ 数据 @100Hz

只解析调用方声明的通道（按名字解析列号后逐行抽取），497 个 5-18MB 文件全量可承受。
"""

from __future__ import annotations

import numpy as np


class RunData:
    """一次 run 的选定通道数据。"""

    def __init__(self, path: str, names: list[str], units: list[str],
                 columns: dict[str, np.ndarray], n_points: int):
        self.path = path
        self.names = names          # 全部通道名（表头）
        self.units = units
        self.columns = columns      # 通道名 -> 1D array（按查找名）
        self.n_points = n_points
        t = columns.get("Time")
        self.time = t if t is not None else np.arange(n_points, dtype=float) / 100.0

    def has(self, name: str) -> bool:
        return name in self.columns

    def get(self, name: str) -> np.ndarray | None:
        return self.columns.get(name)


def find_channel(available: list[str], wanted: str) -> str | None:
    """通道模糊匹配（Stage4 §5.3）：精确 -> 去空格 -> 前缀 -> 包含。

    E8 存在命名差异（如 'Head tracker forward velocity' 无 '(ref point)' 后缀），
    故候选名同时尝试去掉括注的形式。
    """
    import re as _re
    variants = [wanted]
    stripped = _re.sub(r"\s*\([^)]*\)\s*", "", wanted).strip()
    if stripped and stripped != wanted:
        variants.append(stripped)
    for cand in variants:
        norm = {a.replace(" ", "").lower(): a for a in available}
        hit = norm.get(cand.replace(" ", "").lower())
        if hit:
            return hit
        for a in available:
            if a.startswith(cand):
                return a
        w = cand.replace(" ", "").lower()
        for a in available:
            if w in a.replace(" ", "").lower():
                return a
    return None


def read_run(path, wanted: list[str]) -> RunData:
    """读取 .txt，返回 wanted 中能匹配到的通道。

    wanted 中的名字允许与实际通道名不完全一致（走 find_channel）。
    返回的 columns 键 = wanted 中的请求名（调用方按请求名取数）。
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # banner
        pts_line = f.readline()
        names = f.readline().rstrip("\r\n").split("\t")
        units = f.readline().rstrip("\r\n").split("\t")
        n_points = int(pts_line.strip().split("=")[1]) if pts_line.startswith("Points=") else 0

        index = {}
        for w in wanted:
            actual = find_channel(names, w)
            if actual is not None:
                index[names.index(actual)] = w
        idxs = sorted(index)
        if not idxs:
            return RunData(str(path), names, units, {}, n_points)

        cols = {w: [] for w in index.values()}
        n_fields = len(names)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < n_fields:
                continue  # 尾部截断行
            for i in idxs:
                raw = parts[i]
                try:
                    cols[index[i]].append(float(raw))
                except ValueError:
                    cols[index[i]].append(float("nan"))
        columns = {w: np.asarray(v, dtype=float) for w, v in cols.items()}
        return RunData(str(path), names, units, columns, n_points)


WANTED_LABELS = [
    # 附录 G 标签提取所需通道全集（缺失者按存在性门处理）
    "Time",
    "X position", "Y position", "Yaw angle",
    "Forward velocity", "Forward acceleration", "Yaw velocity",
    "Path phase",
    "Time to collision (longitudinal)",
    "Relative longitudinal distance", "Relative lateral distance",
    "Relative resultant distance", "Relative longitudinal velocity",
    "Brake force (unfiltered)", "BR Position",
    "SR path abort", "Abort path speed",
    # 目标物通道用短请求名（find_channel 前缀/去括注匹配实际通道名，兼容 E8 命名差异）
    "Object 1 actual X", "Object 1 actual Y",
    "Object 1 forward velocity", "Object 1 yaw",
    "Head tracker actual X", "Head tracker actual Y",
    "Head tracker forward velocity", "Head tracker yaw",
    # 占位：Time tolerance N 由 labels 按触发器号动态追加
]
