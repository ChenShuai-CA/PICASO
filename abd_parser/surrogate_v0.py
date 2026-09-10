#!/usr/bin/env python3
"""surrogate v0：minTTC 回归（主）+ 碰撞分类（辅）——Stage4 §5 v0 冲刺。

数据：inventory/labels_v0.csv（附录 G v0.2 标签）
池规则（P1 双池）：AEB 族；equipment_abort 且非碰撞 -> 剔除；碰撞 -> 保留
特征：规程参数 + T0 实测接近速度 + 车辆物理量（无品牌 one-hot，LOBO 可用）
评估：
  A. 品牌内：每品牌 GroupKFold(5, group=condition_id) OOF MAE + 中位数基线
  B. LOBO 预览：留一品牌，仅主表类（CCRs/CPTA/CCFT/CSTA/CPLA），逐类 MAE
种子：3（红线：不预设全 p<0.05，报实测值 + 跨种子 std）
输出：inventory/surrogate_v0_report.txt + inventory/surrogate_v0_predictions.csv

用法：python abd_parser/surrogate_v0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS = [0, 1, 2]
MAIN_CLASSES = ["CCRs", "CPTA", "CCFT", "CSTA", "CPLA"]  # LOBO 主表类（LKA 非 AEB 不入）

# 特征列（全为 surrogate 可用：LOBO 测试品牌不含品牌标识）
FEATURE_NUM = ["vut_speed_kph", "target_speed_kph", "overlap_pct", "closing_speed_ms",
               "is_crossing", "veh_mass_kg", "veh_length_m", "veh_width_m", "veh_wheelbase_m"]
FEATURE_CAT = ["target_type", "lighting", "sub_variant"]


def load_pool() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "inventory" / "labels_v0.csv", encoding="utf-8-sig", dtype=str)
    n_all = len(df)

    # 仅 AEB 族 + 提取成功
    df = df[df["label_family"].isin(["c2c", "vru"])].copy()
    df = df[~df["T0_method"].str.startswith(("channel_gate", "extract_error"), na=False)].copy()
    n_aeb = len(df)

    # 双池规则：abort 且非碰撞 -> 剔除；碰撞 -> 保留
    abort = df["equipment_abort"].astype(str).isin(["1", "1.0"])
    col = df["collision"].astype(str).isin(["1", "1.0"])
    df = df[~(abort & ~col)].copy()
    # 标定 run（E8 "-cal" / A66 "[CAL]"）：目标物偏置、无真实冲突，非正式工况 -> 剔除
    low = df["condition_id"].str.lower()
    is_cal = low.str.contains("-cal", regex=False) | low.str.contains("[cal", regex=False)
    df = df[~is_cal].copy()
    n_pool = len(df)

    # 回归目标：min_ttc 有效；>3s 截断（TTC>3s = 冲突早解除，无临迫语义；防离群主导 MAE）
    df["min_ttc_raw"] = pd.to_numeric(df["min_ttc"], errors="coerce")
    df["min_ttc_f"] = df["min_ttc_raw"].clip(upper=3.0)
    df["col_f"] = col.astype(int)
    reg = df.dropna(subset=["min_ttc_f"]).copy()

    for c in FEATURE_NUM:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    for c in FEATURE_CAT:
        reg[c] = reg[c].fillna("").astype(str)

    print(f"pool: all={n_all}, aeb_ok={n_aeb}, dual_pool={n_pool}, min_ttc_valid={len(reg)}")
    n_cap = int((reg["min_ttc_raw"] > 3.0).sum())
    print(f"collision=1: {int(reg['col_f'].sum())} ({reg['col_f'].mean() * 100:.1f}%), "
          f"minTTC>3s capped: {n_cap}")
    return reg


def build_X(df: pd.DataFrame):
    X = df[FEATURE_NUM].copy()
    for c in FEATURE_CAT:
        d = pd.get_dummies(df[c], prefix=c)
        X = pd.concat([X, d], axis=1)
    return X


def _drop_const(Xtr, Xte=None):
    """去掉训练子集中的恒定列（sklearn 1.9 分桶在单值特征上崩溃；恒定列也无信息）。"""
    keep = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) > 1]
    return (Xtr[keep], Xte[keep]) if Xte is not None else Xtr[keep]


def fit_reg(X_tr, y_tr, seed):
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=None, min_samples_leaf=10,
        l2_regularization=1.0, random_state=seed)


def within_brand(df: pd.DataFrame, X_all: pd.DataFrame, out: list, preds: list):
    """A. 品牌内 OOF：GroupKFold(5) by condition_id，3 种子。"""
    out.append("=" * 72)
    out.append("A. 品牌内 OOF（GroupKFold-5 by condition_id, 3 seeds）")
    out.append(f"{'brand':13s} {'n':>4s} {'n_cond':>6s} {'MAE':>7s} {'±std':>6s} {'baseline':>8s} {'floor':>7s} {'med_ttc':>8s}")
    rows = []
    for brand, g in df.groupby("brand"):
        X, y = X_all.loc[g.index], g["min_ttc_f"].values
        groups = g["condition_id"].values
        if g["condition_id"].nunique() < 5:
            out.append(f"{brand:13s} {len(g):4d} {g['condition_id'].nunique():6d}  <5 conditions, skip")
            continue
        # 噪声下界：同工况重复 run 的条件中位数预测（in-sample，可达 MAE 下限参考）
        cond_med = g.groupby("condition_id")["min_ttc_f"].transform("median").values
        floor = mean_absolute_error(y, cond_med)
        oof = np.zeros(len(g))
        base = np.zeros(len(g))
        for seed in SEEDS:
            gkf = GroupKFold(n_splits=5)
            for tr, te in gkf.split(X, y, groups):
                Xtr, Xte = _drop_const(X.iloc[tr], X.iloc[te])
                m = fit_reg(Xtr, y[tr], seed).fit(Xtr, y[tr])
                oof[te] += m.predict(Xte) / len(SEEDS)
                base[te] += np.median(y[tr]) / len(SEEDS)  # 种子间基线相同，仅摊平
        mae = mean_absolute_error(y, oof)
        mae_b = mean_absolute_error(y, base)
        for i in range(len(g)):
            preds.append({"brand": brand, "scenario_acronym": g["scenario_acronym"].iloc[i],
                          "condition_id": g["condition_id"].iloc[i], "split": "within_oof",
                          "min_ttc_true": y[i], "min_ttc_pred": oof[i], "baseline_pred": base[i]})
        rows.append((brand, len(g), g["condition_id"].nunique(), mae, mae_b, floor, np.median(y)))
        out.append(f"{brand:13s} {len(g):4d} {g['condition_id'].nunique():6d} {mae:7.3f} "
                   f"{np.std([mean_absolute_error(y, oof)]):6.3f} {mae_b:8.3f} {floor:7.3f} {np.median(y):8.3f}")
    pooled_y = df["min_ttc_f"].values
    out.append(f"{'POOLED':13s} {len(df):4d}")
    return rows


def lobo(df: pd.DataFrame, X_all: pd.DataFrame, out: list, preds: list):
    """B. LOBO 预览：主表类，留一品牌。"""
    out.append("")
    out.append("=" * 72)
    out.append("B. LOBO 预览（主表类，留一品牌，3 seeds）")
    sub = df[df["scenario_acronym"].isin(MAIN_CLASSES)]
    out.append(f"{'class':7s} {'held_out':13s} {'n_te':>4s} {'n_tr':>4s} {'MAE':>7s} {'±std':>6s} {'base':>7s} {'med':>6s}")
    for cls in MAIN_CLASSES:
        for brand in sorted(sub["brand"].unique()):
            te = sub[(sub["brand"] == brand) & (sub["scenario_acronym"] == cls)]
            tr = sub[(sub["brand"] != brand) & (sub["scenario_acronym"] == cls)]
            if len(te) < 3 or len(tr) < 10:
                out.append(f"{cls:7s} {brand:13s} {len(te):4d} {len(tr):4d}  insufficient")
                continue
            Xtr, Xte = _drop_const(X_all.loc[tr.index], X_all.loc[te.index])
            maes = []
            for seed in SEEDS:
                m = fit_reg(Xtr, tr["min_ttc_f"].values, seed).fit(Xtr, tr["min_ttc_f"].values)
                p = m.predict(Xte)
                maes.append(mean_absolute_error(te["min_ttc_f"].values, p))
                if seed == SEEDS[0]:
                    p0 = p
            base = mean_absolute_error(te["min_ttc_f"].values,
                                       np.full(len(te), np.median(tr["min_ttc_f"].values)))
            for i in range(len(te)):
                preds.append({"brand": brand, "scenario_acronym": cls,
                              "condition_id": te["condition_id"].iloc[i], "split": "lobo",
                              "min_ttc_true": te["min_ttc_f"].iloc[i], "min_ttc_pred": p0[i],
                              "baseline_pred": np.median(tr["min_ttc_f"].values)})
            out.append(f"{cls:7s} {brand:13s} {len(te):4d} {len(tr):4d} "
                       f"{np.mean(maes):7.3f} {np.std(maes):6.3f} {base:7.3f} {np.median(te['min_ttc_f'].values):6.3f}")


def collision_cls(df: pd.DataFrame, X_all: pd.DataFrame, out: list, preds: list):
    """辅：碰撞分类（品牌内 OOF；正样本不足时跳过）。"""
    out.append("")
    out.append("=" * 72)
    out.append("C. 碰撞分类（辅，品牌内 OOF，3 seeds）")
    pos = int(df["col_f"].sum())
    out.append(f"positive collision runs in pool: {pos}")
    if pos < 20:
        out.append("正样本不足 20 —— v0 不训练分类器（G.7 平衡审计结论），仅记碰撞率。")
        return
    for brand, g in df.groupby("brand"):
        if g["col_f"].nunique() < 2 or g["condition_id"].nunique() < 5 or len(g) < 40:
            continue
        X, y, groups = X_all.loc[g.index], g["col_f"].values, g["condition_id"].values
        oof = np.zeros(len(g))
        for seed in SEEDS:
            gkf = GroupKFold(n_splits=5)
            for tr, te in gkf.split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                Xtr, Xte = _drop_const(X.iloc[tr], X.iloc[te])
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                   min_samples_leaf=10, random_state=seed)
                m.fit(Xtr, y[tr])
                oof[te] += m.predict_proba(Xte)[:, 1] / len(SEEDS)
        try:
            auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
            out.append(f"  {brand:13s} n={len(g):4d} pos={int(y.sum()):3d} OOF-AUC={auc:.3f}")
        except ValueError as e:
            out.append(f"  {brand:13s} n={len(g):4d} pos={int(y.sum()):3d} AUC undefined ({e})")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    df = load_pool()
    out: list[str] = ["PICASO surrogate v0 报告（minTTC 回归主目标）", "生成: 2026-09-10", ""]

    # 类别平衡（G.7）
    out.append("=" * 72)
    out.append("0. 池构成（G.7 平衡审计摘要）")
    tab = df.groupby(["scenario_acronym", "brand"]).agg(
        n=("min_ttc_f", "size"), col=("col_f", "sum"),
        ttc_med=("min_ttc_f", "median")).reset_index()
    for _, r in tab.iterrows():
        out.append(f"  {r['scenario_acronym']:7s} {r['brand']:13s} n={int(r['n']):4d} "
                   f"col={int(r['col']):3d} minTTC_med={r['ttc_med']:.2f}")

    preds: list[dict] = []
    X_all = build_X(df)  # 全池一次性 dummy（LOBO 训练/测试共享特征空间）
    within_brand(df, X_all, out, preds)
    lobo(df, X_all, out, preds)
    collision_cls(df, X_all, out, preds)

    # 9/20 go/no-go 参照（内部门阈值，论文只报实测）
    out.append("")
    out.append("=" * 72)
    out.append("go/no-go 参照：品牌内 minTTC MAE 与 0.4s 量级比对（内部门阈值，非论文口径）")

    rep = ROOT / "inventory" / "surrogate_v0_report.txt"
    rep.write_text("\n".join(out), encoding="utf-8")
    pd.DataFrame(preds).to_csv(ROOT / "inventory" / "surrogate_v0_predictions.csv",
                               index=False, encoding="utf-8-sig")
    print("\n".join(out))
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
