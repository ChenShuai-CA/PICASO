#!/usr/bin/env python3
"""少样本校准 LOBO：回答"给新车 4~5 个场景数据，能否预测该车的失效场景"。

设计：对每个 (主表类 × 品牌)：
  zero-shot：仅其他品牌同类数据训练 -> 预测该品牌剩余工况
  few-shot K：其他品牌 + 该品牌 K 个工况（含其重复 run）训练 -> 预测剩余工况
  （工况级划分，测试工况与校准工况不重叠；3 种子 × 随机抽 K 工况）
对照基线：训练集中位数。

输出：inventory/fewshot_lobo_report.txt
用法：python abd_parser/fewshot_lobo.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import surrogate_v0 as sv  # noqa: E402  (复用 load_pool/build_X/_drop_const/SEEDS)


def fit(X, y, seed):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         min_samples_leaf=10, l2_regularization=1.0,
                                         random_state=seed).fit(X, y)


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    df = sv.load_pool()
    X_all = sv.build_X(df)

    out = ["少样本校准 LOBO：K 工况校准 vs zero-shot（minTTC 回归，3 种子）", "",
           "方法：zero = 仅其他品牌训练；few = 其他品牌+K 工况并入 GBM 训练；",
           "      resid = zero-shot 预测 + K 工况残差中位数偏移（少样本校准标准做法）；",
           "      rho = 预测-真值 Spearman 排序相关（选失效场景能力），报 zero/resid。", ""]
    out.append(f"{'class':7s} {'brand':13s} {'K':>2s} {'n_te':>4s} | {'zero':>6s} {'few':>6s} {'resid':>6s} {'base':>6s} | {'rho0':>5s} {'rhoR':>5s}")
    results = []
    for cls in ["CCRs", "CPTA", "CCFT", "CSTA"]:
        sub = df[df["scenario_acronym"] == cls]
        for brand in sorted(sub["brand"].unique()):
            own = sub[sub["brand"] == brand]
            others = sub[sub["brand"] != brand]
            own_conds = sorted(own["condition_id"].unique())
            if len(own) < 8 or len(own_conds) < 6 or len(others) < 10:
                continue
            for K in (2, 4):
                zeros, fews, resids, bases, rho0s, rhoRs = [], [], [], [], [], []
                for seed in sv.SEEDS:
                    rng = random.Random(seed * 31 + K)
                    cal_conds = rng.sample(own_conds, K)
                    cal = own[own["condition_id"].isin(cal_conds)]
                    te = own[~own["condition_id"].isin(cal_conds)]
                    if len(te) < 4:
                        continue
                    Xte, yte = X_all.loc[te.index], te["min_ttc_f"].values

                    # zero-shot
                    Xtr0, Xte0 = sv._drop_const(X_all.loc[others.index], Xte)
                    m0 = fit(Xtr0, others["min_ttc_f"].values, seed)
                    p0 = m0.predict(Xte0)
                    zeros.append(mean_absolute_error(yte, p0))

                    # few-shot（朴素并入训练）
                    tr = pd.concat([others, cal])
                    Xtr1, Xte1 = sv._drop_const(X_all.loc[tr.index], Xte)
                    fews.append(mean_absolute_error(yte, fit(Xtr1, tr["min_ttc_f"].values, seed).predict(Xte1)))

                    # 残差校准：K 工况上的系统偏差 -> 偏移量
                    _, Xcal = sv._drop_const(X_all.loc[others.index], X_all.loc[cal.index])
                    c = float(np.median(cal["min_ttc_f"].values - m0.predict(Xcal)))
                    pr = p0 + c
                    resids.append(mean_absolute_error(yte, pr))

                    bases.append(mean_absolute_error(yte, np.full(len(te), np.median(others["min_ttc_f"].values))))
                    from scipy.stats import spearmanr
                    if len(np.unique(yte)) > 1:
                        rho0s.append(spearmanr(yte, p0).statistic)
                        rhoRs.append(spearmanr(yte, pr).statistic)
                if not zeros:
                    continue
                z, f, r_, b = np.mean(zeros), np.mean(fews), np.mean(resids), np.mean(bases)
                r0 = np.mean(rho0s) if rho0s else float("nan")
                rR = np.mean(rhoRs) if rhoRs else float("nan")
                te_n = int((~own["condition_id"].isin(cal_conds)).sum())
                out.append(f"{cls:7s} {brand:13s} {K:2d} {te_n:4d} | {z:6.3f} {f:6.3f} {r_:6.3f} {b:6.3f} | {r0:5.2f} {rR:5.2f}")
                results.append((cls, brand, K, z, f, r_, b, r0, rR))

    # 汇总
    out.append("")
    out.append("=" * 72)
    for K in (2, 4):
        rows = [r for r in results if r[2] == K]
        if rows:
            zm = np.mean([r[3] for r in rows])
            fm = np.mean([r[4] for r in rows])
            rm = np.mean([r[5] for r in rows])
            bm = np.mean([r[6] for r in rows])
            rho0 = np.nanmean([r[7] for r in rows])
            rhoR = np.nanmean([r[8] for r in rows])
            out.append(f"K={K}: zero {zm:.3f} | few {fm:.3f} | resid {rm:.3f} | base {bm:.3f} "
                       f"|| rho(zero) {rho0:.2f} -> rho(resid) {rhoR:.2f}（{len(rows)} 格子）")
    rep = ROOT / "inventory" / "fewshot_lobo_report.txt"
    rep.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
