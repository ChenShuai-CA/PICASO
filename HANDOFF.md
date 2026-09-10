# PICASO 项目交接文档（Handoff）

> **日期**：2026-09-10（更新）
> **状态**：2026-09-01 叙事重构 + M1 盘点已提交（`7dae27e`）；2026-09-10 可行性评审完成，P1–P5 修订已写入 Stage3/4；surrogate v0 冲刺进行中
> **目标**：Q1 SCI（非开源期刊）。**硬 deadline（2026-09-10 用户裁定）：2026-09-30 前完成论文初稿 + 约 80% 实验；2026-12-31 前投出**，首选 IEEE TIV / T-ITS

---

## 〇、2026-09-10 可行性评审结论（最新，与后文冲突处以本节为准）

**判定：方案方向可行、具备 Q1 潜力、数据底账属实（1,148/497/252/22 独立复算一致）、叙事可防守。** 评审发现两个前提性缺口 + 三个范围/设计问题，当日修复：

| # | 问题 | 状态 |
|---|------|------|
| P1 | Stage4 有效性规则 `SR path abort=1 → 整 Run 作废` 会系统性剔除碰撞样本（碰撞 ⇒ abort，surrogate 将无正样本可学） | ✅ Stage4 §3.3 双池双规则 + 附录 G 多证据碰撞判定 |
| P2 | 三个结果标签（碰撞/minTTC/AEB 触发时刻）尚未提取 + LOBO 诚实范围未显式化 | ✅ Stage4 附录 G 标签提取规范 + §4.1 LOBO 主实验类清单（主表 6 类：CCRs/CPTA/CCFT/CSTA/LKA/CPLA；E8 规程 run 仅 25）+ surrogate 特征含车辆物理量（不用品牌 one-hot） |
| P3 | M4 门统计功效（n≈20，80% 一致率 CI≈[56%,93%]）+ G1 复现组时间漂移混杂 | ✅ 附录 F 门规则预写（点估计 ≥80% 且 Wilson CI 下限 ≥60% 且 minTTC MAE ≤0.3s）+ G1 重复性基线（复测 vs 历史原始 run 三向比对）+ G1 边界选点原则 |
| P4 | 生成器/surrogate/ABD .spec 三层参数化不同构 | ✅ Stage3 新增 §2.3 可执行场景空间 E=(c,P,V,τ,O) |
| P5 | 文档一致性（域标签 AY5T 残留 / coverage 混口径 / 时间线矛盾） | ✅ Stage3 §3.1/§3.3 对齐 Stage4；coverage_matrix 拆 protocol/function 两表；时间线按本节 |

**压缩时间线范围决策**：生成器主线 = **B2+**（规程语法约束合成 + 物理投影 + KFR 审计，输出定义在 E 空间）；Mamba+Waymo 预训练为限时 stretch（9/20 检查点：KFR/多样性不胜 B2+ 则砍）；MACC 首轮为参数空间 do-干预（MACC-lite）；FalseReaction 77 负样本 v0 不并入。**实车补测窗口（10–11 月）是 AI 无法加速的外部关键路径，须本周申请。**

完整评审与执行计划：`C:\Users\chens\.claude\plans\handoff-md-wise-platypus.md`

---

## 一、一句话叙事（当前论文故事线）

> **面向量产 ADAS 的物理可执行安全关键场景生成 + 多品牌真实车辆响应闭环验证**：
> 生成器（Waymo/INTERACTION 预训练保自然性 + C-NCAP 规程语法与 ABD 执行包络硬约束）产出超越标准矩阵的新场景 → VUT 响应 surrogate（场景参数 → 碰撞 / minTTC / AEB 触发时刻，含不确定度）预测危险度并排序 → **ABD 实车补测验证** → 结果反哺 surrogate（主动学习闭环）→ 反事实归因定位失效边界并经实车确认。

**PICASO 命名保持缩写**，展开改为：*Physics-Informed Causal Adversarial Scenario generation with real-wOrld validation*。

---

## 二、用户已确认的四个关键约束（决策前提，勿再推翻）

1. **具备完整实车补测能力**：ABD 机器人 + 场地 + 车辆可调度，能执行生成器产出的规程外新场景并记录真实响应。
2. **ABD 数据还会再加 5 款以上车型**（现有 A66 / E8 / S9 / P7+ 四款，最终约 9–10 款）。
3. **期刊要求 Q1 且非开源**：T-ITS / TIV / TR-C / AAP 均可（均 hybrid）；排除 IEEE Access / Sensors 类 OA。
4. **6 个月内投出，算力可租且充足**——时间是唯一硬约束。

---

## 三、2026-09-01 方案评审核心结论（为什么改叙事）

原方案"开放道路→封闭场地 UDA（GRL-DANN）"是最脆弱环节：

- C-NCAP 目标域是**离散规程网格、条件内方差近零**，经典 UDA 没有可适配的分布；
- 审稿人必杀一问："规程 PDF 已完整定义目标分布，为何需要域适应？"；
- ABD 数据真正的价值是**真实量产车系统响应真值**（AEB/FCW 触发时刻、碰撞结果）+ 实车补测能力——这是 STRIVE/AdvSim/CTG++/CounterScene 全部不具备的。

**模块新角色**：C1 PI-Causal Mamba 保留为生成主干；C2 MACC 反事实升级为明星贡献（surrogate 上搜边界 → 实车验证翻转）；C3 GRL-DANN **降级为跨品牌 surrogate 的可选校准工具**（LOBO 评估），不进标题级叙事。

详细论证见 `C:\Users\chens\.claude\plans\abd-data-c-ncap-sci-typed-turing.md`。

---

## 四、数据真相（M1 盘点实测，非估算）

管线：`abd_inventory.py`（纯标准库）→ `inventory/all_runs_inventory.csv`（1,148 run 全量）+ `inventory/coverage_matrix.csv`。

| 数据池（data_role） | runs | 用途 |
|---|---|---|
| **protocol（C-NCAP 规程）** | **497** | **252 个独立工况 / 22 类规程**；生成评估 + surrogate 训练主池 |
| function_test（企标功能项） | 179 | A66 的 ACC/ICA/TJA/ILC/RCW；可供 surrogate 训练 |
| surrogate_negative（FalseReaction） | 77 | "不应制动"负样本池（surrogate 边界建模） |
| engineering_envelope（调参/标定/预热） | 392 | 仅用于 ABD 执行包络标定 |
| unclassified | 3 | E8 `4-Cal/61-Learn`，待人工确认 |

**M1 门：PASS**（规程有效 run 497 ≥ 100；规程类别 22 ≥ 6）。

数据形态事实：ABD `.txt` = 横幅 + `Points=N` + 通道名（385–415 列，Tab 分隔）+ 单位行 + 数据；**100 Hz**，时长 13–51 s；同名 `.spec/.log/.CRUN` 配套；含 Motion Pack 位姿、相对运动、TTC、触发通道、机器人控制通道。车辆物理参数（质量/轴距等）见各车 `ExpInfo.txt`。

品牌覆盖不对称（写作时注意措辞）：
- **A66 最全**（ADAS+VRU+功能测试 510 runs）；**E8 极度偏科**（CCRs 130/195，无 VRU）；**S9 较均衡**；**P7+** 夹带大量 Conditioning/标定 run（已在盘点中分流）。
- 无任一品牌有完整 VRU+ADAS 全矩阵 → 跨品牌结论按规程类型分层，缺失处只做案例分析。

---

## 五、本次会话已完成的修改（未提交）

| 文件 | 修改要点 |
|---|---|
| `Stage1_研究问题凝练.md` | 顶部新增 2026-09-01 修订块（优先级最高）；SQ3/断裂3/根因句/§4.3 C3/§六贡献3 改写为实车验证口径；删"完全空白/首发优势窗口" |
| `Stage2_综合差距分析与Related_Work.md` | 新增修订块；"完全空白"→"未直接覆盖"；CR 25–35%、KFR>95% 等预承诺数值→"实验后填报"；§4.4 英文 Related Work 的 critical gap 改写；C3 两处重定位；TDPD 统一定义并降级；新增实车一致性指标条目 |
| `Stage2_相关工作梳理与文献调研.md` | §5.2 蓝海预判表软化绝对化表述并加重定位注记 |
| `Stage3_方法与系统设计_PICASO.md` | 标题改名；新增修订块；架构图 Layer 2→"跨品牌响应校准层（可选）"、Layer 3 物理约束口径"解码投影为主"；§2.2 新增 Stage D 实车闭环；§四 Layer 2 降级声明；§5.3.5 删"首个"；§六 MACC 补实车支撑声明；§7.1 删全部预承诺数值 + 新增 D6 实车一致性指标族；§7.2 消融改非承诺口径 + 新增 A9；§8.2 Phase 2 改为 surrogate 训练；页脚创新点确认更新 |
| `Stage4_数据处理与实验方案.md` | 新增修订块；FalseReaction 改为**分流**（生成侧剔除 / surrogate 负样本保留，4 处）；§4.1 切分图重画；新增基线 B6（无 ABD 校准对照）；§4.4 新增实车一致性指标；**新增附录 F：ABD 实车补测协议**（首批 ≥20 场景：G1 规程内复现 ≥8 + G2 规程外新点 ≤20% 外推 ≥12，含安全中止与反哺规则） |

一致性 grep 验证已通过：无"完全空白 / Blue Ocean / 首个在 / KFR>95 / 25–35%"残留（历史修订块除外，均有新块覆盖）；TDPD 定义三处唯一。

---

## 六、红线（写作与实验纪律，历次评审累积）

1. **不预承诺数值**：内部门阈值仅作 go/no-go，论文结果只报实测值 + 95% CI。
2. **不用绝对化表述**："首个/空白/零论文"一律改为 "to the best of our knowledge after systematic search"，投稿前按 IEEE Xplore/Scopus/WoS/Semantic Scholar/OpenAlex/arXiv/GS 复核。
3. **arXiv 预印本不写成已录用会议/期刊**；SOTA 数字引用前回原文核验（Stage2 数字多来自 Gemini/Copilot，未逐条核验）。
4. **ABD 原始数据不外传**；论文中品牌匿名化（Brand A–J），只报聚合统计；Waymo/INTERACTION 管线全开源以支撑可复现性。
5. 不把"GRU→Mamba 替换"当创新点；Mamba 只是组件。
6. 统计：3–5 种子、paired/bootstrap、效应量 + CI、Holm-Bonferroni/FDR 多重校正；不预设全 p<0.05。

---

## 七、压缩路线图与 Go/No-Go 门（2026-09-10 裁定：9/30 初稿+80% 实验，12/31 投稿）

| 时间窗 | 任务 | 门 |
|---|---|---|
| ✅ M1 完成（9/5） | ABD 盘点 + inventory | **PASS**（497 规程 run / 252 工况 / 22 类） |
| ✅ 9/10 | 快照提交（`7dae27e`）+ 文档修订（P1–P5 落稿 Stage3/4） | — |
| 9/10–9/13 | **用户发起实车窗口申请（10–11 月，G1≥8 + G2≥12）** | 窗口锁定 = 外部关键路径 |
| 9/13–9/20 | **surrogate v0 冲刺**：参数提取器 + 标签提取器（附录 G → `labels_v0.csv` + 20 run 人工校验）+ T0 定位 + GBM v0 品牌内 holdout → LOBO 预览（仅主表 6 类） | **9/20 检查点**：品牌内 minTTC MAE 远差于 0.4s 量级 → 新叙事重新评估；stretch（租 A100）Mamba 生成器同日 KFR/多样性 vs B2+ 定去留 |
| 9/20–9/30 | B2+ 合成管线（E 空间语法约束采样 + 物理投影 + KFR 审计）+ MACC-lite 边界点 + 最小消融（B2/B6/A9）+ **论文初稿** | **9/30：初稿 + 80% 实验**（全部历史数据实验；Results 只填实测值，占位标注待补） |
| 10–11 月 | **首批实车补测**（附录 F，含重复性基线）+ sim-to-real 一致性 + 反哺 surrogate | M4 门按附录 F 预写规则：点估计 ≥80% 且 CI 下限 ≥60% 且 minTTC MAE ≤0.3s；窗口滑过 11 月中 → 降级预案（holdout 口径 + TR-C/TVT） |
| 12 月 | 完整统计（3–5 种子/bootstrap/效应量+CI/Holm）+ 优先权检索复核 + 投稿（TIV/T-ITS） | 3 核心消融 + CI 齐全 |

**M4 成败即论文成败**；9/30 复盘点：若初稿+80% 未达成，决策 = 顺延（1 月投）或砍实车批（降级口径），不硬凑。

---

## 八、下一步可立即执行的任务（2026-09-10 重排）

1. **surrogate v0 冲刺（关键路径，9/13–9/20）**：参数提取器（condition 目录名 + .spec → 场景参数表）→ 标签提取器（按 Stage4 附录 G → `labels_v0.csv` + 20 run 人工校验 + 类别平衡审计）→ T0 定位（Stage4 §2.6 多策略）→ GBM v0（minTTC 回归为主、碰撞为辅、特征含车辆物理量）→ 品牌内 holdout → LOBO 预览（仅主表 6 类）。**失败则叙事重新评估，先于一切生成器工作。**
2. **实车窗口申请（用户执行，本周发起）**：10–11 月场地/车辆档期；批规模按附录 F（G1≥8 + G2≥12）；G1 按"历史低 minTTC 边界工况"选点原则。
3. **9/20–9/30**：B2+ 合成管线 + MACC-lite（E 空间 do-干预边界点，G2 送测核心来源）+ 最小消融（B2 朴素扫掠 vs B2+、B6 无 ABD 校准、A9 随机选点 vs surrogate 选点）+ **初稿**。
4. **stretch（租 A100 并行）**：Mamba 生成器 + Waymo 预训练；9/20 按 KFR/多样性 vs B2+ 决定去留。
5. **投稿前检索复核**（"实车验证闭环"优先权）：12 月执行，不阻塞。

---

## 九、关键文件索引

| 文件 | 作用 |
|---|---|
| `C:\Users\chens\.claude\plans\abd-data-c-ncap-sci-typed-turing.md` | 本次方案评审完整版（可行性论证、修改清单、风险登记册） |
| `abd_inventory.py` | M1 盘点管线（重跑：`python abd_inventory.py`） |
| `inventory/all_runs_inventory.csv` | 1,148 run 全量清单（brand/role/工况/条款号/时长/采样率/有效性） |
| `inventory/coverage_matrix.csv` | 规程类 × 品牌覆盖矩阵（**protocol 池口径**，工况数 / run 数；function_test 另见 `coverage_matrix_function_test.csv`，2026-09-10 拆分） |
| `Stage1–4 *.md` | 研究方案（各文件顶部 2026-09-01 修订块为当前有效口径） |
| `四阶段方案可行性审查与修改理由_2026-05-28.md` | 上一轮评审（MVP 分层、红线来源） |
| `Data/ABD_Data/` | 四车 ABD 实测数据（6.6 GB） |
| `Data/ABD_Data/C-NCAP_2024/` | 附录 L / 附录 O / 管理规则 PDF |
| `Data/Waymo`、`Data/INTERACTION` | 公开数据集（生成器预训练用） |

## 十、给下一个会话的最短上手指引

1. 读本文件**〇节（2026-09-10 评审结论）与七节（压缩路线图）** → 读 Stage1/3/4 顶部最新修订块（2026-09-10 > 2026-09-01 > 2026-05-28）。
2. **硬 deadline：9/30 初稿+80% 实验，12/31 投稿**；关键路径 = surrogate v0（标签按 Stage4 附录 G 提取）。
3. 任何与"开放道路→封闭场地 UDA"相关的旧表述均以新叙事为准；不要复活 GRL-DANN 核心地位。
4. 数值一律不预承诺；新增声明一律走"据当前检索"口径。
5. 数据问题先查 `inventory/all_runs_inventory.csv`，不要重新手数文件；结果标签查 `inventory/labels_v0.csv`（生成后）。
