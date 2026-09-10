# Stage 3：方法与系统设计 — Physics-Informed Causal Adversarial Scenario generation with real-wOrld validation (PICASO)

> 依据 `自动驾驶安全关键场景生成方向：工程科研流程与 T-ITS 投稿指南.docx` Stage 3 框架编写
> 基于 `Stage2_综合差距分析与Related_Work.md` 的差距分析矩阵
> 编写日期：2026-05-21

---

## 2026-09-10 评审修订（压缩时间线与范围决策，优先级高于以下所有版本）

用户裁定 deadline：**2026-09-30 前完成论文初稿 + 约 80% 实验；2026-12-31 前投出**。2026-09-10 可行性评审结论与依据见 Stage4 同日修订块；本文件按此执行以下范围决策：

1. **生成器降范围**：主线 = **B2+**（规程语法约束采样 + 解码后物理投影 + KFR 审计 + surrogate 排序，输出定义在 §2.3 可执行场景空间 E 上）；PI-Causal Mamba + Waymo/INTERACTION 预训练降为**限时并行 stretch**（租 A100；**9/20 检查点**：在 KFR 与轨迹形状多样性上不胜 B2+ 则移出主线，仅作附录/未来工作）。Waymo 预训练定位为自然性正则（B6 消融检验增益，不进核心主张）。
2. **新增 §2.3 可执行场景空间 E**：生成器 / surrogate / ABD .spec 三层共享参数化（修复"生成轨迹 vs surrogate 参数 vs .spec 转换"三者不同构的评审问题 P4）。
3. **域标签统一**：§3.1/§3.3 与 Stage4 §5.2 对齐（2=S9, 3=E8, 4=P7+, 5=A66），删除 AY5T 陈旧残留（当前四车数据中无此车型）。
4. **MACC 首轮执行参数空间 do-干预版本（MACC-lite）**：在 surrogate 上对 E 的参数做最小有界干预、搜索结果翻转边界点，供附录 F 的 G2 组送测；多智能体级联反事实为 stretch。
5. §9.4 里程碑时间表由 HANDOFF 2026-09-10 版压缩路线图取代（按周倒排）。

---

## 2026-09-01 方案评审修订：叙事重心调整（执行优先级高于以下所有版本）

1. **C3 重定位**：GRL-DANN"开放道路→封闭场地 UDA"不再是核心贡献——C-NCAP 目标域是离散规程网格、条件内方差近零，经典 UDA 没有可适配的分布。新 C3 = **跨品牌 VUT 响应 surrogate + 实车校准**（LOBO 泛化评估；GRL-DANN/CORAL-MMD/DG 仅作可选对照）。本文 Layer 2 技术内容保留为可选模块文档。
2. **新增 Stage D 实车验证闭环**（见 §2.2）：surrogate 判定高危险且规程矩阵未覆盖的生成场景 → ABD 实车补测 → sim-to-real 一致性（新增 §7.1 D6 指标族）→ 结果反哺 surrogate。已确认具备完整实车补测条件，车型将扩至 9–10 款。
3. **Layer 3 物理约束口径**：架构图与正文统一为"解码后投影 + 物理损失 + KFR 审计"为 MVP 主线；多体 PHNN 硬嵌入仅为增强路线，未验证前不得出现"硬保证"表述。
4. **指标卫生**：§7.1 全部预承诺数值（KFR>95%、CR 25–35%、NMR>30%、minSTTC<1.2s、PET<1.5s、各违反率阈值）改为"实验后填报"；TDPD 定义统一为 `(CR_target − CR_source)/CR_source` 并降级为诊断指标。
5. **命名**：PICASO 展开改为 *Physics-Informed Causal Adversarial Scenario generation with real-wOrld validation*。
6. 详细论证见 `Stage1_研究问题凝练.md` 2026-09-01 修订块。

---

## 2026-05-28 专家审查修订：实施版范围收敛

本方法框架的学术叙事成立，但原设计把多个高风险模块同时设为主路径，工程上不可控。后续实施按以下优先级执行，后文与本节冲突时以本节为准。

### 总体结论

PICASO 应被定义为**自动驾驶安全验证框架**，而不是单纯“PI-Causal Mamba 模型”。首篇论文必须先形成可复现实验闭环：数据解析 -> 统一 Schema -> 生成/干预 -> 物理可执行性评估 -> 目标域验证 -> 消融与统计。Mamba、PHNN、GRL-DANN 是实现组件，不是论文唯一卖点。

### MVP 与增强版分层

| 层级 | 必须完成 | 目的 | 未完成时的论文处理 |
|------|----------|------|--------------------|
| MVP-0 数据层 | Waymo/INTERACTION/ABD 统一 Schema，ABD run inventory，T0 与场景标签可复核 | 保证实验可信 | 不进入模型训练 |
| MVP-1 生成基线 | Mamba 或 Transformer 轨迹生成基线 + 随机/规则/扰动基线 | 建立可比较结果 | 不声称 SOTA |
| MVP-2 物理可执行性 | 解码后可微投影、速度/加速度/曲率/jerk/摩擦圆约束、KFR 与违反类型统计 | 解决“危险但不可执行”问题 | PHNN 内嵌主张降级 |
| MVP-3 反事实干预 | 关键 Agent/时间窗归因 + 有界反事实扰动 + 风险变化一致性 | 支撑因果解释叙事 | 只写反事实敏感性分析 |
| MVP-4 跨品牌校准 | no-DA、CORAL/MMD、GRL-DANN 或 DG 对比；按品牌/规程 holdout（LOBO） | 支撑跨品牌 VUT 响应 surrogate 泛化 | 若数据不足，只写跨品牌诊断 |
| Enhanced 多体 PH/MACC | 多体 PHNN/PHDAE、级联干预、Shapley-in-the-loop | 冲刺高创新 | 作为消融/附录/第二篇论文 |

### 关键技术修订

1. **PH 约束不再承诺隐藏空间硬保证**：首轮实现以“解码后投影 + 物理损失 + ABD 执行约束 + KFR 审计”为主。Port-Hamiltonian 模块可以作为增强实验，但不得在未证明前写“自然保证/100%可行/硬约束”。
2. **因果发现不等于因果识别**：从轨迹中学到的图先定义为“causal interaction proxy / intervention graph”。只有在反事实干预、规程逻辑、人工复核共同支持时，才能提升到因果解释。
3. **域自适应必须有 fallback**：CNCAP 目标域样本少且分布规程化，GRL-DANN 可能过拟合或内容丢失。必须同时实现 no-DA、CORAL/MMD 或域泛化基线，不把 GRL-DANN 作为唯一成功路径。
4. **统计结论以置信区间和效应量为主**：不承诺所有 p<0.05。深度模型主实验 3-5 个随机种子即可；对场景级指标使用 bootstrap 置信区间和 paired test。
5. **基线分层**：必须复现/实现可控基线；对难复现的前沿方法只做概念级对比或标为不可复现，不用不可靠复现结果支撑核心结论。

---

## 一、创新点最终确认（基于 Stage 2 差距分析）

在 Stage 2 的 8 份调研报告交叉验证之后，正式确认以下三大创新点及一个新评估体系：

### C1: PI-Causal Mamba — 物理-因果统一状态空间生成器

**核心主张（修订）**：构建以 Mamba/SSM 为序列骨干的物理可审计、因果可解释生成器，将物理约束损失、解码后可微投影、交互图学习与反事实干预统一到安全关键场景生成 pipeline 中。多主体 Port-Hamiltonian 内嵌作为增强路线，不再作为首轮实现必须满足的硬保证。

**差异化证据链**：

| 比较对象 | 该方法做了 | 该方法没做 | → 我们的突破 |
|---------|-----------|-----------|------------|
| Pi-DiMT (ICRA 2026, 需投稿前核验) | PHNN+Mamba+Diffusion，强调物理可行性 | 主要面向单体/规划或重构；无封闭场地 DA | 物理可审计场景生成 + 目标域验证 |
| CounterScene (2026.03, arXiv) | BEV 世界模型+反事实安全评估 | 物理执行约束与封闭场地迁移不是重点 | 反事实干预 + ABD 执行约束 + 跨域评估 |
| Tamba (CVPR 2025) | Mamba轨迹预测SOTA，4.54M参数 | 非场景生成；无因果/物理 | 场景生成 + 因果 + 物理 |
| GEM (2026) | 变形Mamba LiDAR世界模型生成 | 无因果图；无物理硬约束 | 因果 + 物理嵌入 |
| CausalAF (CoRL 2023) | 因果自回归流+CVM先验 | 依赖专家DAG；非Mamba | 自适应因果发现 + Mamba |

### C2: MACC — 多智能体级联反事实离线闭环

**核心主张**：从 CounterScene 的**单变量干预**扩展到**多智能体因果级联图 + 序列反事实干预**，能够生成"前车A避障→中车B被迫变道→自车Ego追尾"等多层级联因果场景。使用模型自身 World Model 进行反事实推理（离线闭环）。

**差异化证据链**：

| 比较对象 | 该方法做了 | 该方法没做 | → 我们的突破 |
|---------|-----------|-----------|------------|
| CounterScene (2026.03) | 单变量 do(·) 干预，因果交互图CIG | 多变量级联干预；物理约束 | 多智能体因果级联图 + 序列干预 |
| SafeAlign-VLA (2026.05) | 反事实安全配对+GRPO对齐 | 非场景生成，仅策略对齐 | 反事实场景生成 |
| CausalVAD (CVPR 2026) | 后门调整去混淆 | Ego中心，非场景生成 | 场景级因果干预 |

### C3: 跨品牌 VUT 响应 surrogate 与实车校准（2026-09-01 起由"GRL-DANN 多源域自适应"重定位）

**核心主张（2026-09-01 修订）**：C-NCAP 目标域是离散规程网格、条件内方差近零，经典"开放道路→封闭场地"UDA 没有可适配的分布，不再作为核心贡献。本创新点改为：利用多品牌 ABD 实测数据训练 VUT 响应 surrogate（场景参数 → 碰撞 / minTTC / AEB 触发时刻，含 ensemble/conformal 不确定度），以 leave-one-brand-out 评估跨品牌泛化；GRL-DANN / CORAL-MMD / 域泛化仅作为可选校准对照。surrogate 判定高危险且规程矩阵未覆盖的生成场景经 ABD 实车补测验证，构成"生成→预测→实车验证→反哺"闭环。

**差异化证据链**：

| 比较对象 | 该方法做了 | 该方法没做 | → 我们的突破 |
|---------|-----------|-----------|------------|
| **当前检索范围内** | Sim-to-Real（仿真→真实）、Cross-Dataset（Waymo→nuScenes） | 尚未发现用多品牌真实车辆响应闭环验证生成场景的工作 | 先做系统验证，投稿前复核优先权 |
| NeuroNCAP (CVPR 2024) | Euro NCAP 规程对齐 | 仅nuScenes→Euro NCAP重建，无真实车辆响应闭环 | 实车响应 surrogate + 实车补测验证 |
| AdapTraj (ICDE 2024) | 轨迹预测多源DG | 非场景生成；非NCAP规程；无实车响应 | 跨品牌响应校准（LOBO） |

### 评估创新：四维评估矩阵 + 实车一致性指标族

- **KFR (Kinematic Feasibility Rate)**：场景轨迹全程满足物理约束的比例
- **Root-Cause Attribution Score**：基于 Shapley 值的失效归因分
- **Intervention Consistency (IC)**：反事实干预的风险变化一致性
- **实车一致性指标族（2026-09-01 新增，核心）**：surrogate 预测 vs ABD 实测的碰撞判定一致率、minTTC MAE、AEB 触发时刻 MAE
- **Target Domain Performance Drop (TDPD)**（降级为诊断指标）：统一定义为 `(CR_target − CR_source) / CR_source`

---

## 二、PICASO 总体系统架构

### 2.1 五层架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Layer 5: 评估与验证层                              │
│   四维指标矩阵 (保真度 / 安全关键性 / 物理可行性 / 因果可解释性)        │
│   + 消融实验 + 统计显著性检验 + 可视化分析                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ (evaluation signals)
┌───────────────────────────────┴─────────────────────────────────────┐
│                  Layer 4: MACC 反事实干预与场景搜索层                  │
│   ┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐    │
│   │ Shapley 归因  │ → │ do(·) 级联干预 │ → │ 反事实场景变体生成      │    │
│   │ 关键Agent定位 │   │ 最小有界扰动   │   │ 安全边界定向探索        │    │
│   └─────────────┘   └──────────────┘   └───────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ (causal graph + intervention targets)
┌───────────────────────────────┴─────────────────────────────────────┐
│               Layer 3: PI-Causal Mamba 核心生成器                     │
│   ┌───────────────────────┐   ┌──────────────────────────────┐      │
│   │  Port-Hamiltonian 约束  │   │   因果图发现 (CRiTIC 风格 CDN)    │      │
│   │(解码投影为主,硬嵌入为增强)│   │   (变长自适应 DAG 学习)           │      │
│   └───────────┬───────────┘   └──────────────┬───────────────┘      │
│               └───────────┬──────────────────┘                      │
│                           ▼                                         │
│         ┌─────────────────────────────────────┐                     │
│         │   Mamba Selective SSM 时序建模       │                     │
│         │   (O(N) 线性复杂度, 多主体联合解码)   │                     │
│         └─────────────────────────────────────┘                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ (domain-invariant features)
┌───────────────────────────────┴─────────────────────────────────────┐
│         Layer 2: 跨品牌响应校准层（可选: GRL-DANN / CORAL-MMD / DG）    │
│   ┌───────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│   │ 共享特征提取器 F_θ  │  │  域判别器 D_φ      │  │  梯度反转层 GRL │   │
│   │ (交互图 + 轨迹编码) │  │ (品牌A vs 品牌B…)  │  │  (∂L_D/∂θ → -λ) │   │
│   └───────────────────┘  └──────────────────┘  └────────────────┘   │
│   注：仅作跨品牌 VUT surrogate 校准对照；不承担开放道路→封闭场地迁移     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ (unified data schema)
┌───────────────────────────────┴─────────────────────────────────────┐
│                      Layer 1: 统一数据表示层                           │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│   │ Waymo Open Motion│  │   INTERACTION    │  │ CNCAP ABD (匿名化)│   │
│   │ (~487K 场景, 10Hz)│  │ (11 场景, ~55K 轨)│  │(4→9-10品牌,60+信号)│   │
│   └─────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流与训练流程

```
Stage A: 预训练 (Waymo + INTERACTION, 无 CNCAP)
  Input → Unified Schema → PI-Causal Mamba → L_recon + L_PH + L_causal
  Output: 预训练模型参数 θ_pre

Stage B: VUT 响应建模与跨品牌校准 (CNCAP ABD 实测响应, 2026-09-01 重定位)
  Input (场景参数 + 实测 VUT 响应: 碰撞/minTTC/AEB触发时刻; 含 FalseReaction 负样本)
    → surrogate 训练 (GBM/GP/小型MLP + ensemble/conformal 不确定度)
    → 可选跨品牌校准 (GRL-DANN / CORAL-MMD / DG), LOBO 评估
  Output: VUT 响应 surrogate θ_surr

Stage C: 反事实搜索 (冻结 θ_pre 与 θ_surr, MACC 激活)
  Factual Scenario → Shapley Attribution → Causal Graph → 
  do(·) Cascade Intervention → Counterfactual Scenarios → KFR Filter + surrogate 危险度排序
  Output: 安全关键场景库

Stage D: 实车验证闭环 (2026-09-01 新增)
  surrogate 判定高危且规程矩阵未覆盖的场景 → ABD 实车补测 →
  sim-to-real 一致性 (碰撞判定一致率 / minTTC MAE / AEB触发 MAE) →
  结果反哺 surrogate (主动学习)
  Output: 实车验证报告 + 更新后 θ_surr

Stage E: 评估验证
  生成场景库 → CARLA/nuPlan 闭环 → 四维指标矩阵 + 实车一致性 → 消融 + 显著性检验
```

### 2.3 可执行场景空间 E（2026-09-10 新增：生成器 / surrogate / ABD 执行的共享参数空间）

闭环链条"生成 → surrogate 排序 → 实车补测"要求三者共享同一参数化，否则（a）"神经生成器 vs 参数扫掠"对照不同构，（b）生成场景无法转换为 ABD .spec。定义：

```
E = (c, P, V, τ, O)
  c ∈ 规程类（CCRs / CPTA / CCFT / ...，含日/夜间）
  P = 目标路径几何参数（起点、轨迹形状、横向偏移/重叠率、曲率剖面）
  V = 速度剖面参数（目标初速、加减速度剖面、VUT 巡航速度）
  τ = 触发参数（触发方式、TTC 阈值 / 位置 / 时间容差）
  O = 多目标编排参数（目标数量、时序、相对相位；单目标场景退化为空）
```

**三层映射**：
1. 生成器输出轨迹 → 规约拟合到 E（**轨迹形状的连续多样性保留在 P/V 参数中——这是生成器区别于规程网格扫掠的全部价值所在**，也是与 B2 对照实验的核心差异量）；
2. surrogate 输入 = E + 车辆物理特征（质量/轴距/长宽；见 Stage4 §4.1 特征规范）；
3. E → ABD .spec（速度/路径/触发容差；Stage4 附录 F.3 参数化转换）。执行可行性由 **ABD 执行包络**过滤（包络从 engineering_envelope 392 run 标定）。

**对照实验同构性**：B2 朴素扫掠 = E 的网格/随机采样；B2+ = E 上的规程语法约束 + 物理投影 + KFR 审计；神经生成器（stretch）= E 上的条件分布学习。三者输出同一 E、同一评估口径。

---

## 三、Layer 1：统一数据表示层

### 3.1 数据统一 Schema

将 Waymo、INTERACTION、CNCAP ABD 三个异构数据源映射到统一的矢量化场景表示：

```
Scene = {Agent_i}_{i=1..N} ∪ {Lane_j}_{j=1..M} ∪ {TrafficControl_k} ∪ {Meta}

Agent_i:
  - id, type ∈ {vehicle, pedestrian, cyclist, GVT, ABD_robot}
  - history: {x_t, y_t, v_t, a_t, θ_t, κ_t}_{t=1..T_hist}  (10Hz)
  - future:  {x_t, y_t}_{t=1..T_fut}  (ground truth, if available)
  - domain_label ∈ {Waymo_US, INTERACTION_multi, CNCAP_CN}
  - brand_label ∈ {GAC_S9, GAC_E8, XPeng_P7+, GAC_A66, null}  (CNCAP only; 2026-09-10 与 Stage4 §5.2 对齐, 删 AY5T 残留)

Lane_j:
  - polyline: {x_k, y_k}_{k=1..K}
  - type ∈ {solid, dashed, curb, stop_line, crosswalk}
  - direction ∈ {forward, backward, bidirectional}

TrafficControl_k:
  - type ∈ {stop_sign, traffic_light, yield}
  - position, state

Meta:
  - scenario_type: CNCAP规程场景标签 (CCR, SCP, VRU_CPTA, etc.) or INTERACTION场景名
  - TTC_trigger (for CNCAP): TTC阈值 (1.4s, 1.9s, etc.)
  - ABD_precision: {lateral: ±0.02m, speed: ±0.2km/h}
```

### 3.2 矢量化编码设计

借鉴 Tamba (CVPR 2025) 的联合折线编码 (JPE)：

```
Input Vectorization:
  Agent trajectory:  [x_t, y_t, v_x, v_y, a_x, a_y, θ_t, κ_t, type_onehot, domain_onehot]
  Lane polyline:    [x_k, y_k, type_onehot, direction_onehot]
  
Joint Polyline Encoding (JPE):
  For each agent i:
    h_i^init = MLP_joint([Agent_history_i ⊕ nearest_Lane_segments_i])
  → Output: {h_i^init}_{i=1..N}  (shape: [N, T_hist, d_model])
```

### 3.3 域标签设计

| 数据源 | domain_id | 用途 |
|--------|----------|------|
| Waymo Open Motion | 0 (Source) | 预训练主数据（自然性正则） |
| INTERACTION | 1 (Source) | 强交互辅助 |
| CNCAP GAC S9 | 2 | surrogate 数据源（LOBO） |
| CNCAP GAC E8 | 3 | surrogate 数据源（LOBO；规程 run 仅 25，按类分层） |
| CNCAP XPeng P7+ | 4 | surrogate 数据源（LOBO） |
| CNCAP GAC A66 | 5 | surrogate 数据源（LOBO；企标功能项不入主实验） |

> 2026-09-10 修订：domain_id 与 Stage4 §5.2 schema 统一；"域自适应目标"旧口径作废（见 2026-09-01 修订块第 1 条）。

---

## 四、Layer 2：跨品牌响应校准层（可选模块：GRL-DANN / CORAL-MMD / DG）

> **2026-09-01 重定位**：本层从"核心桥接层"降级为**可选的跨品牌校准模块**。C-NCAP 目标域是离散规程网格、条件内方差近零，经典 UDA 没有可适配的分布；本节技术内容保留，仅用于跨品牌 VUT 响应 surrogate 的校准对照实验（LOBO 协议），不再承担"开放道路→封闭场地迁移"的核心叙事。

### 4.1 问题形式化

- **源域 D_S**: Waymo + INTERACTION (美国/国际开放道路，自然交通流)
- **目标域 D_T**: CNCAP 封闭场地 (中国，结构化测试规程，ABD机器人执行)
- **关键域差异**：
  - 交通流分布：自然随机 vs 规程化对抗
  - 场景几何：复杂城市路网 vs 简单封闭测试跑道
  - 车辆行为：人类驾驶多样性 vs ABD 机器人精确定点
  - 传感器配置：多传感器融合 vs 60+ V-CAN 信号通道

### 4.2 架构设计

```
                           ┌─────────────────┐
        x_s (Source) ────→ │                 │
                           │  Feature         │────→ z_s ────→ Decoder ────→ ŷ_s
        x_t (Target) ────→ │  Extractor F_θ   │
                           │  (Shared Weights)│────→ z_t ────→ Decoder ────→ ŷ_t
                           │                 │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  GRL (∂→ -λ·∂)  │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Domain Classifier│
                           │     D_φ          │───→ d̂ ∈ {Source, Target}
                           └─────────────────┘
```

### 4.3 数学形式化

**特征提取器 F_θ**:
- 输入：矢量化场景表示 (Agent轨迹 + Lane几何)
- 输出：域不变交互特征 z_i ∈ R^{d_model} for each agent i
- 结构：基于 Tamba JPE + Mamba SSM 编码器（与 Layer 3 共享前几层）

**域判别器 D_φ**:
- 输入：场景级池化特征 z̄ = Pool({z_i}_{i=1..N})
- 输出：域分类概率 d̂ ∈ [0,1] (0=Source, 1=Target)
- 结构：2层MLP + Sigmoid

**梯度反转层 (GRL)**:
- 前向传播：恒等映射 GRL(z) = z
- 反向传播：GRL'(∂L/∂z) = -λ · ∂L/∂z
- λ 调度策略 (遵循 Ganin et al. 2016)：
  ```
  λ(p) = 2/(1+exp(-γ·p)) - 1
  p = current_iter / total_iter  (从 0 → 1)
  γ = 10  (控制增长速度)
  ```

**域对抗损失**:
```
L_DA = - [ d·log(D_φ(z̄)) + (1-d)·log(1 - D_φ(z̄)) ]
其中 d = 0 for Source, 1 for Target
```

### 4.4 域自适应策略

**训练流程**:
1. **Phase 1 (预训练)**: 仅使用 Source 数据，训练 F_θ + Decoder，无 GRL-DANN
2. **Phase 2 (DA 微调)**: 同时使用 Source (有未来轨迹标签) + Target (仅历史轨迹，无标签)，激活 GRL-DANN
   - Source 流：L_recon + L_PH + L_causal + L_DA
   - Target 流：L_DA (仅域对抗，无重建损失，因为 Target 域的未来轨迹可能不完整)

**伪标签策略** (Target 域):
- 当 Phase 2 训练稳定后（验证损失平台期），对 Target 域数据生成伪未来轨迹
- 基于置信度阈值过滤（保留 top-70% 高置信度样本）
- 将伪标签样本纳入 L_recon 训练
- 迭代：每 5 epochs 更新伪标签

### 4.5 MA-AT 多分支域判别器（针对轨迹预测的扩展设计）

借鉴 Chen et al. (PRCV 2023) 的 MA-AT (Multi-Adversarial Adaptation Transformer) 框架，针对轨迹预测场景中域差异的多源特性，设计**三头域判别器**：

```
MA-AT 判别器架构:

  Feature Extractor F_θ
        │
        ├──→ z_temporal ──→ D_temp (时序模式判别器, 2层MLP)
        ├──→ z_social   ──→ D_soc  (社交交互判别器, 2层MLP)
        └──→ z_env      ──→ D_env  (环境上下文判别器, 2层MLP)
        
  每个判别器独立计算域分类损失 → L_DA = L_temp + L_soc + L_env
```

**设计原理**:
- **时序模式判别器 (D_temp)**: 对齐速度分布、加减速模式等时序特征
  - 源域（开放道路）：随机加减速、频繁变道
  - 目标域（封闭场地）：规程化加减速、定点制动
- **社交交互判别器 (D_soc)**: 对齐多车交互模式
  - 源域：自然交通流交互（跟车、超车、让行）
  - 目标域：ABD机器人精确间距控制、预设碰撞路径
- **环境上下文判别器 (D_env)**: 对齐道路结构语义
  - 源域：复杂城市路网、多车道、交叉口
  - 目标域：简单封闭跑道、单车道、弧形测试区

**优势**: 多分支各自浅层（每分支仅2层MLP, 50-100单元），分散判别能力，避免单判别器过强导致的训练失衡。

### 4.6 GRL-DANN 训练稳定性最佳实践

基于 2022-2026 年间 GRL-DANN 在序列生成任务上的文献调研，总结以下关键训练技巧：

**λ 调度策略**:
- **推荐方案**: Sigmoid 渐进升温，α=5~10
  ```
  λ(p) = 2/(1+exp(-α·p)) - 1,  p = iter/total_iter
  ```
- **初期 (p<0.3)**: λ≈0，模型专注主任务学习稳定特征
- **中期 (p=0.3~0.7)**: λ 平滑上升至接近1
- **后期 (p>0.7)**: λ≈1，充分对抗对齐
- **备选**: 两阶段训练（Stage 1: λ=0 预训练主任务; Stage 2: λ=0.5 恒定对抗微调）

**域判别器架构选择**:
- **推荐**: 2层MLP, 每层50-200单元, ReLU激活
- **正则化**: Dropout 0.3~0.5, L2权重衰减 1e-4
- **原则**: "宁弱勿强"——判别器只需提供足够对抗信号，不应主导训练
- **学习率**: 判别器 lr_d = (1~2)× lr_g（判别器略快以跟上特征提取器）

**批次构造与优化器**:
- 每个 batch 混合等量 Source/Target 样本（各32条序列）
- 源/目标批量配对均衡，避免判别器利用序列长度或padding差异
- 优化器: SGD+momentum (lr_g≈1e-3, lr_d≈2e-3) 或 AdamW (分层学习率)
- 梯度裁剪: global-norm=5.0（防止对抗训练中的梯度爆炸）

**收敛判据**:
- 域判别器准确率 ≈ 50%（域混淆达成，特征无法区分源/目标）
- 同时监控: 主任务损失持续下降 + 域判别器loss逐渐上升（趋近ln2≈0.693）
- 若判别器loss过早饱和为0 → 对抗过强，需减小λ或弱化判别器
- 若判别器loss始终≈ln2 → 对抗未生效，需增加λ或增强判别器

### 4.7 已知失败模式与缓解策略

| 失败模式 | 症状 | 缓解策略 |
|---------|------|---------|
| **模式崩溃** (Mode Collapse) | 只输出单一模式轨迹，丧失多模态多样性 | 降低λ；加入Variety loss鼓励多样化；引入熵正则 |
| **主任务性能降低** | 引入对抗后目标域性能不升反降 | 减弱对抗（λ减小或后期启用）；检查负迁移（目标域差异过大） |
| **域判别无效** | 判别器始终≈50%（从训练开始） | 增加判别器复杂度；确认数据存在真实域差异 |
| **判别器过拟合** | 训练集精度≈100%，验证集精度下降，目标域性能无提升 | 降低判别器层数/宽度；增加Dropout至0.5；提前停止判别器更新；RADA策略重标样本 |
| **判别器过强** | 判别loss快速降为0，生成器loss飙升 | 降低判别器lr；暂停判别器更新数轮；减小λ；One-sided GRL |
| **梯度爆炸/消失** | Loss震荡剧烈或无法下降 | 梯度裁剪(clip=5.0)；降低学习率；检查序列长度padding策略 |

### 4.8 内容保留技术（Content Preservation）

GRL-DANN 本身不保证内容保留——判别器只关注域分类。在跨域轨迹生成中，需额外"守护"内容（运动合理性、交互逻辑）的损失项：

1. **重构/自我一致性损失**: Source域数据自监督重构 → 鼓励模型保留运动信息
   ```
   L_recon_self = Huber(traj_source_reconstructed, traj_source_gt)
   ```
2. **特征解耦设计**: 通过MA-AT三头判别器隐式实现——时序/社交/环境特征分别对齐，结构语义自然保留
3. **物理约束作为内容锚点**: L_PH 同时作用于 Source 和 Target 流（Target 流虽无未来标签，但仍可通过 PH 约束监督动力学一致性），物理一致性本质上是跨域不变的内容
4. **因果结构保持**: L_causal 确保多主体交互逻辑不因域迁移而失真

---

## 五、Layer 3：PI-Causal Mamba 核心生成器

这是 PICASO 的核心方法层，融合了三个理论支柱：(1) Mamba/SSM 时序建模，(2) 物理可执行性约束与审计，(3) SCM/交互图驱动的反事实分析。

### 5.1 Mamba 选择性状态空间建模

**连续时间 SSM**:
```
h'(t) = A·h(t) + B·x(t)    (状态方程)
y(t)  = C·h(t) + D·x(t)    (输出方程)
```
其中 h(t) ∈ R^{d_state} 是隐状态，x(t) ∈ R^{d_in} 是输入，y(t) ∈ R^{d_out} 是输出。

**离散化 (Zero-Order Hold, ZOH)**:
```
给定时间步长 Δ:
Ā = exp(Δ·A)
B̄ = (Δ·A)⁻¹(exp(Δ·A) - I) · Δ·B

离散递推:
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t
```

**选择性机制 (S6 — Mamba 的核心创新)**:
```
Δ_t = softplus(Linear_Δ(x_t) + bias_Δ)
B_t = Linear_B(x_t)
C_t = Linear_C(x_t)

→ Ā_t, B̄_t 依赖于输入 x_t (数据依赖的选择性)
```

这使得 Mamba 能够在每个时间步**选择性**地记住或遗忘信息，类似于 Attention 机制的 query-key 匹配，但复杂度仅为 O(N)。

**多主体 Mamba 编码器**（借鉴 Social-Mamba 的三元组因子分解）：

```
Temporal Scan (个体时序编码):
  对每个 agent i 的历史轨迹独立执行一维 SSM 扫描
  h_i^temp = BiMamba({x_i,t}_{t=1..T})

Ego-centric Scan (交互编码):
  以 Ego 车辆为中心，按空间距离排序邻居 agent
  在排序后的邻居序列上执行一维 SSM 扫描
  h_i^ego = BiMamba([z_ego ⊕ z_neighbor_1 ⊕ ... ⊕ z_neighbor_N-1])

Goal-centric Scan (意图编码):
  将车道中心线未来锚点拼接到序列末尾
  h_i^goal = BiMamba([h_i^temp ⊕ lane_goal_tokens])

融合:
  h_i^enc = MLP_fuse([h_i^temp ⊕ h_i^ego ⊕ h_i^goal])
```

**Mamba 解码器**（借鉴 Tamba 的交叉状态空间注意力）：

```
交叉状态空间注意力 (Cross-SS-Attention):
  - 将所有 agent 的编码特征聚合成共享场景上下文 S
  - 可学习的多模态意图查询 Q_k (k=1..K, K=6 for multi-modal futures)
  - Q_k 在交叉 SS-Attention 层读取 S，并行解码 K 条未来轨迹

解码过程:
  S = Pool({h_i^enc})  (场景上下文)
  For k in 1..K:
    h_k^dec = CrossMamba(Q_k, S)  (交叉 SSM 解码)
    {ŷ_t^(k)}_{t=1..T_fut} = MLP_head(h_k^dec)  (轨迹回归)
```

### 5.2 Port-Hamiltonian 物理约束嵌入（增强路线；MVP 以解码后投影为主）

**Port-Hamiltonian 系统理论**:

一个受控 PH 系统定义为：
```
ẋ = [J(x) - R(x)] · ∇H(x) + G(x) · u

其中：
  x ∈ R^n                 系统状态 (位置、动量)
  H(x): R^n → R            Hamiltonian (总能量 = 动能 + 势能)
  J(x) = -J(x)^T          互连矩阵 (斜对称，描述能量守恒)
  R(x) = R(x)^T ⪰ 0      耗散矩阵 (半正定，描述能量耗散)
  G(x)                     输入矩阵
  u                        外部控制输入
```

能量守恒律：
```
dH/dt = -∇H^T·R·∇H + ∇H^T·G·u
当 u=0 时，dH/dt ≤ 0 (能量被动耗散，系统稳定)
```

**车辆单体的 PH 模型** (简化二轮车模型):

```
状态: x = [p_x, p_y, v, θ]^T
  p_x, p_y: 全局坐标位置
  v: 纵向速度
  θ: 航向角

Hamiltonian: H = ½·m·v²  (动能，忽略势能)

互连/耗散结构:
       [ 0   0  cos(θ)  0 ]        [ 0   0   0   0 ]
  J =  [ 0   0  sin(θ)  0 ]   R =  [ 0   0   0   0 ]
       [ 0   0    0     0 ]        [ 0   0  c_f  0 ]
       [ 0   0    0     0 ]        [ 0   0   0  c_r ]

  其中 c_f, c_r 为前后轴阻尼系数

控制输入:
  u = [F_traction, δ_steering]^T
  G(x): 控制矩阵映射到状态空间
```

**多体 PH 约束嵌入 Mamba 状态空间**:

核心思想：Mamba 的隐状态 h_t 不是直接回归未来位置，而是**编码系统的 Hamiltonian 参数化**：
```
h_t → MLP_H → {Ĥ_t, Ĵ_t, R̂_t, Ĝ_t}
            ↓
    ẋ̂_t = [Ĵ_t - R̂_t]·∇Ĥ_t + Ĝ_t·u_t   (PH 约束的前向演化)
            ↓
    x̂_{t+1} = x̂_t + ẋ̂_t·Δt             (Euler 积分)
```

**PH 物理损失函数**:
```
L_PH = L_energy + L_structure + L_dynamics

(1) 能量守恒损失:
  L_energy = MSE( dĤ/dt, -∇Ĥ^T·R̂·∇Ĥ + ∇Ĥ^T·Ĝ·u )

(2) 结构矩阵约束:
  L_structure = ||Ĵ_t + Ĵ_t^T||_F^2    (J 必须斜对称)
              + ||clip(-R̂_t, 0, ∞)||_F^2  (R 必须半正定，惩罚负特征值)

(3) 动力学一致性损失:
  L_dynamics = MSE( x̂_{t+1}, x_{t+1}^{gt} )
              (PH 演化结果应与真实未来一致)

(4) 硬约束 (推理时强制执行):
  |a_lon| ≤ a_max = μg  (纵向加速度 ≤ 摩擦圆)
  |a_lat| ≤ a_lat_max   (横向加速度 ≤ 0.8g)
  |κ| ≤ κ_max(v)        (速度依赖最大曲率)
  v_min ≤ v ≤ v_max     (速度区间)
  jerk_lon ≤ jerk_max   (纵/横向 jerk 限制)
```

**物理约束架构图**:
```
x_t (历史状态) ──→ Mamba SSM ──→ h_t (隐状态)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               MLP_J            MLP_R           MLP_H
               (Ĵ_t)            (R̂_t)           (Ĥ_t)
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    PH Forward: ẋ̂ = [Ĵ-R̂]·∇Ĥ + Ĝ·u
                                    │
                                    ▼
                              Physics Check:
                     |a| ≤ μg? κ ≤ κ_max? etc.
                                    │
                         ┌─────────┴─────────┐
                         ▼ (pass)            ▼ (fail)
                    x̂_{t+1} = x_t + ẋ̂·Δt   Clamp to feasible set
```

### 5.2.1 多体 PH + 生成模型融合的四大数学障碍

基于 Gemini Deep Research 对 2022-2026 年文献的系统检索，**目前尚无任何一项工作成功实现了将完全耦合的、具有显式多体代数互联结构（Multi-body PHNN/PHDAE）的模块作为内生变换层直接嵌入到 VAE、Diffusion 或 Mamba 模型中**。现存的物理嵌入生成模型（Pi-DiMT, Stochastic PH Diffusion, PH-Dreamer）均局限于**单体/单智能体**或**隐式正则化**层面。多体 PH 与生成模型融合面临以下四个根本性数学矛盾：

**障碍 A：连续高熵随机扩散空间与刚性代数约束流形（High-Index SDAE）的本质冲突**

扩散模型的本质是在连续、高熵的高维欧氏空间中迭代求解反向时间随机微分方程（SDE）。然而，多体 PH 系统由于智能体间的几何边界与机械硬约束，其物理合法相空间仅为低维弯曲的代数流形 M = {x | g(x)=0}。当在隐空间中嵌入多体 PHNN 时，生成过程变为**随机微分代数方程（SDAE）**的求解——扩散去噪中不可避免的高斯扰动会瞬间使隐状态偏离代数流形。若每个去噪步执行高指数 DAE 的代数投影或拉格朗日乘子求解，不仅计算量呈几何级数增加，还会导致剧烈的数值梯度不连续性，引发致命的梯度消失/爆炸。

**障碍 B：拓扑非平稳性（Topological Non-Stationarity）与生成模型固定隐维数的矛盾**

自动驾驶场景中，多车网络是时变开放系统——智能体数量 N 动态增减，邻域交互图 G(t) 随空间距离实时变化。多体 PH 系统要求固定的联合相空间维度 dim(X) = Σ_i dim(x_i) 以显式定义全局一致的狄拉克互联矩阵 D。然而，主流生成模型（VAE, Mamba SSM）的张量通道数和潜变量维度在初始化后必须保持严格静态。这种"物理交互维度的时变性"与"网络特征维度的静态性"存在根本矛盾——试图用静态特征图容纳任意多体 PH 互联会导致狄拉克结构退化，破坏系统的无源性保证（Passivity）。

**障碍 C：热力学随机消散障碍（Stochastic Dissipation Obstacle）与最大似然优化的冲突**

生成模型的核心任务是逼近真实世界的高度不确定性分布，其训练损失（KL 散度、Score Matching）旨在最大化似然概率。多体 PH 系统为保证群组稳定性，必须满足全局耗散矩阵的半正定性 R⪰0 和无源性耗散不等式。当将多体 PHNN 深度嵌入生成网络时，物理系统在复杂路况下的主动变道、避障等"高能主动行为"需要瞬时能量注入——这在数学上等价于要求 R 在某些局部时空呈现"负阻抗"（Active Energy Injection）。但允许 R 负定会导致 Lyapunov 稳定性崩溃与随机发散；强制 R 正半定又会使生成模型的采样分布严重收缩，无法生成富有驾驶多样性的复杂交互轨迹。

**障碍 D：Transformer/Mamba 隐藏层的非局域注意力与 PH 微分连续性的非对称性**

Transformer 依赖多头自注意力在全时序/全空间节点上执行全局非局域关联，Mamba 则通过选择性 SSM 进行线性时间扫视。端口哈密顿动力学是高度局域化、基于时间的连续外微分流演化（取决于瞬时切空间上的梯度 ∇H）。若将多体 PHNN 放置在 Attention/Mamba 的潜特征转换层中，注意力机制的全局非局域跃迁（Teleportation-like state transitions）将直接破坏 PH 系统的哈密顿向量场连续性，导致物理上的瞬时虚假跃迁，破坏动量/能量守恒定律在时间推进上的微积分基础。

### 5.2.2 多智能体 PH 扩展的三种可行技术路线

针对 Pi-DiMT (ICRA 2026) 仅支持单体 PHNN 的局限性，设计以下三种将单体 PHNN 扩展为多体生成模型的可行方案。**推荐采用路线 A + 路线 B 的混合方案**，平衡物理严格性与计算可扩展性。

**路线 A：基于动态图注意力参数化的无源多主体互联（Graph-Attention IDA-PBC Layer）** ★ 首选

保留生成模型中各智能体独立的局部 PHNN 通道，通过**自注意力机制动态参数化多体之间的能量路由**：

```
机制设计:
  1. 利用自注意力层根据各智能体在 Mamba 隐空间中的相对状态计算关联权重 w_ij
  2. 将 w_ij 作为动态自适应狄拉克结构的参数输入，构造智能体 i 与 j 之间的互联矩阵项:
     J_ij = w_ij · (q_i ⊗ p_j - q_j ⊗ p_i)  (反对称, 保证能量守恒)
  3. 阻尼项: R_ij = w_ij · diag(c_lon, c_lat)  (半正定, 随距离自动增强)
  
全局互联矩阵: J_global = [J_ii, J_ij; -J_ij^T, J_jj]  (自然满足反对称性)
```

**优势**: 不增加生成模型特征维度；时变拓扑下严格符合无源性与能量守恒；自注意力权重随车间距自动调节耦合强度。

**路线 B：组合神经端口哈密顿 DAE 投影层（Compositional N-PHDAE Projection Layer）** ★ 推荐混合

借鉴 Neary et al. (CDC 2025) 的 N-PHDAE 框架，在 PI-Causal Mamba 的解码输出端串联一个可微 N-PHDAE 物理修正层：

```
机制设计:
  1. 将多体间的安全几何包络、最小防撞距离公式化为一组代数约束:
     g_ij(x_i, x_j) = ||p_i - p_j|| - d_safety ≥ 0
  
  2. PI-Causal Mamba 解码器输出初步轨迹 x̂_raw 后，作为初始值送入 N-PHDAE 模块
  
  3. N-PHDAE 采用可微自动微分索引消减求解器进行单步隐式数值校正:
     x̂_phys = N-PHDAE_Correction(x̂_raw, {g_ij}, J_interconnect, R_interconnect)
     → 利用神经网络参数化未知的非线性阻抗，通过固定拓扑矩阵进行无损多体组装
     
  4. 该层作为端到端可微神经网络层参与整网反向传播
```

**优势**: 发挥 Mamba 捕捉高维复杂交通长程关联的能力 + N-PHDAE 修正层锁死多车轨迹不发生碰撞的物理底线；约束违背率较普通 N-ODE 降低一个数量级；支持"即插即用"的模型组合。

**路线 C：多体潜在物理相空间影子正则化（Multi-Agent Latent Shadow Regularization）** 备选

借鉴 PH-Dreamer (Luan & Shi, 2026) 的影子物理潜空间正则化思想：

```
机制设计:
  1. 引入可学习的物理投影矩阵 P_q, P_p，将 Mamba 隐藏特征投影到具有广义坐标/动量的低维辛潜空间:
    q_i = P_q · h_i^enc,  p_i = P_p · h_i^enc
    
  2. 在该低维辛潜空间内并行运行多体端口哈密顿影子系统:
    [q̂_{t+1}, p̂_{t+1}] = RK4_Symplectic([q_t, p_t], J_shadow, R_shadow, H_shadow)
    
  3. 训练时增加多体物理影子正则化项:
    L_shadow = MSE([q, p]^enc, sg([q̂, p̂]^shadow))
    其中 sg(·) 为 stop-gradient 算子
    
  4. 渐进式几何退火课程学习: λ_shadow 从 0 逐步退火至最大值
```

**优势**: 零推理延时（影子系统仅在训练时使用）；降低 4-8% 的物理能耗；适合计算资源受限场景。

### 5.2.3 多体 PH 嵌入方案选择建议

| 方案 | 物理严格性 | 计算开销 | 实现复杂度 | 适用场景 |
|------|----------|---------|-----------|---------|
| 路线 A (Graph-Attention IDA-PBC) | ★★★ | ★★ | ★★★ | 多车交互密集场景 (N≤32) |
| 路线 B (N-PHDAE Projection) | ★★★★★ | ★★★ | ★★★★ | 严格物理可行性场景 |
| 路线 C (Latent Shadow Reg.) | ★★ | ★ | ★★ | 计算资源受限, N>50 |
| **A+B 混合 (推荐)** | ★★★★★ | ★★★ | ★★★★ | 综合最优: 训练时用A保证交互物理, 推理时用B保底 |

### 5.2.4 关键参考文献

- **Pi-DiMT** (Zhou et al., arXiv 2602.00808，venue 待核验): 单体 PHNN + Diffusion + Mamba，强调动力学可行轨迹
- **N-PHDAE** (Neary et al., CDC 2025): 组合式神经端口哈密顿微分代数方程，多体系统的可微组装
- **PH-Dreamer** (Luan & Shi, 2026): RSSM潜空间的影子PH动力学正则化
- **Stochastic PH Diffusion** (2026): 将扩散去噪过程解释为随机端口哈密顿系统
- **PIPHEN** (Zhou et al., 2026): 物理交互预测网络+哈密顿能量网络，多机器人协同
- **LEMURS/pH-MARL** (Sebastian et al., 2025): 自注意力+IDA-PBC，零样本sim-to-real迁移

### 5.3 可微因果图发现：CRiTIC 风格摊销因果网络

**动机**：CausalAF (CoRL 2023) 需要专家手动定义因果图 (CVM)。CounterScene (2026) 的 Causal Interaction Graph (CIG) 本质上是**基于启发式规则（相对速度、TTC、最小间距）构建的先验图，而非从数据中学习得到**。我们需要一种**端到端可微、从数据中自适应发现因果结构、支持变长节点数**的方法。

基于 Copilot 对可微因果发现算法的系统对比（NOTEARS系列、Gumbel-Softmax、Granger深度方法、VAE因果表示学习、LLM辅助因果发现），确定以下分级方案：

#### 5.3.1 首选方案：CRiTIC 风格摊销因果发现网络 (ICRA 2025) ★

**CRiTIC** (Causal tRajecTory predICtion, ICRA 2025) 是首个将可微因果结构学习直接嵌入多智能体轨迹预测的工作。核心理念是**摊销因果发现（Amortized Causal Discovery）**：训练一个通用的因果结构推断模型，不需要在每个新场景重新优化。

**因果发现网络 (Causal Discovery Network, CDN)**:
```
输入: 多车历史轨迹 {traj_i}_{i=1..N}

1. 消息传递编码 (MPNN):
   For each agent pair (i, j):
     e_ij = MLP_edge([h_i ⊕ h_j ⊕ Δx_ij ⊕ Δv_ij])  // 时空关系编码
   h_i^graph = MPNN({h_i}, {e_ij})                    // 图消息传递

2. 汇总因果图 (Summary Causal Graph, SCG):
   A_ij = σ(MLP_score([h_i^graph ⊕ h_j^graph]))     // 因果强度 ∈ [0,1]
   注: SCG 允许加权边，不要求严格 DAG（跨时间窗口，无环约束适当松弛）

3. 因果正则化:
   - 信息瓶颈 (Information Bottleneck): 迫使 CDN 专注最相关的因果关联
     L_IB = I(Z; X) - β·I(Z; Y)
   - 图结构自监督学习 (GSL): 通过移除/扰动边进行去噪训练，增强对真实因果的敏感度
     L_GSL = ||A - Â||_F^2  (原始图 vs 扰动重建图)
```

**因果注意力门控 (Causal Attention Gating)**:
```
机制: 将 CDN 产出的因果图用于动态调整 Mamba/Transformer 的注意力分配

  A_causal[i,j] = A_ij · Attention(Q_i, K_j)  // 因果加权注意力
  h_i^causal = Σ_j A_causal[i,j] · V_j       // 仅因果相连的 agent 参与信息融合
  
效果:
  - 对非因果车辆的轨迹扰动: 预测稳定性提升 54% (Argoverse + INTERACTION)
  - 跨域泛化 (不同城市): 预测性能提升 29%
  - 对抗攻击鲁棒性: 显著优于无因果门控的 baseline
```

**CRiTIC 适配 PICASO 的设计调整**:
1. 将 CDN 的 MPNN 层替换为 Mamba 三元组扫描的输出特征 {h_i^enc}（无需独立 MPNN 编码器）
2. 因果注意力门控直接嵌入 Mamba 解码器的交叉 SSM 中：M_causal = A ⊗ 1_{T×T}
3. 加入 PH 一致性检查：A_ij > 0 当且仅当 i 和 j 之间存在物理可达的交互路径
4. 损失项整合：
   ```
   L_causal = L_IB + λ_gsl·L_GSL + λ_sparse·||A||_1 + L_consistency
   
   L_consistency: 如果 A_ij > 0 (i→j 有因果)，则 i 的行为改变应在 j 的未来中可见
     L_consistency = -Corr(Δh_i^enc, Δh_j^enc) for A_ij > threshold
   ```

#### 5.3.2 备选方案 A：NTS-NOTEARS (AISTATS 2023)

适用场景：离线分析、对因果结构准确性要求极高

```
核心: 在 NOTEARS 基础上引入 1D CNN 表达非线性父子依赖
  - 支持非线性因果关系 (CNN 作为函数逼近器)
  - 可整合先验知识 (强制/禁止某些边)
  - 精度: 相比参数/非参数基线因果图质量提升 10-20% (F1)
  
局限:
  - 需固定节点数 (每场景须重新优化)
  - 计算复杂度 O(N³) (矩阵指数运算)
  - 非端到端联合深度生成模型训练
```

#### 5.3.3 备选方案 B：Gumbel-Softmax 可微 DAG (DP-DAG/VI-DP-DAG)

适用场景：需要全可微 DAG 采样的离线/在线混合方案

```
核心: 通过 Gumbel-Sinkhorn 可微采样拓扑排序 + Gumbel-Softmax 采样边
  - 始终输出有效 DAG (无需拉格朗日惩罚调整)
  - 100 节点 DAG 训练约 190 秒 (vs MCMC 12小时)
  - 全可微，支持端到端联合训练
  - 计算复杂度 O(N²)
  
局限:
  - 需固定节点数 (每次初始化需预定 N)
  - 无自动驾驶场景验证 (通用因果结构学习方法)
  - 停留在学术证明与算法层面
```

#### 5.3.4 因果发现算法综合对比

| 方法类别 | 变长节点 | 非线性因果 | 计算复杂度 | 自动驾驶验证 | 端到端可微 | PICASO 适配性 |
|---------|---------|-----------|-----------|------------|-----------|-------------|
| **CRiTIC (ICRA 2025)** ★ | ✅ 天然支持 | ✅ 神经网络 | O(N²) 摊销 | ✅ Argoverse+INTERACTION | ✅ | ★★★★★ 首选 |
| NTS-NOTEARS (AISTATS 2023) | ❌ 需固定N | ✅ 1D CNN | O(N³) | ❌ 金融/分子生物 | ⚠️ 部分 | ★★★ 离线先验 |
| DP-DAG/VI-DP-DAG (2023) | ❌ 需固定N | ✅ 神经网络 | O(N²) | ❌ 通用方法 | ✅ | ★★★ 离线/在线混合 |
| Neural Granger (cLSTM/TCDF) | ⚠️ RNN天然可变 | ✅ 神经网络 | O(N)~O(N²) | ⚠️ EEG/传感网 | ✅ | ★★ 轻量备选 |
| VAE因果表示 (CausalVAE/DEAR) | ❌ 隐变量固定 | ✅ 神经SCM | O(N)/步 | ❌ CV/NLP | ✅ | ★★ 潜空间建模 |
| LLM辅助因果 (CausalFusion) | ✅ 灵活 | ⚠️ LLM推理 | 依赖LLM | ❌ 实验阶段 | ❌ 不可微 | ★ 先验补充 |

#### 5.3.5 最终推荐：三级因果发现体系

```
Level 1 — 在线摊销因果发现 (CRiTIC 风格 CDN):
  训练时嵌入 PI-Causal Mamba，端到端联合优化
  每个场景一次前向传播即可获得因果图
  用于: 实时因果门控 + MACC 级联干预的因果链构建

Level 2 — 离线高精度因果验证 (NTS-NOTEARS 备选):
  对 CNCAP 关键场景离线计算高精度 DAG
  用于: 验证 Level 1 因果图的准确性 + 提供强先验

Level 3 — 领域知识先验注入:
  交通规则 + C-NCAP 碰撞逻辑 + 物理可达性
  用于: 约束 Level 1 和 Level 2 的搜索空间
```

**注**：CounterScene 的 CIG 是规则驱动的（基于 TTC、相对速度、最小间距等启发式指标计算危险贡献分数），不涉及数据驱动的因果结构学习。据当前检索，本方案较早将**端到端可微因果发现**引入场景生成；最终优先权表述以投稿前系统检索为准，论文中统一使用 "to the best of our knowledge" 口径。

### 5.4 PI-Causal Mamba 完整模块

**编码器完整流程**:
```
Input: Scene {Agent_i, Lane_j}

1. 矢量化编码 (JPE): 
   h_i^vec = MLP_joint([traj_i ⊕ nearest_lanes_i])

2. 三元组 Mamba 编码:
   h_i^temp = BiMamba_temporal(h_i^vec)           # 时序扫描
   h_i^ego  = BiMamba_ego([h_ego ⊕ sorted_neighbors]) # 空间扫描
   h_i^goal = BiMamba_goal([h_i^temp ⊕ lane_goals])   # 意图扫描

3. 特征融合:
   h_i^enc = MLP_fuse([h_i^temp ⊕ h_i^ego ⊕ h_i^goal])

4. 因果图发现 (CRiTIC 风格 CDN):
   A = CDN({h_i^enc}_{i=1..N})                     # 摊销因果发现, N×N 汇总因果图
	   A = A ⊙ Mask_physical_reachability             # 物理可达性掩码过滤({h_i^enc}_{i=1..N})  # N×N 因果邻接矩阵

5. 因果注意力门控 (Causal Attention Gating):
   h_i^causal = Σ_j (A_ij · Attention(Q_i, K_j)) · V_j  # 因果加权特征融合

6. PH 状态编码:
   From h_i^causal → MLP → {Ĵ, R̂, Ĥ, Ĝ}           # PH 结构参数
```

**解码器完整流程**:
```
1. 场景上下文聚合:
   S = CrossAttention_Pool({h_i^causal}_{i=1..N})

2. 多模态意图查询:
   For k in 1..K:
     Q_k = LearnableIntentQuery[k]                # 可学习意图原型

3. 交叉 SSM 解码:
   For each agent i:
     For k in 1..K:
       h_{i,k}^dec = CrossSSM_Decode(Q_k, S, h_i^causal)
       traj_{i,k} = MLP_traj(h_{i,k}^dec)         # 第k条未来轨迹

4. PH 投影 (物理可行性保证):
   For k in 1..K:
     traj_{i,k}^phys = PH_Project(traj_{i,k})     # 投影到 PH 可行流形
     π_{i,k} = MLP_score(h_{i,k}^dec)             # 轨迹概率得分

Output: {traj_{i,k}^phys, π_{i,k}}_{i=1..N, k=1..K}
```

### 5.5 模型超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 256 | 特征维度 |
| d_state | 64 | Mamba SSM 状态维度 |
| d_ff | 512 | FFN 维度 |
| N_Mamba_blocks | 6 | Mamba 层数 (编/解码器各6) |
| T_hist | 10 (1s @10Hz) | 历史帧数 |
| T_fut | 80 (8s @10Hz) | 未来帧数 |
| K | 6 | 多模态轨迹数 |
| N_max_agents | 128 | 最大 agent 数 (与Waymo对齐) |
| τ_init → τ_final | 1.0 → 0.1 | CDN 因果边温度退火 (备选: Gumbel-Softmax 回退方案) |
| λ_PH | 1.0 | PH 物理损失权重 |
| λ_dag | 0.1 | DAG 无环性损失权重 |
| λ_sparse | 0.01 | 因果图稀疏性权重 |
| λ_DA | 0.1 → 1.0 (递增) | 域自适应损失权重 |
| learning_rate | 1e-3 (预训练) / 1e-4 (DA微调) | AdamW |
| batch_size | 64 (A100-80G) / 16 (A100-40G) | 按 GPU 调整 |

---

## 六、Layer 4：MACC 多智能体级联反事实干预

> **2026-09-01 补充**：MACC 的归因与干预结论不再仅凭 World Model 自评成立——最终结论需经 VUT 响应 surrogate 预测 + ABD 实车补测共同支撑（见 §2.2 Stage D 与 §7.1 D6 实车一致性指标）。

### 6.1 从 CounterScene 的单变量干预到 MACC 的级联干预

**CounterScene 的局限性**:
- 仅识别**一个**关键 agent，施加**单点**时空扰动
- 无法建模"agent A 的行为改变 → agent B 被迫调整 → agent C 响应"的因果级联
- 干预方式：在扩散去噪过程中增加空间/时间引力

**MACC 的级联干预框架**:
- 学习完整的 N×N 因果 DAG（Layer 3 已提供）
- 沿因果 DAG 的拓扑排序，逐级施加序列干预
- 使用 do-calculus 形式化级联干预的效应传播

### 6.2 MACC 算法流程

```
Algorithm: MACC Cascade Counterfactual Search

Input:
  - Factual scene S = {Agent_i} with ground-truth trajectories
  - Causal DAG A (from Layer 3)
  - PI-Causal Mamba model M (frozen)
  - Intervention budget B (max perturbation magnitude)
  - Safety threshold ε (minimum TTC to consider "dangerous")

Output:
  - Counterfactual scenario set C = {S'_1, ..., S'_M}
  - Root-cause attribution report

Step 1: Shapley Attribution (identify key agents)
  For each agent i:
    Compute φ_i = ShapleyValue(Risk, i, all_subsets)
    // φ_i 衡量 agent i 对场景总风险的边际贡献
  KeyAgents = Top-k({φ_i})

Step 2: Causal Cascade Construction
  Given KeyAgents and DAG A:
    Sort agents by topological order in A
    Cascade = [] 
    For each key agent i (by topological order):
      affected = {j : A[i,j] > threshold}  // i 的因果下游
      Cascade.append((i, affected))
    // Cascade: [(cause_1, [effect_1a, effect_1b, ...]), (cause_2, [...]), ...]

Step 3: Sequential Cascade Intervention
  S' ← S  (start from factual scene)
  For each (cause_i, affected_i) in Cascade:
    // Step 3a: Intervention on cause_i
    do_i = Optimize:
      min_{δ_i} ||δ_i||  (最小有界扰动)
      s.t.
        Risk(S' with agent i perturbed by δ_i) > Risk(S')  (增加风险)
        KFR(S') > 0.95  (保持物理可行性)
        ||δ_i||_∞ ≤ B  (扰动幅度限制)
    
    Apply do_i to agent i's trajectory in S'
    
    // Step 3b: Cascade propagation (使用 M 作为 World Model)
    S' ← M.rollout(S', intervention_mask = [i])
    // World Model 自动预测 affected agents 将如何响应 i 的改变
    
    // Step 3c: Recursive check for secondary effects
    For each j in affected_i:
      If A[j, k] > threshold for some k and Risk(S') < target:
        Cascade.append_secondary((j, [k]))  // 追加次级级联

Step 4: Feasibility Filter
  For each generated S'_m:
    If KFR(S'_m) < 0.95: discard
    If SolutionRate(S'_m) < 0.5: discard  // 需要至少50%可解
  Sort remaining by Risk Score descending

Step 5: Root-Cause Report
  For each accepted S'_m:
    Compute RootCauseScore_i = φ_i / Σ_j φ_j  (归一化归因)
    Compute IC = consistency of Risk under repeated similar interventions

Return: C (filtered counterfactual scenarios), Attribution Report
```

### 6.3 Shapley 值归因的蒙特卡洛近似

精确 Shapley 值需要 O(2^N) 次模型调用。使用**采样近似**：

```
ShapleyValue_MC(Risk, i, M_samples, model M):
  φ_i = 0
  For m in 1..M_samples:
    S_subset = RandomSubset(Agents \ {i})
    Risk_without_i = M.evaluate_scenario(S_subset)        // 不含 agent i
    Risk_with_i = M.evaluate_scenario(S_subset ∪ {i})    // 含 agent i
    φ_i += (Risk_with_i - Risk_without_i)
  Return φ_i / M_samples
```

计算量：每个场景 N×M_samples 次 model rollout（~10-50 agents × 100 samples = 1K-5K forward passes，在 Mamba 的 O(N) 效率下可接受）。

### 6.4 反事实干预的数学形式化

使用 Pearl 的 do-calculus：

```
因果效应:
  P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)  (后门调整)

在 MACC 中:
  - X: 原因 agent 的轨迹
  - Y: Ego 的碰撞概率
  - Z: 其他 agent (可能的混淆因子)

级联干预 (truncated factorization):
  P(Y | do(X_1=x_1), do(X_2=x_2), ...) 
  = ∫ P(Y | X_1,...,X_k, Z) · Π_j P(Z_j | Pa(Z_j)) dZ
  
  其中 Pa(Z_j) 由因果 DAG A 定义
```

### 6.5 离线闭环的 World Model 实现

```
WorldModel.rollout(scene, intervention_mask):
  """
  使用 PI-Causal Mamba 作为 World Model 进行多步前向推演
  
  Args:
    scene: 当前场景状态 (Agent 历史 + 地图)
    intervention_mask: [i] 表示 agent i 的轨迹被强制修改
  
  Returns:
    updated_scene with all agents' future trajectories
  """
  
  For t in 1..T_fut:
    // Step 1: 编码当前场景
    {h_i^enc} = Encoder(scene)
    
    // Step 2: 因果图更新
    A_t = CausalDiscovery({h_i^enc})
    
    // Step 3: 干预检查
    For each agent j:
      If j in intervention_mask:
        // 使用干预轨迹，跳过生成
        continue
      If any ancestor k of j (A_t[k,j] > threshold) is intervened:
        // j 的状态受级联影响，使用因果门控特征
        h_j^causal = Σ_p A_t[p,j] · h_p^enc
      Else:
        h_j^causal = h_j^enc
    
    // Step 4: 解码下一步
    For each non-intervened agent j:
      x̂_j^{t+1} = Decoder(h_j^causal, scene.context)
      x̂_j^{t+1} = PH_Project(x̂_j^{t+1})  // 物理投影
    
    // Step 5: 更新场景状态
    scene = scene.append(x̂^{t+1})
  
  Return scene
```

---

## 七、Layer 5：评估与验证层

### 7.1 四维评估指标矩阵

#### D1: 轨迹保真度 (Trajectory Fidelity)

| 指标 | 公式 | 用途 |
|------|------|------|
| minADE₆ | min_k (1/T) Σ_t ‖traj_t^(k) - traj_t^gt‖ | 最佳轨迹平均误差 |
| minFDE₆ | min_k ‖traj_T^(k) - traj_T^gt‖ | 最佳轨迹终点误差 |
| MR₆ | P(min_k FDE_k > 2.0m) | 漏检率 |
| DAO | 轨迹在可行驶区域比例 | 道路合规性 |

#### D2: 安全关键性 (Safety-Criticality)

| 指标 | 公式 | 备注 |
|------|------|----------|
| CR (Collision Rate) | N_collision / N_scenarios | 实验后填报（均值±std + 95% CI，不预承诺数值） |
| NMR (Near-Miss Rate) | N_TTC<3s_no_collision / N_scenarios | 实验后填报 |
| median minSTTC | median(per-scenario min TTC) | 实验后填报 |
| PET minimum | min(PET) per intersection scenario | 实验后填报 |

#### D3: 物理可行性 (Physical Feasibility) — **[PROPOSED NEW]**

| 指标 | 公式 | 备注 |
|------|------|----------|
| **KFR** | (∑_t I(所有物理约束满足)_t) / T_total | 实验后填报（与无物理约束基线对照，不预承诺数值） |
| a_violation_rate | (∑_t I(|a|>μg)_t) / T_total | 实验后填报 |
| κ_violation_rate | (∑_t I(|κ|>κ_max)_t) / T_total | 实验后填报 |
| jerk_violation_rate | (∑_t I(|jerk|>j_max)_t) / T_total | 实验后填报 |
| friction_circle_violation | (∑_t I(a_x²+a_y²>(μg)²)_t) / T_total | 实验后填报 |

#### D4: 因果可解释性 (Causal Interpretability) — **[PROPOSED NEW]**

| 指标 | 公式 | 用途 |
|------|------|------|
| **Root-Cause Attribution Score** | φ_i / Σ_j φ_j (归一化Shapley值) | 识别导致碰撞的关键agent |
| **Intervention Consistency (IC)** | 1 - Var(ΔRisk)/E(ΔRisk)² | 干预效果的可预测性 |
| Causal Graph Sparsity | ‖A‖₀ / N² | 因果图的简洁性 |

#### D5: 跨品牌校准诊断 (Cross-Brand Calibration)

| 指标 | 公式 | 用途 |
|------|------|------|
| **TDPD (Target Domain Performance Drop)** | (CR_target − CR_source) / CR_source | 2026-09-01 统一定义（与 Stage4 §4.4 对齐）；降级为跨品牌校准诊断指标，不作核心贡献 |
| Wasserstein Distance | W(P_gen, P_real) | 生成分布与真实分布的差异 |
| MMD | MMD²(P_gen, P_real) | 核方法分布差异 |

#### D6: 实车一致性 (Real-Vehicle Agreement) — **[2026-09-01 新增核心维度]**

| 指标 | 公式 | 用途 |
|------|------|------|
| **碰撞判定一致率** | (1/N) Σ I(ŷ_collision = y_collision^ABD) | surrogate 预测 vs ABD 实车实测（规程内复现组与规程外新点组分别报告） |
| **minTTC MAE** | (1/N) Σ \|minTTC_pred − minTTC_ABD\| | 实车一致性（回归） |
| **AEB 触发时刻 MAE** | (1/N) Σ \|t_AEB_pred − t_AEB_ABD\| | 实车一致性（事件时序） |
| **实车验证命中率** | 实车确认高危场景数 / surrogate 判定高危且送测场景数 | 闭环选点效率（对照：随机选点） |

### 7.2 消融实验设计

| 消融编号 | 移除组件 | 预期影响 | 验证创新点 |
|---------|---------|---------|-----------|
| A1 | 移除 PH 约束 (L_PH=0) | 预期 KFR 显著下降、不可执行场景增多（幅度实验后填报） | C1 物理约束 |
| A2 | 移除因果发现 (A = I, 全连接) | Root-Cause Score 无法计算, 场景退化 | C1 因果结构 |
| A3 | 移除跨品牌校准 (brand-agnostic surrogate) | 预期 LOBO holdout 品牌上响应预测退化（幅度实验后填报） | C3 跨品牌校准 |
| A4 | 单变量干预 (仅 top-1 agent, 非级联) | 无法生成级联场景, NMR 下降 | C2 级联因果 |
| A5 | 移除 PH 投影层 (仅软约束) | 预期 KFR 下降、边界场景物理违反增多（幅度实验后填报） | C1 硬约束 |
| A6 | 仅单品牌 surrogate (无跨品牌校准) | 预期校准收益消失、外品牌误差上升（幅度实验后填报） | C3 校准对照 |
| A7 | 随机归因 (替代 Shapley) | Key Agent 定位不准, IC 下降 | C2 归因 |
| A8 | GRU 替代 Mamba | 推理速度下降, 长序列 minFDE 恶化 | C1 架构 |
| A9 | 移除 surrogate 排序（随机选点送实车验证） | 预期实车验证命中率下降（幅度实验后填报） | C3 实车闭环效率 |

### 7.3 统计显著性检验

- **独立实验次数**: 深度模型主实验 3-5 次随机种子；轻量级/解析型实验可扩大到 10 次以上
- **报告格式**: 均值 ± 标准差 + 95% bootstrap 置信区间；场景级指标报告分位数而非只报均值
- **显著性检验**: 优先使用 paired test（同一场景上的模型对比）；分布明显非正态时使用 Wilcoxon signed-rank 或 bootstrap 差值区间
- **效应量**: Cohen's d / Cliff's delta / 相对提升百分比，必须和 p 值一起报告
- **多重比较**: 多 baseline / 多消融时做 Holm-Bonferroni 或 FDR 修正
- **写作规则**: 不预设“所有 p<0.05”。若无显著性，照实报告并讨论样本量、场景覆盖和效应方向。

---

## 八、统一训练目标与优化策略

### 8.1 多任务损失函数

```
L_total = L_recon + λ_PH·L_PH + λ_causal·L_causal + λ_DA·L_DA (+ λ_safety·L_safety)

(1) 轨迹重建损失:
  L_recon = -Σ_i Σ_k π_{i,k} · log P(traj_i^gt | traj_{i,k}, S)
          = Huber(traj_best - traj_gt) + λ_ent·Entropy(π)
  // 负对数似然 + 熵正则 (鼓励多模态)

(2) PH 物理约束损失 (见 §5.2):
  L_PH = L_energy + L_structure + L_dynamics

(3) 因果结构损失 (见 §5.3):
  L_causal = L_dag + λ_sparse·L_sparse + L_consistency

(4) 域自适应损失 (见 §4.3):
  L_DA = -[d·log(D_φ(z̄)) + (1-d)·log(1-D_φ(z̄))]

(5) 安全关键引导损失 (推理时可选):
  L_safety = λ_risk·RiskGuidedGradient
  // 在监督训练阶段不使用 (防止模型偏向产生不安全轨迹)
  // 仅在 Stage C (反事实搜索) 的优化中使用
```

### 8.2 三阶段训练策略

```
Phase 1: 预训练 (Waymo + INTERACTION，先 subset 后 full)
  目标: 学习自然驾驶的统计分布 + 物理先验
  Data: Waymo subset 20K-100K + INTERACTION；full Waymo 仅在 MVP 指标稳定后扩展
  Loss: L_recon + L_PH + L_causal
  Epochs: 以 early stopping 为准，不固定 100 epoch
  GPU: RTX 4090/A100-40G 可跑 subset；full dataset 再申请 A100-80G
  Time: subset 24-48h；full 48-72h 仅作为可选扩展
  Output: θ_pre

Phase 2: VUT 响应 surrogate 训练 + 跨品牌校准 (2026-09-01 重定位)
  目标: 学习"场景参数 → 真实车辆响应"映射（碰撞 / minTTC / AEB 触发时刻），含不确定度
  Data: CNCAP ABD 多品牌实测（含 FalseReaction 负样本；调参/标定类仅用于执行包络标定）
  Model: 规程参数 + 物理特征 → GBM/GP/小型 MLP + ensemble/conformal 不确定度
  校准对照（可选）: brand-agnostic vs CORAL-MMD vs GRL-DANN 跨品牌特征校准
  评估: leave-one-brand-out；报告效应量 + CI，不预设固定阈值
  Output: θ_surr (+ 可选校准参数)
  备注: 原"开放道路→封闭场地 GRL-DANN 微调"降级为可选对照；CNCAP 目标域为离散规程网格、
        条件内方差近零，不作为生成器的 UDA 目标域

Phase 3: MACC 反事实场景生成 (推理)
  目标: 定向生成安全关键场景
  θ_da frozen
  For each factual scene:
    Shapley attribution → Cascade intervention → KFR filter → Scene bank
  Output: curated safety-critical scenario library
```

### 8.3 学习率与优化器配置

| 超参数 | Phase 1 | Phase 2 |
|--------|---------|---------|
| Optimizer | AdamW | AdamW |
| lr (peak) | 1e-3 | 1e-4 |
| lr schedule | Cosine Annealing + Linear Warmup (5 epochs) | Cosine Annealing |
| Weight decay | 1e-4 | 1e-4 |
| Gradient clipping | 1.0 | 1.0 |
| Batch size | 64 | 32 (含 Target) |
| Mixed precision | BF16 | BF16 |
| Gradient accumulation | 2 steps | 4 steps |

---

## 九、实现策略与技术栈

### 9.1 代码架构 (从零构建)

```
causal-scenario-generation/
├── configs/                    # Hydra 配置文件
│   ├── data/                   # 数据配置 (Waymo, INTERACTION, CNCAP)
│   ├── model/                  # 模型配置 (encoder, decoder, PH, causal)
│   └── train/                  # 训练配置 (phase1, phase2, phase3)
├── data/                       # 数据处理模块
│   ├── unified_schema.py       # 统一数据表示
│   ├── waymo_loader.py         # Waymo Open Motion Dataset 加载器
│   ├── interaction_loader.py   # INTERACTION Dataset 加载器
│   ├── cencap_loader.py        # CNCAP ABD 数据加载器 (匿名化)
│   └── augmentation.py         # 数据增强 (旋转、平移、agent dropout)
├── models/                     # 模型模块
│   ├── mamba/                  # Mamba SSM 核心
│   │   ├── selective_scan.py   # S6 选择性扫描算子
│   │   ├── mamba_block.py      # Mamba 残差块
│   │   └── bimamba.py          # 双向 Mamba
│   ├── encoder/                # 编码器
│   │   ├── jpe.py              # 联合折线编码 (JPE)
│   │   ├── temporal_scan.py    # 时序扫描
│   │   ├── ego_centric.py      # 自我中心社交网格
│   │   └── goal_centric.py     # 目标中心意图编码
│   ├── decoder/                # 解码器
│   │   ├── cross_ssm.py        # 交叉状态空间注意力
│   │   └── intent_queries.py   # 多模态意图查询
│   ├── physics/                # 物理约束
│   │   ├── phnn.py             # Port-Hamiltonian NN
│   │   ├── vehicle_model.py    # 车辆动力学模型
│   │   └── feasibility.py      # KFR 计算与硬约束投影
│   ├── causal/                 # 因果发现 (CRiTIC 风格)
│   │   ├── cdn.py              # 摊销因果发现网络 (CDN)
│   │   ├── causal_gating.py    # 因果注意力门控
│   │   ├── notears.py          # NTS-NOTEARS 离线高精度DAG (备选)
│   │   ├── gumbel_dag.py       # DP-DAG Gumbel-Softmax DAG (备选)
│   │   └── shapley.py          # Shapley 值归因
│   ├── physics/body/           # 多体PH扩展 (新增)
│   │   ├── ida_pbc.py          # 路线A: Graph-Attention IDA-PBC Layer
│   │   ├── n_phdae.py          # 路线B: Compositional N-PHDAE Projection
│   │   └── latent_shadow.py    # 路线C: Multi-Agent Latent Shadow Reg.
│   └── domain_adapt/           # 域自适应
│       ├── grl.py              # 梯度反转层
│       ├── discriminator.py    # 多分支域判别器 (MA-AT)
│       └── content_preserve.py # 内容保留损失
├── training/                   # 训练模块
│   ├── phase1_pretrain.py      # Phase 1: 预训练
│   ├── phase2_domain_adapt.py  # Phase 2: 域自适应
│   └── phase3_counterfactual.py # Phase 3: MACC 反事实搜索
├── evaluation/                 # 评估模块
│   ├── metrics/                # 指标计算
│   │   ├── fidelity.py         # ADE/FDE/MR/DAO
│   │   ├── safety.py           # CR/NMR/TTC/PET
│   │   ├── physical.py         # KFR/a_violation/etc
│   │   └── causal.py           # RootCause/IC
│   └── statistical.py          # 统计显著性检验
├── visualization/              # 可视化
│   ├── trajectory_plot.py
│   ├── causal_graph_viz.py
│   └── t_sne_domain.py         # 域特征 t-SNE 可视化
├── scripts/                    # 运行脚本
│   ├── preprocess_data.sh
│   ├── train_phase1.sh
│   ├── train_phase2.sh
│   └── generate_scenarios.sh
└── utils/                      # 工具函数
    ├── geometry.py              # 坐标变换、曲率计算
    ├── io.py                    # 数据序列化
    └── logging.py               # WandB/TensorBoard 集成
```

### 9.2 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 深度学习框架 | PyTorch 2.x | 生态成熟, BF16支持, torch.compile |
| SSM 算子 | mamba-ssm (官方) + 自定义 Triton kernel | 性能最优 |
| 数据加载 | Waymo Open Dataset API + torch_geometric | 官方API + 图数据处理 |
| 配置管理 | Hydra + OmegaConf | 模块化实验配置 |
| 日志追踪 | Weights & Biases (WandB) | 实验追踪和可视化 |
| 仿真验证 | nuPlan (闭环) + scenario_plotting | 标准评估平台 |
| 因果发现 | CRiTIC 风格摊销CDN (首选) + NTS-NOTEARS (离线备选) + DP-DAG (备选) | 摊销因果发现 + 端到端可微 + 变长节点支持 |
| PH 求解 | 自定义 PyTorch 实现 | 完全控制梯度流 |

> **保密修订**: 若 ABD/CNCAP 数据受公司保密约束，日志追踪默认使用本地 TensorBoard/MLflow 或 WandB offline mode；不得将原始轨迹、品牌标签、文件名、测试时间戳上传到外部平台。

### 9.3 GPU 算力需求

| 训练阶段 | 数据集规模 | 最小 GPU | 推荐 GPU | 预计时间 |
|---------|-----------|---------|---------|---------|
| Phase 0 (数据/解析器/MVP smoke test) | ABD + Waymo tiny subset | RTX 4060Ti / CPU | RTX 4060Ti | 本地完成 |
| Phase 1 (预训练) | Waymo subset 20K-100K | RTX 4090 / A100-40G | A100-40G | 24-48h |
| Phase 1 (全量预训练, 可选) | Waymo full ~487K | A100-40G × 1 | A100-80G × 1 | 48-72h @ A100-80G |
| Phase 2 (DA/DG 微调) | Waymo 20K-50K + CNCAP | RTX 4090 / A100-40G | A100-40G | 12-24h |
| Phase 3 (反事实推理) | CNCAP scenes | RTX 4060Ti | — | 1-2h (本地) |
| 消融实验 (分层执行) | Subset | A100-40G × 1 | A100-80G × 2 | 1-2周 |

**推荐云平台**: AutoDL (A100-80G, ¥8-12/小时) 或 智星云 (A100-40G, ¥5-8/小时)
**预算估算**: MVP 阶段优先控制在 50-100 GPU-hours；确认指标有效后再扩展到 200-300 GPU-hours 的全量训练与消融。避免在解析器和指标未冻结前消耗云算力。

### 9.4 开发里程碑

| 里程碑 | 时间 (周) | 产出 | 验证标准 |
|--------|----------|------|---------|
| M1: 数据 pipeline | 1-2 | 统一 schema + 3个 data loader | 成功加载 Waymo/INTERACTION/CNCAP |
| M2: Mamba 编码器 | 3-4 | 三元组 Mamba 编码器 | 在 Waymo minFDE₆ 对标 Tamba (差距 <10%) |
| M3: Mamba 解码器 | 5-6 | 交叉 SSM 解码器 + 多模态输出 | K=6 轨迹, minFDE₆ < 1.5m |
| M4: 物理可执行性层 | 7-8 | 解码后投影 + 物理损失 + KFR/违反统计；PHNN 作为增强 | KFR 明显高于无物理约束基线，违反类型可解释 |
| M5: 因果/反事实层 | 9-10 | 关键 Agent/时间窗归因 + 有界干预；Shapley 可作为离线解释 | 干预前后风险变化方向稳定，能被规程逻辑复核 |
| M6: DA/DG 对比 | 11-12 | no-DA + CORAL/MMD + GRL-DANN 或 DG | 目标域 holdout 上报告效应量和 CI，不强求固定阈值 |
| M7: MACC 增强 | 13-14 | 级联反事实搜索（若 M5 稳定） | 可生成少量可解释级联案例即可进入案例分析 |
| M8: 消融实验 | 15-16 | 分层消融 + 统计检验 | 核心消融结论可重复；不要求所有 p<0.05 |
| M9: 论文撰写 | 17-20 | 初稿 | 符合 T-ITS/TR Part C/AAP 的数据、实验、可复现要求 |

---

## 十、补充调研完成状态（Gemini / Copilot 报告已整合）

以下 3 个方向的补充调研已于 2026-05-21 完成，关键发现已整合至对应章节：

### ✅ Prompt 1：多体 Port-Hamiltonian 系统 → 已整合至 §5.2.1-§5.2.3

**执行工具**: Gemini Deep Research (Google Scholar, 2022-2026)
**关键发现**:
- **检索状态**: 据该次检索，尚未发现将完全耦合的多体 PHNN/PHDAE 嵌入生成模型的工作 —— 投稿前需按修订检索协议复核，论文中不得表述为"零突破/Blue Ocean"
- **四大数学障碍**: SDAE 冲突、拓扑非平稳性、随机消散障碍、非局域注意力 vs PH 连续性（详见 §5.2.1）
- **三种扩展路线**: Graph-Attention IDA-PBC (路线A)、Compositional N-PHDAE Projection (路线B)、Latent Shadow Regularization (路线C)（详见 §5.2.2）
- **关键文献**: N-PHDAE (Neary et al., CDC 2025)、PH-Dreamer (Luan & Shi, 2026)、Stochastic PH Diffusion (2026)、PIPHEN (2026)、LEMURS/pH-MARL (2025)

### ✅ Prompt 2：可微因果发现算法对比 → 已整合至 §5.3

**执行工具**: Copilot 研究助手原报告；需按 2026-05-28 修订检索协议重新核验
**关键发现**:
- **CRiTIC (ICRA 2025) 确定为首选**: 摊销因果发现 + 因果注意力门控，54% 扰动鲁棒性提升，29% 跨域泛化增益
- **CounterScene CIG 是规则驱动的**: 基于 TTC、相对速度等启发式指标，非数据驱动学习 —— 明确差异化点
- **三级因果体系**: Level 1 在线摊销(CRiTIC) → Level 2 离线高精度(NTS-NOTEARS) → Level 3 领域先验注入
- **综合对比表**: 6 类方法在变长节点、非线性、复杂度、自动驾驶验证、端到端可微 5 个维度的系统对比（详见 §5.3.4）

### ✅ Prompt 3：GRL-DANN 训练稳定性 → 已整合至 §4.5-§4.8

**执行工具**: Copilot 研究助手原报告；需按 2026-05-28 修订检索协议重新核验
**关键发现**:
- **MA-AT 多分支判别器**: 时序/社交/环境三头判别器设计，避免单判别器过强（详见 §4.5）
- **训练最佳实践**: Sigmoid λ 升温(α=5~10)、2层MLP判别器(50-200单元)、Dropout 0.3~0.5、梯度裁剪 5.0（详见 §4.6）
- **六种失败模式**: 模式崩溃、主任务性能降低、域判别无效、判别器过拟合、判别器过强、梯度爆炸（详见 §4.7）
- **内容保留技术**: 自监督重构损失、特征解耦、PH 约束作为物理内容锚点、因果结构保持（详见 §4.8）

---
## 十一、风险识别与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| Mamba + PHNN 联合训练不稳定 | 中 | 高 | 先分阶段训练（先 Mamba 后 PH），再联合微调；使用梯度裁剪和 warmup |
| 多体 PHNN 嵌入 Mamba 隐空间时的 SDAE 梯度不连续 (障碍A) | 中 | 高 | 采用路线B (N-PHDAE 输出端投影层) 避免隐空间内 SDAE 求解；将 PH 约束放在解码器之后而非 Mamba 内部 |
| GRL-DANN 模式崩溃/判别器过强导致生成质量退化 | 中 | 中 | Sigmoid λ 渐进升温(α=10)；Dropout 0.3~0.5 正则化判别器；RADA 策略动态调整对抗难度；监控判别器准确率维持≈50% |
| GRL-DANN 内容丢失 (Content Loss) | 中 | 中 | 加入自监督重构损失 L_recon_self；因果结构保持损失；PH 约束作为跨域不变的物理内容锚点 |
| CNCAP 数据量不足（每品牌 < 100 场景） | 高 | 高 | 数据增强（轨迹扰动、场景重组）；伪标签迭代（每5 epochs更新）；降低 DA 维度（仅对齐 high-level 分布） |
| CRiTIC 风格 CDN 在极小数据量下摊销不足 | 低 | 中 | 保留 NTS-NOTEARS 离线高精度 DAG 作为备选；领域知识先验注入约束搜索空间 |
| 因果注意力门控在 N>50 时计算开销过大 | 低 | 低 | 物理可达性掩码预过滤 (仅保留空间邻近的 agent 对)；Mamba O(N) 效率保证 |
| 反事实干预违反物理约束 | 中 | 中 | N-PHDAE 投影层作为最后防线；干预 δ 的搜索空间受 PH 可行域约束 |
| A100 租用成本超预算 | 低 | 中 | Phase 1 可先使用 Waymo subset (100K) 验证；A100-40G 而非 80G |

---

## 文件版本与后续步骤

> **文件版本**: v2.0 (基于 Stage 3 三份调研报告更新)
> **前置依赖**: Stage1 (研究问题凝练 v2.0) + Stage2 (综合差距分析 v1.0)
> **创新点确认**: C1 PI-Causal Mamba, C2 MACC 级联反事实, C3 跨品牌 VUT 响应 surrogate 与实车校准（2026-09-01 由 GRL-DANN 多源 DA 重定位）
> **评估创新**: KFR, Root-Cause Attribution Score, Intervention Consistency, 实车一致性指标族（碰撞判定一致率 / minTTC MAE / AEB 触发 MAE）, TDPD（诊断用）
> **v2.0 更新内容**:
>   - §4.5-§4.8: 新增 MA-AT 多分支判别器、训练稳定性最佳实践、失败模式与缓解、内容保留技术
>   - §5.2.1-§5.2.4: 新增四大数学障碍、三种多体 PH 扩展路线 (A/B/C)、关键参考文献
>   - §5.3: 因果发现方案从 Gumbel-Softmax 升级为 CRiTIC 风格摊销因果网络 (首选) + NTS-NOTEARS (备选) + DP-DAG (备选) 三级体系
>   - §5.4: 编码器/解码器流程适配 CRiTIC CDN + 因果注意力门控
>   - §9.1-§9.2: 代码架构与技术栈更新 (cdn.py, causal_gating.py, ida_pbc.py, n_phdae.py 等)
>   - §10: 三份调研报告完成状态标注，关键发现汇总
>   - §11: 风险表扩展 (增加 SDAE 梯度不连续、GRL-DANN 模式崩溃/内容丢失等风险)
> **下一步**:
>   1. 进入 **Stage 4（数据处理与实验方案）** — 详细设计统一 Schema、数据预处理流水线、增强策略
>   2. 启动代码 Phase 0：搭建 PyTorch 项目骨架 + 数据加载 pipeline
>   3. 复现 Tamba (CVPR 2025) baseline 以验证 Mamba 编码器在 Waymo 上的性能
