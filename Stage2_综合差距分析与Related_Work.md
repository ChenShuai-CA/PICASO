# Stage 2：综合差距分析与 Related Work 章节

> 整合 8 份 Gemini/Copilot 研究助手调研报告
> 编写日期：2026-05-21
> 状态：Revised Draft（2026-05-28 专家审查后，进入投稿前源核验）

---

## 2026-05-28 专家审查修订：差距结论与 Related Work 使用边界

本文件的总体框架可用，但不能以当前形态直接写入 SCI/T-ITS 论文。核心问题是：部分“零论文、首次、SOTA 数值、会议归属”来自研究助手整合，尚未完成出版商/官方页面/DOI/arXiv 的逐条核验。后续执行按以下修订：

1. **差距结论降级为可防守表述**：统一使用“To the best of our knowledge after a systematic search...”或“在公开可检索文献中尚未发现直接研究……”。不要写成绝对的“全领域零论文”。
2. **相邻前沿工作分类**：已出版/已录用工作可作为正式对比；arXiv 预印本只作为 concurrent/preprint 讨论；无法核验的工作不进入主 Related Work。
3. **定量结果不得提前承诺**：表中“Q1目标值”改为“实验设计目标/期望方向”，论文结果表必须等实验完成后填真实值和置信区间。
4. **PICASO 定位调整**：主张不是“首个四合一模型”，而是“面向安全验证的统一框架，将物理可执行性、反事实因果分析与开放道路到封闭场地迁移放入同一可评估 pipeline”。
5. **Related Work 改写原则**：每段结尾只指出与本文问题定义的差距，不攻击现有工作；避免“cannot / never / entirely absent”等不可证绝对词，改用“rarely reported / not directly addressed / remains underexplored”。

---

## 一、8 份调研报告核心发现摘要

### 1.1 场景生成方法全景（Gemini Task 1）

**核心发现**：

| 范式 | 代表方法 | 关键局限 |
|------|---------|---------|
| 数据驱动生成 | TrafficGen (ICRA 2023), Scenario Dreamer (CVPR 2025), DiffScene (AAAI 2025) | 无物理约束/非闭环/无因果诊断 |
| 对抗与强化学习 | SaFeR (2026), Adv-BMT (NeurIPS 2025), FREA (CoRL 2024) | LFR仅限单对抗车、离线RL泛化差、非闭环 |
| 知识与LLM驱动 | ChatScene (CVPR 2024), LLM-attacker (TITS 2025), SERA (2025) | 离散控制/轨迹粗糙/多车动力学不一致 |
| 标准对齐与HIL验证 | NeuroNCAP (CVPR 2024), RiskMV-DPO (2026) | 传感器级高时延/黑盒动力学 |

**PICASO框架定位**：该报告基于我们的研究计划，正式提出了 **PICASO (Physics-Informed Causal Domain-Adaptive Scenario Generation)** 五层架构：混合数据输入层 → 域自适应过渡层(GRL-DANN) → PI-Causal Mamba核心层 → 场景优化与干预层(双层博弈) → 工业级物理验证层(HIL)。该框架是对我们3个创新点(物理+因果+DA)统一架构的系统表达。

**关键判断**：当前学术界的场景生成方法在"物理约束+因果引导+域自适应"三元交叉地带存在系统性空白。

### 1.2 因果推断 × 自动驾驶（Gemini Task 2）

**核心发现**：五大因果方法分类：

| 方法类型 | 代表工作 | 与我们计划的差异 |
|---------|---------|---------------|
| SCM + 后门/前门调整 | CausalVAD (CVPR 2026), CICR | 侧重Ego去混淆，非多车级联反事实生成 |
| **反事实推理** | **CounterScene (2026.03)**, CF-VLA, SafeAlign-VLA | CounterScene仅单变量干预，无级联因果链 |
| 因果发现 | CRiTIC (ICRA 2025), TR-JCDN, Baidu Apollo Study | 仅作事故后诊断，非在线生成引导 |
| 因果表示学习 | PCM (ICRA 2026) | 单车预测跨域，非多车联合物理一致性 |
| Shapley/中介分析 | Shapley-based Analysis, Double-ML Mediation | 仅被动诊断，非在线归因 |

**关键空白**：
1. **多智能体级联因果链条生成能力缺失**：CounterScene仅单变量干预，无法生成"前车A避障→中车B紧急变道→Ego追尾"等多层级联场景
2. **VLA自回归推理时延**：CF-VLA/CausalDrive的推理频率难以满足实时闭环
3. **物理-视觉一致性**：强反事实扰动下的BEV生成常出现穿模/滑移伪影

**MACC框架**：报告提出了多智能体级联反事实(Multi-Agent Cascade Counterfactual)框架概念，直接针对上述空白。

### 1.3 物理约束生成模型（Copilot Task 3）

**核心发现**：三类物理约束嵌入方式：

| 约束类型 | 代表方法 | 约束粒度 |
|---------|---------|---------|
| **软约束（PINN）- 损失惩罚** | DiffScene, PINN扩散模型(ICLR 2025) | 单对抗车 vs Ego |
| **硬约束（PHNN/LNN）- 结构嵌入** | **Pi-DiMT (ICRA 2026)**, LK-SDE | **仅单车轨迹，非多车交互** |
| **可行域控制 - 后验筛选** | SaFeR/LFR, FREA/LFR | 单背景车 vs Ego |

**关键判断（修订）**：
- Pi-DiMT (Physics-informed Diffusion Mamba Transformer, arXiv 2602.00808，venue 需投稿前核验) 是重要相邻工作，但其任务边界、数据集和多主体程度需回到原文确认。
- 当前检索尚未发现同时覆盖"多主体物理可执行性审计 + 反事实因果分析 + 开放道路到封闭场地迁移"的公开场景生成框架。
- 与 SaFeR/LFR 类方法对比时，应强调 ABD 执行约束、KFR 违反类型审计和目标域验证，不应预先声称对方在复杂交叉口数值崩溃。

**对比表摘要（修订）**：AdvSim/STRIVE 偏对抗或数据驱动；DiffScene 偏软约束；FREA/SaFeR 采用可行域/可行性控制；CounterScene 强调反事实因果。PICASO 的可防守定位是将物理可执行性审计、反事实因果分析、多主体场景生成和封闭场地跨域验证整合到统一 pipeline，而不是直接宣称“首个”。

### 1.4 域自适应 × 自动驾驶（Copilot Task 4）

**核心发现**：六类域迁移方向调研结果：

| 域迁移方向 | 研究状态 | 代表工作 |
|-----------|---------|---------|
| **★ 开放道路 → 封闭场地** | 当前检索未发现直接针对场景生成的公开工作；投稿前需复核 | 待最终检索确认 |
| 仿真 → 真实 (Sim-to-Real) | 成熟，大量研究 | PanDA(CVPR 2026), UNIT-GAN, 对抗DA |
| 跨数据集泛化 | 活跃，多源DA/DG | AdapTraj(ICDE 2024), MUSDA |
| 地区/国家差异 | 起步阶段 | 风格迁移+自适应归一化 |
| 无人机航拍 → 车载视角 | 已有突破 | CROVIA(TGRS 2024), 跨视角3D DA |

**关键确认（修订）**：截至当前检索，尚未发现直接面向**"开放道路自然驾驶数据集 → C-NCAP/E-NCAP封闭场地测试场景生成"**的公开论文。该判断必须在投稿前用 IEEE Xplore、Scopus/WoS、Semantic Scholar、OpenAlex、arXiv 与 Google Scholar 再核验一次；论文中不得写成无条件的“全领域为零”，而应写成“to the best of our knowledge after systematic search”。

### 1.5 Mamba/SSM 应用全景（Gemini Task 5）

**核心发现**：

| 子任务 | 代表工作 | 性能特征 |
|--------|---------|---------|
| 轨迹预测 | **Tamba (CVPR 2025)** | b-minFDE6=1.89, 4.54M参数, 比QCNet少40.7% |
| 轨迹预测 | Social-Mamba (ICRA 2026) | Cycle Mamba + 三元组因子分解, ETH/UCY SOTA |
| **场景生成** | **GEM (2026)** | 变形Mamba, LiDAR点云连续生成, nuScenes/KITTI |
| 运动规划 | Pi-DiMT (arXiv 2602.00808, venue 待核验) | Diffusion Mamba + PHNN, 强调物理可行性 |
| 端到端驾驶 | DRAMA (2024), ME³-BEV | Mamba替代Attention, 更低延迟和显存 |
| 交通流预测 | ST-Mamba, DST-Mamba | GFLOPs线性增长, 较GNN提速61.11% |

**"Mamba + 因果 + 物理 + 场景生成"四合一研判**：
- **结论：完全未被占据（Blue Ocean）**
- GEM：Mamba + 场景生成，但无因果图、无物理硬约束
- Pi-DiMT：Mamba + 物理约束，但仅单车规划器，非场景生成
- CRAJ (TVT 2025)：Mamba + 因果推断，但仅离线RL决策层
- CausalDrive：因果 + 场景预测 + LLM，但用Transformer非Mamba，无物理约束

**关键参考**：Tamba 的联合折线编码(JPE) + 交叉状态空间注意力解码器设计、Social-Mamba 的自我中心社交网格 + 三元组因子分解、GEM 的多路径变形Mamba架构，均为我们PI-Causal Mamba的架构设计提供直接参考。

### 1.6 SOTA 定量结果对标（Gemini Task 6）

**核心发现**：

**Waymo Open Motion Dataset**：
- MTR v3 (Ensemble 1st): Soft mAP=0.4967, minFDE₆=1.1062m, MR=0.1098
- Tamba (Single): minFDE₆=1.24m, MR=0.17, 4.54M params

**INTERACTION**：
- FJMP: ADE=0.81m, FDE=1.89m, CR=0.23%
- DSA (KAN-based): 风格自适应，细分操作精细化评估

**nuScenes**：
- OmniScene: L2=规划误差, CR极低; MomAD在Turning-nuScenes碰撞率相对SparseDrive大幅下降

**Argoverse 2**：
- DeMo (Ensemble): minFDE₆=1.11m, MR=0.12
- SEAM: 流式实时预测，avgMinFDE₆=1.21m

**安全关键指标基准**：

| 指标 | SOTA范围 | Q1期刊目标值 |
|------|---------|------------|
| 碰撞率(CR) | STRIVE ~20-30%, Any2Critical ~46% | ~25-35% (保持JSD低) |
| 近距事件率(NMR) | STRIVE ~25%, FlowVAE ~30% | >FlowVAE上限 |
| 中位数 minSTTC | STRIVE ~1.1s, FlowVAE ~1.7s | <1.2s |
| 动力学JSD(速度) | FlowVAE 0.083, STRIVE 0.135 | <0.10 |
| 动力学JSD(加速度) | FlowVAE 0.142, STRIVE 0.209 | <0.15 |
| 动力学JSD(Jerk) | FlowVAE 0.215, STRIVE 0.157 | <0.20 |

### 1.7 评估指标体系（Copilot Task 7）

**核心发现**：六大类指标体系：

| 类别 | 代表指标 | 成熟度 | 报告率 |
|------|---------|--------|--------|
| a) 轨迹保真度 | ADE, FDE, minADE, minFDE, MR, DAO | **Standard** | 普遍 |
| b) 安全关键性 | CR, NMR, TTC, PET, DRF, RSS Compliance | Standard/Emerging | 部分 |
| c) **物理可行性** | **加速度/曲率/摩擦圆违反率, KFR** | **Emerging/Proposed** | **极少** |
| d) 多样性与覆盖率 | Diversity Score, 参数空间覆盖率 | Emerging | 零星 |
| e) **因果可解释性** | **Root-Cause Score, Intervention Consistency** | **Proposed** | **无** |
| f) 域迁移与泛化 | Wasserstein距离, MMD, 目标域性能损失 | Standard/Emerging/Proposed | 少量 |

**关键判断**：
- 物理可行性指标（如KFR）在场景生成论文中几乎不被系统报告
- 因果可解释性指标（Root-Cause Score, Intervention Consistency）完全未被提出
- 现有方法存在"四重断裂"：高碰撞/低误差 vs 物理可执行性缺失 vs 因果黑盒 vs 域泛化盲区
- 我们提出的KFR、Root-Cause Attribution Score、Intervention Consistency填补了系统性空白

### 1.8 CNCAP/NCAP/ISO 标准约束（Copilot Task 8）

**核心发现**：

**C-NCAP 2024 可量化硬约束**：
- VUT速度区间：AEB 20-40 km/h, FCW 50-80 km/h, CCRH 80/120 km/h
- 目标物速度：GVT 40/50 km/h (横穿), 行人5/6.5 km/h, 骑行者15 km/h
- TTC触发条件：行人误触发TTC≤1.4s, 弯道超越TTC≤1.9s
- 路径精度：横向偏差≤±0.1m
- 隐含约束：摩擦圆|a|≤μg (约1g纵向, 0.8g横向)

**Euro NCAP 2023/2026**：类似C-NCAP，速度离散阶梯测试（10/30/50 km/h），目标物参数标准化，精度≤±0.1m。

**ISO 34504:2024**：提供参数维度框架（速度/曲率/密度分段标签），但不提供具体数值约束。对生成模型的价值在于指明参数空间覆盖维度。

**ISO 21448 (SOTIF)**：强调"触发条件"和"性能限制"的概念约束，引导生成模型向已知/未知不安全场景空间探索。

**学术转化现状**：
- 绝大多数主流学术方法**未将NCAP/ISO标准约束作为生成模型的内嵌先验**
- 标准约束常被视为后处理验证步骤，而非生成过程的内生约束
- 少数例外：Lukas Birkemeyer等(ITSC 2023)的SOTIF合规场景生成、Finkeldei等(IV 2023)的时序逻辑形式化
- **研究空白确认**：如何系统性地将NCAP/ISO标准约束形式化为可嵌入生成模型的数学约束，是一个未被充分探索的方向

---

## 二、六维度交叉差距分析矩阵

### 2.1 差距矩阵总览

下表横轴为6个研究维度，纵轴为评估维度，标注各方法的覆盖状态：
- ● = 已充分覆盖
- ◐ = 部分覆盖/有局限
- ○ = 完全未覆盖/空白

| 方法/维度 | D1:场景生成 | D2:因果推断 | D3:物理约束 | D4:域自适应 | D5:Mamba/SSM | D6:标准评估 |
|-----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| STRIVE (CVPR 2022) | ● | ○ | ○ | ○ | ○ | ○ |
| TrafficGen (ICRA 2023) | ● | ○ | ○ | ○ | ○ | ○ |
| DiffScene (AAAI 2025) | ● | ○ | ◐软约束 | ○ | ○ | ◐ |
| AdvSim (CVPR 2023) | ● | ○ | ○ | ○ | ○ | ○ |
| **CounterScene (2026.03)** | ● | **●单变量** | ○ | ○ | ○ | ◐ |
| **SaFeR (2026)** | ● | ◐(区分可避/不可避免) | **●LFR硬约束** | ○ | ○ | ◐ |
| Adv-BMT (NeurIPS 2025) | ● | ◐(逆向因果) | ◐后校验 | ◐跨框架 | ○ | ○ |
| CausalAF (CoRL 2023) | ◐ | ●(专家先验DAG) | ◐因果掩码 | ◐跨场景 | ○ | ○ |
| ChatScene (CVPR 2024) | ● | ◐语义归因 | ◐Scenic语法 | ◐ | ○ | ◐ |
| LLM-attacker (TITS 2025) | ● | ◐碰撞因果评估 | ◐DenseTNT约束 | ○ | ○ | ○ |
| **Pi-DiMT (ICRA 2026)** | ○ | ○ | **●PHNN硬约束** | ○ | **●** | ○ |
| **Tamba (CVPR 2025)** | ○ | ○ | ○ | ○ | **●** | ○ |
| GEM (2026) | ● | ○ | ○ | ○ | **●变形Mamba** | ○ |
| CausalVAD (CVPR 2026) | ○ | ●(后门调整) | ○ | ○ | ○ | ○ |
| RiskMV-DPO (2026) | ● | ◐风险建模 | ◐几何锚点 | ◐多视角 | ○ | ●(CNCAP对齐) |
| NeuroNCAP (CVPR 2024) | ◐ | ○ | ○ | ○ | ○ | ●(Euro NCAP) |

**我们的PI-Causal Mamba目标**：

| D1 | D2 | D3 | D4 | D5 | D6 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| ● | **●多变量级联** | **●PHNN硬约束+多主体** | **●GRL-DANN多源** | **●Mamba+因果+物理** | **●KFR+因果指标+CNCAP嵌入** |

### 2.2 交叉空白矩阵（Cross-Gap Matrix）

标注每对维度的"交叉空白"状态——即**同时解决两维度的工作是否存在**：

| 交叉维度 | D1-场景生成 | D2-因果 | D3-物理 | D4-DA | D5-Mamba |
|---------|:----------:|:------:|:------:|:----:|:------:|
| **D2-因果** | CounterScene ◐(单变量) | — | | | |
| **D3-物理** | SaFeR/DiffScene ◐(单对抗) | ○ **完全空白** | — | | |
| **D4-DA** | ○ **完全空白** | ○ **完全空白** | ○ **完全空白** | — | |
| **D5-Mamba** | GEM ◐(无因果/物理) | CRAJ ◐(仅RL) | Pi-DiMT ◐(仅单车) | ○ **完全空白** | — |
| **D6-标准** | RiskMV-DPO ◐(仅视觉) | ○ **完全空白** | ○ **完全空白** | ○ **完全空白** | ○ **完全空白** |

**关键解读**：
- **三元差距 (D2+D3+任意)**：因果推断+物理约束在安全关键场景生成中的结合仍不充分
- **四元差距 (D2+D3+D4+D5)**：当前检索尚未发现同时覆盖物理、因果、跨域与 SSM 生成的封闭场地验证框架
- **五元差距 (D2+D3+D4+D5+D6)**：加入标准评估维度后，PICASO 的差异化主要体现在工业目标域和可执行性审计

---

## 三、蓝海空间确认与修正

### 3.1 预判验证结果

| 序号 | Stage2预判蓝海空间 | 验证状态 | 证据来源 | 修正说明 |
|------|-------------------|---------|---------|---------|
| 1 | 物理约束+因果引导+生成模型的统一框架 | **✅ 确认** | Task 1表2 + Task 3表1 + Task 5§4 | Pi-DiMT有物理无因果，CounterScene有因果无物理，GEM有场景无因果/物理 |
| 2 | 场景生成中的多源域自适应 | **需投稿前复核** | Task 4 §2.A | 当前检索未发现"开放道路→封闭场地"直接工作 |
| 3 | 开放道路→封闭场地的跨域场景迁移 | **需投稿前复核** | Task 4全文 + Task 1 §4 | 场景生成方向暂未发现直接工作；相邻领域需继续检索 |
| 4 | 超越ADE/FDE的安全关键场景评估矩阵 | **✅ 确认** | Task 7 §3.c-d | 物理可行性指标极少报告，因果可解释性指标完全未被提出 |
| 5 | Mamba与因果交互的深度结合 | **需核验** | Task 5 §3 + 表6 | 不以“首个提出者”为核心卖点，转为强调安全验证框架中的工程闭环 |
| 6 | CNCAP规程约束的形式化嵌入 | **✅ 确认并强化** | Task 8 §5 | 仅极少数工作(Birkemeyer等ITSC 2023)尝试SOTIF合规生成，CNCAP嵌入为零 |

### 3.2 新发现的蓝海空间（调研后新增）

| 序号 | 新增蓝海空间 | 发现来源 | 价值评估 |
|------|------------|---------|---------|
| N1 | **多智能体级联因果链(Causal Cascade Chain)生成** | Task 2 §4 + MACC框架 | **极高** — CounterScene单变量干预后，学界尚未突破多层级联干预；可直接对标"因果推断顶级方法" |
| N2 | **Mamba状态空间中的物理可执行性约束** | Task 3 §2 + Task 5 §3 | **高** — 先以解码后投影/KFR 审计为可执行主线，多体 PHNN 作为增强 |
| N3 | **可行域方法的工程替代/补充方案** | Task 3 §3 vs Task 1表2 | **中高** — 与 SaFeR/LFR 类方法对比时，应强调 ABD 执行约束和可复现工程验证 |
| N4 | **C-NCAP ABD机器人执行约束的形式化** | Task 8 §2.1 + ABD精度数据 | **高** — 将ABD的±0.02m路径精度、<0.2km/h速度精度形式化为生成模型的解码器约束 |
| N5 | **因果归因+生成+评估的完全闭环(Shapley-in-the-Loop)** | Task 2 §5 + Task 7 §3.e | **中高** — 实现"测试场景生成→规划器压测崩溃→实时因果失效定位→生成变体再测试"的全自动闭环 |

### 3.3 差异化创新点强化路径

基于全部8份报告的交叉验证，我们的三大创新点在以下方向获得了**更强的证据支撑**：

**C1: PI-Causal Mamba (物理-因果统一生成器)**
- 强化证据：Pi-DiMT、CounterScene、GEM 等相邻工作分别覆盖物理、因果或生成子问题；PICASO 的差异在于安全验证框架内的统一评估与目标域验证
- 新增论证维度：物理投影/KFR 审计可作为 LFR 类可行域方法的工程互补路径
- 对比基线：Pi-DiMT (物理基线)、CounterScene (因果基线)、Tamba (架构基线)

**C2: Counterfactual Offline Closed-Loop (反事实离线闭环)**
- 强化证据：CounterScene仅单变量干预→多智能体级联反事实是下一步必然方向
- 新增论证维度：MACC级联因果图发现+Joint Spatial-Temporal ODE+Closed-loop Diff-WM
- 对比基线：CounterScene (单变量反事实基线)、CausalAF (专家先验CVM基线)

**C3: Multi-Source Domain Adaptation (多源域自适应)**
- 强化证据：Task 4 检索暂未发现直接研究"开放道路→封闭场地场景生成"的论文，投稿前需按修订检索协议复核
- 新增论证维度：独有的多品牌CNCAP ABD数据资产是此方向不可替代的竞争壁垒
- 对比基线：无直接可比的跨域场景生成基线（可用No-DA ablation + 单源训练baseline对比）

---

## 四、Related Work 章节草稿

> 以下为SCI Q1期刊论文的Related Work章节草稿，按六维度组织，可直接用于后续Stage 4论文撰写。

### 4.1 Safety-Critical Scenario Generation

The generation of safety-critical driving scenarios has evolved from rule-based parameter sampling to deep generative models, forming three main paradigms: data-driven generation, adversarial/RL-based generation, and knowledge/LLM-driven generation [FM-AD-Survey, 2025].

**Data-driven methods** learn statistical distributions from real-world driving logs. TrafficGen [ICRA 2023] pioneered autoregressive Transformer-based traffic scene generation from Waymo Open Dataset. Scenario Dreamer [CVPR 2025] advanced this with vectorized latent diffusion, achieving 83% lower generation latency than rasterized baselines. DiffScene [AAAI 2025] introduced multi-objective guided diffusion to bias generation toward collision-prone regions. However, these methods lack explicit physics constraints and causal interpretability — they learn correlations, not causations, and cannot explain *why* a generated scenario is dangerous.

**Adversarial and RL-based methods** formulate scenario generation as a game between the ego vehicle and adversarial agents. AdvSim [CVPR 2023] demonstrated that adversarial perturbation of agent behaviors can expose autonomy failures, but often produces physically infeasible trajectories. SaFeR [2026] addressed this with feasibility-constrained token resampling grounded in Largest Feasible Region (LFR) theory, achieving 86.5% scenario solvability while maintaining 76.1% collision rate. FREA [CoRL 2024] similarly used LFR to bound adversarial behavior. However, LFR-based methods are limited to single-adversary settings and rely on computationally expensive offline RL value functions that may suffer numerical instability at complex intersections.

**Knowledge and LLM-driven methods** leverage large language models to translate natural language specifications into simulation scenarios. ChatScene [CVPR 2024] uses LLM agents with a Retrieval-RAG database to generate Scenic programs for CARLA, but provides only discrete high-level behavior assignments rather than continuous dynamic trajectories. LLM-attacker [TITS 2025] demonstrated 92.92% attack success rate with multi-agent LLM collaboration, yet trajectory optimization assumes independent background vehicles, causing physically inconsistent multi-vehicle conflicts.

**Critical gaps**: Existing methods still leave several issues insufficiently addressed for closed-field safety verification: (1) physical feasibility is often handled by post-filtering, soft penalties, or limited feasible-region control rather than being evaluated against hardware execution limits; (2) causal reasoning for scenario generation remains early, with CounterScene representing a closely related counterfactual direction; (3) domain adaptation from open-road datasets to standardized test protocols (C-NCAP/Euro NCAP) has not been directly addressed in the public literature we have found so far; (4) multi-agent cascade causal chains (e.g., Vehicle A swerves to avoid obstacle -> Vehicle B is forced into emergency lane change -> Ego collision) remain difficult to generate and diagnose in a reproducible way.

### 4.2 Causal Inference for Autonomous Driving

Causal inference has recently emerged as a principled framework to address the fundamental limitations of correlation-based learning in autonomous driving [CausalVAD, CVPR 2026].

**Structural Causal Models (SCM)** have been applied to de-confound end-to-end driving policies. CausalVAD [CVPR 2026] proposed Sparse Causal Intervention Scheme (SCIS) using backdoor adjustment to eliminate spurious correlations between scene context and planning trajectories. CICR [2026] employed front-door adjustment for pedestrian trajectory prediction debiasing. These works focus on *ego-centric de-confounding* rather than *scene-level causal generation*.

**Counterfactual reasoning** represents the most relevant direction to our work. CounterScene [2026.03] pioneered counterfactual causal reasoning in generative BEV world models for safety-critical closed-loop evaluation, achieving collision rate improvement from 12.3% to 22.7% on nuScenes. However, CounterScene is limited to **single-variable intervention** — it selects one critical agent and removes its safety margin. Real-world chain-reaction accidents involve **multi-agent causal cascades** (e.g., lead vehicle A braking → vehicle B swerving → ego collision), which require joint multi-variable intervention that current methods cannot perform.

**Causal discovery** methods including CRiTIC [ICRA 2025] and TR-JCDN [2026] have been developed to identify causal structures from traffic time series, but they are primarily used for post-hoc accident diagnosis rather than online generative guidance.

**Key research gap**: A framework that (a) estimates causal interaction structure for variable-length multi-agent scenes, (b) supports multi-variable cascade counterfactual intervention, and (c) evaluates the generated scenarios under physical and test-protocol constraints is not directly covered by the public literature reviewed so far.

### 4.3 Physics-Informed Generative Models

Ensuring physical feasibility of generated trajectories is critical for scenario validity. Three main approaches exist:

**Soft constraint (PINN-based)** methods add physics violation penalties to the loss function. Physics-constrained diffusion models [ICLR 2025] reduced PDE residuals by two orders of magnitude through loss-based regularization. DiffScene [AAAI 2025] uses gradient guidance during diffusion denoising to softly constrain acceleration limits. However, soft penalties cannot guarantee strict constraint satisfaction — the trade-off between diversity and feasibility depends on penalty weight tuning.

**Hard constraint (structure-embedded)** methods encode physical laws directly into network architecture. Port-Hamiltonian Neural Networks (PHNN) enforce energy conservation and dissipation laws through architectural design. Physics-informed Diffusion Mamba Transformer [Pi-DiMT, ICRA 2026] first combined PHNN with Mamba SSM and diffusion models for trajectory generation. However, Pi-DiMT is limited to **single-vehicle trajectory reconstruction** — it treats surrounding agents as context rather than jointly constrained interactive entities. Lagrangian Neural Networks (LNN) and Kinematics-aware Latent SDE [2023] similarly focus on single-agent dynamics.

**Feasible region control** methods enforce constraints during generation: SaFeR and FREA use LFR-based post-sampling rejection or token resampling. These are computationally expensive and limited to single-adversary settings.

**Key research gap**: For the first implementation, the defensible gap is not a guaranteed fully coupled multi-agent PH system inside Mamba, but a practical pipeline that makes physical feasibility auditable: differentiable physical losses, decoder-side projection, ABD execution constraints, and KFR/violation reporting. Stronger claims about native multi-agent Port-Hamiltonian embedding should be kept as an advanced design or ablation until experimentally validated.

### 4.4 Domain Adaptation for Autonomous Driving

Domain adaptation (DA) and domain generalization (DG) have been extensively studied for perception tasks in autonomous driving [PanDA, CVPR 2026], but their application to scenario generation remains unexplored.

**Sim-to-Real adaptation** is the most mature direction, with adversarial DA [Li et al., WACV 2023] achieving significant detection AP improvements in adverse weather, and PanDA [CVPR 2026] enabling multi-modal 3D panoptic segmentation UDA.

**Cross-dataset generalization** has been advanced by AdapTraj [ICDE 2024] using causal feature decomposition for multi-source trajectory prediction domain generalization, and MUSDA [2026] for multi-source 3D detection.

**Cross-view adaptation** from UAV aerial to vehicle-mounted perspective was pioneered by CROVIA [TGRS 2024] using geometric consistency constraints.

**Critical gap**: After the current search across perception, prediction, planning, and scenario generation literature, we have not found a published work that directly addresses the domain shift from open-road naturalistic driving datasets (Waymo, nuScenes) to standardized closed-field test protocols (C-NCAP, Euro NCAP) for scenario generation. This gap is plausible because closed-field test data is proprietary, the domain shift is large, and the problem requires adaptation of both scene geometry and agent behavior distributions. We therefore position our work as a first systematic attempt, while keeping the final priority claim subject to the submission-time literature search.

### 4.5 Sequence Modeling Architectures: From Transformer to Mamba

The evolution of sequence modeling architectures has direct implications for trajectory generation efficiency.

**Transformer-based methods** dominate current trajectory prediction and generation but face the O(N²) complexity bottleneck of self-attention. QCNet [CVPR 2023] and MTR [NeurIPS 2022] achieve strong accuracy at the cost of quadratic scaling with agent count and sequence length.

**State Space Models (SSM)** offer a compelling alternative. Mamba [Gu & Dao, 2023] introduced selective SSM (S6) with input-dependent parameters, achieving linear O(N) complexity while maintaining long-range dependency modeling. Tamba [CVPR 2025] was the first to apply Mamba to multi-agent trajectory prediction on Argoverse, achieving 1.89 b-minFDE₆ with 4.54M parameters — 40.7% fewer than QCNet (7.66M). Social-Mamba [ICRA 2026] further addressed spatial permutation invariance in SSM through ego-centric social grid sequentialization and social triplet factorization. GEM [2026] extended Mamba to 3D LiDAR world model generation using deformable Mamba architecture.

**Mamba-causal-physics integration** remains in early stages. CRAJ [TVT 2025] combined Mamba with unsupervised confidence-based causal inference for RL decision-making. MADiff used motion-driven selective scan for ego-motion-conditioned trajectory generation. Pi-DiMT combined Mamba with PHNN for physically-constrained single-vehicle planning. No work has integrated Mamba, causal inference, and multi-agent physics constraints into a unified generative framework — the "four-in-one" integration we propose.

### 4.6 Evaluation Metrics and Standard Constraints

**Evaluation metrics** for scenario generation have been dominated by trajectory fidelity metrics (ADE, FDE). Recent work has added safety-criticality metrics (collision rate, near-miss rate), but physical feasibility metrics (acceleration/curvature violation rates) are still not consistently reported, and causal interpretability metrics are not standardized. We identify this as a critical evaluation gap and propose Kinematic Feasibility Rate (KFR), Root-Cause Attribution Score, and Intervention Consistency as auditable evaluation dimensions.

**Standard constraints** from C-NCAP 2024, Euro NCAP 2023/2026, and ISO 34504/21448 define explicit operational boundaries — vehicle speed ranges, target motion parameters, TTC trigger conditions, and ABD robot execution precision (±0.02 m lateral, <0.2 km/h speed). However, current academic scenario generation methods largely ignore these constraints, treating them as post-hoc verification rather than embedded generative priors. Only isolated works [Birkemeyer et al., ITSC 2023] have attempted SOTIF-compliant scenario generation. Systematic formalization of NCAP/ISO constraints as mathematical priors for generative models remains an open problem.

---

## 五、对 Stage 3（方法与系统设计）的建议

### 5.1 创新点最终确认（基于差距分析）

基于全部8份报告的交叉验证，建议在Stage 3中确认以下三大创新点：

**C1: PI-Causal Mamba — 物理-因果统一状态空间生成器**
- 核心主张：在Mamba选择性状态空间中同时嵌入PHNN多体物理守恒律 + 变长自适应因果图发现
- 差异化：Pi-DiMT仅单车物理、CounterScene仅单变量因果、GEM无物理/因果
- 技术突破：实现"多主体物理可执行性审计 + 因果交互学习 + 线性复杂度序列建模"的统一实验框架

**C2: Multi-Agent Cascade Counterfactual (MACC) — 多智能体级联反事实闭环**
- 核心主张：从CounterScene的单变量干预扩展到多智能体级联因果图+序列反事实干预
- 差异化：CounterScene单变量、CausalAF专家先验DAG、SafeAlign-VLA仅安全对齐
- 技术突破：生成并分析"前车避障→中车急转→自车碰撞"等多层级联反事实案例，是否构成普适机制由实验决定

**C3: GRL-DANN Multi-Source Domain Adaptation — 开放道路→封闭场地跨域迁移**
- 核心主张：利用GRL-DANN将Waymo/INTERACTION域与CNCAP ABD域联合对抗训练
- 差异化：当前检索暂未发现直接研究"开放道路→封闭场地场景生成"的公开工作；最终优先权以投稿前系统检索为准
- 竞争壁垒：独有CNCAP多品牌ABD测试数据（60+信号通道）

### 5.2 方法设计关键参考

| 参考来源 | 可借鉴设计 | 适用模块 |
|---------|-----------|---------|
| Tamba (CVPR 2025) | 联合折线编码(JPE) + 交叉状态空间注意力解码器 | C1 Mamba编码器-解码器架构 |
| Social-Mamba (ICRA 2026) | 自我中心社交网格 + 三元组因子分解 | C1 多主体交互建模 |
| Pi-DiMT (ICRA 2026) | PHNN模块嵌入Diffusion Mamba | C1 物理约束层 |
| CounterScene (2026.03) | 因果交互图(CIG) + 时空反事实引导 | C2 因果发现与干预 |
| GEM (2026) | 变形Mamba + 动静分离器 | C1 场景分词与生成 |
| CausalAF (CoRL 2023) | 因果掩码约束智能体状态关联 | C2 因果图结构化 |
| CausalVAD (CVPR 2026) | SCIS后门调整参数化 | C2 干预机制 |
| AdapTraj (ICDE 2024) | 因果特征分解→域共享/域特有 | C3 域不变特征学习 |
| PanDA (CVPR 2026) | 伪标签+多模态增强UDA | C3 目标域伪标签策略 |

### 5.3 实验设计关键对标

| 数据集 | 核心指标 | 对标方法 | Q1目标 |
|--------|---------|---------|--------|
| Waymo Open Motion | minFDE₆, MR, CR | Tamba, MTR v3, QCNet | minFDE₆~1.2m, MR<0.15 |
| INTERACTION | ADE, FDE, CR | FJMP, DSA, IPP* | ADE<0.8m, CR>25% |
| CNCAP ABD (目标域) | CR, NMR, KFR, 目标域性能损失 | No-DA基线, Single-source基线 | KFR>95%, CR提升>10% over No-DA |
| nuPlan (闭环) | SR (求解率), CR, 碰撞率@闭环 | SaFeR, CounterScene | SR>0.85, CR>20% |

### 5.4 评估体系设计

建议在Stage 3中正式定义以下新指标：

1. **KFR (Kinematic Feasibility Rate)**：场景轨迹全程满足所有物理约束（加速度/曲率/摩擦圆/ABD执行限制）的时长比例
2. **Root-Cause Attribution Score**：基于Shapley值的各智能体对碰撞风险贡献的定量分解
3. **Intervention Consistency (IC)**：反事实干预前后场景危险性变化的一致性
4. **Target Domain Performance Drop (TDPD)**：源域→目标域的性能衰减率，量化域自适应必要性

---

## 六、文件版本与后续步骤

> **文件版本**: v1.0
> **数据来源**: 8份 Gemini/Copilot 研究助手调研报告（Stage 2 文件夹）
> **下一步**: 进入 **Stage 3（方法与系统设计）**，基于本报告的差距分析确认创新点，设计PI-Causal Mamba详细架构、MACC级联反事实算法流程、GRL-DANN域自适应网络结构
