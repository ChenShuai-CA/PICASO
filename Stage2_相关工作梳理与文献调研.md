# Stage 2：相关工作梳理与文献调研

> 依据 `自动驾驶安全关键场景生成方向：工程科研流程与 T-ITS 投稿指南.docx` Stage 2 框架编写
> 编写日期：2026-05-20

---

## 2026-05-28 专家审查修订：文献检索协议必须更新

原方案的调研方向是正确的，但检索协议存在一个会影响可信度的问题：**Microsoft Academic 已停止服务，不能再作为 2026 年系统检索工具**。后续所有文献结论，尤其是“尚未发现/零论文/首次”等空白判断，必须按以下协议重新核验：

1. **数据库组合**：IEEE Xplore、Scopus 或 Web of Science、Semantic Scholar、OpenAlex、arXiv、Google Scholar；CVPR/ICCV/ECCV/NeurIPS/ICLR/CoRL/ICRA/ITSC/T-ITS/TVT/TR Part C 以官方页面或出版商页面为准。
2. **每篇文献必须记录**：题名、作者、年份、venue/status（已录用/预印本/under review）、DOI 或 arXiv ID、代码链接、数据集、指标、与 PICASO 的直接差异。
3. **空白判断写法**：不得写“全领域零论文”作为无条件事实；统一写成“截至投稿前最终检索，在公开可检索文献中尚未发现直接研究开放道路自然驾驶数据到 C-NCAP/E-NCAP 封闭场地场景生成迁移的工作”。
4. **2026 预印本处理**：arXiv 论文可以作为前沿相关工作，但除非有会议/期刊官网佐证，不得写成 CVPR/ICRA/T-ITS 已录用论文。
5. **Related Work 进入论文前必须复核**：Stage 2 中所有由研究助手给出的定量结果、会议归属、SOTA 数值、代码可用性都要做一次人工核对，未核对项在论文草稿中标注 `[TBD: verify]`。

---

## 一、文献调研维度矩阵

本文的研究问题（物理约束因果场景生成 + 域自适应跨域迁移）横跨 5 个研究方向的交叉地带。文献调研需覆盖以下 6 个维度：

| 维度 | 核心问题 | 产出 |
|------|---------|------|
| **D1: 安全关键场景生成** | 现有方法如何生成安全关键场景？分类、优缺点、未解决的问题？ | 场景生成方法全景对比表 |
| **D2: 因果推断在自动驾驶中的应用** | 因果发现/反事实推理如何用于驾驶安全？与 D1 的交集？ | 因果方法在 ADS 中的前沿定位 |
| **D3: 物理约束生成模型** | PINN/PHNN 等物理约束方法如何嵌入生成模型？可执行性约束的最新方案？ | 物理约束方法技术路线图 |
| **D4: 域自适应与跨域迁移** | DA/UDA/DG 方法在自动驾驶中的应用？是否存在"开放道路→封闭场地"的迁移工作？ | 域自适应方法列表 + 空白确认 |
| **D5: 序列建模架构演进** | Transformer → Mamba/SSM → Diffusion+Mamba 的技术路线？Mamba 在轨迹建模中的 SOTA？ | 架构对比时间线 |
| **D6: 数据集与评估标准** | 各数据集的规模和特点？CNCAP/NCAP 规程对场景生成的约束？评估指标的演进？ | 数据集全景表 + 评估指标矩阵 |

---

## 二、已知文献梳理（Claude Code 已完成）

以下文献在已有文档中被反复引用，属于必读核心论文。

### 2.1 场景生成（D1）

| 论文 | 年份/出处 | 方法类型 | 核心贡献 | 与我们工作的关系 |
|------|----------|---------|---------|---------------|
| **STRIVE** (Rempe et al.) | CVPR 2022 | CVAE + 交通先验 | 学习交通先验后生成危险场景 | **必须对比的 Baseline**。基于 CVAE，无因果理解，无物理约束，无 DA |
| **AdvSim** (Wang et al.) | CVPR 2023 | 对抗扰动 | 修改 agent 行为以增加自动驾驶系统失败概率 | 以碰撞率最大化为目标，但忽略物理可行性和因果可解释性 |
| **TrafficGen** (Feng et al.) | 2023 | GAN | 基于 GAN 的交通场景生成 | GAN 范式代表，缺乏风险感知和域适应 |
| **CTG++** (Zhong et al.) | 2024 | Diffusion | 基于扩散模型的条件场景生成 | Diffusion 范式代表，推理速度慢，无因果引导 |
| **CounterScene** (Jing et al.) | arXiv 2603.21104 | Causal World Model | 将反事实因果推理引入安全关键场景生成；因果对抗 agent 识别 + 最小干预 + 交互式世界模型 | **最相关的前沿工作**。nuScenes+nuPlan 验证；其报告的碰撞率等数值需以 arXiv 原文为准。差异：物理执行约束、封闭场地 DA 与 ABD 验证不是重点 |
| **SaFeR** (Cui et al.) | arXiv 2603.04071 | Token Resampling | 可行性约束 Token 重采样；Largest Feasible Region (LFR) 离线 RL 保证可避撞性 | **物理可行性约束的 SOTA 方案**。但基于 Transformer+离散 Token，无因果推理，无 DA |
| **ChatScene** | CVPR 2024 | LLM + 知识驱动 | 用 LLM 从自然语言规范生成 OpenSCENARIO 场景 | 知识驱动范式代表，但依赖手工模板，覆盖率受限于知识库 |

### 2.2 因果推断（D2）

| 论文 | 年份/出处 | 方法类型 | 核心贡献 | 与我们工作的关系 |
|------|----------|---------|---------|---------------|
| **CausalVAD** | arXiv 2603.18561 | SCM + Backdoor Adjustment | 通过后门调整消除自动驾驶端到端模型中的伪相关 | 因果去偏方法，非场景生成。但 SCM 使用方式可参考 |
| **Causal Composition Diffusion** (Lin et al.) | CVPR 2025 | Causal Diffusion | 因果组合扩散用于闭环交通生成 | 因果+扩散范式的代表，但聚焦于交通流，非安全关键场景 |
| **Pearl (2009)** | 经典著作 | SCM 理论 | *Causality: Models, Reasoning, and Inference* | 因果推断的理论基础 |
| **Schölkopf et al. (2021)** | Proc. IEEE | Causal Representation Learning | 因果表示学习的综述 | 因果与 ML 结合的框架性参考 |

### 2.3 物理约束生成（D3）

| 论文 | 年份/出处 | 方法类型 | 核心贡献 | 与我们工作的关系 |
|------|----------|---------|---------|---------------|
| **Physics-informed Diffusion Mamba Transformer** (Zhou et al.) | arXiv 2602.00808（venue 需投稿前核验） | Diffusion Mamba + PHNN | 将 Port-Hamiltonian 约束、Mamba 与 Diffusion 结合用于驾驶轨迹/规划相关任务 | **直接相关**。但其任务边界、数据集、是否为多主体安全关键场景生成需以原文核验后再写入论文 |
| **Raissi et al. (2019)** | J. Comp. Physics | PINN | 物理信息神经网络的奠基性工作 | PINN 方法论基础 |
| **Trajectron++** (Salzmann et al.) | ECCV 2020 | CVAE + 动态可行性 | 将车辆运动学约束嵌入轨迹预测 | 早期物理约束+轨迹预测的代表，但约束形式较简单 |

### 2.4 域自适应（D4）

| 论文 | 年份/出处 | 方法类型 | 核心贡献 | 与我们工作的关系 |
|------|----------|---------|---------|---------------|
| **Ganin et al. (2016)** | JMLR | DANN/GRL | 域对抗训练的开创性工作 | GRL-DANN 的方法论基础 |
| 域自适应 × 场景生成 | — | — | **文献检索中未发现相关工作** | **这是我们的空白机会** |

### 2.5 序列建模架构（D5）

| 论文 | 年份/出处 | 核心贡献 | 与我们工作的关系 |
|------|----------|---------|---------------|
| **Mamba** (Gu & Dao) | arXiv 2312.00752 | 选择性状态空间模型；线性复杂度；5× Transformer 推理吞吐 | SSM 理论基础 |
| **Trajectory Mamba (Tamba)** | CVPR 2025 | 首个将 Mamba 用于多主体轨迹预测（Argoverse）；Attention-Mamba 混合架构 | Mamba 在轨迹预测中的 SOTA |
| **DriveMamba** | OpenReview | Mamba 用于端到端自动驾驶 | Mamba 在 ADS 中的应用案例 |
| **SocialVAE** (Xu et al.) | ECCV 2022 | Timewise VAE + GRU 用于行人轨迹预测 | 传统 VAE 范式代表（我们不做的基础基线） |

### 2.6 数据集与标准（D6）

| 数据集/标准 | 规模 | 场景类型 | 用途 |
|------------|------|---------|------|
| **Waymo Open Motion Dataset** | ~487K 训练场景, 9.1s/场景, 10Hz | 城市开放道路，多主体+地图 | 主训练集 |
| **INTERACTION** | ~55K 轨迹, 11 场景 | 交叉口/环岛/合流（跨国） | 辅助训练 + 强交互验证 |
| **CNCAP 封闭场地** | 多品牌 ABD 实测 | 结构化测试规程 | 域自适应目标域 + 工业验证 |
| **ISO 34504:2024** | 标准 | 自动驾驶场景分类 | 场景标签体系参考 |
| **C-NCAP 2024 附录 L/O** | 标准 | ADAS/VRU 试验规程 | 物理约束边界定义 |

---

## 三、需补充调研的空白区域（委托 Gemini / Copilot）

以下列出了已有文档中**未充分覆盖、需要系统性检索**的 10 个具体调研问题，每个都附有精确的搜索提示词。

---

### 任务 1：场景生成方法全景调研（委托 Gemini Deep Research）

```
请使用 Google Scholar 进行系统性文献调研，回答以下问题：

**调研主题**: 自动驾驶安全关键场景生成方法的完整技术全景

**具体任务**:
1. 按照 Wenhao Ding 等人的分类框架（数据驱动生成 / 对抗与强化学习生成 / 知识驱动生成）或更
   新的分类框架（如 TUM-AVS 的 Foundation Models for Scenario Generation and Analysis 
   survey, arXiv:2506.11526），系统列举每类方法的核心论文（每类至少 5 篇，含年份、出处、
   方法名、一句话核心贡献）。

2. 重点覆盖以下会议/期刊近 3 年（2023-2026）的论文：
   - CVPR / ICCV / ECCV / NeurIPS / ICRA / IROS
   - IEEE T-ITS / T-IV / TVT
   - ITSC / IV Symposium
   - 其他交通/自动驾驶领域重要期刊

3. 对每篇论文，请提取：
   - 生成范式（VAE/GAN/Diffusion/LLM/RL/Optimization-based/Other）
   - 使用的训练数据集和评估数据集
   - 是否包含物理可行性约束
   - 是否考虑因果可解释性
   - 是否有多数据集/跨域迁移实验
   - 核心指标的定量结果（碰撞率、ADE/FDE 等）

4. 特别检索：是否存在将深度学习生成模型应用于 C-NCAP/E-NCAP/ISO 标准测试场景的论文？
   是否存在使用封闭场地/硬件在环数据进行验证的论文？

5. 输出格式：按分类整理成表格，并标注每篇论文与我们研究的"差异化空间"（即该论文未解决的、
   而我们计划解决的问题）。
```

### 任务 2：因果推断 × 自动驾驶交叉领域（委托 Gemini Deep Research）

```
请使用 Google Scholar 进行系统性文献调研，回答以下问题：

**调研主题**: 因果推断与反事实推理在自动驾驶安全验证中的前沿应用

**具体任务**:
1. 检索 2023-2026 年间，在自动驾驶/智能交通领域应用以下因果方法的论文：
   - 结构因果模型 (Structural Causal Models, SCM)
   - 反事实推理 (Counterfactual Reasoning)
   - 因果发现 (Causal Discovery, 包括 PC/LiNGAM 及其现代替代方法)
   - 因果表示学习 (Causal Representation Learning)
   - Shapley 值 / 因果中介分析 (Causal Mediation Analysis)

2. 特别关注以下子方向：
   a) 使用因果推断进行场景生成的论文
   b) 使用反事实推理进行自动驾驶安全边界探索的论文
   c) 使用因果归因分析自动驾驶系统失效根因的论文
   d) 使用因果方法解决轨迹预测/交互建模中分布外泛化 (OOD Generalization) 的论文

3. 对每篇论文，请提取：
   - 因果方法的具体类型和技术细节
   - 解决的具体问题（场景生成/预测/安全分析/其他）
   - 使用的因果发现/推理算法
   - 实验验证方式和数据集
   - 局限性

4. 关键问题：在场景生成领域，"因果引导"是否已被充分探索？CounterScene (2026.03) 
   之后，是否有其他团队跟进反事实场景生成？目前的研究空白是什么？

5. 输出格式：按因果方法类型分类，列出关键论文、核心方法、与我们计划的差异。
```

### 任务 3：物理约束生成模型前沿（委托 Copilot 研究助手）

```
请使用 IEEE Xplore / Scopus 或 Web of Science / Semantic Scholar / OpenAlex / arXiv / Google Scholar 进行交叉检索，回答以下问题：

**调研主题**: 物理信息约束在自动驾驶轨迹/场景生成中的最新方法

**具体任务**:
1. 检索 2023-2026 年间，将以下物理约束方法嵌入深度生成模型的论文：
   - Physics-Informed Neural Networks (PINN)
   - Port-Hamiltonian Neural Networks (PHNN)
   - Lagrangian Neural Networks (LNN)
   - 其他基于物理定律的约束方法

2. 重点关注：
   a) 物理约束的具体形式（硬约束 Hard Constraints vs 软惩罚 Soft Penalties）
   b) 物理约束嵌入生成模型的方式（损失函数惩罚 / 架构设计 / 投影层 / 其他）
   c) 考虑的物理量（速度/加速度/jerk/曲率/横摆角速度/摩擦圆/轮胎模型 等）
   d) 是否针对多主体交互场景
   e) 是否有实验证明物理约束确实提升了生成轨迹的可行性

3. 特别检索：
   - Physics-informed Diffusion Mamba Transformer (ICRA 2026) 的后续引用或相关工作
   - 是否有论文将 PHNN 与 Mamba/SSM 结合用于自动驾驶
   - 是否有论文专门讨论"如何保证生成轨迹在 ABD 驾驶机器人等硬件平台上可执行"

4. 对比维度：现有物理约束方法中，约束粒度（单体运动学 vs 多体交互约束）、
   与场景生成（而非轨迹预测）的融合深度

5. 输出格式：按物理约束类型分类的论文列表，标注与我们 PI-Causal Mamba 方案的差异。
```

### 任务 4：域自适应 × 自动驾驶数据迁移（委托 Copilot 研究助手）

```
请使用 IEEE Xplore / Scopus 或 Web of Science / Semantic Scholar / OpenAlex / arXiv / Google Scholar 进行交叉检索，回答以下问题：

**调研主题**: 域自适应与域泛化方法在自动驾驶数据迁移中的应用

**具体任务**:
1. 检索 2023-2026 年间，以下域自适应方法在自动驾驶中的应用论文：
   - 对抗性域自适应 (Adversarial Domain Adaptation, DANN/GRL-based)
   - 基于差异的域自适应 (Discrepancy-based, MMD/CORAL)
   - 域泛化 (Domain Generalization)
   - 无监督/半监督域自适应 (UDA/SSDA)
   - 多源域自适应 (Multi-Source Domain Adaptation, MSDA)
   - 领域随机化 (Domain Randomization)

2. 特别关注以下几种域迁移（每个方向检索是否有相关工作）：
   a) 开放道路数据集 → 封闭场地测试数据的迁移
   b) 仿真数据 → 真实传感器数据的迁移 (Sim-to-Real)
   c) 一个数据集训练的模型 → 另一个数据集的泛化 (Cross-Dataset Generalization)
   d) 一个国家的驾驶数据 → 另一个国家的驾驶数据（如 US → China）
   e) 无人机航拍数据 → 车载传感器数据的迁移

3. 关键问题：
   - **是否存在任何将开放道路数据训练的场景生成模型迁移到 C-NCAP/E-NCAP 封闭场地的论文？**
   - 如果没有，相邻领域（如目标检测、轨迹预测）中是否有"开放道路→测试规程"的迁移工作？
   - 场景生成领域的跨数据集泛化相关论文有哪些？

4. 对每篇相关论文，提取：
   - 源域和目标域的具体定义
   - DA/DG 方法类型
   - 迁移效果（目标域性能提升幅度）
   - 是否涉及真实测试设备的数据

5. 输出格式：按迁移类型分类的论文矩阵，重点标注"开放道路→封闭场地迁移"相关论文
   （如果存在）或明确确认此方向为空白。
```

### 任务 5：Mamba/SSM 在自动驾驶中的最新应用（委托 Gemini Deep Research）

```
请使用 Google Scholar 进行系统性文献调研，回答以下问题：

**调研主题**: Mamba 与选择性状态空间模型在自动驾驶轨迹建模中的最新进展

**具体任务**:
1. 检索 2024-2026 年间，将 Mamba/SSM 应用于以下自动驾驶子任务的论文：
   - 轨迹预测 (Trajectory Prediction)
   - 场景生成 (Scenario Generation)
   - 运动规划 (Motion Planning)
   - 端到端自动驾驶 (End-to-End Driving)
   - 交通流预测/生成 (Traffic Flow Prediction/Generation)

2. 重点关注：
   a) 各论文使用的 Mamba 变体（原始 Mamba / Mamba-2 / Vision Mamba / 其他改进）
   b) Mamba 与 Attention/CNN/GNN 的混合架构设计
   c) 在轨迹建模方面的参数量、训练时间、推理速度对比
   d) 相对于 Transformer/GRU 基线的性能提升
   e) 是否有 Mamba 在多主体交互建模中的特定设计

3. 关键对比：
   - Trajectory Mamba (Tamba, CVPR 2025) 的详细方法与结果
   - 是否有将 Mamba 与因果推断结合的论文？
   - 是否有将 Mamba 与物理约束结合的论文（除 Physics-informed Diffusion Mamba 外）？

4. 输出格式：按子任务分类的 Mamba 应用全景表 + 
   "Mamba + 因果 + 物理 + 场景生成"四合一方向是否已被占据的结论。
```

### 任务 6：SOTA 定量结果对标（委托 Gemini Deep Research）

```
请使用 Google Scholar 进行调研，回答以下问题：

**调研主题**: 安全关键场景生成领域 SOTA 方法的定量结果对标

**具体任务**:
1. 收集以下数据集的场景生成/轨迹预测 SOTA 定量结果：
   - Waymo Open Motion Dataset (WOMD) — ADE, FDE, minADE, minFDE, miss rate
   - INTERACTION — ADE, FDE 各场景类型
   - nuScenes — 场景生成相关的定量结果
   - Argoverse 2 — 轨迹预测最新 SOTA

2. 对每个数据集，列出至少 Top-5 方法的定量结果

3. 特别收集以下"安全关键性"指标的文献结果：
   - 碰撞率 (Collision Rate)
   - 近距事件率 (Near-Miss Rate)  
   - TTC/PET 分布统计数据
   - 物理可行性违反率 (Kinematic Feasibility Violation Rate)
   - 场景多样性指标 (Diversity Score)

4. 这些指标在 SOTA 论文中通常报告什么数值范围？
   我们的方法如果要在 Q1 期刊上立足，需要在这些指标上达到什么水平？

5. 输出格式：按数据集分组的定量结果对标表，标注各指标的可比范围和目标值。
```

### 任务 7：评估指标体系调研（委托 Copilot 研究助手）

```
请使用 IEEE Xplore / Scopus 或 Web of Science / Semantic Scholar / OpenAlex / arXiv / Google Scholar 进行交叉检索，回答以下问题：

**调研主题**: 自动驾驶安全关键场景生成的评估指标体系

**具体任务**:
1. 检索关于场景生成评估指标的最新论文（2022-2026），包括但不限于：
   - 专门讨论评估指标的论文
   - 提出新指标的论文
   - 安全标准中的评估框架（ISO 26262, ISO 21448/SOTIF, ISO 34504, NATM 等）
   
2. 整理完整的评估指标分类体系：
   a) 轨迹保真度指标（ADE, FDE, minADE, minFDE, MR, DAO 等）
   b) 安全关键性指标（CR, NMR, TTC, PET, DRF, RSS compliance 等）
   c) 物理可行性指标（加速度违反率、曲率违反率、摩擦圆违反率 等）
   d) 多样性与覆盖率指标（Diversity Score, Parameter Space Coverage, Actor Coverage 等）
   e) 因果可解释性指标（Root-Cause Attribution Score, Intervention Consistency 等）
   f) 域迁移指标（Wasserstein Distance, MMD, Target Domain Performance Drop 等）
   
3. 对每个指标，提取：
   - 标准定义和计算公式
   - 首次提出该指标的论文
   - 在使用中的争议或局限
   
4. 关键问题：现有的安全关键场景生成论文是否普遍报告物理可行性指标和因果可解释性指标？
   如果不是，这说明什么？

5. 输出格式：完整的评估指标分类表，标注哪些是 "Standard"（普遍使用）、
   "Emerging"（开始出现）、"Proposed"（我们计划新提出）。
```

### 任务 8：CNCAP 规程约束调研（委托 Copilot 研究助手）

```
请使用官方标准文档、IEEE Xplore、Scopus 或 Web of Science、Semantic Scholar、OpenAlex 与 Google Scholar 进行交叉调研，回答以下问题：

**调研主题**: C-NCAP/E-NCAP/ISO 标准对自动驾驶场景生成的约束条件

**具体任务**:
1. 检索以下标准文档中与"测试场景"相关的约束条件：
   - C-NCAP 管理规则 2024 年版（及 2027 版规划）
   - Euro NCAP 2023/2026 测试规程
   - ISO 34504:2024（自动驾驶场景分类）
   - ISO 21448/SOTIF（预期功能安全）
   - NHTSA 预碰撞场景分类

2. 提取可供生成模型使用的"硬约束"：
   - 测试车辆的物理边界（速度范围、加速度限制、转向限制）
   - 目标物（GVT/骑行者/行人）的运动参数范围
   - 触发条件（TTC/距离/时间窗口 等）
   - 安全执行边界（多大 TTC 以下必须中止测试）
   - ABD 驾驶机器人/软目标平台的执行精度限制

3. 调研：是否有学术论文将上述标准约束形式化为可嵌入生成模型的数学约束？
   如果有，是如何做的？如果没有，为什么这是一个空白？

4. 输出格式：按标准分类的约束条件清单，标注可量化为数学约束的部分。
```

---

## 四、委托策略

| 任务 | 委托工具 | 理由 |
|------|---------|------|
| 任务 1: 场景生成全景 | **Gemini Deep Research** | 需要 Google Scholar 广泛覆盖 CV/robotics 会议论文 |
| 任务 2: 因果推断交叉 | **Gemini Deep Research** | 因果推断是跨学科领域，Google Scholar 覆盖面更广 |
| 任务 3: 物理约束生成 | **Copilot 研究助手 + IEEE/Scopus/WoS/Scholar 交叉核验** | 物理约束研究跨工程、应用数学和机器人领域，必须用多个数据库交叉核验 |
| 任务 4: 域自适应迁移 | **Copilot 研究助手 + IEEE/Scopus/WoS/Scholar 交叉核验** | 需要系统性确认空白；“不存在”的证明只能写成检索范围内未发现 |
| 任务 5: Mamba/SSM 应用 | **Gemini Deep Research** | CS/ML 领域，Google Scholar 覆盖最全 |
| 任务 6: SOTA 定量对标 | **Gemini Deep Research** | 需要跨数据集检索，Google Scholar 更适合 |
| 任务 7: 评估指标体系 | **Copilot 研究助手 + 标准/出版商原文核验** | 标准和工程文献必须回到官方标准或出版商原文确认 |
| 任务 8: CNCAP 规程 | **Copilot 研究助手 + 官方标准文档** | 规程约束以 C-NCAP/E-NCAP/ISO 官方文件为准，论文只作为补充 |

---

## 五、已知的差异化空间（预判）

在等待 Gemini/Copilot 调研结果的同时，基于已有文献，以下是预判的差异化空间：

### 5.1 已被占据的赛道（避免进入）

| 赛道 | 占据者 | 为何避免 |
|------|--------|---------|
| "将 GRU 替换为 Mamba" | Tamba (CVPR 2025), DriveMamba 等 | 非创新，Q1 审稿人秒拒 |
| 纯 VAE/GAN 场景生成 | STRIVE (CVPR 2022), TrafficGen (2023) | 赛道已饱和 |
| 无约束的对抗扰动 | AdvSim (CVPR 2023) | 忽略物理可行性，方法被 CounterScene/SaFeR 超越 |
| PC/LiNGAM 因果发现 | 已被 SCM + Counterfactual 范式取代 | 方法过时 |
| 模仿学习 + 碰撞惩罚 | 大量论文 | 缺乏方法论深度 |

### 5.2 蓝海空间（我们的机会）

| 空白 | 证据 | 价值 |
|------|------|------|
| **物理约束 + 因果引导 + 生成模型的统一框架** | CounterScene 有因果无物理；Physics-informed Diffusion Mamba 有物理无因果；SaFeR 有物理可行性但基于离散 Token | 高——三元交叉地带未被占据 |
| **场景生成中的多源域自适应** | 文献中零发现 | **极高**——首发优势 + 释放 CNCAP 数据价值 |
| **开放道路→封闭场地的跨域场景生成迁移** | 文献中零发现 | **极高**——与中汽研独特优势直接匹配 |
| **超越 ADE/FDE 的安全关键场景评估矩阵** | SaFeR/CounterScene 增加了 CR 等指标但不成体系 | 中高——形成可被后续工作引用的标准 |
| **Mamba 与因果交互的深度结合** | 无相关工作 | 中——架构创新，但需与 Tamba 划清界限 |
| **CNCAP 规程约束的形式化嵌入** | 无相关工作 | 高——标准-学术的桥梁，Q1 期刊重视 |

### 5.3 需调研确认的关键问题（如果空白成立则价值极高）

1. CounterScene 是否已在修订版中加入物理约束？→ **需 Gemini 检索**
2. 是否有团队正在做"开放道路→CNCAP"的迁移（可能用其他术语描述）？→ **需 Copilot 检索**
3. SaFeR 团队是否有后续工作将 Token Resampling 与 Mamba 结合？→ **需 Gemini 检索**
4. TUM-AVS 的 FM-AD-Survey 中是否已将"域自适应场景生成"列为待研究方向？→ **需 Copilot 检索**

---

## 六、预期产出

| 产出 | 格式 | 负责人 |
|------|------|--------|
| 场景生成方法全景表（40+ 篇论文，分类标注） | Markdown 表格 | Gemini 任务1 + Claude 整合 |
| 因果推断 × ADS 交叉文献地图 | Markdown 表格 | Gemini 任务2 + Claude 整合 |
| 物理约束方法对比表 | Markdown 表格 | Copilot 任务3 + Claude 整合 |
| 域自适应方法列表 + 空白确认 | Markdown 表格 | Copilot 任务4 + Claude 整合 |
| Mamba/SSM 应用全景 | Markdown 表格 | Gemini 任务5 + Claude 整合 |
| SOTA 定量结果对标表 | Markdown 表格 | Gemini 任务6 + Claude 整合 |
| 评估指标体系分类 | Markdown 表格 | Copilot 任务7 + Claude 整合 |
| CNCAP 规程约束清单 | Markdown 表格 | Copilot 任务8 + Claude 整合 |
| 综合差距分析矩阵 | 对比矩阵 | Claude（基于所有调研结果） |
| Related Work 章节草稿 | LaTeX/Word | Claude（基于差距分析） |

---

## 七、下一步

1. 将第 3 节中的 8 个任务提示词分别发送给 Gemini Deep Research 和 Copilot 研究助手
2. 收集调研结果后，Claude Code 负责：
   - 整合 8 份调研报告
   - 输出差距分析矩阵
   - 确认或修正预判的差异化空间
   - 撰写 Related Work 章节草稿
3. 进入 Stage 3（方法与系统设计）

---

> **文件版本**: v1.0
> **下一步**: 请将第 3 节的提示词发送给 Gemini Deep Research（任务 1/2/5/6）和 Copilot 研究助手（任务 3/4/7/8）。待调研结果返回后，Claude Code 进行整合分析。
