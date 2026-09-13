# P3.3.2 名义模型与训练管线 smoke 报告

- 日期：2026-09-13；执行者：GLM 5.3（Claude Code）；配置版本 `p33.0-v1.1` + `p33-model-smoke-v1`
- 结论一句话：**管线工程 smoke 全部通过（12/12 checks）**；模型本身按预期严重欠训练
  （200 updates × 256 有效批量 ≈ 5.1 万样本，约为 waymo train 一个 epoch 的 1.1 倍），
  不构成对 AR-Scene-v1 表示能力的任何判断。
- `status=pass` 仅表示可进入人工结果评审，**不表示 G1**，不表示优于 CV/GRU。

## 1. 交付物与验证状态

| 交付物 | 状态 |
|---|---|
| `scenario_lab/p33_dataset.py`（D0） | 26 shards / 31,215 样本 / 4,998 组审计与 P3.3.1 逐项一致 |
| `scenario_lab/p33_metrics.py`（B0） | 全 4,416 dev 样本 CV 完成，min@6≡单采样确定性检查通过 |
| `scenario_lab/p33_model.py`（M0） | 13,126,282 参数（区间 10–25M）；静态键隔离通过严格反事实测试 |
| `research_tasks/train_p33_nominal.py`（M1/M2） | M1 gate PASS；M2 gate PASS（gate 语义见 §4） |
| `research_tasks/evaluate_p33_t1.py`（B0/M3） | 全 dev 模型诊断 + 8 张叠加图完成 |
| `configs/p33/model_smoke_v1.json` | smoke 预算与 gate 语义，不覆盖冻结训练超参 |
| `tests/test_p33_{dataset,metrics,model,train_checkpoint}.py` | 29 项契约测试全过（全仓 177 = 148 基线 + 27 p33 + 2 checkpoint） |
| `SMOKE_VALIDATION.json` | 12 checks 全 pass + limitations |

## 2. Q1 参数量 / 显存 / 吞吐 / 训练时间（RTX 4060 Ti 16G, WSL2）

| 量 | 值 |
|---|---|
| 参数量 | **13,126,282** = 13.13M（冻结区间 10–25M ✓） |
| M1 overfit（500 updates，fp32） | 44.4 s |
| M2 smoke（200 updates，bf16，有效批 256） | 715.9 s，107.3 样本/s |
| M2 峰值显存 | 976.6 MB（16 GB 卡余量 ~15×） |
| M3 全 dev 评估（4,416 样本，前向 + 6 样本 rollout） | 188.9 s |
| CV 基线全 dev | 5.9 s（CPU） |

## 3. Q2 CV 基线（group-level，全部 4,416 dev 样本 / 46,845 agent 记录）

| 来源 | 独立组数 | agent 记录 | group ADE | group FDE | pooled ADE |
|---|---|---|---|---|---|
| waymo | 481 | 2,631 | 3.969 | 10.965 | 4.052 |
| interaction | **4** | 44,214 | 3.246 | 8.557 | 1.733 |

- INTERACTION dev 仅 **4 个独立 location::case 组**（3,935 个重叠窗口样本，26,233 个
  重叠窗口绝不能写成独立场景数）；group 与 pooled 差 ~1.9 m，组级聚合是必要条件。
- minADE@6 ≡ ADE、minFDE@6 ≡ FDE（CV 复制 6 份的确定性检查通过）。
- 66/46,845 条记录 FDE 未定义（endpoint 帧无效；57+7+2 按 type 对账）。
- **smoke dev 中不存在 role-0（other）agent**（46,845 条全部 role∈{1,2}），"other" 角色
  路径在真实数据上未被行使——100-shard 阶段需确认其出现率。

## 4. Q4 M1 / M2 gates

**M1 单批 overfit**（16 样本、500 updates、门槛 last20 ≤ 0.5×first20）：
first-20 均值 14.33 → last-20 均值 0.149，**比例 0.0104，PASS**；
token accuracy 0.5% → 98.7%，CE 4.95 → 0.065——模型能记住单批数据，梯度通路完整。

**M2 GPU smoke + resume**（200 updates、bf16、micro 16×16=有效 256）：
- 训练健康：损失 10.246 → 3.290（u100）→ 2.203（u200），全程有限；715.9 s、
  107.3 样本/s、峰值显存 976.6 MB。
- **resume gate PASS**（gate 语义与全部阈值见 `configs/p33/model_smoke_v1.json: resume_gate`）：
  1. **restore-point 逐位相等** ✓：恢复后第 1 个 update 损失与不间断运行逐位相同
     （本 run 前 8 步均逐位相等）。该损失由恢复的权重 + 恢复的批次 + 恢复的 dropout RNG
     的一次前向决定——任何 checkpoint 内容错误（权重/Adam 矩/loader 位置/RNG）必然在此暴露；
     bf16 舍入吸收亚 ULP 核差异，全部 8 个观察副本（4 次完整 run + 3 个探针副本）在此
     index 均逐位相等。
  2. **20-update 窗口轨迹相等** ✓：窗口内 max 差 0.0077 ≤ 0.165（5% × 窗口均值，21× 裕度）。
     该视界内混沌噪声 ≤~8e-3（四次 run 观测），而任何真实状态错误在前几步即宏观分歧。
  3. **样本流匹配** ✓：恢复分支消费的样本序与 checkpoint 快照完全一致。
  4. **恢复分支 sanity** ✓：损失有限且末 20 步均值较恢复点水平下降 >20%。
  - 诊断量（不设 gate）：100-update 尾段 max 漂移 = **0.27 / 0.39 / 1.12 / 0.98（四次
    相同 run）**——幅度是 run 间随机分布的混沌量；最终权重差最大 0.0131（token_head，
    ~2% 参数尺度）。
- **为什么 gate 不要求长视界逐位/有界相等**（实验证据链，详见文档 §7 缺陷 5/6）：
  1. 严格模式 `torch.use_deterministic_algorithms(True)` 在本训练环路不报任何无确定性
     实现的算子 → 排除内核非确定性（初版归因 "embedding atomicAdd" 是错的，已纠正）。
  2. 分配器扰动探针（`ALLOCATOR_PROBE_RESULT.json`）：同一恢复状态、同一 RNG、确定性内核，
     仅改变 CUDA 分配器布局（不触碰任何数学）→ 首 update 仍逐位相等、第 2 步即分歧 4.1；
     干净重建则复现完整 run 的"前缀逐位相等 + 1e-5 级微差混沌放大"形态。机制：重建进程
     分配器布局不同 → 张量对齐不同 → cuBLAS 按对齐选核 → 浮点累加顺序变化 → 混沌放大。
  3. 分歧起点位置与幅度均随机（四次 run 起点在尾段 index 9/4/6/8，max 漂移见上），且
     phase A 自身跨进程终值也不同（2.08/1.75/2.08/2.20）——对混沌量设绝对界之前先看它的
     run 间分布。
  - checkpoint 快照语义（防"活引用假阳性"缺陷类复发）由
     `tests/test_p33_train_checkpoint.py` 单元级锁死（快照不可变性 + 恢复点逐位复现，
     CPU 上确定性成立）。

## 5. Q3 模型 teacher-forced NLL 与 6 样本 rollout（M3，全 dev）

| 量 | waymo（481 组） | interaction（4 组） | CV 对照（同源 group ADE） |
|---|---|---|---|
| minADE@6 | 11.71 | 6.36 | 3.969 / 3.246 |
| minFDE@6 | 22.52 | 12.01 | 10.965 / 8.557 |
| 单采样 ADE/FDE（K=1 语义） | 17.09 / 32.65 | 10.29 / 19.17 | — |

- teacher-forced token NLL 1.575（均匀基线 ln128=4.85），accuracy 70.0%；
  sampled NLL 1.872（模型自采样分布的熵，top-p=0.95 下有效词数 ~6.5——不是拟合优度）。
- 6 样本相对单采样改善 31%（17.09→11.71）：采样有多样性，但全体样本都差。
- 运动学诊断（工程阈值，非 G1）：速度违规率 0.002%（281,070 条轨迹）；加速度/jerk
  "100%"——逐 chunk 分段线性解码在 chunk 边界的结构性速度跳变，几何代理被结构性抬高，
  不作告警解读；offroad 在 scene-shard 契约下不可辨识，如实记为不可算。
- **teacher-forced 解码探针**（`TEACHER_DECODE_PROBE.json`，把解码路径与自回归 token
  漂移分离，group-level ADE/FDE）：

  | 设置 | interaction | waymo | CV 对照 |
  |---|---|---|---|
  | 真值 token + 预测残差（解码路径隔离） | **1.30 / 2.44** | **2.54 / 4.93** | 3.25 / 3.97（ADE） |
  | argmax token（+贪心 token 误差，无采样无漂移） | **2.52 / 3.73** | **3.76 / 4.80** | 8.56 / 10.97（FDE） |
  | 自回归 6 样本 rollout（单采样） | 10.29 / 19.17 | 17.09 / 32.65 | — |

  读法：解码路径（codebook+残差+cumsum）健全且误差小于 CV；教师强制 argmax 域两来源
  均已优于 CV 同项；rollout 的灾难性劣化（×4.5）来自自回归条件漂移与采样的复合
  （每 chunk 70% 精度 × 10 chunk 连乘 ≈ 3% 全对率，与观测一致）——是训练深度与暴露
  偏差问题，**不是表示/解码缺陷的证据**。注意：教师强制域数字不可与 rollout 域数字
  互换解读，也不构成"模型优于 CV"的声称。
- 模型 200 updates 严重欠训练（~1.1 个 waymo epoch），上述数字**远差于 CV 属预期**，
  不构成表示能力结论；诊断优先级见 §8。

## 6. Q5 截断 vs 未截断（CV，分来源；TRUNCATION_ANALYSIS.json）

| 来源×截断 | 样本 | agent 记录 | CV ADE | CV FDE |
|---|---|---|---|---|
| interaction × truncated | 2,016 | 32,000 | 1.071 | 2.856 |
| interaction × untruncated | 1,919 | 12,214 | 3.465 | 9.138 |
| waymo × truncated | 383 | 2,268 | 3.994 | 11.032 |
| waymo × untruncated | 98 | 363 | 4.413 | 12.043 |

- 边缘分层（truncated ADE 1.26 vs untruncated 3.49）主要由 interaction 占比（44,214/46,845）
  驱动；**分来源后方向一致**：截断（>16 agent、更稠密、静止 agent 多）的 CV 误差更低。
- 模型侧（`MODEL_DEV_METRICS.json` strata，注意是全 dev 混合的边缘分层而非分来源）：
  truncated minADE 3.86（34,268 条） vs untruncated 7.01（12,577 条）——与 CV 方向一致，
  幅度更大；map_truncated=False 仅 181 条（minADE 16.8），样本过小不可解读。

## 7. Q6/Q7 token 分布与来源方向

- token 频率：真值 top-1 token（#23，近零运动）占 28.0%，占用 127/128；argmax 预测侧
  top-1 占 36.9%（×1.32 过度预测）；macro recall（argmax，按类平均）0.530。
- **长尾失守如期**：6 个真值计数 2–17 的稀有 token 的 argmax 预测计数为 0（#11/19/51/65/90/123）；
  采样 macro recall 0.0096——top-p 采样把质量摊到 ~6.5 个有效 token 上，稀有类几乎不可能被
  精确命中（这是采样结构的性质，配合 200-update 欠训练）。
- 来源方向：两来源同向劣于 CV、无方向反转——waymo minADE@6/CV = 2.95×，interaction = 1.96×；
  interaction（低速稠密、近零 token 主导）相对更接近 CV，与"模型先学会低速/静止模式"一致
  （interaction 仅 4 组，解释力以组数为准，不与 waymo 481 组等量齐观）。

## 8. Q8 是否进入 100-shard architecture？

**建议：进入 100-shard architecture（工程 GO），不需要先改表示或指标。** 理由：

1. **工程边界全部成立**（12/12 checks）：loader 有界且来源平衡、训练有限且可断点、
   评估全 dev 可跑、吞吐 107.3 样本/s、峰值显存 977 MB——16 GB 卡对 100-shard 的批量与
   有效批量有 ~15× 余量。
2. **表示与解码路径经诊断健全**（teacher-decode 探针）：真值 token 解码 ADE 1.30/2.54、
   argmax 解码 2.52/3.76——教师强制域两来源均优于 CV 同项。这是 smoke 能给出的最强正面
   证据，且严格限定在教师强制域，不外推。
3. **唯一重大缺口有清晰归因**：rollout 单采样 17.09/10.29 vs argmax-teacher 3.76/2.52
   ≈ ×4.5，来自自回归漂移+采样复合（每 chunk 70% 精度、10 chunk 连乘）。这是训练深度
   与暴露偏差问题；100-shard + 正式训练量正是检验"更长训练是否闭合该缺口"的实验——
   若先改表示反而没有对照。
4. **护栏（建议预注册进 P3.3）**：训练中按固定间隔记录 teacher-argmax 解码指标与
   rollout 指标之比作为漂移预警；若正式训练量下 rollout/teacher-argmax 差距仍 >2×，
   优先评估采样策略（温度/top-p/beam）与暴露偏差缓解，而非改表示。
5. **100-shard 阶段需监测（不改架构）**：role-0 agent 出现率（smoke dev 为 0，该路径
   仅合成测试行使）；稀有 token（6/128 argmax 全零，频率驱动，随数据量复核）；
   INTERACTION 组级不确定性（dev 仅 4 组；100-shard 的 split 设计需保证两组来源的
   组数均衡）。

## 9. Q9 结论分级

| 级别 | 内容 |
|---|---|
| **verified fact**（测试/全量数据证实） | 参数量 13,126,282；loader 有界性（每来源 ≤1 驻留 shard，26 shards/31,215 样本审计一致）；隐藏信息反事实不变性与 decoder 因果性（29 项测试）；CV 全 dev 组级数字；M1 比例 0.0104；M2 restore-point 逐位相等 + 流匹配 + 有界漂移；checkpoint 快照不可变性（单元测试）；分配器→核选择→混沌放大机制（探针复现） |
| **engineering diagnostic**（工程观测，非科学结论） | bf16 吞吐 107.3 样本/s、峰值显存 976.6 MB；NLL 1.575 / minADE@6 11.71 等模型数字（欠训练状态下）；截断分层差异；token 频率与 macro recall；运动学违规率（几何代理）；teacher-decode 探针分解 |
| **hypothesis**（有证据倾向、未对照验证） | 分配器布局差异是跨进程逐位不可复现的充分原因（探针支持，未做穷尽排除）；静态键隔离对交互建模的表达代价可接受（M1 可 overfit，但无对照量化） |
| **尚未验证的 scientific claim** | AR-Scene-v1 是否达到 G1 真实度；是否优于 CV/GRU（100-shard + 正式训练后才能评估）；token 化对连续轨迹精度的上界影响；visibility proxy t=0 快照 vs 时变可见性的差异 |

## 10. heldout 纪律

未读取任何 heldout 内容（Waymo validation shards 00002–00149、INTERACTION 四个 heldout
location、P2.12 seed77000、ABD NPC 用途）；`heldout_trajectory_content_read=false`。
dev 仅用于 smoke 工程验证，未做任何基于 dev 的超参或结构选择（唯一一次 gate 语义修订
针对检验对象本身，附三轮机制证据，不涉及模型/超参）。

## 11. 限制（与 SMOKE_VALIDATION.limitations 一致）

1. `status=pass` 只表示工程 smoke 边界成立、可进入人工结果评审；不表示 G1 真实度结果，
   不表示模型优于 CV/GRU。
2. 200-update 模型严重欠训练（有效样本约 5.1 万 ≈ 1.1 个 waymo epoch）；M3 全部模型
   数字是工程诊断。教师强制域优于 CV 严格限定在教师强制域，不可外推到 rollout 域。
3. checkpoint_resume gate 语义 = restore-point 逐位相等 + 20-update 窗口 + 流匹配 +
   sanity；长视界逐位/绝对有界相等在本 CUDA 栈不可实现（分配器布局→GEMM 核选择→ULP
   差异混沌放大；四次相同 run 的 100-update 尾段 max 漂移 0.27/0.39/1.12/0.98）。
   gate 语义的三次修订全部针对检验对象本身并附可证伪实验，从未触及模型/超参/主指标。
4. INTERACTION dev 仅 4 个独立组；其组级数字带 4 组不确定性；26,233 个重叠窗口从未被
   计为独立场景数。
5. 加速度/jerk 几何代理受逐 chunk 分段线性解码结构性抬高（100% "violations"），速度
   违规率 0.002%；这些阈值是工程诊断不是 G1 门槛。
6. smoke dev 无 role-0（other）agent，该路径仅由合成测试行使。
7. sampled_token_nll=1.872 是模型自采样分布的熵（有效词数 ~6.5），不是拟合优度。
8. dev 仅用于 smoke 工程验证；未做任何基于 dev 的超参或结构选择。
