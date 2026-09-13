# P3.3.2a 可见性修复重跑报告

- 日期：2026-09-13；执行者：GLM 5.3（Claude Code）；配置版本 `p33.0-v1.1` +
  `p33-model-smoke-v1`（新增 `revision_note` 与 `overlay_plot_per_source`）。
- 上游：独立复核 `runs/20260913_p332_model_smoke/CODEX_REVIEW.md`（commit 0f28bcc，
  REQUIRES_CORRECTION）——保留数据/CV/训练基础设施，修正模型与确定性后全量重跑。
  修复内容见 `docs/p332a_visibility_fix.md`；本报告只记重跑数字与结论。
- 结论一句话：**六个修正项全部落实，M1/M2/M3 全量重跑通过，且跨进程逐位复现成立**；
  模型仍为 200-update 欠训练状态，模型数字仍是工程诊断。
- `status=pass` 仅表示可进入人工结果评审，不表示 G1、不表示优于 CV/GRU。

## 1. 交付物与验证状态

| 交付物 | 状态 |
|---|---|
| `scenario_lab/p33_model.py`（W1） | per-query 视图键（逐 query 逐时刻可见性池化）；参数量不变 13,126,282 |
| `tests/test_p33_model.py`（W2） | 部分历史反事实**逐位**不变 + t=0 遮挡下过去帧可达 + 全遮挡有限性 |
| `scenario_lab/p33_dataset.py`（W3） | `get()` 行级拷贝；backing-array 释放经 weakref+gc 测试锁死 |
| `research_tasks/train_p33_nominal.py`（W4） | strict `warn_only=False` + math SDPA + `restore_optimizer_isolated` |
| `scenario_lab/p33_metrics.py`（W5） | joint-scene best-of-6 次级指标 + 分来源图 + NLL 定义落盘 |
| 全仓测试 | **183 通过**（177 旧 + 6 新）；p33 子集 35/35 |

## 2. 确定性与 resume（复核第 2、3 项的核心质询）

| 检查 | 结果 |
|---|---|
| M2 run1 resume gate | PASS：restore-point 逐位相等，**100-update 恢复尾段 max diff = 0.0**（进程内全逐位） |
| M2 run2（独立进程，同 seed 同配置） | gate PASS，loss 曲线与 run1 **字节级相同**（SHA-256 `6850228…af52b9`，见 `cross_process_check.txt`） |
| 分配器探针（重建版） | R1 活体 / R0 干净重建 / R2 分配器扰动：30/30 update 逐位相等——**"分配器→核选择"归因撤回** |
| 真实非确定源 | bf16 memory-efficient attention 反向（旧 run log 有警告；strict+math 后消除） |
| 代价 | 吞吐 95.2 vs 103.65 样本/s（~8%）；M2 峰值显存 1,371.8 MB（per-query 键物化，16 GB 卡余量 ~11×） |

**跨进程逐位复现现在成立**（旧报告"本 CUDA 栈不可实现"的结论以确定性配置为前提被
推翻；旧探针的大分歧由优化器 payload 存储别名污染解释）。

## 3. M1 / M2 数字（RTX 4060 Ti 16G，WSL2）

- **M1 单批 overfit**（16 样本、500 updates、fp32）：first-20 均值 → last-20 均值
  比例 **0.01265**（门槛 ≤0.5），65.7 s，峰值显存 1,158 MB。
- **M2 GPU smoke**：损失 10.244 → 3.393（u100）→ **1.867**（u200），全程有限；
  870.7 s；吞吐 88.2 样本/s（含恢复分支重建开销；纯训练段 95.2）。
- **exposure（修正口径）**：有效样本 25,600/来源 = **5.69 个 waymo train pass /
  1.15 个 interaction train pass**；elapsed = 300 update 当量（主分支 200 + 恢复分支 100）。

## 4. CV 基线回归（应与旧值完全一致——确认）

| 来源 | 组数 | agent 记录 | group ADE | group FDE | pooled ADE |
|---|---|---|---|---|---|
| waymo | 481 | 2,631 | 3.969 | 10.965 | 4.052 |
| interaction | 4 | 44,214 | 3.246 | 8.557 | 1.733 |

与 P3.3.2 逐位一致（可见性修复不触及 CV 路径的回归检查）。CV×截断分析同理不变，
见旧目录 `runs/20260913_p332_model_smoke/TRUNCATION_ANALYSIS.json`（其 CV 侧数字
仍有效）；模型侧截断分层在本目录 `MODEL_DEV_METRICS.json` 的 `strata` 内
（truncated minADE 3.63 vs untruncated 6.20，与 CV 方向一致）。

## 5. M3 模型诊断（全 4,416 dev 样本，191.6 s，修正后模型）

| 量 | waymo（481 组） | interaction（4 组） | CV 对照 |
|---|---|---|---|
| minADE@6（主端点，冻结） | 10.07 | 5.65 | 3.97 / 3.25 |
| minFDE@6 | 18.86 | 10.20 | 10.97 / 8.56 |
| joint-scene best-of-6（次级） | 13.39 / 25.70 | 8.80 / 16.76 | — |
| 单采样 ADE/FDE（K=1） | 16.78 / 32.18 | 11.19 / 21.15 | — |

- teacher-forced token NLL **1.422**（均匀基线 ln128=4.85），accuracy **71.1%**；
  sampled_token_nll 1.646 = top-p 截断后采样 token 按全 softmax log-prob 的均值 NLL
  （截断采样分布相对于原模型分布的交叉熵的 MC 估计，**不是**采样分布熵，`exp(·)` 不称有效词数）。
- joint > per-agent min 是数学必然（同一 k* 服务全场景 agent，弱于逐 agent oracle），
  二者差距随训练收敛应缩小——只作次级诊断，主端点不变。
- 运动学（工程阈值非 G1）：速度违规率 0.011%（281,070 条）；加速度/jerk 100% 为
  逐 chunk 分段线性解码的结构性边界跳变，不作告警解读；offroad 无多边形不可算。
- 分来源叠加图 **waymo 4 张 + interaction 4 张**（`overlay_plots/`；首版实现内层
  循环漏配额检查导致 8 张全 waymo，已修复重跑——该 bug 属本次 smoke 修正范围）。
- token 分布：真值 top-1 占 28.0%、argmax 预测侧占比上升；macro recall（argmax，
  127 个占用类）0.550；稀有 token 长尾欠训练状态下仍失守（频率驱动，随数据量复核）。

## 6. teacher-forced 解码探针（修正后模型，全 dev）

| 设置（group ADE/FDE） | interaction | waymo | CV 对照 |
|---|---|---|---|
| 真值 token + 预测残差（解码路径隔离） | **1.11 / 1.81** | **2.25 / 4.06** | 3.25 / 3.97（ADE） |
| argmax token（+贪心 token 误差） | **2.35 / 3.04** | **4.03 / 5.03** | 8.56 / 10.97（FDE） |
| 自回归 6 样本 rollout（单采样 K=1） | 11.19 / 21.15 | 16.78 / 32.18 | — |

复核勘误：真值 token + 预测残差的 ADE 两来源均低于 CV；teacher-argmax 的
Waymo ADE 4.03 略高于 CV 3.969，并高于旧模型 3.76，不能称为全部改善。
rollout 与教师强制结果的差距支持暴露偏差/采样误差假设，但不能唯一归因，
也不能排除离分布前缀下的表示、残差或解码问题。
**教师强制域数字不可与 rollout 域互换，不构成"模型优于 CV"的声称**；0.7^10 式
连乘只是启发式，不是实测全对率。新旧结果有改善也有退步，且后端与可见性实现
同时变化，不能据此推断去除泄漏对模型质量的独立因果作用。

## 7. 事实分级

| 级别 | 内容 |
|---|---|
| **verified fact** | per-query 视图键的部分历史反事实逐位不变性（合成 + 真实数据抽验 `torch.equal`）；t=0 遮挡下过去帧可达；参数量 13,126,282；183 项测试；M1 比例 0.01265；M2 restore-point 逐位相等 + 恢复尾段全逐位（max 0.0）；**跨进程逐位复现**（SHA-256 相同）；CV 全 dev 与旧值逐位一致；backing-array 释放（weakref+gc）；优化器恢复无别名 |
| **engineering diagnostic** | M2 吞吐 95.2 样本/s（strict+math）、峰值显存 1,371.8 MB；token NLL 1.422 / minADE@6 10.07/5.65 / joint 13.39/8.80（欠训练状态）；teacher-decode 三行分解；token 频率与 macro recall；运动学几何代理；分层差异 |
| **hypothesis** | rollout 缺口可能随更多数据/训练收窄；暴露偏差与采样是候选机制，尚未隔离验证 |
| **已撤回** | ~~分配器布局→GEMM 核选择→ULP 混沌放大~~（旧探针受优化器 payload 别名污染；干净探针三配置 30/30 逐位相等）；~~跨进程逐位不可实现~~（strict+math 下成立） |
| **尚未验证** | AR-Scene-v1 是否达 G1；是否优于 CV/GRU（需 100-shard + 正式训练）；token 化对连续轨迹的上界影响 |

## 8. heldout 纪律

未读取任何 heldout 内容（Waymo validation shards 00002–00149、INTERACTION 四个
heldout location、P2.12 seed77000、ABD NPC 用途）；`heldout_trajectory_content_read=false`。
dev 仅用于 smoke 工程验证；未做任何基于 dev 的超参/结构选择（模型改动仅落实复核
指出的 spec 违反，不涉及超参/主指标/split/模型规模）。

## 9. 限制

1. `status=pass` 只表示工程边界成立、可进入人工结果评审；不表示 G1、不表示优于 CV/GRU。
2. 200-update 严重欠训练（25,600 样本/来源 = 5.69 waymo pass / 1.15 interaction
   pass）；M3 全部模型数字是工程诊断；教师强制域优于 CV 严格限定在教师强制域。
3. 确定性结论限定于本机 CUDA 栈 + strict `warn_only=False` + math SDPA 配置；
   其他后端/精度组合未声明。
4. INTERACTION dev 仅 4 个独立组；其组级数字带 4 组不确定性；26,233 个重叠窗口
   从未计为独立场景数。
5. 加速度/jerk 几何代理被分段线性解码结构性抬高（100%）；速度违规率 0.011%；
   阈值是工程诊断不是 G1 门槛。
6. sampled_token_nll 是截断采样分布相对于原模型分布的交叉熵的 MC 估计，不是采样分布熵或真值拟合优度。
7. decoder 交叉注意门仍取 t=0 可见性（identity 键无历史，无时序泄漏面）——保守
   选择，若 100-shard 阶段需要时变交叉门控再评估。
