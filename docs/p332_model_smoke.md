# P3.3.2 名义模型与训练管线工程 smoke

> **P3.3.2a 修正（2026-09-13）**：独立复核确认本阶段模型存在时序可见性泄漏（§3 描述的
> "静态 token + t=0 门控"实现违反 spec §4.5 逐 query 逐时刻掩码），§4/§7.6 的确定性归因
> （分配器→核选择）也被干净探针否定；§8 的 role-0 监测项基于 role/type 混淆。修正后的
> 模型、探针、重跑结果与勘误清单见 `docs/p332a_visibility_fix.md` 与
> `runs/20260913_p332a_visibility_fix/`。本文其余部分保留为当时的历史记录。

- 阶段文档：`docs/p332_model_smoke.md`；配置版本 p33.0-v1.1 + `p33-model-smoke-v1`，2026-09-13；
  状态：工程 smoke 完成（见 `runs/20260913_p332_model_smoke/SMOKE_VALIDATION.json`）。
- 本阶段只证明模型/训练/评估的**工程边界**成立；`status=pass` 不代表 G1 真实度结果，
  不代表 Transformer 优于 CV 基线，也不代表可进入正式训练结论。

## 0. 交付物

| 文件 | 内容 |
|---|---|
| `scenario_lab/p33_dataset.py` | manifest 驱动、单驻留 shard、来源平衡 loader（D0） |
| `scenario_lab/p33_metrics.py` | CV 基线、ADE/FDE/min@6、组级聚合、运动学诊断（B0） |
| `scenario_lab/p33_model.py` | 最小 AR-Scene-v1（M0） |
| `research_tasks/train_p33_nominal.py` | M1 单批 overfit + M2 GPU smoke/resume |
| `research_tasks/evaluate_p33_t1.py` | B0 CV 全 dev + M3 模型诊断 |
| `configs/p33/model_smoke_v1.json` | smoke 预算（不覆盖冻结训练超参） |
| `tests/test_p33_{dataset,metrics,model}.py` | 27 项契约测试 |
| `runs/20260913_p332_model_smoke/` | 结果工件（checkpoint 与 loss 曲线不入库） |

## 1. 数据装载（D0）

- `ShardCatalog.from_manifest` 按 manifest+jsonl 逐行核对来源/split/schema；**Waymo shard 内
  混合 train/dev 行**（split 按 scenario 哈希而非文件），因此 split 在行级跟踪。
- `ShardReader` 每来源最多一个驻留 shard；`SceneLoader` 每批 8 waymo + 8 interaction，
  顺序由 (seed, source, pass) 决定，epoch = 较小来源（waymo train 4,501）的一遍。
- `audit_catalog`（真实数据）：26 shards、31,215 样本、4,998 组、0 跨 split 组、
  waymo 4501/481、interaction 22298/3935 —— 与 P3.3.1 上游完全一致。
- 新增 `advance_epoch()`：`iter_batches` 是单 epoch 生成器；训练循环在 StopIteration 处
  跨 epoch 接续（pass 顺序确定性延续，epoch 计入 loader state 支持 checkpoint resume）。

## 2. CV 基线（B0，全部 4,416 dev 样本）

CV 从**最后有效状态**外推（t_last 可能 < 0），不假设 t=0 可观测。

| 来源 | 组数 | agent 记录 | group ADE | group FDE | pooled ADE |
|---|---|---|---|---|---|
| waymo | 481 | 2,631 | 3.969 | 10.965 | 4.052 |
| interaction | 4 | 44,214 | 3.246 | 8.557 | 1.733 |

- **INTERACTION dev 仅 4 个独立 location::case 组**（3,935 个重叠窗口样本）。group 与
  pooled 相差 ~1.9 m —— 组级聚合不是学术洁癖，是本数据上的必要条件。
- minADE@6 ≡ ADE、minFDE@6 ≡ FDE（CV 复制 6 份采样的确定性检查通过）。
- 66/46,845 条 agent 记录 FDE 未定义（endpoint 帧无效），按 agent_type 分解 57+7+2 完全对账。

## 3. 模型（M0）：静态键隔离

冻结规格（d=256、6+6 层、8 头、FFN 1024、pre-LN、GELU、10–25M 参数）之外，严格隐藏信息
边界（query logits 对不可见 source-agent 历史的精确不变性）迫使我们放弃标准的"层间更新
K/V"注意力——那会产生 a→b→s 的转发泄漏路径（b 可见 s、a 可见 b，则 s 的历史经 b 的
表示渗入 a）。本实现：

- **encoder**：agent-agent 注意力的 K/V = 仅由**自身原始历史**计算的静态 token（逐层不变）；
  query 逐层精化；agent-map 注意力 K/V = masked-pool 的静态地图 token + 一个恒可注意的
  NULL 键（防全遮蔽行 NaN）。可见性掩码取 t=0 的 `pairwise_visibility_mask`，对角自注意恒允许。
- **decoder**：chunk k 输入编码 token k-1（k=0 为 BOS）+ centroid 投影 + chunk 位置 +
  encoder context；注意力分两支——(i) 每 agent 自身 10 chunk 的因果自注意，
  (ii) 跨 agent 交叉注意的 K/V = **纯 token 恒等嵌入**（BOS/token embedding + centroid 投影，
  不含任何历史或 context），掩码为 present ∧ 非自身 ∧ t=0 可见 ∧ j≥1 ∧ j≤k + NULL 键。
  任何从 s 到 a 的信息路径都必须经过直接注意力边，而该边被可见性掩码移除。
- **解码语义**：与 `batched_motion_vectors` 严格一致——从原始 t=0 历史位置累加
  centroid+residual；轨迹损失的掩码用**累积链有效性**（current 有效且沿途 future 全有效），
  与 token chunk 有效性同构。
- **损失**：CE（ignore 255，且 valid 掩码折入 target——归一化与被忽略元素严格同集）、
  轨迹 Huber（delta=1.0 m，工程选择）、endpoint Huber（t=5s，主评测 agent），各自先按有效
  元素归一再以 1.0/0.5/0.2 加权。
- 参数量 13,126,282 = 13.13M（区间 10–25M）；centroid 输入归一化 /5.0 m、residual 头直接输出米制。

## 4. 训练 smoke（M1/M2）

- **M1 单批 overfit**（16 样本、seed 7、500 updates、GPU fp32）：first-20 均值 14.33 →
  last-20 均值 0.149，比例 0.010（门槛 ≤0.5）；token accuracy 0.5%→98.7%，CE 4.95→0.065。
- **M2 GPU smoke**（200 updates、bf16、micro 16×16=有效 256、warmup 5%、clip 1.0、
  确定性算法）：损失 10.246 → 3.290(u100) → 2.203(u200)，全程有限；715.9 s、
  吞吐 107.3 样本/s、峰值显存 976.6 MB（RTX 4060 Ti 16G）——正式训练的显存余量极大。
  （终值损失跨进程有 1.75–2.20 的混沌波动，见 §7 缺陷 6。）
- **resume 检查**（gate 语义见 `configs/p33/model_smoke_v1.json` 的 `resume_gate`）：
  从 update-100 checkpoint 重建（模型/优化器/loader/RNG 全状态快照）接续到 200。gate =
  (i) **restore-point 逐位相等**——恢复后第 1 个 update 损失与不间断运行逐位相同，任何
  checkpoint 内容错误（权重/Adam 矩/loader 位置/RNG）必然在此暴露；(ii) **20-update
  窗口轨迹相等**（≤5% 窗口均值；该视界内混沌噪声 ≤~5e-3，真实状态错误在前几步即宏观
  分歧）；(iii) 样本流与快照一致；(iv) 恢复分支损失有限且继续下降。长视界漂移是
  run 间随机分布的混沌量（同样 100-update 尾段，三次运行 max 漂移 0.27 / 0.39 / 1.12），
  与最终权重差（~1-2% 参数尺度）一起仅作诊断记录、不设 gate。
- 训练中发现并修复的工程缺陷详见 §7。

## 5. 评估 smoke（M3，全部 4,416 dev 样本，188.9 s）

- teacher-forced token NLL 1.575（均匀基线 4.85）、accuracy 70.0%；6 样本 top-p=0.95
  T=1.0 rollout 组级：waymo minADE@6 11.71 / minFDE@6 22.52，interaction 6.36 / 12.01
  （CV 对照 3.97/10.97 与 3.25/8.56）——欠训练状态下远差于 CV，属预期。
- **teacher-decode 探针**（`TEACHER_DECODE_PROBE.json`）把误差分解为：解码路径
  （真值 token + 预测残差）ADE 1.30 / 2.54、FDE 2.44 / 4.93 ——**健全且优于 CV**；
  argmax token（+贪心误差）2.52 / 3.76 ——教师强制域已具竞争力；缺口纯粹来自自回归
  漂移+采样复合（rollout 单采样 17.09/10.29，×4.5）。归因链支持进入 100-shard，
  不支持先改表示（详见 REPORT.md §8）。
- token 频率：真值 top-1（近零运动 #23）28.0%，argmax 预测 36.9%（×1.32）；
  6 个稀有 token（真值 2–17 次）argmax 全零；macro recall（argmax）0.530、采样 0.0096。
- 速度违规率 0.002%（281,070 条轨迹）；加速度/jerk "100%" 为逐 chunk 分段线性解码的
  结构性边界跳变，几何代理不作 G1 解读。offroad 在 scene-shard 契约下**不可辨识**
  （无可行驶区域多边形），如实记为不可算，不报 0。
- 分层与来源方向、叠加图（8 张，CV vs 6 样本 rollout）见 `MODEL_DEV_METRICS.json`、
  `overlay_plots/` 与 REPORT.md；两来源同向劣于 CV（waymo ×2.95、interaction ×1.96），
  无方向反转。

## 6. heldout 纪律

本阶段未读取任何 heldout 内容（Waymo validation shards、INTERACTION 四个 heldout location、
P2.12 seed77000、ABD NPC 用途）；`heldout_trajectory_content_read=false`。dev 仅用于 smoke
工程验证，未做任何基于 dev 的超参或结构选择。

## 7. 发现的工程缺陷（诚实记录）

1. **Waymo shard 混 split**：loader 初版假设 shard 单 split，真实数据立即暴露；改为行级
   split 跟踪（`rows_by_split`）。这正是 D0 必须在真实数据上跑的原因。
2. **epoch 边界 StopIteration**：`iter_batches` 是单 epoch 生成器，训练循环初版当无限流用，
   在 batch 563（waymo 4501 轮空）崩溃；新增 `advance_epoch()` 语义。
3. **checkpoint 活引用假通过**：初版 `checkpoint_payload` 直接存 `optimizer.state_dict()`
   （张量为活引用）与 history 列表引用——phase A 继续训练时 checkpoint 内容原地变异，
   resume 分支实际加载了 update-200 的 Adam 矩（错误状态），而"尾段损失逐位相等"的比较
   对象其实是同一数组的别名（假阳性）。快照语义（deepcopy/clone/list()）修复后，
   resume 检查给出真实结论；该假阳性由 `final_weights_match=false` 的诚实信号暴露。
4. **CE 归一化不一致**：`ignore_index=255` 不会忽略"valid=False 但 target≠255"的行，
   归一化分母与被忽略集合可能不同（真实数据中二者恰好等价，合成数据暴露）；改为把
   valid 掩码折入 CE target。
5. **CUDA 逐位 resume 相等不可实现（归因纠正）**：修复快照语义后，200-update 的尾段
   损失仍相差 ~0.27。初步归因"embedding 反向 atomicAdd 非确定"**是错的**——严格模式
   `torch.use_deterministic_algorithms(True)` 在本训练环路（含 embedding 前反向）不报
   任何无确定性实现的算子，内核层面已全确定。真机制见第 6 条。保留本条以记录归因错误
   本身：对无法逐位复现的现象，先用可证伪实验定机制，再改 gate 语义。
6. **分配器布局 → GEMM 核选择 → ULP 差异混沌放大**（`allocator_probe.py` +
   `ALLOCATOR_PROBE_RESULT.json`）：从同一 10-update 状态出发，(a) 干净重建（真实
   phase B 场景）首 update 逐位相等、第 2 步 1.4e-5、30 步混沌增长到 ~1e-3——与完整
   200-update run 的形态一致；(b) 仅扰动 CUDA 分配器（分配/释放奇尺寸张量 + 清缓存，
   不触碰任何数学）后，首 update 仍逐位相等、第 2 步即分歧 4.1。结论：重建进程的
   分配器布局改变张量对齐，cuBLAS 按对齐选核，浮点累加顺序随之变化，混沌动力学放大
   到宏观损失差。**checkpoint 状态恢复是精确的**（首 update 逐位相等是权重/优化器/
   loader/RNG 全对的最强证据；分歧起点位置与幅度均随机——三次运行起点分别在尾段
   index 9/4/6，100 步 max 漂移 0.27/0.39/1.12，phase A 自身跨进程终值也不同）。
   据此 resume gate 收敛为可检验且稳健的对象：restore-point 逐位相等 + 短窗口轨迹
   相等 + 流匹配 + 恢复分支 sanity（`resume_gate` 配置键内含完整 semantics）；
   长视界漂移与权重差为诊断量。教训：**跨进程逐位复现不是 checkpoint 正确性的必要
   条件，不要把平台属性当成状态 bug 追；对混沌量设绝对界之前先看它的 run 间分布**。

## 8. 对 P3.3.3 的建议

**工程 GO：进入 100-shard architecture，不先改表示/指标。** 依据（完整论证见
REPORT.md §8）：工程边界 12/12 成立；teacher-decode 探针证明表示与解码路径健全
（教师强制域两来源优于 CV）；唯一重大缺口（rollout ×4.5 劣化）归因于自回归漂移+
采样复合，是训练深度/暴露偏差问题，100-shard + 正式训练量正是它的对照实验。
护栏：固定间隔记录 teacher-argmax vs rollout 指标比为漂移预警；若正式训练量下差距
仍 >2×，先评估采样策略与暴露偏差缓解，不动表示。监测项：role-0 agent 出现率
（smoke dev 为 0）、稀有 token 频率、两来源组数均衡。

## 9. P3.3.2a 勘误与修正（2026-09-13）

独立复核（`runs/20260913_p332_model_smoke/CODEX_REVIEW.md`，commit 0f28bcc）确认的
问题与 P3.3.2a 落实的更正：

1. **§3 模型描述已过时**：原实现把每个 source 的全部历史帧池化成静态 token、仅按 t=0
   可见性门控注意力——t=0 可见但过去被遮挡的 source 会把遮挡帧信息泄漏给 query（部分
   历史反事实 logits 改变 ~e-5..e-4 量级；dev ~90% 样本存在此类 pair）。修正为逐查询
   视图键：pair (q,s) 的键按 `pairwise_visibility[q,:,s] ∧ state_valid[s,:]` 逐帧加权
   池化，隐藏帧权重恰为 0（部分历史反事实**逐位**不变）。
2. **§4/§7 缺陷 6 的机制叙述错误**：m2_run.log 实际包含 "Memory Efficient attention
   defaults to a non-deterministic algorithm" 警告（warn_only=True 把严格模式报错降级
   为警告），"严格模式不报错→排除内核非确定性"不成立；且分配器探针的 R0/R2 共用同一
   优化器 payload，`load_state_dict` 的存储别名使 R2 初始状态被 R0 训练污染——旧探针的
   大分歧由状态污染解释，与分配器无关。修正后探针（strict `warn_only=False` +
   math SDPA + `restore_optimizer_isolated`）显示：干净重建、分配器扰动、活体继续三者
   30/30 update 逐位相等；确定性配置吞吐代价 ~8%（95.2 vs 103.7 样本/s）。
3. **§8 监测项勘误**：role-0 是 padding；"other" 是 agent_type=4 且 smoke dev 有
   8,692 条记录——该路径在真实数据上已被行使，不作为"未行使路径"监测。
4. **exposure 口径**：M2 有效样本 25,600/来源 = **5.69 个 waymo train pass / 1.15 个
   interaction train pass**（不是 "1.1 个 waymo epoch"）；715.9 s 覆盖主分支 200 +
   恢复分支 100 = **300 update 当量**。
5. **指标补充**：per-agent minADE@6 可为不同 agent 选中不同的联合样本；已加 joint-scene
   best-of-6 次级指标（同一 k* 服务场景内全部 eval agent；主端点仍为 spec 冻结的
   per-agent minADE@6）。sampled_token_nll 是 top-p 截断后采样、按全 softmax log-prob
   计算的样本均值 NLL（平均熵的单样本无偏估计），不是采样分布的严格熵，
   `exp(·)` 不可称有效词数。
