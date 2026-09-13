# P3.3.2 交接：最小 AR-Scene-v1、基线与 GPU smoke

_交接给 GLM 5.3；2026-09-13；上游 commit：`bd44be4`_

## 0. 本阶段的唯一目标

在 P3.3.1 已通过 G0 的 smoke 数据上，实现并验证：

1. 有界、可恢复、按来源平衡的数据加载器；
2. constant-velocity（CV）确定性基线及正式 T1 指标实现；
3. 冻结规范中的最小 `AR-Scene-v1`；
4. 单 batch overfit、CUDA smoke training、checkpoint resume、六样本自回归 rollout；
5. 全部 smoke dev 的按来源诊断和少量可视化。

本阶段是 **模型与训练链路的工程 smoke**，不是 100-shard architecture experiment，不执行 G1
统计判定，不读取任何新 heldout/final-confirmation 内容，也不开始安全引导、ABD 扰动或 CARLA。

完成后必须能回答：数据是否正确送入模型、隐藏信息是否被 mask、损失能否下降、checkpoint 能否恢复、
自回归输出是否有限且形状正确、CV 与 Transformer 的 smoke dev 指标能否复现。即使 Transformer 指标
很差，只要如实落盘并完成诊断，也算有效结果；不得通过反复调参把 smoke 包装成优越性证据。

## 1. 工作区与不可破坏边界

- 唯一工作区：`D:\Projects\Scenario_Generation_Research`
- WSL 路径：`/mnt/d/Projects/Scenario_Generation_Research`
- 所有 Python、测试和训练必须在 WSL2 Ubuntu 24.04 内运行。
- Python：`/home/shuai/.venvs/scenario-gpu/bin/python`
- 当前 GPU 环境：PyTorch `2.14.0+cu130`，CUDA 可用，RTX 4060 Ti 16 GB。
- 当前仓库存在大量与本阶段无关的修改、删除和未跟踪结果。**禁止 reset、clean、checkout 覆盖或顺手
  提交它们。** 最终只 stage P3.3.2 新增/修改文件。
- 上游里程碑应为 `bd44be4`。先执行 `git log -1 --oneline` 和 `git status --short`，但不要因工作树脏
  而清理它。
- 大文件必须有界读取；不要把全部 31,215 个样本载入内存。
- 不安装第二套 Python 环境，不把项目复制到 `/home/shuai/projects`。
- 未得到用户明确授权，不 push。

建议首先运行：

```bash
cd /mnt/d/Projects/Scenario_Generation_Research
/home/shuai/.venvs/scenario-gpu/bin/python research_tasks/verify_p33_data.py \
  --output runs/20260913_p331_data_pipeline/smoke
/home/shuai/.venvs/scenario-gpu/bin/python -m pytest -q
```

期望上游状态：`VERIFICATION.json status=pass`，全仓 `148 passed`。如果本地
`smoke/units/` 不存在，才允许用下列命令从冻结 smoke inventory 重建；不得扩大 selection：

```bash
/home/shuai/.venvs/scenario-gpu/bin/python research_tasks/convert_p33_data.py \
  --output runs/20260913_p331_data_pipeline/smoke
```

## 2. 必须先读的文件

按顺序阅读，机器可读配置高于本交接中的解释：

1. `AGENTS.md`
2. `docs/research_claims.md` 第 9 节，尤其 9.4–9.10
3. `docs/p33_task_spec.md`
4. `docs/p331_data_pipeline.md`
5. `configs/p33/ar_scene_v1.json`
6. `configs/p33/data_pipeline_v1.json`
7. `runs/20260913_p330_spec_v2/FROZEN.json`
8. `runs/20260913_p331_data_pipeline/smoke/{RUN_CONFIG,DATASET_MANIFEST,CODEBOOK_MANIFEST,G0_VALIDATION,VERIFICATION}.json`
9. `scenario_lab/p33_spec.py`
10. `scenario_lab/p33_pipeline.py`
11. `research_tasks/convert_p33_data.py`
12. `tests/test_p33_spec.py` 与 `tests/test_p33_pipeline.py`

若本交接与 `ar_scene_v1.json` 冲突，以配置和 P3.3.0 冻结规范为准；如果配置内部无法实现或相互矛盾，
停止相应实验，记录具体冲突，不可静默改定义。

## 3. 已完成的输入及其边界

正式输入目录：`runs/20260913_p331_data_pipeline/smoke/units/`。该目录约 481 MB，由局部
`.gitignore` 排除，不要提交 NPZ/JSONL 数据 shard。

| 项目 | 数量 |
|---|---:|
| Waymo train/dev | 4,501 / 481 个独立 Scenario |
| INTERACTION train/dev | 22,298 / 3,935 个重叠窗样本 |
| INTERACTION 独立 recording case | 16（train 12、dev 4） |
| 全部 train/dev 样本 | 26,799 / 4,416 |
| 全部独立组 | 4,998 |
| motion-token 有效目标 | 3,371,044 |
| codebook | 128 个质心，train-only 拟合 |

INTERACTION 重叠窗不能被当作独立统计样本。训练可以消费重叠窗，但 dev 汇总必须先在
`(location,case_id)` 内汇总，再把 case 当成独立单位。Waymo 的独立单位是 `scenario_id/group_id`。

已知数据风险必须进入报告：

- 7,423/31,215 个样本发生 agent 截断；
- 30,685/31,215 个样本发生 map-polyline 截断；
- 最大 token 占比 21.58%，有明显长尾，最少 token 只有 14 个标注；
- INTERACTION `pedestrian/bicycle` 无法可靠拆型，当前统一为 `other`；
- public visibility 只是动态 actor box 几何代理，不含建筑物遮挡；
- smoke dev 只有 481 个 Waymo Scenario 和 4 个 INTERACTION case，远低于 G1 的 5,000/100。

## 4. 严禁读取的内容

P3.3.2 不需要任何 heldout。禁止读取内容、计算哈希、抽样预览或用文件大小间接选模：

- Waymo `validation` 目录中的 P3.3 final-confirmation shard，尤其编号 00002–00149；
- INTERACTION 地点 `DR_CHN_Merging_ZS`、`DR_CHN_Roundabout_LN`、`DR_DEU_Merging_MT`、
  `TC_BGR_Intersection_VA`；
- P2.12 `seed77000` 结果用于模型选择、阈值选择或早停；
- ABD 数据用于训练 NPC 生成模型。

运行 manifest 必须继续写 `heldout_trajectory_content_read=false`。若误读，立即停止，在报告中列出确切
路径和操作；不得删除痕迹后继续称“未读”。

## 5. 建议新增文件

文件名可小幅调整，但职责必须保持清楚：

```text
scenario_lab/p33_dataset.py             # shard iterator、source-balanced batches、metadata
scenario_lab/p33_model.py               # AR-Scene-v1、loss、teacher forcing、rollout
scenario_lab/p33_metrics.py             # CV、ADE/FDE/NLL、分组聚合、运动学诊断
research_tasks/train_p33_nominal.py      # overfit/smoke train、resume、manifest
research_tasks/evaluate_p33_t1.py        # CV/model 全 smoke-dev 评价
configs/p33/model_smoke_v1.json          # 只放 smoke 执行预算，不复制/改写冻结科学定义
tests/test_p33_dataset.py
tests/test_p33_model.py
tests/test_p33_metrics.py
docs/p332_model_smoke.md
runs/<实际日期>_p332_model_smoke/
```

结果目录至少包含：

```text
RUN_CONFIG.json
ENVIRONMENT.json
DATA_AUDIT.json
MODEL_MANIFEST.json
OVERFIT_METRICS.json
TRAINING_CURVE.jsonl
CHECKPOINT_MANIFEST.json
CV_DEV_METRICS.json
MODEL_DEV_METRICS.json
SMOKE_VALIDATION.json
REPORT.md
preview_waymo_rollout.(png|gif)
preview_interaction_rollout.(png|gif)
checkpoints/                 # 大文件本地保留并 gitignore
```

## 6. 数据加载器的硬性要求

### 6.1 有界读取

NPZ 使用压缩格式，不能假装 `mmap` 已生效。每个 worker 最多缓存一个 shard；读完释放，不建立全局
样本对象列表。JSONL metadata 可以逐 shard 读取。每次训练前验证：

- `DATASET_MANIFEST.validation.shards[].sha256`；
- NPZ 行数与 JSONL 行数；
- `schema_version=scene-shard-v1`；
- source/split/group/sample ID；
- 数组 shape/dtype 和 token/mask 契约。

允许缓存已验证 shard 的 hash 结果到 run manifest，resume 时只有输入 hash 和配置 hash 全同才能复用。

### 6.2 Source-balanced batch

默认 micro-batch 为 16，Waymo 与 INTERACTION 各 8；若某次尾 batch 不足，允许确定性循环较小来源，
但必须记录每个 epoch 的实际 source exposure。不得让 INTERACTION 的 22,298 个窗口按自然比例压倒
Waymo。shuffle seed 必须固定并进入 manifest。

训练抽样与统计独立性是两件事。smoke 可使用 source-balanced example sampling；报告中必须同时给出
每个 INTERACTION case 的曝光量，不能把窗口数量写成独立样本量。

### 6.3 Batch 输出

至少返回全部 12 个冻结数组，加：

```text
source_id: int64 [B]                 # 明确词表，例如 Waymo=0, INTERACTION=1
sample_id: list[str]
group_id: list[str]
split: list[str]
truncation metadata
```

主训练监督可使用所有 `motion_token_valid_mask=true` 的 agent/chunk。主评价 agent mask 固定为：

```text
agent_present_mask
AND agent_role in {anchor=1, prediction_target=2}
AND future supervision exists
```

INTERACTION 没有 Waymo 的 `tracks_to_predict`，因此其主要评价对象是每个窗口明确选择的 vehicle anchor。
若增加“所有可监督 context agent”指标，只能作为 secondary，必须分开命名。

## 7. 隐藏信息边界：必须用测试锁死

`agent_history` 保存完整监督状态，不等于每个 query 都可以看到全部 agent。模型输入必须先结合
`pairwise_visibility_mask [B,Q,T,S]` 构造 query-specific history；隐藏 source-agent 的所有 feature
先置零，再进入任何 attention、pooling、残差或共享 scene token。

必须写以下反事实测试：

1. 固定模型为 eval、固定全部可见输入；大幅修改某 query 不可见 source-agent 的历史值，该 query 的
   context/logits 必须在数值容差内完全不变；
2. 修改可见 source-agent，query 输出应发生变化；
3. invalid/padded agent 不得影响任一有效 query；
4. decoder 在 chunk `k` 的 logits 不得随 chunk `k` 或未来 teacher token 改变；只能依赖 `<k`。

不能先对完整 agent truth 做全局 pooling 再 mask，因为那已经泄漏。地图是共享静态输入，但必须遵守
`map_point_mask`。

## 8. Constant-velocity 基线

CV 必须先于 Transformer 完成并落盘，作为指标实现的独立验证：

```text
p_hat(t) = p_current + v_current * t,  t=0.1,...,5.0 s
```

当前位置和速度取最后一个有效历史状态，全部已经在 anchor-local 坐标中。不要再按 agent heading
旋转。确定性 CV 复制为 6 个相同样本时，`minADE@6/minFDE@6` 应与 ADE/FDE 相同；写测试验证。

指标至少包括：

- ADE、FDE、`minADE@6`、`minFDE@6`；
- 按 source、agent type、agent role、agent/map 是否截断分层；
- Waymo 按 Scenario，INTERACTION 先按 case 聚合；
- 有效 agent 数、有效 future 点数、无完整终点而不能计算 FDE 的数量。

严禁用 sample-level bootstrap 处理 INTERACTION 重叠窗。P3.3.2 不运行 G1 bootstrap，但聚合代码必须
保留 group-level 表，供 architecture 阶段使用。

## 9. AR-Scene-v1 最小实现

冻结结构来自 `configs/p33/ar_scene_v1.json`：`d_model=256`、6 层、8 heads、FFN=1024、dropout=0.1、
pre-layer norm、GELU、source/type/role embedding、polyline point MLP + masked pooling、agent-agent 与
agent-map interaction。不得为快速 smoke 静默缩小主模型；测试 fixture 可以用 tiny config。

参数总量必须实际计算并落在 10M–25M。若按合理实现仍不在区间，报告精确数字和结构组成，先修实现；
若必须改变冻结结构，则升级 spec，而不是添加不参与 forward 的“凑数参数”。

### 9.1 输入与 encoder

- history：`[B,16,11,8]`，结合 query-specific visibility 使用；
- map：`[B,64,20,6]`，point MLP 后 masked pool；
- type/role/source：独立 embedding；
- 所有 attention 必须传显式 padding mask；全 mask 行须安全处理，不能产生 NaN；
- 输出至少提供每个 active query-agent 的 context embedding。

### 9.2 时间自回归 decoder

未来为 10 个 0.5 s chunk，同一 chunk 内 agent 并行生成。推荐用 shifted teacher token：chunk `k` 只看
BOS 和 `<k` token。禁止让 agent ID 顺序形成同 chunk 内自回归条件。

接口至少为：

```python
encode_context(batch) -> context
forward(batch, teacher_tokens) -> {
    "motion_token_logits": FloatTensor[B,16,10,128],
    "delta_xy_residual": FloatTensor[B,16,10,10],
}
rollout(batch, num_samples=6, temperature=1.0, top_p=0.95, seed=...) -> {
    "trajectories": FloatTensor[B,6,16,50,2],
    "tokens": LongTensor[B,6,16,10],
    "token_log_prob": FloatTensor[B,6,16,10],
}
```

codebook 向量是 **anchor-local 的连续 `delta_xy`**，形状 `[128,10]`，每个 token 展开成 5 个二维
位移。连续预测为 `centroid[token] + residual`，再从当前坐标累加得到未来位置。不要误改成每个 agent
heading-local，也不要把 absolute future 当 residual。

### 9.3 Loss

严格实现冻结权重：

```text
total = 1.0 * token_cross_entropy
      + 0.5 * trajectory_huber
      + 0.2 * endpoint_huber
```

- CE 只在 `motion_token_valid_mask` 上计算，ignore index 255；
- continuous target 是真实 anchor-local displacement 与对应 codebook centroid 的差；
- trajectory Huber 比较累积后的 50 帧位置，只用 `future_valid_mask`；
- endpoint Huber 只用第 50 帧有效的主评价 agent；
- 每项先按有效元素归一化，再乘权重，不能让 agent 数或来源样本数隐式改变权重；
- 每步记录三项 raw loss、total、token accuracy、梯度范数和学习率。

## 10. Smoke 执行顺序和停止条件

### D0：loader audit

遍历 train/dev manifest，不训练；确认计数与上游完全一致，最大驻留样本不超过一个 shard，source-balanced
batch 比例正确，固定 seed 的前 N 个 sample ID 可复现。失败则停止。

### B0：CV 全 smoke-dev

在全部 4,416 个 dev 样本上运行 CV，保存逐 group 中间表和按来源结果。指标出现 NaN、FDE 样本数解释
不清或确定性 `minADE@6 != ADE` 时停止。

### M0：forward/backward contract

用 synthetic/tiny fixture 和真实 1 个 batch 检查：输出 shape/dtype、finite、mask、causality、hidden-state
反事实、梯度有限、参数量。失败则停止。

### M1：单 batch overfit

- 固定 16–32 个 train 样本，seed=7；
- 最多 500 update；
- 不使用 dev 选择停止点；
- 通过建议：末 20 步 total loss 均值不高于初 20 步的 50%，token accuracy 明显上升，所有 loss/gradient
  有限；
- 保存曲线和预测预览。

若不通过，先查 shift、mask、residual target、codebook decode 和 loss normalization。不得直接增加正式
训练量掩盖实现错误。

### M2：GPU smoke training

- 仅 seed=7；
- source-balanced micro-batch 16；目标 effective batch 256，用 gradient accumulation；
- AdamW、lr `3e-4`、weight decay `0.01`、warmup 5%、clip norm 1.0；
- AMP 优先 bf16（若设备/算子不支持则 fp16 + GradScaler，并记录）；
- 建议 200 optimizer updates，目的只验证链路；
- 至少在中间保存 checkpoint，恢复后继续，验证 step、optimizer、scheduler、scaler、RNG 状态均恢复；
- 记录峰值显存、wall time、samples/s、data wait、GPU utilization、有效 token/s。

OOM 时只允许先调低 micro-batch 并提高 accumulation、启用 activation checkpointing；保持 effective
batch、主模型维度和科学定义不变。任何变更都写入 effective config。

### M3：全 smoke-dev 诊断

冻结 smoke checkpoint，在全部 smoke dev 上执行 teacher-forced token NLL 和六样本开放环 rollout。
固定 sampling seed，分别报告 Waymo/INTERACTION、group-level 指标和截断分层。至少生成一张 Waymo、
一张 INTERACTION 的 history/ground truth/CV/六样本预测叠加图；最好同时生成短 GIF。

**这里没有“必须优于 CV”的通过条件。** 如果模型不如 CV，报告负结果和最可能的实现/数据原因，仍可
完成 P3.3.2；是否进入 100-shard architecture 由结果评审决定。

## 11. 运动学指标的诚实边界

P3.3.2 至少实现由预测 xy 可直接辨识的速度、加速度、jerk 诊断，并记录阈值。碰撞可用当前尺寸和由
位移推导的 heading 做近似，但必须标为 geometric proxy。当前 `scene-shard-v1` 没有完整 drivable-area
polygon，不能把“距 lane center 很近”直接称为正式 offroad 率。

若正式 G1 确实需要 offroad，`SMOKE_VALIDATION.json` 应把它标为
`not_identifiable_from_current_scene_shard`，并提出在读取更多数据前升级 P3.3.0 schema 的具体方案。
不得输出一个看似完整但语义错误的 0% offroad。

## 12. 测试要求

只写能锁住关键错误的测试，至少覆盖：

1. shard/hash/count 与固定顺序复现；
2. source-balanced batch 和无跨 split；
3. CV 手算 ADE/FDE，确定性 min@6 等价；
4. model 输出 shape、参数量范围、finite forward/backward；
5. hidden-agent 反事实不变性；
6. padded/invalid agent 不影响输出；
7. teacher-token causal shift，无当前/未来 token 泄漏；
8. codebook decode + residual + cumulative sum 的手算例；
9. loss mask/normalization；
10. checkpoint resume 恢复 step 和 RNG；
11. 固定 seed 的 rollout 可复现，不同 seed 能产生不同样本（若分布非退化）。

完成后运行全仓：

```bash
/home/shuai/.venvs/scenario-gpu/bin/python -m pytest -q
git diff --check
```

## 13. `SMOKE_VALIDATION.json` 最低结构

```json
{
  "phase": "P3.3.2",
  "status": "pass_or_fail",
  "engineering_only": true,
  "g1_evaluated": false,
  "heldout_trajectory_content_read": false,
  "checks": {
    "upstream_p331_verification": "pass",
    "bounded_loader": "pass_or_fail",
    "source_balanced_batches": "pass_or_fail",
    "cv_metric_contract": "pass_or_fail",
    "actor_visible_counterfactual": "pass_or_fail",
    "decoder_causality": "pass_or_fail",
    "finite_forward_backward": "pass_or_fail",
    "parameter_count_in_range": "pass_or_fail",
    "single_batch_overfit": "pass_or_fail",
    "checkpoint_resume": "pass_or_fail",
    "six_sample_rollout": "pass_or_fail",
    "full_smoke_dev_evaluated": "pass_or_fail"
  },
  "limitations": []
}
```

`status=pass` 只表示可进入人工结果评审，不表示 G1 pass。

## 14. 报告必须明确回答的问题

1. 精确模型参数量、峰值显存、吞吐量和训练时间是多少？
2. CV 在两个来源上的 group-level ADE/FDE 是多少？
3. Transformer 的 teacher-forced NLL 与六样本 minADE/minFDE 是多少？
4. M1 loss 是否按预定幅度下降？M2 是否有限、可恢复？
5. 截断与未截断样本的误差差异多大？
6. token 长尾是否导致稀有 token 几乎不被预测？至少报告真实/预测 token 频率与 macro recall。
7. Waymo 与 INTERACTION 是否出现方向相反的效果？禁止只给 pooled 指标。
8. 当前结果支持进入 100-shard architecture，还是应先修表示/指标？
9. 哪些结论是 verified fact、engineering diagnostic、hypothesis、尚未验证的 scientific claim？

## 15. 禁止事项与决策规则

- 不读取 heldout/final confirmation。
- 不运行 100/500/1000 shard 转换或训练。
- 不做 PPO、CEM safety guidance、P2 router、ABD 敏感性、CARLA。
- 不用 dev 反复调到好看；smoke 只修明确工程错误。
- 不把 26,233 个 INTERACTION 窗口写成独立场景数。
- 不声称用了全部 Waymo/INTERACTION 数据。
- 不声称 Transformer 优于 CV/GRU、达到 G1、SOTA、NCAP 合规或 sim-real 校准。
- 不在失败后打开 final 数据救结果。
- 不改 `p33.0-v1.1` 主指标、split、门槛或模型规模而不升级版本并说明原因。

如果 smoke 模型明显不如 CV，优先诊断：token shift/解码错误、source imbalance、地图截断、agent 截断、
INTERACTION 重叠窗口权重、静止 token 主导、visibility proxy 过强/过弱、anchor-only evaluation。不能用
“数据量还小”作为唯一解释，也不能直接跳到 full training。

## 16. 提交边界与最终汇报格式

提交代码、配置、测试、报告、JSON/JSONL 指标和小型预览。不要提交 checkpoints、完整逐样本预测、NPZ
shard 或大动画；在结果目录的 `.gitignore` 中明确排除，并在 manifest 记录本地路径和 SHA-256。

建议 commit message：

```text
Implement P3.3.2 nominal model smoke pipeline
```

最终汇报必须包含：commit、变更文件、全仓测试数、各 smoke gate、CV/模型按来源指标、参数量、显存、
耗时、heldout 未读证明、失败/限制，以及是否建议进入 P3.3 architecture。不要只说“完成”或只报告
训练 loss。
