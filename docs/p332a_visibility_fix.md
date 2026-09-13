# P3.3.2a：可见性泄漏修复与确定性重建（2026-09-13）

- 上游：P3.3.2（`docs/p332_model_smoke.md`，已加勘误横幅）；触发 = 独立复核
  `runs/20260913_p332_model_smoke/CODEX_REVIEW.md`（commit 0f28bcc，REQUIRES_CORRECTION）。
- 本阶段范围：**修正 + 重跑**，不动冻结项（p33.0-v1.1 主指标 per-agent minADE@6、
  split、模型规模 10–25M、数据契约 scene-shard-v1）。重跑数字见
  `runs/20260913_p332a_visibility_fix/REPORT.md`；本文记录改了什么、为什么、如何验证。

## 1. 模型修复：per-query 视图键（scenario_lab/p33_model.py）

**缺陷**：原实现把每个 source 的全部 10 帧历史池化成一个静态 token，注意力仅按
t=0 的 `pairwise_visibility_mask[:, :, -1, :]` 门控。t=0 可见但过去帧被遮挡的 source
会把**遮挡期历史**泄漏给 query——违反 spec §4.5（逐 query、逐时刻掩码，池化前生效）。
真实数据量化：抽验 pair 的部分历史反事实 logits 改变 3.99e-5；dev 3,976/4,416 样本
存在至少一个此类 (query, source) pair。原 `actor_visible_counterfactual` 检查只扰动
"t=0 起整段不可见"的 source，恰好不触发泄漏路径——假阳性。

**修复**（`_pair_weights` / `_pair_view_tokens`）：

- pair (q, s) 的键不再共用 source 静态 token，而是按
  `W[q,t,s] = pairwise_visibility[q,t,s] ∧ state_valid[s,t]`（对角恒允许）逐帧加权：
  masked mean + last-visible 两支池化（`einsum("bqts,bstd->bqsd")`，分母 clamp≥1），
  加 type/role/source embedding 后投影为 **16×16 组 per-query 键**。
- 隐藏帧权重恰为 0 → 部分历史反事实**逐位**不变（IEEE 0·x=0，无累加路径）。
- no-relay 性质保持：键由原始历史计算、逐层不更新，a→b→s 转发路径仍不存在；
  encoder MHA 重排为 `[B*A,1,d]` query × `[B*A,A,d]` key，`blocked` 作
  key_padding_mask；对角恒可注意（own 完整历史）。
- 参数量不变（13,126,282）；decoder 交叉注意门保持 t=0（identity 键不含历史，
  无时序泄漏面，保守选择记录在案）。

**验证**（`tests/test_p33_model.py`，13 项）：
`test_partial_history_counterfactual_bitwise_invariance`（遮挡帧扰动 +50/+20 m，
query logits `torch.equal` 逐位不变；可见帧扰动必须改变）与
`test_past_visible_frames_flow_under_t0_occlusion`（t=0 遮挡但过去可见的 source，
其过去帧信息必须可达）。真实数据抽验：pair (b0, q1, s6) frame 0 扰动 →
`torch.equal True, delta 0.0`。

## 2. shard 生命周期修复（scenario_lab/p33_dataset.py）

`ShardReader.get()` 原返回底层 numpy 数组的行视图，跨 shard 换驻留后旧视图钉住
backing array。改为 `value[index].copy()`（行级拷贝，docstring 说明 view-pinning 依据）。
新测试 `test_row_copies_do_not_pin_previous_shard_backing_arrays`：weakref 跟踪
`reader._arrays`，跨 shard 加载 + `gc.collect()` 后断言全部回收 + `OWNDATA` 标志。

## 3. 确定性重建（research_tasks/train_p33_nominal.py + allocator_probe.py）

**旧探针为什么不可信**（复核确认的两条）：

1. `warn_only=True` 把严格模式报错降级为警告——m2_run.log 实际含有
   "Memory Efficient attention defaults to a non-deterministic algorithm" 警告，
   "严格模式不报错→内核已确定"的推理不成立。真实非确定源 = bf16
   memory-efficient attention 反向。
2. `Optimizer.load_state_dict` 在 device/dtype 匹配时与 payload **共享张量存储**——
   旧探针 R0（干净重建）先训练即原地污染 R2（分配器扰动）的初始优化器状态，
   R2 的巨大分歧由状态污染解释，与分配器无关。

**修复**：

- 训练脚本：`torch.use_deterministic_algorithms(True, warn_only=False)` +
  `enable_mem_efficient_sdp(False)` + `enable_flash_sdp(False)`（math 后端），
  结果 JSON 落盘 `deterministic_algorithms` 与 `sdpa_backends` 字段。
- 新函数 `restore_optimizer_isolated`（load 后逐 state tensor clone，断绝别名），
  phase B 与探针统一使用；单元测试
  `test_optimizer_restore_does_not_alias_the_payload` 锁死。
- 探针重建（`runs/20260913_p332a_visibility_fix/allocator_probe.py`）：R1 活体、
  R0 干净重建、R2 分配器扰动（奇尺寸张量存活 + 碎片化 + empty_cache）三者从同一
  10-update 状态继续 30 update——**全部逐位相等**；吞吐对照 95.2（strict+math）
  vs 103.65（mem-eff+warn_only）样本/s，确定性代价 ~8%。
- 结论：跨进程逐位复现在本 CUDA 栈**可实现**（此前"分配器→核选择→混沌放大"的
  归因撤回）；resume gate v3 语义不变（restore-point 逐位 + 20-update 窗口 + 流匹配
  + sanity），但其依据改为确定性配置成立，而非"混沌不可控"。

## 4. 指标与图修正（scenario_lab/p33_metrics.py + evaluate_p33_t1.py）

- **joint-scene best-of-K**：k* = 使场景内 eval-agent 平均 ADE 最小的样本，全部
  agent 报告第 k* 样本的 ADE/FDE（`min_ade_joint`/`min_fde_joint`，组级聚合）。
  次级指标——主端点仍为 spec 冻结的 **per-agent minADE@6**（可为不同 agent 选不同
  样本）。测试：`test_joint_scene_best_of_k_uses_one_sample_index`。
- **分来源叠加图**：`overlay_plot_per_source: {waymo: 4, interaction: 4}`，文件名
  `overlay_{source}_{ii}.png`；CV 只在需要出图的样本上惰性计算。
- **sampled_token_nll 措辞**：top-p 截断/重归一后采样 token、按**全 softmax**
  log-prob 计算的均值 NLL——截断采样分布相对于原模型分布的交叉熵的 MC 估计，不是采样分布熵，
  `exp(·)` 不可称有效词数（结果 JSON 附 `sampled_token_nll_definition`）。
- **exposure 口径**：M2 有效样本 25,600/来源 = 5.69 个 waymo train pass / 1.15 个
  interaction train pass；elapsed = 300 update 当量（主 200 + 恢复分支 100）。
- **role/type 勘误**：role-0 = padding；"other" = agent_type=4（dev 8,692 条记录），
  对应路径在真实数据上已被行使，从"未行使监测项"中移除。

## 5. 测试与验证状态

- 全仓 183 项测试通过（旧 177 + 新 6：部分历史反事实逐位不变 ×1、过去帧可达 ×1、
  全遮挡有限性 ×1、backing-array 回收 ×1、优化器恢复无别名 ×1、joint best-of-K ×1）。
- M1 单批 overfit 重跑通过（见 REPORT.md）；M2 双跑跨进程逐位对照、CV 全 dev、
  M3 全 dev、teacher-decode 探针全部重跑，数字与 gate 见
  `runs/20260913_p332a_visibility_fix/REPORT.md` 与 `SMOKE_VALIDATION.json`。

## 6. 边界声明（不变）

`status=pass` 仅表示工程 smoke 边界成立、可进入人工结果评审；不表示 G1、不表示优于
CV/GRU、不构成进入 100-shard 的模型质量结论。heldout 未读取
（`heldout_trajectory_content_read=false`）；dev 仅用于工程验证，未做基于 dev 的
超参/结构选择。P3.3.2 旧目录的模型侧数字与 checkpoint 已标记 superseded，CV 结果
不受影响。
