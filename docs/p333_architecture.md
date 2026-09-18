# P3.3.3：Architecture 档（100-shard 名义模型，2026-09-14/15）

- 上游：P3.3.2a ENGINEERING_GO（`docs/p332a_visibility_fix.md`；复核
  `runs/20260913_p332a_visibility_fix/CODEX_REVIEW.md`，修正 commit 8bc95ae）。
- 本档 = P3.3.0 冻结梯度第三档（100 waymo training shards + INTERACTION 25%/dev location），
  用途 `model_and_representation_selection`；完整数字与事实分级见
  `runs/20260913_p333_architecture/REPORT.md`（验收字段：ARCH_VALIDATION.json；产物哈希：
  ARTIFACT_MANIFEST.json）。本文记录做了什么、关键数字、遗留问题。

## 1. 做了什么

1. **数据**：`configs/p33/data_pipeline_architecture_v1.json`（= v1 + `stage: architecture`）；
   转换 147 unit / 121,224 examples（train 109,638 / dev 11,586），G0 pass、
   `cross_split_groups: 0`、`heldout_trajectory_content_unread: true`。转换经历 3 次被杀 +
   1 次整机断电，unit 级 done 标记断点续跑全部兜住（残留空目录 → rmtree 重建已修入
   `research_tasks/convert_p33_data.py`）。
2. **训练**：`research_tasks/train_p33_nominal.py --mode architecture`，冻结预算
   （有效 batch 256、30 epochs、早停 `source_macro_minade_at_6`/patience 5、seed 7、
   strict 确定性）。链：`post_convert_chain.sh`（G0 verify → catalog audit → CV 基线）。
3. **评估**：`research_tasks/evaluate_p33_t1.py --mode model --checkpoint-name
   arch_best_checkpoint.pt`，per-agent + joint-scene + 运动学双口径 + 分层 overlay 图。

## 2. 关键数字（dev，group-level）

| | macro minADE@6 | waymo | interaction |
|---|---|---|---|
| 模型（best ep29） | **1.3352** | 1.2504 | 1.4199 |
| CV 参考 | 3.6026 | 3.9764 | 3.2289 |

- 模型 macro 为 CV 参考的 37.1%；teacher token NLL 0.5757 / acc 80.4%
  （与训练验证一致）。GRU 基线本档未跑（留 scale 档）；**不声称 G1**。
- guardrail（rollout/teacher-argmax）收敛至 1.56/1.48——仅固定教师 token 观测隔离意义下成立。

## 3. 遗留问题（带入 scale 档）

1. **运动学有效性（负结果，根因已由 P3.3.4 Phase A 定位）**：boundary-excluded 后
   chunk 内 accel/jerk 违规仍 91.4%/100%（逐轨迹口径）——非边界伪影。五臂消融
   （`docs/p334_kinematics.md`）定位：**无约束残差头主导**（teacher-forced 下仅残差
   即把 GT 干净的 INTERACTION 拉到 accel 41%/jerk 98% 逐时间步违规）；码本量化
   次之（+2.4/+12.4 pp，集中 chunk 边界）；token 预测误差与自回归漂移 ≈0。
   off-road 不可识别，speed 干净 ≠ 可行域干净。边界跳变 mean 5.69 m/s（p99 12.36）。
2. ep28–29 仍小幅改善（+1.2%），30-epoch 预算可能略紧，增益边际，是否上调由 scale 档定。
3. 两次工程教训已记忆化：宿主并发其他算法会 OOM 杀 WSL 长任务（跑长任务时暂停其他任务）；
   断电/被杀靠 unit 标记 + epoch checkpoint 全量兜底。
