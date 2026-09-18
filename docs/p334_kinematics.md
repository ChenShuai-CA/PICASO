# P3.3.4：运动学违规定位与修复（2026-09-15 起）

- 上游：P3.3.3 负结果（`docs/p333_architecture.md` §3.1）；预注册 SPEC：
  `runs/20260915_p334_kinematics/SPEC.md`（门槛先冻结后读数，heldout 封存）。
- 外部方案审查（GPT6）确定的执行顺序：A 定位消融 → B 冻结模型约束投影 →
  C 条件化运动学解码头（重训）→ 联合门槛评审 → 通过才进 scale。

## Phase A：五臂归因（已完成 2026-09-15）

完整数字与协议合规记录：`runs/20260915_p334_kinematics/PHASE_A_REPORT.md`。
结论（逐时间步口径，相对 GT 地板的增量）：

| 来源 | 码本量化 | token 预测误差 | 残差头 | 自回归漂移 |
|---|---|---|---|---|
| interaction/vehicle（GT 干净） | accel +2.4 / jerk +12.4 pp | ≈0 | **accel +40.0 / jerk +83.9 pp** | ≈0 |
| waymo/vehicle（GT 本身噪声大） | accel +2.9 / jerk −33 pp（幅度更差） | ≈0 | **accel +42.3 / jerk +78.8 pp** | ≈0 |

1. **残差头是主导违规源**：teacher-forced（token 近乎完美）下仅残差即把
   INTERACTION 从 0%/1.16% 拉到 41%/98%。
2. **码本量化是第二违规源**：GT 干净组注入 +2.4/+12.4 pp，违规集中 chunk 边界，
   越限幅度 p95 更大（299 vs 135 m/s³）；起点连续性 implied 超限 0%→10.8%。
3. token 预测误差与自回归漂移各 ≤1.7 pp，近乎无罪——token 主干不需动。
4. 结构性地板（A1，已冻结 SPEC §4）：Waymo GT 车辆 jerk 82.3% 轨迹越限
   （jerk 20 m/s³ 低于 Waymo 噪声地板）；INTERACTION GT accel 0%——两源
   的可行域不同，门槛分源冻结。

## Phase B：冻结模型约束投影（已完成 2026-09-15，四类门槛全过）

方法最终形态 = **可行热启动 + blend 回溯投影**（非罚函数平衡，三版罚项/Adam
设计均因 jerk 梯度链 ×10³ 失衡或 scale-transfer 病理被否，详见
`runs/20260915_p334_kinematics/PHASE_B_REPORT.md` §1）：位置 MA → 首速:=历史
末端速度 → 二阶受限滤波（jerk≤19 按构造）→ 纯偏差 GD 沿 warm→pred 回溯取最大
可行比例。

dev 全量结果（11,586 样本，ep29，同采样）：

| | macro minADE@6 | accel/jerk step（5 组） | 多样性 | solver_failure |
|---|---|---|---|---|
| orig | 1.3352 | 33–47% / 98–99% | 98.88 | — |
| **proj(B)** | **1.3280（−0.54%）** | **全部 0.00%** | 96.30（97.4%） | 0.012% |
| zero_res 兜底 | 1.4861（+11.3%，不过门槛） | 0.1–3.3% / 5.5–14.4% | 105.01 | — |

修复位移 mean 0.169 m / p95 0.520 m；17.9 ms/样本。**B 单独满足 SPEC §4 全部
四类门槛**；是否先训 C 再联合评审由用户定（C ≈ 一次重训成本）。

## Phase C：条件化解码头（已完成 2026-09-16，冻结 SPEC §4 口径四类门槛全过）

实施细则已预注册于 SPEC §6（训练启动前冻结）。落地形态（`scenario_lab/p33_model.py`
的 `ARSceneV1K` + `ar_scene_loss_c`，配置 `configs/p33/model_kinematic_c_v1.json`）：

- **结构**：token 主干不动；删除 `residual_head`（Phase A 主导违规源），换成
  state-carrying 头（`state_mlp`+`jerk_head`+`init_head`，输入 = decoder hidden +
  携带速度/加速度状态 + 历史末端速度），参数 13,126,282 → 约 13.26M（仍在冻结区间）。
- **积分**（半隐式，fp32，头/积分在 autocast 外）：`j_t = J·tanh(‖u‖)·u/‖u‖`（范数
  有界）→ `a_{t+1} = Π_A(a_t + h·j_t)`（圆盘投影）→ `v_{t+1} = v_t + h·a_{t+1}` →
  `P_{t+1} = P_t + h·v_{t+1}`；`v_0` = 历史末端速度（起点连续按构造）；起步对
  `‖a_1+a_2‖ ≤ 9.5` 联合限幅。分类型：vehicle J19/A9.5、pedestrian 5/3、
  cyclist 10/5、other 10/5。
- **按构造的保证**（对齐指标差分口径证明，单测覆盖）：implied accel ≤ A、
  implied jerk ≤ J（投影 1-Lipschitz）、起点跳变 implied ≤ 9.5 < 10；速度非构造
  界 → 损失罚项 + 如实报告。
- **损失**：冻结三项逐字保留 + 0.1·意图对齐（chunk 位移 vs 质心位移）+
  0.05·速度超限罚。纯平滑对照变体（`construction="penalty"`）已实现，条件执行。
- **训练**：与 P3.3.3 完全同预算（30 epochs/早停 5/seed 7），全新初始化，输出
  `runs/20260915_p334_kinematics/phase_c_train/`；先 30-updates GPU 探针。
- 单测 8 项（构造界/rollout=并行解码一致性/零头=CV 解析等/起步限幅/损失/罚项
  对照/参数量区间），全套 208 通过。

**结果（`runs/20260915_p334_kinematics/PHASE_C_REPORT.md`，同 B 协议 eval）**：
训练 30/30 epochs 跑满（9/15 12:04 启动，13:27 被宿主事件杀死，ep2 checkpoint
断点续训零损失，总 17.5 h），best ep30 macro **1.4015**。联合门槛（vs orig 1.3352）：
macro +4.97%（压线过）、minFDE waymo +3.07%/interaction −0.95%、joint 全部
≤+2.15%、**5 组 accel/jerk 逐时间步违规全部 0.0000%**（构造保证在自由 rollout
兑现）、多样性 90.4%、无有效候选 0、无 solver。**口径声明**：SPEC §4 冻结文本
不辖分源 minADE；若按 B 报告表格的更严口径，waymo minADE +8.21% 超一行
（误差代价集中在 GT 噪声大的 Waymo 源），interaction minADE +2.11% 过——
随联合评审裁决。**结论：B/C 双双满足冻结门槛 → 进 scale 成立，首选 C**（训练内
保证、零推理期开销），B 为零重训替代。penalty 对照变体（§6.5）条件已满足，
跑否由用户定（≈17.5 h）。heldout 全程未读。
