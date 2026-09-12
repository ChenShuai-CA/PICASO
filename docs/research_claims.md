# 研究主张与证据表（当前论文主线）

版本：v0.2（2026-09-12）

当前论文主线为：**条件化安全关键场景生成与预算化候选选择**。单目标和双目标均为正式实验分支。
学习系统根据 actor-visible 初始历史对冻结的可执行扰动原型排序，在固定候选 rollout 预算下提高
`complete && valid && dangerous` 场景覆盖率。

本版本取代 v0.1 中以共享 MAPPO actor 和 ABD 校准降低 sim-real 偏差为核心的主张。此前 PPO、
BC-MAPPO 和多解蒸馏的负结果继续保留并报告，但不再要求通过增加相同训练配方的步数挽救。

## 1. 当前研究问题

| ID | 研究问题 | 分支 |
|---|---|---|
| RQ1 | 在相同单候选预算下，条件路由能否比最强固定单原型和脚本获得更高的有效危险覆盖率？ | single、dual 分别回答 |
| RQ2 | 在相同双候选预算下，条件路由能否比最强固定双原型排序获得更高的有效危险覆盖率？ | single、dual 分别回答 |
| RQ3 | 候选预算从 K=1 增至 K=2 时，危险覆盖增益与额外交互成本如何变化？ | single、dual 分层报告 |
| RQ4 | 上述排序增益在数值敏感性扰动下是否稳定，同时保持角色约束与候选有效性？ | single、dual 分别回答 |

single 指 ego 与横穿行人；dual 指 ego、横穿行人与运动遮挡车。single 不是基线，不能只报告
dual 或用合并平均掩盖任一分支失败。

## 2. 方法定义与主张边界

最终候选方法由两部分组成：

1. P2.7 学到并冻结的 branch-specific 四原型库；
2. P2.8 训练并冻结的 ridge router，只读取最初五帧 actor-visible history、visibility mask 和
   presence mask，对四个原型排序。

K=1 表示只执行排序第一的一个原型；K=2 表示按排序最多执行两个原型，首次获得有效危险结果后停止。
router 是场景级条件选择器。它不是在整个 episode 中持续根据新观测更新动作的反应式 actor，因此论文
使用“条件化候选选择”或“单候选/双候选 rollout”，不使用“持续反馈策略”或“单次反应式闭环策略”。

主要指标为条件级有效危险覆盖率。必须同时报告：

- candidate valid rate；
- pedestrian/occluder 角色违规数；
- target-target collision 等无效原因；
- 到首次成功或候选预算耗尽的 decision steps；
- 每条件允许的候选 rollout 数。

## 3. 当前 Claims-to-Evidence 表

| Claim | 当前状态 | 证据与限制 |
|---|---|---|
| C1：统一环境、观测边界和候选接口可同时支持 single/dual | 工程确认 | 单元/集成测试与 P2 系列运行；不是主要科学贡献 |
| C2：K=2 条件排序优于固定 K=2 排序 | development 支持，heldout 待确认 | P2.10 新 screen 和条件式 fresh development 双分支通过；最终论文主张仍需一次性 heldout |
| C3：K=1 条件排序优于固定 K=1 和脚本 | heldout 确认 | P2.12 一次性 seed77000 heldout 双分支通过；single 差 0.041 [0.012, 0.071]，dual 差 0.029 [0.001, 0.058] |
| C4：K=2 相比 K=1 的覆盖增益值得额外交互成本 | 待预算曲线汇总 | 分别报告覆盖差、平均 steps 和总 episode 数；不预设哪一档更优 |
| C5：条件排序不是输入置乱或边际原型频率造成 | heldout 确认 | P2.12 相对条件置换均值差 single 0.114、dual 0.057，单侧 p 均为 0.0002；P2.8 原负结果仍保留 |
| C6：方法在既定数值敏感性域保持有效性和角色约束 | heldout 确认 | P2.12 candidate valid rate 为 1.000/0.942，角色违规均为 0；不能扩写为 ABD 校准或实车泛化 |

C3、C5 和 C6 已按冻结协议通过一次性 heldout，可进入论文主结果。C2 的 K=2 证据仍限于
development，只作为预算扩展结果报告，不能写成 heldout 确认。

## 4. 基线与预算公平性

| 基线 | 用途 | 公平比较 |
|---|---|---|
| fixed-1 | 训练域平均成功率最高的单个冻结原型 | K=1 的主要等预算基线 |
| fixed-2 | 训练域平均成功率最高的两个冻结原型 | K=2 的主要等预算基线 |
| script | 零动作/规则参考 | K=1 时为单 episode；K=2 时必须注明候选预算不同 |
| permuted router | 保留路由排序边际分布但打乱条件对应 | 检验输入与排序的真实对应关系 |
| uniform random candidates | P2.7 机制参考 | 只在相同 K 和相同筛选规则下比较 |
| CEM | 搜索上限和交互成本参考 | 不与固定 K 方法伪装成等成本比较 |

所有正式比较固定 sampler、physics、`lane_locked`、条件生成规则、扰动种子、候选有效性定义和
bootstrap 单位。条件只可按 nominal script 的 complete/valid/safe 状态筛选，不能依据候选 outcome
删除失败条件。

## 5. 观测与信息边界

- router 只能读取 actor-visible history 及显式 mask；隐藏目标保持为零，不得 truth-fill。
- centralized critic 的历史实验不构成当前 router 的推理输入。
- scenario ID、未来轨迹、候选 outcome、扰动 realization 和隐藏目标真值不得作为 router 特征。
- 每次正式实验必须记录输入模型哈希、条件指纹交集和 heldout 访问状态。

## 6. ABD 与实车数据的当前定位

ABD 不再承担当前论文的核心 RQ 或成功门槛。现有数据可以支持：

- 历史数据解析和通道审计；
- `observed_braking_onset`、等效减速度和停车响应的有限案例分析；
- 通过 `.spec` 中 CAN User Defined→Time Tolerance Trigger 映射恢复的
  `T_FCW_audio_observed`；历史 FCW-only 测试中的后续制动不用于 AEB 分析；
- 数值敏感性范围的动机与量级检查。

现有数据不能支持：

- 实测 ECU AEB request/active 时刻或由历史数据拟合的 request-to-response delay；
- AEB 校准概率分布；
- 车辆或车型级保留验证；
- “ABD 校准降低 sim-real 偏差”的主张；
- AEB 触发时刻 MAE、NCAP 合规或实车闭环部署结论。

测试团队可提供 0.15–0.35 s（名义 0.25 s）的 request-to-−0.3 m/s²响应工程先验，并据此报告
`T_AEB_proxy` 区间；该量只能标记为专家/工程先验，不能写成历史 TXT 实测 ECU 时刻。

原因是测试没有连接车辆 CAN，且缺少独立踏板/压力通道。全目录复核已经确认大量运行具有同步的
Object/Head tracker reference/actual 通道，可用于目标执行误差分析，但仍需剔除固定参考点偏置和
异常残差。经测试人员复核，当前 AEB 响应池为 44 条唯一运行：43 条无人工接管，1 条在 AEB 完成
首次刹停后才人工接管（分析窗截断在首次刹停）。这些运行可支持观测制动响应包络和工程先验
`T_AEB_proxy`，仍不能升级为直接 ECU AEB 触发证据。另有 188 条 FCW-only 声音上升沿，其中
33 条 BR-zero 制动均确认是报警后人工接管，因此不进入 AEB 响应或代理校准。
当前 `abd_supported_v1` 数值仅保留为历史可复现的 sensitivity envelope，论文统一称为
“numerical sensitivity domain”。

当前可继续扩展 observed braking response 与目标平台执行误差分析；车辆 ECU 时延的直接校准仍需
新增 CAN 或权威外部触发通道。这不是当前论文提交的前置条件。

## 7. 正式评价状态

1. P2.11 已在全新 screen 上确认 K=1，single/dual 均通过；heldout 未读取。
2. K=1 已按预注册决策成为首选最终方法，冻结为 P2.11 router-1。
3. P2.12 在授权后执行唯一一次 seed77000 final confirmation，single/dual 均通过全部冻结门槛。
4. 方法、K、分母和 gate 在 heldout 后均未改变；完成标记禁止重跑。
5. 论文主结果使用 P2.12 router-1，P2.10 router-2 仅作 development 预算扩展证据。

旧 `heldout_seed41000` 属于较早 P2 设计，规模与 P2.11 的最低合格样本要求不匹配，保持未读取且不进入
P2.12。最终确认使用预注册的新 seed77000，只生成并执行一次，没有按结果重采样。

## 8. 禁止扩写的结论

- 不把 K=2 覆盖率写成单次 rollout 性能。
- 不把条件原型选择器写成持续反馈 actor。
- 不把 dangerous scene generation 写成控制器缺陷检出率或 NCAP 通过率。
- 不把 nominal script-safe 条件写成物理上可避免碰撞的充分证明。
- 不把 development 结果写成 heldout 确认。
- 不声称显著优于 SOTA，除非另有同协议、同预算、同条件的正式比较。
- 不隐藏 PPO、BC-MAPPO、P2.8 原门槛等负结果。
