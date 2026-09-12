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
| C3：K=1 条件排序优于固定 K=1 和脚本 | independent screen 支持，P2.12 待授权 | P2.11 全新 seed76000 条件集上双分支通过预注册判据；single 差 0.044 [0.012, 0.078]，dual 差 0.037 [0.003, 0.073] |
| C4：K=2 相比 K=1 的覆盖增益值得额外交互成本 | 待预算曲线汇总 | 分别报告覆盖差、平均 steps 和总 episode 数；不预设哪一档更优 |
| C5：条件排序不是输入置乱或边际原型频率造成 | development 支持，heldout 待确认 | P2.9/P2.10 完整条件置换；P2.8 单次 shuffled 负控曾失败并如实保留 |
| C6：方法在既定数值敏感性域保持有效性和角色约束 | development 支持，heldout 待确认 | 不能把当前扰动称为 ABD 校准分布或实车泛化 |

只有 heldout 按冻结协议通过后，C2 或 C3 才能进入论文主结果。若 heldout 失败，development 结果仍可
作为方法开发证据，但论文必须把最终确认记为负结果。

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
- 数值敏感性范围的动机与量级检查。

现有数据不能支持：

- ECU AEB request/active 时刻或 request-to-response delay；
- AEB 校准概率分布；
- 车辆或车型级保留验证；
- “ABD 校准降低 sim-real 偏差”的主张；
- AEB 触发时刻 MAE、NCAP 合规或实车闭环部署结论。

原因是测试没有连接车辆 CAN，且缺少独立踏板/压力与同步目标平台 command/actual 通道。四条
14-BZ3X BR-zero smoke run 只表示未观察到人工接管运动学特征，不能升级为直接 AEB 触发证据。
当前 `abd_supported_v1` 数值仅保留为历史可复现的 sensitivity envelope，论文统一称为
“numerical sensitivity domain”。

未来若能补采数据，可另行研究 observed braking response 与目标平台执行误差；这不是当前论文提交的
前置条件。

## 7. 当前正式评价顺序

1. P2.11 已在全新 screen 上确认 K=1，single/dual 均通过；heldout 未读取。
2. K=1 已按预注册决策成为首选最终方法，冻结为 P2.11 router-1。
3. P2.12 已冻结最终方法、seed77000、样本量、主指标、无效判据和统计脚本；预检未生成或读取 heldout。
4. 经用户明确授权后，执行一次 P2.12 final confirmation。
5. heldout 结果无论正负均完整落盘，不允许换模型、换 K、换分母或删除失败条件。

旧 `heldout_seed41000` 属于较早 P2 设计，规模与 P2.11 的最低合格样本要求不匹配，继续封存且不进入
P2.12。最终确认使用预注册的新 seed77000；它只在授权后的单次执行中生成，不能按结果重采样。

## 8. 禁止扩写的结论

- 不把 K=2 覆盖率写成单次 rollout 性能。
- 不把条件原型选择器写成持续反馈 actor。
- 不把 dangerous scene generation 写成控制器缺陷检出率或 NCAP 通过率。
- 不把 nominal script-safe 条件写成物理上可避免碰撞的充分证明。
- 不把 development 结果写成 heldout 确认。
- 不声称显著优于 SOTA，除非另有同协议、同预算、同条件的正式比较。
- 不隐藏 PPO、BC-MAPPO、P2.8 原门槛等负结果。
