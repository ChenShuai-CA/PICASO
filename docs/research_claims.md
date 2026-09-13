# 研究主张与证据表（当前论文主线）

版本：v0.3（2026-09-13）

当前论文主线为：**条件化安全关键场景生成与预算化候选选择**。单目标和双目标均为正式实验分支。
学习系统根据 actor-visible 初始历史对冻结的可执行扰动原型排序，在固定候选 rollout 预算下提高
`complete && valid && dangerous` 场景覆盖率。

本版本取代 v0.1 中以共享 MAPPO actor 和 ABD 校准降低 sim-real 偏差为核心的主张。此前 PPO、
BC-MAPPO 和多解蒸馏的负结果继续保留并报告，但不再要求通过增加相同训练配方的步数挽救。

v0.3 新增 P3.3 大规模公开数据交互先验的任务规范和最小模型设计。P3.3 是尚未运行的新研究扩展，
不追溯修改 P2.12 的方法、阈值或一次性 heldout 结论。P3.3 采用地图感知的自回归多智能体
Transformer；此前 10,770 窗口上的小型单步动作回归模型只保留为诊断基线。

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
`T_AEB_proxy` 区间；request 与 ECU active 在无 CAN 时不可分辨，两个代理使用相同敏感性带。
这些量只能标记为专家/工程先验，不能写成历史 TXT 实测 ECU 时刻。

原因是测试没有连接车辆 CAN。全目录复核已经确认大量运行具有同步的 Object/Head tracker
reference/actual 通道，可用于目标执行误差分析，但仍需剔除固定参考点偏置和异常残差。经测试人员
复核，43 条无人工接管运行可作为人工确认锚点；另有 637 条通过 BR 踏板力学签名和残余风险规则
筛出的响应候选，合计形成 `aeb_dataset_v1=680`。其中 670 条在观测 onset 后 10 s 内停稳，按纵向
接触代理排除 contact 后的 612 条给出等效减速度敏感性包络 U(5.801,8.951) m/s²。637 条机器筛选
候选不能写成 ECU 确认 AEB，池化包络也不是车型均衡的概率分布。另有 188 条 FCW-only 声音上升沿，其中
33 条 BR-zero 制动均确认是报警后人工接管，因此不进入 AEB 响应或代理校准。
当前 `abd_supported_v1` 仅保留用于历史回放；新实验可使用
`abd_derived_v2_sensitivity`，论文统一称为“ABD-supported numerical sensitivity domain”。
仿真已将冻结控制器预瞄与 AEB 执行时延拆分，避免控制器随每次时延扰动同步改变触发点。

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

## 9. P3.3 大规模公开数据交互先验（设计已确定，结果待实验）

### 9.1 决策、目的与现状

P3.3 的主模型确定为**地图感知、自回归、多智能体 Transformer**。研究目的不是单独提高一步动作
回归精度，而是学习真实交通条件下的多模态交互分布，并将其用于安全关键候选生成、合理性约束和
固定预算候选选择。

P1 v5 语料仅含 10,770 个窗口（train 7,404），来源为 32 个 INTERACTION CSV 和 4 个 Waymo
TFRecord shard；Waymo 每个 shard 最多读取 32 条 Scenario。原 `Actor(hidden=64)` 只有 43,016 个
参数，输入为八帧历史和至多一个同类/异类邻居，不解码地图、交通信号、`objects_of_interest` 或
`tracks_to_predict`，监督目标是单步加速度/转向变化。该模型及 neighbor 变体未超过零动作基线，
因此只构成“原任务和模型不足”的负结果，不能作为公开数据先验有效的证据。

P3.3 不重复放大同一单步回归配方。公共数据、仿真搜索和 ABD 的监督角色保持分离：

- Waymo：地图感知的多智能体运动分布与真实度学习；
- INTERACTION：路口、环岛、合流和车辆—行人交互，以及跨地点泛化；
- P2.4–P2.12 CEM/候选 attempts：危险性、可行性和同条件候选排序监督；
- ABD：ego 制动执行与响应的数值敏感性评价，不监督 NPC 行为或真实车辆 AEB 决策。

在 P3.3 产生结果前，当前论文主结果仍是 P2.12 的冻结四原型与 ridge router。不得把 P3.3 的
设计、预计规模或未运行代码写成已经获得的证据。

### 9.2 任务定义

给定初始条件 $c$，模型生成未来多智能体轨迹 $\tau$：

- 约 1 s、10 Hz 的历史状态；
- ego、车辆、行人的类型、尺寸、速度、朝向、presence mask 和 visibility mask；
- 车道中心线、道路边界、路口、横道和可用交通信号状态；
- 场景分支（single/dual）、角色、lane-locked 等执行约束；
- 可选安全目标，包括目标 TTC 区间、near/contact 目标和风险强度。

模型输出：

- 未来 5 s、10 Hz 的联合轨迹分布；
- 离散运动原语的条件概率及连续轨迹；
- 候选真实度/似然分数；
- 独立的可行性、危险性和排序分数；
- 可供闭环执行的动作/轨迹及全部显式 mask。

概念目标为：

$$
\max_{\tau} R_{\mathrm{safety}}(\tau)
\quad \mathrm{s.t.}\quad
\log p_{\mathrm{real}}(\tau\mid c) \ge \gamma,
\quad g_{\mathrm{physical}}(\tau) \le 0.
$$

`p_real` 只由公开真实轨迹学习；`R_safety` 和可行性/排序头由训练域仿真 attempts 学习。安全引导
不得通过读取 heldout outcome、未来真值或隐藏目标状态实现。

P3.3 分为三个可单独证伪的任务：

1. **T1 真实交通建模**：学习地图条件下的多模态多智能体未来分布；
2. **T2 安全关键生成**：在真实度和物理约束内将名义分布引导到高风险尾部；
3. **T3 预算化选择**：从 $M$ 个冻结候选中选择 K=1/K=2，并在相同 rollout 预算下评价。

T1 失败时不得用 T2/T3 的仿真成功率宣称数据驱动生成有效；T2 失败时保留 T1 运动模型结果，但不
进入安全关键生成主张；T3 失败时只报告生成器结果，不宣称预算选择增益。

### 9.3 数据范围、划分与泄漏控制

数据管线必须改为 Scenario/record 级流式处理，按分片写入，训练按 chunk 读取；禁止将约 463 GiB
Waymo 原始数据或全部派生窗口同时载入内存。每条样本保留 source、文件、Scenario/case、track、
时间窗、内容哈希、split 和变换版本。重复导出、重叠时间窗和同一独立场景不得跨 split。

规模阶梯固定为：

| 阶段 | Waymo training shard | INTERACTION | 用途 |
|---|---:|---|---|
| smoke | 10 | 8 个开发地点各至多 2 个文件 | 只验证解析、mask、地图和闭环接口 |
| architecture | 100 | 338 个开发文件的确定性 25% 子集 | 选择模型和运动表示 |
| scale | 500 | 全部 338 个开发文件 | 规模曲线和超参数冻结 |
| full | 1,000 | 全部 338 个开发文件 | 多种子正式训练 |
| final confirmation | 不训练；读取冻结的 validation | 不训练；读取 35 个 location-heldout 文件 | 一次性独立确认 |

Waymo 1,000 个 training shard 内按 Scenario ID 的稳定哈希分 train/dev；不得把 official validation
当训练替代。此前已读取的 validation shard `00000`、`00001` 只可作为 development 数据，不进入新
final confirmation；其余 148 个 official validation shard 在模型、阈值、种子和评价代码冻结前
保持未读。

INTERACTION 的开发划分沿用已知地点：

- train：`DR_USA_Roundabout_EP`、`DR_USA_Intersection_EP1`、
  `DR_USA_Intersection_GL`、`DR_USA_Roundabout_FT`、
  `DR_USA_Intersection_MA`、`DR_USA_Intersection_EP0`；
- dev：`DR_USA_Roundabout_SR`、`DR_DEU_Roundabout_OF`；
- location-heldout：`DR_CHN_Merging_ZS`、`DR_CHN_Roundabout_LN`、
  `DR_DEU_Merging_MT`、`TC_BGR_Intersection_VA`。

location-heldout 目录在 final protocol 冻结前只允许清点文件名、大小和校验和，不读取轨迹内容。
Waymo 与 INTERACTION 使用共享场景表示，但保留 source embedding、分来源 batch 配额和分来源指标，
不能把两者混成一个无来源标签池，也不能只报告由 Waymo 数量主导的合并结果。

“使用全部公开数据”在本文中指全部文件被预先分配到 train、dev 或一次性 final confirmation，并在
相应阶段产生可审计结果；不表示把 validation/location-heldout 用于参数更新。INTERACTION 的 338 个
开发文件进入训练或开发评价，剩余 35 个文件只在 final confirmation 读取。

### 9.4 最小主模型设计（AR-Scene-v1）

第一版主模型固定为可在 RTX 4060 Ti 16 GB 上训练的中等规模模型：

| 组件 | 最小设计 |
|---|---|
| agent history | 1 s、10 Hz；局部坐标中的位置、速度、朝向、尺寸、类型与 mask |
| agent 数量 | 最多 16；按与 ego/目标的交互相关性确定，截断和缺失必须计数 |
| vector map | 最多 64 条 polyline，每条最多 20 点；保留 lane/crosswalk/boundary 类型 |
| motion token | 先在 train split 学习 128 类、每类约 0.5 s 的运动原语；禁止用 dev/final 学 codebook |
| encoder/decoder | `d_model=256`、6 层、8 heads、FFN=1024、dropout=0.1 |
| rollout | 自回归生成 10 个 token，解码成 5 s、10 Hz 连续轨迹 |
| source handling | source embedding + source-balanced mini-batch；分别报告两个来源 |
| capacity | 预计 10M–25M 参数；实现后记录精确参数量、显存峰值和吞吐量 |

地图编码器和 agent 编码器先分别编码，再通过 agent-agent、agent-map 注意力融合。解码器按时间自回归，
在每个 0.5 s 运动块后更新场景状态。训练使用 teacher forcing；dev 同时执行开放环预测和闭环滚动，
不得只报告 teacher-forced loss。

真实度模型可在完整候选生成后读取候选轨迹计算似然；在线生成器和预算选择器仍须遵守 actor-visible
边界。隐藏目标 token 必须置零并 mask，不能用真实历史填充。地图和公开数据中的完整场景信息不能
通过共享缓存泄漏到 router 输入。

### 9.5 训练顺序与现有系统接口

1. **数据与 token 阶段**：仅用 train split 学运动 codebook，生成流式分片和数据审计报告；
2. **名义预训练**：Waymo/INTERACTION source-balanced token cross-entropy，保存按来源和角色的 dev
   NLL、minADE、minFDE 与闭环运动学指标；
3. **安全引导阶段**：冻结或低学习率更新名义模型，用训练域 CEM attempts 学习风险、可行性和成对
   排序；以名义 log-likelihood 下限约束安全引导；
4. **候选生成阶段**：每个条件产生固定 $M$ 个候选，固化 candidate bank、模型哈希和随机种子；
5. **预算选择阶段**：在冻结 candidate bank 上比较 K=1/K=2；P2.12 ridge router 保留为已有基线；
6. **闭环与敏感性阶段**：`scenario_lab` 重算 complete/valid/dangerous 和角色约束，再使用
   `abd_derived_v2_sensitivity` 检查结论稳定性；CARLA 只在冻结后执行重放/跨仿真器验证。

最小版本不进行 PPO 或在线强化学习。只有当自回归模型通过 T1、但 T2 的安全引导仍不足时，才另行
预注册是否加入闭环策略优化；不能把 PPO 作为无条件的后续步骤。

### 9.6 基线、消融和预算公平性

T1 至少比较：

- constant-velocity / zero-action；
- P1 v5 43,016 参数 GRU（self 与 neighbor）；
- 连续输出 CVAE 或 mixture-density 基线；
- AR-Scene-v1 无地图；
- AR-Scene-v1 完整模型。

T2/T3 至少比较：

- script、固定原型、均匀随机候选和 CEM 搜索参考；
- 名义自回归采样（无安全引导）；
- 只有风险引导、没有真实度约束；
- 安全引导 + 真实度约束；
- 固定排序、P2.12 ridge router 和新选择器。

所有生成方法固定相同的初始条件、候选数 $M$、闭环 rollout 数 K、仿真步数上限、有效性规则和
bootstrap 单位。生成模型的前向/采样成本、闭环 rollout 成本和 CEM 搜索成本分账报告。

必须做数据消融：Waymo-only、INTERACTION-only、combined，并增加 matched-size 比较以区分“来源
互补”与“单纯样本更多”。single/dual、vehicle/pedestrian、来源和地点分别报告，不得只用合并均值。

### 9.7 开发门槛与一次性确认规则

在进入新 final confirmation 前同时满足：

1. **G0 工程门槛**：全部分片可断点续跑；内容哈希去重；同一独立场景零跨 split；无 NaN/Inf；
   地图/轨迹坐标变换可逆；数据和模型 manifest 完整；
2. **G1 真实度门槛**：AR-Scene-v1 在 Waymo dev 和 INTERACTION dev 上相对
   constant-velocity/P1 GRU 中的最强基线均改善 `minADE@6 >=5%`，且按独立 Scenario/地点聚类的
   配对 bootstrap 95% CI 下界大于 0；`minFDE@6` 点估计不得劣于最强基线，主要运动学违规率不得
   恶化超过 1 个百分点；
3. **G2 安全生成门槛**：在相同 $M$ 和同条件下，“安全引导 + 真实度约束”的有效危险覆盖率高于
   名义采样；95% CI 支持正向差异，真实度主指标相对名义采样下降不超过 5%，角色违规为 0；
4. **G3 预算门槛**：K=1 和 K=2 分别与等预算固定/随机选择比较；single、dual 均报告，其中任一
   分支失败即如实标为该分支失败，不以合并平均补救；
5. **G4 复现门槛**：正式 full 训练至少 3 个种子；报告均值、种子离散度、模型/数据哈希和完整命令。

G1 的 minADE、minFDE、token NLL、闭环碰撞/越界/加速度/jerk 指标必须并列报告，不能只报告主指标。
G2 中 5% 真实度容差是开发期预注册阈值；如需修改，必须在读取 location-heldout 和新
Waymo confirmation 前形成新版本并说明理由。

只有 G0–G4 和样本量最低要求全部通过，才允许执行一次新的 P3.3 final confirmation。P2.12
seed77000 已读结果不能用于 P3.3 的选模、阈值或停止决策；P3.3 使用新的条件种子、数据确认集和
完成标记，禁止按结果重训后复测同一 confirmation。

### 9.8 可视化、论文声明与停止规则

所有最终入库及 K=1/K=2 选中的场景必须生成逐帧二维动画和 CARLA 三维重放，叠加 actor ID、
visibility、TTC、最小间距、运动 token、候选排序和 ABD 扰动。CARLA 的第一阶段是确定性运动学
重放，用于证明视频对应冻结日志；后续物理闭环结果单列为跨仿真器验证，不能反向筛选 confirmation。

P3.3 完成前禁止声称：

- 已使用全部 Waymo/INTERACTION 完成训练；
- 自回归 Transformer 已优于基线；
- 真实数据提高了安全关键生成率或 sim-real 一致性；
- CARLA 重放等价于真实车辆验证；
- 达到 Waymo leaderboard、SOTA、NCAP 合规或生产实时性能。

若 100-shard architecture 阶段 G1 明确失败，先诊断地图使用、运动 token、开放环/闭环误差和来源
失衡，不直接进入 1,000-shard full。若完整模型在 full 数据上仍未通过 G1，则保留规模负结果，
停止用该模型支撑安全生成主张；不得通过读取 final confirmation 或增加相同配方训练轮数挽救。

### 9.9 P3.3 预期交付物

- 版本化的任务规范、数据 schema、split manifest 和数据卡；
- 流式 Waymo/INTERACTION 转换器及 10/100/500/1000 shard 审计报告；
- 运动 codebook、AR-Scene-v1 模型卡、训练曲线和三个种子 bundle；
- 按来源/地点/角色/分支统计的开放环与闭环评价；
- 冻结候选库、K=1/K=2 预算曲线、失败路线消融和成本分账；
- ABD-supported 数值敏感性结果；
- 二维动画、CARLA 三维视频及轨迹/事件一致性报告；
- 新 final confirmation 完成标记，以及可复现实验命令和哈希。

### 9.10 P3.3.1 当前证据状态（2026-09-13）

P3.3.1 smoke 数据管线已经完成 G0 工程验证：冻结的 10 个 Waymo training shard 和 16 个
INTERACTION 记录 case 被转换为 31,215 个 `scene-shard-v1` 样本，train/dev 为 26,799/4,416。
Waymo 有 4,982 个独立 Scenario；INTERACTION 有 16 个独立 `(location,case_id)` 组，其 26,233 个
重叠窗不能作为独立样本计数。heldout 轨迹内容未读。

最终 26 个 shard 的内容哈希、行数、sample ID、metadata、数组/mask 和 motion-token 契约均通过独立
复核；零独立组跨 split，坐标正反变换最大误差为 `4.56e-12 m`。train-only 运动码本的 128 个 token
在 3,371,044 个有效目标中全部出现。以上只支持“可流式、可恢复地形成训练输入”这一工程结论，不
支持 Transformer 已学到真实交通分布或优于基线。

当前 smoke dev 只有 481 个 Waymo Scenario 和 4 个 INTERACTION case，未达到 G1 的 5,000/100
最低独立样本量。另有 7,423 个样本发生 agent 截断、30,685 个样本发生地图截断；这两项必须在
P3.3.2/architecture 阶段分层评估。下一步是数据加载器、constant-velocity 基线和最小
AR-Scene-v1 的单 batch overfit/smoke training，而不是读取 final-confirmation 数据。
