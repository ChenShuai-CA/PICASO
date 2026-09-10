# Stage 4：数据处理与实验方案 — PICASO

> 依据 `Stage3_方法与系统设计_PICASO.md` 的 5 层架构与 Layer 1 统一 Schema
> 基于 C-NCAP 2024 Appendix L (ADAS) + Appendix O (VRU) 规程深度阅读
> 基于 ABD 场景类型数据深度抽查 (CCRs, CPLA, CSFAO, CBNAO, SCP, SCPO, CCFT, BSD)
> FalseReaction 场景：生成/统一评估侧剔除（无统一 T0 定义）；作为"不应制动"负样本保留给 VUT surrogate 边界建模（2026-09-01 修订）
> 基于 ABD Robot Controller Software Manual (RM-S-01 Issue 23) + Path Following User Manual (RM-S-02 Issue 17) 完整阅读
> 编写日期：2026-05-28

---

## 2026-09-01 方案评审修订（执行优先级最高）

1. **ABD 数据第一定位**：真实车辆响应 ground truth，用于训练/校准 VUT 响应 surrogate 与实车验证闭环；不再作为"开放道路→封闭场地 UDA"的目标域。
2. **FalseReaction 分流**：生成/统一评估侧剔除（无统一 T0），但作为"不应制动"负样本进入 surrogate 负样本池（替代原"整体剔除"决策）。
3. **实验切分**：主实验 leave-one-brand-out（surrogate 跨品牌泛化）+ S9 工程验证支线不变；新增 **Stage D 实车补测集**（M4 起，含规程内复现点 + 规程外新点两组，协议见附录 F）。
4. **指标**：TDPD 统一定义为 `(CR_target − CR_source)/CR_source` 并降级为诊断指标；新增实车一致性指标族（碰撞判定一致率 / minTTC MAE / AEB 触发时刻 MAE / 实车验证命中率）为核心指标。
5. **基线**：新增必做对照 B6（纯 Waymo/INTERACTION 训练、无 ABD 校准）。

---

## 2026-05-28 专家审查修订：数据口径与实验切分锁定前置条件

本阶段方案总体可执行，且 ABD 手册、C-NCAP 规程和样例数据的结合较充分。但正式实验前必须先统一数据口径，避免“原始文件数、run 数、有效场景数、品牌数”混用。

### 数据口径修订

1. **原始池**：以文件系统审计为准，ABD 目录下当前可见 4 个品牌/车型来源（GAC S9、GAC E8、XPeng P7+、GAC A66）以及 C-NCAP 规程资料。原始 `.spec/.log/.CRUN/.txt` 文件数只能说明采集规模，不能直接等同有效 run。
2. **可用池**：必须由 `all_runs_inventory.csv` 输出，字段至少包括 `brand`, `scenario_type`, `standard_group`, `has_spec`, `txt_columns`, `t0_method`, `target_source`, `validity_label`, `exclude_reason`。
3. **实验池**：只使用通过 parser 验证、通道完整、T0 可定位、目标物来源明确、规程类型可确认的 run。论文中所有数据量均以实验池为准。
4. **A66 处理**：A66 企标功能项测试继续排除于主 CNCAP 实验；A66 CNCAP 目录可以作为独立外部验证，但其有效数量必须由 inventory 确认后再写入论文。
5. **不要提前写定有效 run 数**：现阶段所有 `~888 runs`、`~80 runs` 等均为规划估算，不得进入论文结果表。

### 实验切分修订

原方案把 GAC S9 作为唯一 DA Adapt、E8/P7+/A66 作为全量 holdout，逻辑清晰但有过拟合 S9 的风险。正式实验采用两层切分：

| 层级 | 切分方式 | 用途 |
|------|----------|------|
| 主实验 | leave-one-brand-out 或 leave-one-family-out（按实际有效场景覆盖选择） | 评估跨品牌泛化 |
| 工程验证 | S9 作为适配集，E8/P7+/A66 作为 holdout | 保留原方案的工业叙事 |
| 场景分层 | 每个 split 内按 CCRs/SCP/CCFT/VRU 等规程类型分层 | 避免某品牌缺失某类场景导致假泛化 |

若某品牌某规程有效样本不足，只能报告该规程的案例分析或单次验证，不能做显著性结论。

---

## 一、数据资产全景

### 1.1 三源数据总览

| 数据集 | 规模 | 场景类型 | 采集环境 | PICASO 角色 |
|--------|------|---------|---------|------------|
| **Waymo Open Motion** | 496.6 GB, 1150 文件 (training + validation) | 开放道路自然驾驶交互轨迹 | 美国城市开放道路, 多传感器融合 | Layer 3 主训练数据 (Source 0) |
| **INTERACTION** | ~135 MB (压缩), ~30 场景 × train/val/test 三切分 | 结构化交叉口/匝道/环岛多智能体交互 | 中国/德国/美国, 无人机+高精地图 | Layer 3 辅助训练数据 (Source 1) |
| **ABD CNCAP** | 原始池含 4 个品牌/车型来源；最终有效 run 数待 `all_runs_inventory.csv` 确认 | 封闭场地标准测试规程 (C2C + VRU) | 中国封闭测试场, ABD 驾驶机器人 | Layer 2 DA/DG 目标域 + 工业验证 |

> **排除说明**: GAC A66 企标功能项测试 (目录 12-ACC/13-ICA/14-TJA/15-ILC/16-RCW) 不属于 CNCAP 规程, 直接从数据池中移除。GAC A66 CNCAP 数据 (目录 2/3/8/9/11) 完整保留, 作为第四品牌独立验证数据源。

**四品牌数据来源**:

| 品牌 | 代号 | CNCAP 数据量 (估) | 角色 |
|------|------|------------------|------|
| GAC S9 | S9 | 待 inventory 确认 | 工程验证适配候选 (VRU+ADAS 较完整) |
| GAC E8 | E8 | 待 inventory 确认 | 跨品牌 holdout 候选 (主要 ADAS, 少量 VRU) |
| XPeng P7+ | P7+ | 待 inventory 确认 | 跨品牌 holdout 候选 (ADAS + 部分 VRU) |
| GAC A66 | A66 | 待 inventory 确认 | 独立验证候选 (CNCAP ADAS + VRU, 企标排除) |

### 1.2 多品牌 ABD 数据完整性矩阵（S9/E8/P7+ 已细化，A66 待 inventory 锁定后并表）

#### CNCAP VRU 2024 (行人/自行车/踏板车)

| 场景规程 | 缩写 | CNCAP 章节 | GAC S9 | GAC E8 | XPeng P7+ |
|---------|------|-----------|--------|--------|-----------|
| 近端成人纵向 | CPLA-25 | O.6.1.6 | 4r/4v | 7r/4v¹ | — |
| 近端儿童纵向遮挡 | CPNCO-25 | O.6.1.7 | 4r/3v | — | — |
| 远端成人纵向遮挡 | CPFAO-25 | O.6.1.8 | 9r/7v | — | 5r/3v |
| 近端成人转向 | CPTA-LN-50 | O.6.1.9 | 10r/8v | 8r/4v¹ | 18r/8v |
| 远端成人转向 | CPTA-LF/RF-50 | O.6.1.10/11 | 含于上 | — | — |
| 近端自行车纵向遮挡 | CBNAO-50 | O.6.1.12 | 4r/3v | — | — |
| 远端自行车纵向 | CBLA-25 | O.6.1.13 | — | 2r/2v¹ | — |
| 近端踏板车纵向遮挡 | CSFAO-50 | O.6.1.14 | 3r/3v | — | — |
| 远端踏板车转向 | CSTA-LN/RN-50 | O.6.1.15 | 8r/5v | — | 6r/5v |

> r = run 总数, v = 变体数 (速度条件 × 重叠率组合 × 日/夜间)。
> ¹ GAC E8 的 CPLA/CPTA/CBLA 数据在 Cal/企标目录下, 非标准 CNCAP 分组, 需人工核实规程匹配。

#### CNCAP ADAS 2024 (Car-to-Car)

| 场景规程 | 缩写 | CNCAP 章节 | GAC S9 | GAC E8 | XPeng P7+ |
|---------|------|-----------|--------|--------|-----------|
| 静止车尾 AEB/FCW | CCRs | L.6.1.5 | 8r/7v | 9r/7v | 10r/7v |
| 高速跟车 AEB/FCW | CCRH | L.6.1.6 | 4r/2v | 1r/1v | 2r/2v |
| 横向穿车 AEB/FCW | SCP | L.6.1.7 | 10r/6v | 1r/1v | 11r/5v |
| 对向穿车 FCW | SCPO | L.6.1.8 | 4r/2v | — | — |
| 穿车假目标 AEB | CCFT | L.6.1.9 | 9r/6v | 4r/2v | 3r/3v |
| 盲区监测 | BSD | L.6.5.4 | 12r/4v | — | — |

> **分流处理（2026-09-01 修订）**: FalseReaction 误作用场景 (L.6.2) 从生成/统一评估框架剔除（10 种子场景无统一 T0 定义）；但作为"系统不应制动"的负样本保留，供 VUT surrogate 边界建模使用。

#### 关键覆盖缺口

| 缺口 | 影响 | 缓解 |
|------|------|------|
| GAC E8 无 VRU 组团 | 无法在该品牌验证 VRU 场景的跨品牌泛化 | 主要用 S9 + P7+ VRU 做 VRU holdout |
| P7+ 无 CBNAO/CBLA/CSFAO | 缺少自行车/踏板车遮挡场景的跨品牌数据 | 仅在 S9 上评估该类场景 |
| E8/P7+ 无 SCPO | 对向穿车仅 S9 有数据 | SCPO 仅在 S9 上评估 |
| S9/E8/P7+ 均无完整 VRU+ADAS 全矩阵 | 跨品牌 holdout 需按场景类型分层 | 按 ScenarioType 分层切分；A66 待 inventory 后补充 |

---

## 二、ABD 数据格式深度规范

### 2.1 文件组成

每次测试 Run 由四个文件构成:

| 文件 | 格式 | 内容 | PICASO 用途 |
|------|------|------|------------|
| `V*_T*_R*.txt` | 制表符分隔文本 | 时序数据, 300-416 通道, 100Hz 采样 | 主数据源 |
| `V*_T*_R*.spec` | XML-like 文本 | 测试配置 (车辆参数, 坐标基准, 路径定义, 控制参数) | **必需** — 完整测试均含 .spec; 无 .spec = 不完整测试 |
| `V*_T*_R*.log` | 文本 | 事件时间线 (触发时刻, 阶段转换, 异常记录) | 异常检测辅助 |
| `V*_T*_R*.CRUN` | 二进制 | ABD Robot Controller 原生格式 | **不使用** (.txt 已含所有通道) |
| `CurrentTestSpec.txt` | 文本 | 导出时的测试配置摘要 | **不使用** (配置信息从 .spec 读取) |

> 依据: RC Software Manual §8.3 (Exporting results in ASCII format, p.143-144)

### 2.2 .spec 文件关键字段 (经实测验证)

.spec 文件在不同场景类型下结构有差异，但均包含以下关键配置段:

**依据**: RC Software Manual §6.1.4.1 (SR triggers, p.63-64), §6.1.4.5 (Time-tolerance parameters, p.69), §6.1.4.7 (Synchro, p.72-75)

#### A. 同步配置 (Synchro)

实测发现 **同步配置因测试类型而异**:

**CCRs AEB 测试 (GAC S9 V4_T27_R1, VUT 20 kph, CCRs -50% AEB)**:
```
UseSynchronizationMode=False          ← 不使用同步
Type=SR/AR Combination                ← 组合测试 (转向+油门机器人)
  [AR Speed Throttle Event]:
    StartTrigger=Time-tolerance trigger 1    ← 时间容差触发器1启动
    EndTrigger=Speed, EndTriggerValue=15.5   ← 速度触发结束
    TimeTolerance1Channel1=Time to point of interest
    TimeTolerance1MinTrigger1=0
    TimeTolerance1MaxTrigger1=3        ← POI时间在0-3秒之间触发
    TimeTolerance1TriggerTime=0        ← 最小触发时间0.5ms
    TimeTolerance1TriggerDelayTime=0   ← 无延迟
  StartTrigger=Closed Loop (PF Standard)
```

**BSD 测试 (GAC A66 V3_T396_R1, Lane Change Right, VUT 50 kph)**:
```
UseSynchronizationMode=False          ← 不使用同步
Type=PF Standard                       ← 路径跟随标准测试
  StartTrigger=Closed Loop
  SpeedControl=PathFile*100%
```

**ELK 测试 (GAC A66, 前次分析)**:
```
UseSynchronizationMode=True           ← 使用同步
SynchronizationMode=Subject vehicle   ← VUT 作为 Subject
SyncEndTriggerType=Time-tolerance trigger 4  ← 时间容差触发器4结束同步
StartTrigger=Standing start
```

> **结论**: `UseSynchronizationMode` 的值**取决于具体测试规程需求**, 不能一概而论。AEB CCRs (静止目标) 和 BSD (车道变更预警) 不需要同步; ELK (紧急车道保持, 涉及目标车切入) 需要同步。**必须在数据处理时从 .spec 文件读取同步模式, 作为 T0 定位策略的选择依据。**

#### B. 车辆参数 (VehicleParameters)

```
WheelBase=2.93           ← 轴距 (m), 用于运动学模型
Mass=1300                ← 质量 (kg) — 默认值, 实际未填写
SteerRatio=15.4          ← 转向比 — 默认值, 实际未填写
VehicleLength=5.06       ← 车长 (m) — 关键: 用于碰撞判定
VehicleWidth=1.95        ← 车宽 (m)
```

> **参数可用性评估 (用户确认)**:
> - **Mass (质量)**: **重要参数** — Layer 5 物理可行性验证中 KFR 需要计算摩擦圆约束 `F_fric ≤ μ × m × g`, 质量直接影响法向力 F_N。用户已提供实际测试质量 (整备质量 + 乘员 200kg + ABD设备/惯导 35kg, 非精确值, 工程可用):
>   - GAC E8: **2410 kg** (2175 + 200 + 35)
>   - GAC S9: **2600 kg** (2365 + 200 + 35)
>   - GAC A66: **2535 kg** (2300 + 200 + 35)
>   - XPeng P7+: **2395 kg** (2160 + 200 + 35)
> - **SteerRatio (转向比)**: **不重要** — PICASO 使用 GPS/IMU 实测轨迹 (x, y, v, a, yaw, yaw_rate), 不依赖转向输入。转向比仅在需要将方向盘转角转换为车轮转角时使用, 而 PICASO 直接从轨迹曲率 `κ = yaw_rate / v` 获取路径信息, 无需转向比。此参数不纳入处理流程。
> - **VehicleLength/VehicleWidth**: **重要** — 碰撞判定和边界约束需要车体尺寸。
> - **WheelBase**: **中等** — 运动学一致性验证, 直接使用 .spec 中的值即可。 

#### C. 坐标基准 (CoordinateData)

```
CurrentDatumLatitude     ← 测试场地基准纬度
CurrentDatumLongitude    ← 测试场地基准经度
CurrentDatumBearing      ← 场地坐标系方位角 (°)
```

### 2.3 .txt 文件结构

```
行 1: "Anthony Best Dynamics Ltd"                    ← 厂商标识
行 2: "Points=N"                                      ← 数据点数 (如 2204 点 = 22.04s @ 100Hz)
行 3: 通道名称行 (300-416 列, 制表符分隔)              ← 通道名
行 4: 单位行                                          ← 物理单位
行 5+: 数据行 (每行一个时间步, 制表符分隔, 100Hz)       ← 时序数据
```

> 依据: RC Software Manual §8.3 (p.143-144): "The ASCII file consists of a header... The channel titles and units are tab-separated. Following the header is the data with one row of data per measurement point (normally spaced at 20 ms intervals)"

#### 通道数列数分析 (经 22 个 CCFT 文件 + 多场景类型实测)

| 通道数 | 出现场景 | 原因 | 处理方式 |
|--------|---------|------|---------|
| **415-416** | CCRs, CPLA, SCP, SCPO, BSD, CCFT 完整 Run, ELK, LDW | 全通道导出 (Object 1 + Head tracker + 相对运动 + 触发通道) | FULL 模式 — 正常处理 |
| **< 400** | CCFT R1 及其他场景的首次尝试/中止 Run | 测试不完整, 通道缺失 | **直接排除** (按用户决策: 不做运动学反推) |
| **381-395** | GAC E8 部分场景 | 品牌/配置特定, 但仍含 Object 1 通道 | 按列数匹配处理, 检查关键通道存在性 |

> **修正**: CCFT 并非"仅有 384 通道"。22 个 CCFT .txt 文件分析结果: 14 个完整 Run (415 列, 含 Object 1 直接测量), 4 个 R1 文件 (<400 列, 直接排除)。用户确认: "所有动态场景的目标物姿态数据都不需要反推, 软件都会记录各自姿态数据。不完整 Run 直接剔除。"

### 2.4 ABD Synchro 同步机制 (基于 RC Manual 原文)

> **用户需求**: "Synchro trigger 是啥, 我具体不清楚, 请你给出说明, 告诉我在哪份说明文档里面"

#### 2.4.1 什么是 Synchro (同步)

**依据**: RC Software Manual §6.1.4.7 Synchro (p.72-75)

Synchro 是 ABD 的多车同步机制, 用于在测试中协调 Subject (主车/VUT) 和 Tracker (从车/目标物) 之间的运动关系。有四种工作模式:

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Synchro start/abort only** | 仅同步启动和停止, 不持续调整 (p.72) | 所有车辆均被机器人精确控制 |
| **POI Synchro start** | 延迟到最后一刻才启动 Tracker, 使各车同时到达各自的 Point of Interest (p.73) | Subject 非机器人控制 (驾驶员或车辆自身) |
| **Full Synchro** | 全几何同步 — Tracker 实时调整速度和位置以补偿 Subject 的误差 (p.74) | **碰撞测试、近距通过测试** (Subject 为人类驾驶员或自身系统控制) |
| **POI Full Synchro** | 同 Full Synchro, 但时间同步到 Point of Interest (p.74) | 需要路径上特定点精确同步 |

#### 2.4.2 Full Synchro 工作原理

> **RC Manual p.74**: "It can be imagined as the vehicles being 'geared' together. It does this by adjusting the timing and speeds of the Trackers to match the timing error and speed error of the Subject vehicle... So, if the Subject vehicle is 10% overspeed, the Trackers will move at 10% above their programmed speed, and vice versa."

Full Synchro 将 Subject 和 Tracker 像"齿轮啮合"一样耦合在一起:
- Subject (VUT) 先启动, 沿预设路径行驶
- Tracker (目标物) 持续接收 Subject 的速度/位置误差, 实时等比调整自身速度
- Subject 快 10% → Tracker 也快 10%, Subject 慢 10% → Tracker 也慢 10%
- **目的**: 确保即使 Subject 的行驶不精确, 两者之间的几何关系也能精确维持

#### 2.4.3 "条件触发退出同步" 机制

> **RC Manual p.74**: "If Use trigger to end synchronization mode is checked, then the vehicles set to Tracker mode will stop being synchronized after the trigger occurs... When a trigger occurs on the Subject vehicle in a PF Standard test, then a signal is sent to the Tracker vehicles."

这是用户描述的测试流程的技术实现:
1. VUT (Subject) 先出发, 机器人控制沿预设路径行驶
2. Tracker (目标物) 处于 Full Synchro 模式, 与 VUT 保持几何同步
3. 当 VUT 满足触发条件 (如到达特定速度/位置/TTC 阈值), **触发器在 Subject 上触发**
4. 触发信号通过 Ethernet 广播给 Tracker
5. Tracker 收到信号后**退出同步模式, 开始执行独立路径** (碰撞路径)
6. 如果没有碰撞, 目标物完成整个路径; 如果碰撞, 在碰撞点停止

可在 Subject 上使用的触发类型 (RC Manual p.74-75):
- Start button, Input ADC, Digital input, Ethernet from subject
- **Speed** — 速度触发
- **Time-tolerance triggers** — 时间容差触发 (最常用)
- **X position / Y position** — 位置触发
- **Distance travelled** — 行驶距离触发
- CAN user defined, **Advanced Channel**

#### 2.4.4 Synchro Trigger 信号详解

**这里有容易混淆的两个概念:**

**概念 A: "Synchro object X trigger" (同步对象触发器)**

> **依据**: RC Manual §6.1.4.1 (p.64): "The pre-defined digitals currently available for advanced triggering are: ... Synchro object 0-15 trigger"
> **依据**: RC Manual Triggers 1 CAN message (p.193): Bits 48-63 = "Synchro object 0-15 trigger" (1-bit digital signals)

这些是 **Triggers 1 CAN 消息** (p.193-194) 中预定义的 16 个 1-bit 数字信号 (bits 48-63), 属于 **Advanced Channel 触发** 的预定义数字选项。它们是:
- 用于**启动测试**的触发源 (不是用于结束同步)
- 在 Advanced Channel 触发设置中可选择 `Synchro object X` 作为触发通道
- **每个 bit 代表一个独立对象 (Object 0-15) 的触发状态**

> **RC Manual p.75**: "Broadcast synchro trigger signal allows any robot in an advanced multi-object system to be triggered by a signal from another robot in the system... Any robot that needs to use this trigger can have a trigger type of Advanced Channel, choosing the General / Synchro object X."

**概念 B: "Use trigger to end synchronization" (用于结束同步的触发器)**

> **依据**: RC Manual §6.1.4.7 (p.74)

这是 Synchro 配置中的一个**复选框选项**, 允许使用任何标准触发器 (Speed, Time-tolerance, X/Y position 等) 来**结束同步模式**。当 Subject 上的触发器条件满足时:
1. 触发信号通过 Ethernet 发送给 Tracker
2. Tracker 的同步模式结束
3. Tracker 开始执行其独立编程路径

**实测数据中的体现**:
- 在 CCRs AEB 测试 (V4_T27_R1) 中, 虽然 `UseSynchronizationMode=False`, 但 Time-tolerance trigger 1 **确实在 T=14.120s 触发, 且恰好与 TTC=3.0s 同时**
- Synchro object X trigger 通道在所有测试中均保持为 0 — 因为:
  - 这些是"启动测试"用的触发源, 而非"结束同步"信号
  - 如果测试未使用 Advanced Channel 中的 Synchro object X 作为启动触发, 则这些 bit 始终为 0
  - 即使使用了 Full Synchro, **结束同步的信号是通过 Ethernet 广播的, 不经过 Synchro object X trigger bit**

#### 2.4.5 用户关切: 目标物被触发时的通道值

> **用户问题**: "我看了导出的数据文件, 似乎没有目标物被触发时的通道值"

**答案**: 目标物被触发的时刻**有多个通道可以检测**:

| 检测方法 | 通道 | 可靠性 | 依据 |
|---------|------|--------|------|
| **Time-tolerance trigger 触发** | `Time tolerance X` / `Time tolerance X (excl. delay)` / `Time tolerance X (within tolerances)` | **最高** — CCRs 实测 T=14.120s 精确对应 TTC=3.0s | RC Manual p.69 (§6.1.4.5) + 实测数据 |
| **Path Phase 状态转换** | `Path phase` | 高 — 对 SPT 测试显示同步状态 (-5/-25 → 1) | Path Following Manual §8 (p.77-78, Table 8.1) |
| **目标物开始运动** | `Object 1 forward velocity` / `Head tracker forward velocity` | 高 — 目标从静止开始移动 | 实测数据 |
| **TTC 反算** | `Relative longitudinal distance` / `Relative longitudinal velocity` | 中 — 依赖通道数据质量 | C-NCAP 2024 L.1.54 |
| Synchro subject/object X trigger | `Synchro subject trigger` / `Synchro object X trigger` | **低** — 用于"启动"触发而非"结束同步", 且为短暂脉冲 | RC Manual p.193 |

> **推荐 T0 定位策略**: **优先使用 Time-tolerance trigger 通道** (从 .spec 文件中确定触发同步结束的 Trigger 编号), **辅以 Path Phase 状态转换 + 目标运动检测 + TTC 反算**进行交叉验证。

### 2.5 Object 1 / Head Tracker 通道动态行为 (实测发现)

实测 CPLA 场景的 Object 1 数据变化:
```
时间窗口           Object 1 X 范围     Object 1 前向速度     说明
─────────────────────────────────────────────────────────────────
行 0-500 (~0-5s)   -10.544 (常数)      0 m/s                目标静止等待
行 1467-1967       -10.544 → -5.076    0 → 1.48 m/s         目标开始加速
行 2201-2701       -1.775 → 5.033      1.355-1.390 m/s      = 5 km/h 稳态 (符合 CNCAP)
行 2435-2935       1.487 → 6.484       0.765-1.390 m/s      目标持续移动
```

> **关键推论**:
> 1. **Object 1 和 Head tracker 数据完全一致** (相同数值, 相同列模式) — 两者反映同一目标物, 仅在不同测试模式下命名不同。
> 2. 目标物在 T0 附近开始按规程移动, 早期的 ~5s 是测试准备期
> 3. 场景数据应提取**目标运动窗口**而非全 Run
> 4. **数据源优先级 (用户决策)**: C2C 场景 → **Object 1** (语义更准确, GVT 机器人直接测量); VRU 场景 → **Head tracker** (C-NCAP 要求 VRU 速度精度 ±0.01 km/h)

### 2.6 C-NCAP 2024 规程关键参数

#### A. T0 时刻定义

| 规程 | T0 定义 | 来源 |
|------|--------|------|
| AEB (C2C) | TTC = 3s 时刻 | L.1.54 |
| AEB (VRU) | TTC = 3s 时刻 | O.1.40 |
| BSD 超车 | VUT 与 GVT 纵向距离 = 33m | L.6.5.4.2 |
| BSD 变道 | GVT 开始变道时刻 | L.6.5.4.3 |
| FCW | TTC = 最晚警告点 (由制造商申报) | L.1.56 |

> **分流处理（2026-09-01 修订）**: FalseReaction 场景从生成/统一评估侧剔除（10 种子场景无统一 T0 定义）；作为"不应制动"负样本保留给 surrogate 边界建模。

#### B. ABD 数据中的 T0 定位 (多策略分级)

**策略 1 (最优) — Time-tolerance trigger 通道直接检测:**

> **依据**: RC Manual §6.1.4.5 Time-tolerance parameters (p.69)
> **依据**: RC Manual Triggers 2 CAN message (p.194-195): 16 个 Time-tolerance trigger, 每种有 3 个变体通道 (trigger / excl. delay / within tolerances)

```
方法:
  1. 读取 .spec 文件, 确定测试配置的触发方式:
     - 如果 UseSynchronizationMode=True → 查找 SyncEndTriggerType (如 "Time-tolerance trigger 4")
     - 如果 UseSynchronizationMode=False → 查找 StartTrigger (如 "Time-tolerance trigger 1")
  2. 在 .txt 文件中定位对应的 "Time tolerance X" 通道
  3. 找到该通道值 0→1 的第一帧 → T0
  4. 交叉验证: TTC 在该时刻应约为 3.0s (AEB 场景)
```

> **实测验证 — CCRs AEB V4_T27_R1 (VUT 20 kph, CCRs -50% AEB):**
> - .spec: `StartTrigger=Time-tolerance trigger 1`, `TimeTolerance1Channel1=Time to point of interest`, `Min=0 Max=3`
> - **Time tolerance 1: 0→1 at T=14.120s**
> - **TTC (longitudinal): 精确在 T=14.120s 时为 3.000s**
> - 触发持续 7.42s (T=14.120s → 21.540s), 在此期间 POI time 保持在 0-3s 范围内
> - **结论: Time-tolerance trigger 通道是可靠的 T0 标记, 在 100Hz 采样下完全可捕获**

**策略 2 (备选) — Path Phase 状态转换 + 目标运动检测:**

> **依据**: Path Following Manual §8 (p.77-78, Table 8.1): Path Phase 完整状态定义

```
Path Phase 关键状态 (Table 8.1):
  -99 = system inactive
  -5  = Synchronized (fully) [SPT test only]
  -25 = Synchronized (longitudinal) [SPT test only]
  -24 = Synchronized (lateral) [SPT test only]
  -6  = Finished synchronization [SPT test only]
  0   = Lead-in to path
  1   = Unsynchronized [SPT] / On normal path [PF Standard]

对 SPT (VRU) 测试: 检测 Path Phase 从 -5/-25 转换到 1 的时刻 = 同步结束, 即目标触发时刻
对 PF Standard (C2C) 测试: 检测 Path Phase 从 0 转换到 1 的时刻 = 测试路径开始
```

> **实测验证 — CCRs AEB (PF Standard):** Path phase: 0→1 at T=7.140s (测试开始), 此后 TTC 逐步降低, 直到 T=14.120s 时 TTC=3.0s
> **实测验证 — BSD (PF Standard):** Path phase: 0→1 at T=7.290s, Test phase: 0→396 at T=7.290s

**策略 3 (兜底) — 运动学 TTC 反算:**

```
TTC_est = Relative_longitudinal_distance / max(|Relative_longitudinal_velocity|, 0.1)
T0 = argmin(|TTC_est - 3.0|)
```

**策略 4 (CCFT 专用) — 相对运动起始:**

> 对于 CCFT (完整 Run 有 415 列, 含 Object 1 直接测量), 目标运动起始可直接检测。
> 对于不完整 Run (384 列, 无 Object 1), 使用 Relative velocity 变化检测。

#### C. CCFT 场景特殊性

> **依据**: C-NCAP 2024 L.6.1.9 (Crossing Car False Target)

- VUT 以左转/直行路径行驶, GVT 为虚假目标 (布艺车辆模型)
- 完整 CCFT Run (14/22) 包含全部 415 列通道, 含 Object 1 直接测量
- 不完整 Run (4/22, 384 列) 为 R1 首次尝试 (可能因碰撞/中止/设置错误未完成)
- **处理策略**: **直接排除所有不完整 Run** (通道数 < 400 列), 不做运动学反推。按用户决策: "不完整 Run 的 CCFT 或者其他场景不需要运动学反推, 直接将这些不完整 Run 的场景全部剔除"
- 完整 Run 使用 FULL 模式 (Object 1 直接测量)
- 速度组合: VUT 10/20/30 km/h, GVT 20/40/50 km/h

#### D. 测试有效性规则 (来自 C-NCAP 2024 L.1.57-1.60, O.1.44-1.47)

| 条件 | 测试次数 | 说明 |
|------|---------|------|
| 制造商未申报预测结果 | **每测试点 1 次** | 单次执行, 无重复 |
| 制造商申报了预测结果 | 最多 **3 次重复** | 复杂有效性判定流程 |
| 硬件故障/环境超限 | 允许排除重做 | 不计入测试次数 |

> **→ 数据处理启示**: 同一变体下 runs 数量波动 (1~3) 是正常的。若某变体只有 1 个 run, 可能因为车辆 AEB 功能正常 (一次通过) 或功能异常 (提前终止)。

#### E. 数据精度要求 (来自 C-NCAP 2024 附录 A/B)

| 参数 | 精度要求 | ABD 对应通道 |
|------|---------|------------|
| VUT 速度 | ±0.1 km/h | `Forward velocity` |
| VUT 位置 | ±0.03 m | `X position`, `Y position` |
| VUT 偏航角速度 | ±0.1 °/s | `Yaw velocity` |
| VRU 目标速度 (PT/Bicycle/Scooter) | ±0.01 km/h (比 VUT 严一个数量级!) | `Head tracker forward velocity (ref point)` |

### 2.7 PICASO 关键通道映射

#### A. VUT 位姿 — Motion Pack (直接 GPS+IMU 测量)

> **依据**: RC Manual §8.1.4 (p.95-96), 绿色通道列表 (p.146-148)

| ABD 通道 | 物理量 | Layer 1 Schema 映射 |
|------|--------|-------------------|
| `X position` | VUT 全局 X 坐标 (m) | `Agent.history.x_t` |
| `Y position` | VUT 全局 Y 坐标 (m) | `Agent.history.y_t` |
| `Z position` | VUT 海拔 (m) | Layer 5 物理验证 |
| `Forward velocity` | 前向速度 (m/s) | `Agent.history.v_t` |
| `Lateral velocity` | 侧向速度 (m/s) | `Agent.history.v_lat` |
| `Forward acceleration` | 前向加速度 (m/s²) | `Agent.history.a_t` |
| `Lateral acceleration` | 侧向加速度 (m/s²) | 摩擦圆约束验证 |
| `Vertical acceleration` | 垂向加速度 (m/s²) | 路面质量参考 |
| `Yaw angle` | 偏航角 (°) | `Agent.history.θ_t` |
| `Yaw velocity` | 偏航角速度 (°/s) | `Agent.history.θ̇_t` |
| `Yaw acceleration` | 偏航角加速度 (°/s²) | (可选) |
| `Roll angle` / `Pitch angle` | 姿态角 (°) | 物理可行性参考 |
| `Slip angle` | 侧偏角 (°) | 车辆稳定性状态 |
| `Tracking angle` | 轨迹角 (°) | 路径跟踪方向 |
| `Longitude` / `Latitude` | GNSS 经纬度 | 全局参考系对齐 |
| `Bearing (heading from North)` | GNSS 方位角 (°) | 全局朝向 |
| `Satellites` | GNSS 卫星数 | 定位质量标记 |

#### B. 目标物 — Object 1 系列 (GVT 机器人直接 GPS+IMU 测量)

> **依据**: RC Manual §8.1.4 导出通道列表 (p.145-146), "Other" channels

**在 C2C 场景和完整 CCFT Run (415 列) 均可用**:

| ABD 通道 | 物理量 | 说明 |
|------|--------|------|
| `Object 1 actual X (front axle)` | 目标全局 X (m) | 前轴中点实际位置 |
| `Object 1 actual Y (front axle)` | 目标全局 Y (m) | 前轴中点实际位置 |
| `Object 1 forward velocity (ref point)` | 目标前向速度 (m/s) | 参考点处速度 |
| `Object 1 forward acceleration` | 目标前向加速度 (m/s²) | |
| `Object 1 yaw` | 目标偏航角 (°) | |
| `Object 1 reference X/Y position` | 目标参考轨迹基准点 | |
| `Object 1 time to collision (longitudinal)` | 目标物视角 TTC (s) | 与 VUT TTC 交叉验证 |
| `Object 1 incoming data extrapolation time` | 数据外推延迟 (ms) | 无线通信延迟标记 |

#### C. VRU 假人头部 — Head Tracker 系列 (SPT 直接测量)

> **依据**: RC Manual §8.1.4 (p.95-96), Path Following Manual §8 (p.77-78): SPT channels

| ABD 通道 | 物理量 |
|------|--------|
| `Head tracker actual X (front axle)` | 头部全局 X (m) |
| `Head tracker actual Y (front axle)` | 头部全局 Y (m) |
| `Head tracker forward velocity (ref point)` | 头部前向速度 (m/s) — C-NCAP 精度 ±0.01 km/h |
| `Head tracker forward acceleration` | 头部前向加速度 (m/s²) |
| `Head tracker lateral acceleration` | 头部侧向加速度 (m/s²) |
| `Head tracker lateral velocity` | 头部侧向速度 (m/s) |
| `Head tracker yaw` | 头部偏航角 (°) |
| `Head tracker yaw velocity` | 头部偏航角速度 (°/s) |
| `Head tracker pitch` | 头部俯仰角 (°) |
| `Head tracker reference X/Y position` | 头部参考轨迹基准点 |
| `Head tracker lateral error` | 头部侧向跟踪误差 (m) |
| `Tracker status` | 跟踪器状态 (**实测 = 9**: SPT 惯导状态良好, 但 SPT 仅使用惯导的 GPS 授时功能) |
| `Tracker time error` | 同步时间误差 (s) |

#### D. VUT 控制响应 — 机器人状态

| ABD 通道 | 物理量 | 用途 |
|------|--------|------|
| `Steering Angle Feedback` | 方向盘转角 (°) | 转向输入 |
| `Speed` | VUT 实际速度 (km/h) | 纵向控制响应 |
| `BR Position` | 制动踏板位置 (mm) | 制动强度 |
| `AR Position` | 油门踏板位置 (%) | 油门强度 |
| `Engine RPM` | 发动机转速 | 动力系统状态 |
| `Brake force (unfiltered)` | 制动管路压力 (N) | 制动强度冗余 |
| `SR Torque` | 转向扭矩 (Nm) | 转向行为分析 |

#### E. 相对运动量 — 校验与 CCFT 回退

> **依据**: RC Manual §8.1.4 Range channels (p.95-96), Range 1-8 CAN messages (p.192-193)

在 Object 1 数据可用时, 这些通道用于交叉校验。**在 CCFT 不完整 Run (384 列) 时, 是目标物运动信息的唯一来源**:

| ABD 通道 | 说明 |
|------|------|
| `Relative longitudinal distance` | 相对纵向距离 (m) |
| `Relative lateral distance` | 相对横向距离 (m) |
| `Relative resultant distance` | 相对合距离 (m) |
| `Relative lateral velocity` | 相对侧向速度 (m/s) |
| `Relative longitudinal velocity` | 相对纵向速度 (m/s) |
| `Relative yaw` | 相对偏航角 (°) |
| `Relative velocity` | 相对合速度 (m/s) |
| `Time to collision (longitudinal)` | VUT 视角 TTC (s) — **用于 T0 交叉验证** |

> **已移除**: CCFT 目标物运动反推。按用户决策, 不完整 Run 直接剔除, 完整 Run 使用 Object 1 直接测量。相对运动通道仅用于 TTC 计算和交叉校验。

#### F. 测试状态与同步旗标 — Run 级质量判定

> **依据**: RC Manual §8.1.4 (p.95-96), Triggers 1 CAN message (p.193-194), Triggers 2 CAN message (p.194-195)
> **依据**: Path Following Manual §8 (p.77-78, Table 8.1): Path Phase / Test Phase

| 通道 | 类型 | 判定逻辑 | 用途 |
|------|------|---------|------|
| `Path phase` | 状态机 (-99~4) | 见 Path Following Manual Table 8.1 | **主状态检测** (测试进度/同步状态) |
| `Test phase` | 阶段标识 | 从准备→测试→结束完整流转 | 测试阶段判定 |
| `Time tolerance 1-16` | 数字 0/1 | 0→1 = 触发条件满足 | **T0 主要定位 (推荐)** |
| `Time tolerance X (excl. delay)` | 数字 0/1 | 排除延迟的触发信号 | T0 辅助验证 |
| `Time tolerance X (within tolerances)` | 数字 0/1 | 条件在容差范围内的持续信号 | 触发窗口验证 |
| `Sync end` | 数字 0/1 | 1 = 同步结束 | 同步结束确认 |
| `Subject end sync flag` | 数字 0/1 | 1 = Subject 结束同步 | 仅 Synchro=True 时有效 |
| `SR path abort` | 数字 0/1 | 1 = 机器人主动中止 | **关键异常: 碰撞/安全风险** |
| `Abort path speed` | 数字 0/1 | 1 = 紧急中止 | 驾驶员或机器人紧急干预 |
| `Stop point go` | 数字 0/1 | 1 = 手动触发停止 | 人为干预中止 |
| `Critical section start/end` | 数字 0/1 | 标记关键区间 | PARTIAL 判定 |
| `SR/BR test` | 数字 0/1 | 1 = 在测试模式 | — |
| `Motion Going SR/BR/AR` | 数字 0/1 | 1 = 该轴运行中 | 全 0 >2s → 机器人停摆 |
| `Gyro status` | 状态值 | 非正常值 | IMU 数据不可靠 |
| `Subject status` | 状态值 | 0 = VUT 数据不可用 | 持续 0 >1s → 数据无效 |
| `Tracker status` | 状态值 | **实测 = 9 (恒定)** | **≠ 0 目标可用, 具体含义 ABD 文档未找到** |
| `Subject time error` | 连续值 (s) | Subject 路径时间误差 | Synchro 质量评估 |
| `Tracker time error` | 连续值 (s) | Tracker 路径时间误差 | Synchro 质量评估 |
| `Subject velocity ratio` | 连续值 | Subject 速度比率 | Synchro 耦合强度 |
| `Synchro subject trigger` | 数字 0/1 | Subject 同步触发脉冲 | **不建议作为 T0**: 用于启动测试 (Advanced Channel), 非结束同步信号 |
| `Synchro object 1-15 trigger` | 数字 0/1 | Object X 同步触发脉冲 | **不建议作为 T0**: 同上原因 |
| `Satellites` | 连续值 | < 4 >2s = GNSS 失锁 | 定位质量标记 |

#### G. 路径跟随通道 — 仅用于异常检测, 不作为 PICASO 特征

> **依据**: Path Following Manual §8 (p.77-78, Table 8.1): Path following data capture fields

**原理**: 路径跟随通道反映 ABD 机器人对预设路径的跟踪偏差, 而非 VUT 自身的运动学/动力学特性。

| 通道 | 质量判定用途 |
|------|------------|
| `Path following error` | 侧向偏差 > 0.5m → 机器人未沿预设路径 |
| `Controller error` | 过大 → 车辆动态超出机器人可执行边界 |
| `Distance error` | 纵向距离误差 > 2m → 速度控制异常 |
| `Path phase` | 负值 → 控制器故障 (已在阶段 B 旗标检测中处理) |
| `Actual X/Y (front axle)` | 与 Motion Pack X/Y 完全冗余, 取 Motion Pack |
| `Desired X (lead axle)` / `Desired Y (lead axle)` | 保留用于**车道中心线几何重建** → `Lane.polyline` |

---

## 三、数据质量筛选工作流

### 3.1 总体流程

```
原始 ABD Run 目录 (仅 CNCAP 规程, A66 企标数据排除, A66 CNCAP 完整保留)
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 A: 结构完整性检查                                        │
│  检查: 文件存在, 格式正确, .spec 存在, 通道 ≥ 400 列            │
│  通道 < 400 列 → 直接排除 (不完整 Run)                         │
│  输出: PASS → 阶段 B / FAIL → REJECT                          │
└──────────────────┬───────────────────────────┘
                   │ PASS
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 B: Run 有效性判定                                        │
│  检查: 时长, 异常旗标, 目标运动窗口, 传感器数据, 物理合理性       │
│  读取 .spec 确定 Sync 模式 + Trigger 配置                       │
│  输出: VALID / PARTIAL / ABORTED / CORRUPT / STATIC           │
│  VALID + PARTIAL → 阶段 C                                     │
│  ABORTED + CORRUPT + STATIC → 排除                            │
└──────────────────┬───────────────────────────┘
                   │ VALID ∪ PARTIAL
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 C: 场景聚合与覆盖矩阵                                     │
│  按 {品牌, 规程类型, 场景变体} 聚合                              │
│  输出: 覆盖状态 COMPLETE / SINGLE / DEGRADED                   │
│  输出: 覆盖矩阵 CSV                                            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 D: PICASO 特征提取与 Schema 统一化                        │
│  提取 Agent 轨迹, Lane 几何, Meta 标签                          │
│  降采样 100Hz→10Hz, 坐标对齐                                    │
│  T0 定位: Time-tolerance trigger → Path Phase → TTC 反算       │
│  输出: Unified Scene 数据集 (HDF5)                             │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 阶段 A：结构完整性检查

> 每项检查依据 ABD 文档中定义的文件格式规范

对每个 Run 目录执行:

```
[A0] 文件完整性 (新增):
     - V*_T*_R*.spec 文件必须存在
     - 依据: 实测验证 — 完整测试场景均有对应 .spec 文件 (GAC S9: 208/208, E8: 195/195, P7+: 235/235)
     - 无 .spec → 标记为 INCOMPLETE, 不进入后续阶段
     
[A1] .txt 文件存在且大小 > 1 KB
    依据: RC Manual §8.3 (p.143-144) — ASCII 导出包含 header + 数据行

[A2] 文件头行 1 = "Anthony Best Dynamics Ltd"
    依据: RC Manual §8.3 (p.143) — ASCII 文件首行为厂商标识

[A3] 行 2 符合 "Points=N" 格式, N > 100 (最少 1 秒 @ 100Hz)
    依据: RC Manual §8.3 (p.143) — "Points=3800" 为标配行

[A4] 行 3 通道名称行与后续数据行列数一致:
     行 3 列数 = 行 4 列数 = 行 5 列数 (允许行 5+ 尾部元数据行)
    依据: RC Manual §8.3 (p.143) — "one row of data per measurement point...
          each row containing a column for each chosen channel"

[A5] 通道数列数判定:
     - ≥ 400 列 → 完整导出 (Object 1 + Head tracker + 相对运动 + 触发), 进入阶段 B
     - < 400 列  → 不完整 Run (含 CCFT R1/其他场景首次尝试/中止运行), 直接排除
    依据: 22 个 CCFT 文件 + 多种场景类型的实测列数分布
    用户决策: "不完整 Run 不需要运动学反推, 直接剔除"

[A6] 关键通道列存在性检查:
     必须存在: "X position", "Y position", "Forward velocity",
               "Path phase", "Test phase"
     期望存在: "Object 1 actual X (front axle)" (C2C 场景),
               "Head tracker actual X (front axle)" (VRU 场景),
               "Time to collision (longitudinal)",
               "Time tolerance 1-16" (用于 T0 定位)

结果: ALL_PASS → 阶段 B / ANY_FAIL → REJECT (记录拒绝原因)
```

### 3.3 阶段 B：Run 有效性判定

#### B1 基础完整性

```
[B1.1] Points > 500 (至少 5 秒 @ 100Hz)
[B1.2] 时间列单调递增, 无跳变 (帧间差值 = 0.01s ± 10%)
      依据: RC Manual §8.3 (p.143) — "normally spaced at 20 ms intervals"
[B1.3] 实际数据行数与 Points 一致 (±0.5%)
```

#### B2 异常旗标检测

> 依据: RC Manual §6.1.4.10 (SR abort trigger, p.76), Triggers 1 CAN (p.193-194)

```
[B2.1] SR path abort = 1 in ANY timestep             → CRITICAL_ABORT
       依据: RC Manual p.193 — bit 9 = SR path abort
       (碰撞/安全风险导致路径中止, 整 Run 作废)

[B2.2] Abort path speed = 1 in ANY timestep          → CRITICAL_ABORT
       依据: RC Manual p.193 — bit 11 = Abort path speed

[B2.3] Stop point go = 1 (关键区间前触发)             → MANUAL_STOP

[B2.4] Motion Going SR/BR/AR 全部 = 0 持续 > 2s       → ROBOT_STALLED

[B2.5] Path phase < 0 持续 > 1s                       → PATH_CONTROLLER_FAULT
       依据: Path Following Manual §8 Table 8.1 — 负值含义
```

#### B3 传感器数据有效性

```
[B3.1] X position 或 Y position 全程不变 (std < 0.001m)    → SENSOR_DEAD
[B3.2] Forward velocity 全程 < 0.1 m/s                      → STATIC (非动态测试)
[B3.3] Subject status 持续 = 0 (> 1s)                       → VUT_DATA_INVALID
       依据: Subject status channel from Range 5 CAN message (p.192)
[B3.4] Tracker status = 0 持续 (> 1s)                       → TARGET_DATA_INVALID
       (实测 Tracker status 始终=9, 含义在 ABD 文档中未找到;
        此检查为防御性编程)
[B3.5] Gyro status ≠ 正常值 持续 > 1s                       → IMU_CORRUPT
[B3.6] Satellites < 4 持续 > 2s (GNSS 失锁)                 → GNSS_LOST

[B3.7] 目标运动检测:
       对 VRU 场景: Head tracker forward velocity 全程 std < 0.05 m/s → TARGET_STATIONARY
       对 C2C 场景: Object 1 forward velocity 全程 std < 0.05 m/s → TARGET_STATIONARY
       (目标物未移动 = 测试未执行或不完整; 384 列不完整 Run 已在阶段 A 排除)
```

#### B4 关键区间完整性判定

```
[B4.1] 目标运动窗口检测:
       - 找到目标速度首次 > 0.1 m/s 的帧索引 → t_motion_start
       - 找到目标速度最后一次 > 0.1 m/s 的帧索引 → t_motion_end
       - 若 (t_motion_end - t_motion_start) < 1.0s → 目标运动不足, 判定为 ABORTED

[B4.2] 关键区间判定 (针对提前中止的 Run):
       若 [B2] 触发了中止类异常:
         - 若 Critical section start = 1 已触发
           且 Critical section end = 1 已触发
           且 (t_end - t_start) ≥ 1.0s
           → PARTIAL (关键区间可用)
         - 否则 → ABORTED

[B4.3] 路径跟随质量检查:
       - Path following error 全程 max > 0.5m  → PATH_FOLLOWING_DEGRADED
         不直接拒绝, 但标记为 quality_flag=WARN
       - Distance error 全程 max > 2.0m → SPEED_CONTROL_DEGRADED
```

#### B5 物理合理性快速筛查

```
[B5.1] Forward velocity 帧间差分 > 50 km/h/s (≈1.4g) 连续 ≥ 3 帧 → SENSOR_NOISE
[B5.2] Yaw angle 帧间 0°-360° 跳变 > 30° → GPS_GLITCH
[B5.3] Object 1 与 Head tracker 一致性检查 (两者均存在时):
       |Object1_X - Head_X| 均值 > 0.1m → TARGET_MISMATCH
       实测: 两者数据完全一致 (差值 < 1e-6)
```

#### 最终标签判定

```
┌──────────┬──────────────────────────────────────────────┐
│ VALID    │ 所有检查通过, 目标运动窗口 ≥ 1s, 全 Run 可用    │
│ PARTIAL  │ 关键区间完整但对目标运动前/后有提前终止          │
│ ABORTED  │ 测试中止, 关键区间不完整                        │
│ CORRUPT  │ 传感器/数据损坏                                │
│ STATIC   │ 目标物或 VUT 未移动 (非有效动态测试)            │
└──────────┴──────────────────────────────────────────────┘

处理策略:
  VALID   → 目标运动窗口 [t_motion_start - 1s, t_motion_end + 1s] 提取, 进入阶段 C
  PARTIAL → 仅提取 [t_critical_start, t_critical_end], 标注 partial=True, 进入阶段 C
  其他    → 排除, 记录在 rejected_runs.csv
```

### 3.4 阶段 C：场景聚合与覆盖矩阵

```
[C1] 场景身份识别
  从目录路径结构 + .spec 文件提取层次标签:
    Brand          ∈ {GAC_S9, GAC_E8, XPeng_P7+, GAC_A66}
    TestCategory   ∈ {CNCAP_VRU, CNCAP_ADAS}
      注: 企标数据已排除, 仅保留 CNCAP 规程. A66 CNCAP 完整保留.
    ScenarioType   ∈ {CPLA, CPNCO, CPFAO, CPTA, CBNAO, CBLA,
                      CSFAO, CSTA, CCRs, CCRH, SCP, SCPO, CCFT,
                      BSD}
      注: FalseReaction 生成侧剔除, 但作为负样本保留给 surrogate (2026-09-01 修订)
    VariantLabel   ← 从目录名 + .spec 速度段解析
    SpeedCondition ← 从 VariantLabel 解析 VUT/Target 速度

[C2] 变体聚合
  按 {Brand, TestCategory, ScenarioType, VariantLabel} 四元组聚合
  每个四元组内统计:
    total_runs     : 原始文件数
    valid_runs     : VALID 标签数
    partial_runs   : PARTIAL 标签数
    rejected_runs  : ABORTED + CORRUPT + STATIC 标签数
    completeness   : valid_runs / total_runs (该变体的有效率)

[C3] 覆盖状态标记
  COMPLETE  : valid_runs ≥ 2  (可重复)
  SINGLE    : valid_runs = 1  (单次有效, 可验证, 不用于统计推断)
  DEGRADED  : valid_runs = 0 但 partial_runs > 0
  EMPTY     : 无有效数据

[C4] 输出
  - 覆盖矩阵 CSV: coverage_matrix.csv (行=ScenarioType, 列=Brand)
  - Run 清单 CSV: all_runs_inventory.csv (含路径, 标签, 场景标签, 异常原因)
```

### 3.5 阶段 D：PICASO 特征提取与 Schema 统一化

#### T0 定位策略 (多优先级)

```python
def locate_t0(run, spec_config):
    """
    T0 定位策略 (优先级从高到低):
    
    Strategy 1 [优先]: Time-tolerance trigger 通道直接检测
      依据: RC Manual §6.1.4.5 (p.69), Triggers 2 CAN (p.194-195)
      已验证: CCRs AEB — Time tolerance 1 fires at T=14.120s, TTC=3.000s
    
    Strategy 2 [备选]: Path Phase 状态转换
      依据: Path Following Manual §8 (p.77-78, Table 8.1)
      SPT: -5/-25 → 1 (同步结束)
      PF Standard: 0 → 1 (测试开始)
    
    Strategy 3 [兜底]: TTC 运动学反算
      TTC = distance / velocity → 找 TTC=3.0s
    
    Strategy 4 [兜底]: 目标运动起始
      目标绝对速度首次 > 0.1 m/s
    """
    # Strategy 1: Time-tolerance trigger
    trigger_num = spec_config.get_trigger_number()
    if trigger_num:
        tt_col = f"Time tolerance {trigger_num}"
        tt_data = run.get_channel(tt_col)
        transitions = find_transitions(tt_data, 0, 1)
        if len(transitions) > 0:
            t0_candidate = transitions[0] / run.sample_rate
            # 交叉验证: TTC 应约等于 3.0s (AEB 场景) 或满足 BSD 距离条件
            ttc_at_t0 = run.get_ttc_at(t0_candidate)
            if abs(ttc_at_t0 - 3.0) < 0.5:
                return t0_candidate, "Time-tolerance trigger (verified by TTC)"
    
    # Strategy 2: Path Phase
    pp = run.get_channel("Path phase")
    # SPT test: synchro end
    pp_transitions = find_transitions(pp, from_states=[-5,-25,-24,-6], to_states=[1])
    if len(pp_transitions) > 0:
        return pp_transitions[0] / run.sample_rate, "Path Phase synchro end"
    # PF Standard: path start
    pp_transitions = find_transitions(pp, from_states=[0], to_states=[1])
    if len(pp_transitions) > 0:
        return pp_transitions[0] / run.sample_rate, "Path Phase path start"
    
    # Strategy 3: TTC kinematics
    rel_dist = run.get_channel("Relative longitudinal distance")
    rel_vel = run.get_channel("Relative longitudinal velocity")
    rel_vel_safe = np.where(np.abs(rel_vel) > 0.1, rel_vel, 0.1)
    ttc_est = rel_dist / np.abs(rel_vel_safe)
    t0_idx = np.argmin(np.abs(ttc_est - 3.0))
    return t0_idx / run.sample_rate, "TTC kinematics (TTC=3.0s)"
```

#### 主提取流程

```python
for each run in VALID ∪ PARTIAL:
    # [D1] 读取 .spec 配置
    spec = SpecReader(run.spec_path)
    sync_mode = spec.get("UseSynchronizationMode")  # True/False
    trigger_num = spec.get_trigger_number()  # 从 StartTrigger 或 SyncEndTriggerType

    # [D2] 确定通道模式 (< 400 列已在阶段 A 排除, 此处均为 FULL)
    channel_mode = "FULL"  # 所有进入阶段 D 的 Run 均为 ≥ 400 列完整导出

    # [D3] 确定时间窗口
    if run.label == PARTIAL:
        t_start, t_end = run.t_critical_start, run.t_critical_end
    else:
        t_motion_start, t_motion_end = find_target_motion_window(run, spec)
        t_start = max(0, t_motion_start - 1.0)
        t_end = min(run.t_max, t_motion_end + 1.0)

    # [D4] T0 定位 (多优先级策略, 见上)
    T0, T0_method = locate_t0(run, spec)

    # [D5] 提取 VUT Agent 轨迹
    vut_agent = extract_vut_agent(run, t_start, t_end, brand)

    # [D6] 提取 Target Agent 轨迹
    # 数据源优先级 (用户决策):
    #   C2C 场景 → Object 1 (GVT 机器人直接测量, 语义准确)
    #   VRU 场景 → Head tracker (SPT 直接测量, C-NCAP 精度 ±0.01 km/h)
    if is_vru_scenario:
        target_agent = extract_target_from_head_tracker(run, t_start, t_end)
    else:
        target_agent = extract_target_from_object1(run, t_start, t_end)

    # [D7] 降采样: 100Hz → 10Hz (每 10 帧取 median)
    for agent in [vut_agent, target_agent]:
        agent.resample(target_rate=10, method="median")

    # [D8] 坐标对齐: 原点 = VUT T0 时刻位置, x 轴 = VUT T0 时刻朝向
    for agent in [vut_agent, target_agent]:
        agent.transform_to_local(vut_agent.x[T0_idx], vut_agent.y[T0_idx], vut_agent.yaw[T0_idx])

    # [D9] Lane 几何重建 (从 Desired X/Y 路径点)
    lane = LanePolyline.from_desired_path(
        run.get_window("Desired X (lead axle)", t_start, t_end),
        run.get_window("Desired Y (lead axle)", t_start, t_end),
    )

    # [D10] Meta 填充
    scene = Scene(
        agents=[vut_agent, target_agent],
        lane=lane,
        meta=Meta(
            scenario_type=spec.scenario_type,
            test_category=spec.test_category,
            vut_speed=spec.vut_speed,
            target_speed=spec.target_speed,
            overlap=spec.overlap,
            ttc_trigger=3.0,
            t0_method=T0_method,
            abd_precision={"lateral": 0.03, "speed": 0.1, "vru_speed": 0.01},
            data_source=src,
            partial=(run.label == "PARTIAL"),
            channel_mode=channel_mode,
            sync_mode=sync_mode,
        ),
        source="CNCAP_ABD",
    )
    dataset.append(scene)
```

---

## 四、实验设计

### 4.1 数据切分策略

```
┌──────────────────────────────────────────────────────────────────┐
│                       全量数据池                                   │
├────────────────┬──────────────────────┬───────────────────────────┤
│  Waymo +       │  CNCAP ABD 多品牌     │  ABD 实车补测批次          │
│  INTERACTION   │  (响应建模, LOBO)     │  (M4 起, 附录 F 协议)      │
├────────────────┴──────────────────────┴───────────────────────────┤
│                                                                   │
│  ┌─────────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │ Pretrain Set    │  │ Surrogate Set │  │ Real-Validation Set│  │
│  │ (Source only)   │  │ (LOBO splits) │  │ (Stage D 实车闭环) │  │
│  │                 │  │               │  │                    │  │
│  │ 用途: Stage A   │  │ 用途: Stage B │  │ 用途: Stage D      │  │
│  │ 生成器预训练     │  │ surrogate训练 │  │ 实车验证闭环       │  │
│  │                 │  │ + 跨品牌校准  │  │                    │  │
│  │ Waymo: 训练集   │  │ 4→9-10 品牌   │  │ 规程内复现点        │  │
│  │ INTERACTION:    │  │ 留一品牌外推  │  │ + 规程外新点       │  │
│  │   训练+验证     │  │ (LOBO)        │  │ (surrogate 高危)   │  │
│  └─────────────────┘  └───────────────┘  └────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**切分逻辑（修订）** (A66 企标数据已排除, A66 CNCAP 数据作为独立验证候选):
- **主实验优先**: 采用 leave-one-brand-out / leave-one-family-out，按实际有效样本覆盖选择 holdout 品牌；不要默认只让 S9 参与适配训练。
- **工程验证保留**: 可保留“GAC S9 适配，GAC E8 + XPeng P7+ + GAC A66 CNCAP holdout”的设置作为工业验证实验，但需明确它不是唯一切分。
- **按 ScenarioType 分层**: 确保 train 和 holdout 在 CCRs, SCP, CPLA, CPTA 等主要规程类型上尽量同类对比；缺失场景只做案例分析，不做统计显著性。
- **Source/Target 严格隔离**: 生成器预训练仅 Waymo + INTERACTION；CNCAP ABD 数据只用于 surrogate 训练/校准、反事实评估、实车验证闭环与最终 holdout。
- **数据泄漏检查**: 同一物理测试序列、同一 `.CRUN/.spec` 派生出的片段不能同时进入训练和 holdout。

### 4.2 数据量估算

| 集合 | 品牌 | 场景变体数 (估) | 等效有效 runs | 说明 |
|------|------|---------------|-------------|------|
| Pretrain (Waymo) | — | ~487K 场景片段 | ~135,000 小时 | 9s 片段 @ 10Hz |
| Pretrain (INTERACTION) | — | ~55K 轨迹 | ~150 小时 | 11 个路口/匝道 |
| Surrogate 训练/校准 | 按 LOBO split 决定 | 待 inventory 确认 | 待 inventory 确认 | 不提前锁死品牌 |
| Holdout | leave-one-brand-out 品牌 | 待 inventory 确认 | 待 inventory 确认 | 用于跨品牌泛化 |
| 工程验证 Adapt | GAC S9（候选） | 待 inventory 确认 | 待 inventory 确认 | 若 S9 有最完整 VRU+ADAS，可作为工业适配实验 |
| 实车补测集 (Stage D) | 具备补测条件的品牌 | M4 起按 surrogate 选点 | ≥20 场景/首批 | 规程内复现 + 规程外新点两组（附录 F） |

### 4.3 对比基线

| 基线 | 类型 | 代表方法 | 复现策略 |
|------|------|---------|---------|
| B1: Random Sampling | 数据驱动下限 | 从 Waymo 随机采样场景片段 | 自行实现 |
| B2: Rule/Parameter Sweep | 规程参数采样 | 基于 C-NCAP 速度/重叠率/TTC 范围采样 | 必须自行实现 |
| B3: STRIVE / TrafficGen | 数据驱动或对抗生成 | 取可复现代码中最稳定者 | 有开源且能跑通才列入主表 |
| B4: AdvSim | 对抗生成 | LiDAR/仿真依赖较强 | 若复现成本过高，作为文献对比或附录 |
| B5: CounterScene-style | 因果反事实 | 单变量 do(·) 反事实基线 | 可实现轻量版，不冒充原论文完整复现 |
| B6: No-ABD-Calibration | 无校准对照 | 纯 Waymo/INTERACTION 训练，生成器与 surrogate 均不做 ABD 校准 | 必做对照（2026-09-01 新增） |
| **PICASO (Ours)** | — | 完整 5 层架构 | — |
| PICASO w/o C1 | 消融 | 无 PHNN 物理约束 | 去掉 L_PH |
| PICASO w/o C2 | 消融 | 无 MACC 级联干预 | 去掉因果图 + do(·) |
| PICASO w/o C3 | 消融 | 无跨品牌校准 | surrogate 退化为 brand-agnostic |

### 4.4 评估指标 (Layer 5 四维矩阵)

| 维度 | 指标 | 计算方式 | 预期改进方向 |
|------|------|---------|------------|
| **安全关键性** | Collision Rate (CR) | 生成场景中发生碰撞的比例 | 高于所有基线 |
| | Min TTC 分布 | 每个场景的最低 TTC 统计 | 更低的 5th percentile |
| | Exposed Safety Violations | 基线未发现的独特安全违规场景数 | 高于 CounterScene |
| **物理可行性** | KFR | 全程满足 a ≤ a_max, |κ| ≤ κ_max, F_fric ≤ μF_N | 高于无物理约束基线；目标值待实验确认 |
| | Jerk Violation Rate | 加加速度超限次数/场景 | 低于无物理约束/无投影消融版，报告 CI |
| **保真度** | MMD | 生成 vs 真实场景轨迹分布距离 | 低于 TrafficGen |
| **跨品牌诊断** | TDPD | (CR_target − CR_source) / CR_source | 2026-09-01 统一定义；仅诊断用，不作核心指标 |
| **实车一致性（核心）** | 碰撞判定一致率 | surrogate 预测 vs ABD 实测一致比例 | 实验后填报（规程内/规程外分组报告） |
| | minTTC MAE | \|minTTC_pred − minTTC_ABD\| 均值 | 实验后填报 |
| | AEB 触发时刻 MAE | \|t_pred − t_ABD\| 均值 | 实验后填报 |
| | 实车验证命中率 | 实车确认高危 / 送测高危场景数 | 高于随机选点对照（A9） |
| **因果可解释性** | Root-Cause Score | Shapley/归因结果与规程逻辑或人工复核一致性 | 作为新指标报告，不提前承诺阈值 |
| | IC | do(x)→do(y) 风险变化方向一致性 | 报告均值、CI 与失败案例 |

---

## 五、代码架构

### 5.1 模块结构

```
D:\Scenario_Generation_Research\
  │
  ├── abd_parser/
  │   ├── __init__.py
  │   ├── parser.py           # ABD .txt 读取 + 通道解析 (≥400 列完整 Run)
  │   ├── spec_reader.py      # .spec 文件解析 (提取 Synchro 配置, Trigger 编号, 车辆参数)
  │   ├── trigger_analyzer.py # Time-tolerance trigger 检测 + T0 多优先级定位
  │   ├── validator.py        # 阶段 A+B: 结构检查 + 有效性标签 + 目标运动窗口检测
  │   ├── aggregator.py       # 阶段 C: 场景聚合与覆盖矩阵
  │   ├── extractor.py        # 阶段 D: PICASO 特征提取 (Agent 轨迹 + Lane 几何)
  │   ├── schema.py           # Layer 1 统一 Schema 数据结构 (dataclass)
  │   ├── channel_map.py      # 通道名模糊匹配 (兼容不同 ABD 版本命名差异)
  │   └── utils.py            # 坐标变换, 降采样, 窗口提取, 运动段检测
  │
  ├── config/
  │   └── stage4_config.yaml  # 通道映射表, 阈值, 路径, 域标签定义
  │
  └── Stage4_数据处理与实验方案.md  (本文件)
```

### 5.2 核心数据结构

```python
# schema.py — 与 Stage 3 Layer 1 定义严格对齐

@dataclass
class AgentTrajectory:
    id: str                       # "VUT" | "Target"
    type: str                     # "vehicle" | "GVT" | "VRU" | "FalseTarget"
    x: np.ndarray                 # [T] @ 10Hz, 场景内相对坐标
    y: np.ndarray                 # [T]
    v_f: np.ndarray               # [T] 前向速度 m/s
    v_l: np.ndarray | None        # [T] 侧向速度 m/s
    a_f: np.ndarray | None        # [T] 前向加速度 m/s²
    a_l: np.ndarray | None        # [T] 侧向加速度 m/s²
    yaw: np.ndarray | None        # [T] 偏航角 rad
    yaw_rate: np.ndarray | None   # [T] 偏航角速度 rad/s
    domain_label: int             # 0=Waymo, 1=INTERACTION, 2=S9, 3=E8, 4=P7+, 5=A66
    brand_label: str | None

@dataclass
class LanePolyline:
    points: np.ndarray            # [K, 2] 路径点序列 (从 Desired X/Y 重建)
    type: str                     # "centerline" | "desired_path"

@dataclass
class Meta:
    scenario_type: str            # "CCRs", "CPLA", "CCFT", etc.
    test_category: str            # "CNCAP_VRU" | "CNCAP_ADAS"
    vut_speed: float              # km/h
    target_speed: float | None    # km/h
    overlap: float | None         # 重叠率 (C2C), None for VRU
    ttc_trigger: float            # 3.0 for AEB
    t0_method: str                # "Time-tolerance trigger" | "Path Phase" | "TTC kinematics" | "Target motion onset"
    sync_mode: bool               # 从 .spec UseSynchronizationMode 读取
    abd_precision: dict           # {"lateral": 0.03, "speed": 0.1, "vru_speed": 0.01}
    data_source: str              # "Object_1" (C2C) | "Head_tracker" (VRU)
    partial: bool
    channel_mode: str             # "FULL" (所有进入阶段 D 的 Run 均 ≥ 400 列)

@dataclass
class Scene:
    agents: list[AgentTrajectory]
    lane: LanePolyline | None
    meta: Meta
    source: str                   # "Waymo" | "INTERACTION" | "CNCAP_ABD"
```

### 5.3 通道模糊匹配策略

> **依据**: RC Manual §8.1.4 (p.95-96) 列出约 150 个通道, 不同 ABD 版本/配置可能存在命名差异
> **依据**: RC Manual §9.3.3 Export settings (p.187) — 用户可选择导出哪些通道

```python
def find_channel(available_channels: list[str], canonical_name: str) -> str | None:
    """
    1. 精确匹配
    2. 去空格 + case-insensitive 匹配
    3. canonical_name 前缀匹配 (如 "Object 1" 匹配 "Object 1 actual X (front axle)")
    4. 编辑距离 < 3 的模糊匹配
    失败返回 None, 记录 unmatched_channels.log
    """
```

---

## 六、实施计划与依赖

| 步骤 | 产出 | 依赖 | 状态 |
|------|------|------|------|
| 4.0 数据审计 | 覆盖矩阵初版 (含 8 种场景类型深度抽查) | 文件系统遍历 + Python 数据分析 | ✅ 已完成 |
| 4.1 C-NCAP 规程阅读 | 82 页 PDF (Appendix L + O) 完整阅读 | C-NCAP 2024 原版 PDF | ✅ 已完成 |
| 4.2 ABD 文档完整阅读 | RC Software Manual (354pp) + Path Following Manual (81pp) 关键章节 | ABD Documentation | ✅ 已完成 |
| 4.3 ABD 数据深度抽查 | 8 种场景类型 × 多时间窗口 + Trigger 通道验证 | 4.0 + 4.1 + 4.2 | ✅ 已完成 |
| 4.4 ABD 解析器开发 | `abd_parser/*.py` 全部模块 | 4.1 + 4.2 + 4.3 | ⬜ 待开发 |
| 4.5 有效性筛查执行 | `all_runs_inventory.csv` + `coverage_matrix.csv` | 4.4 | ⬜ 待执行 |
| 4.6 特征提取 | Unified Schema 数据集 (HDF5) | 4.5 | ⬜ 待执行 |
| 4.7 实验配置锁定 | train/val/holdout 最终切分 + 基线选择确认 | 4.6 | ⬜ 待执行 |

---

## 七、已决策事项与仍需确认事项

### 7.1 已由用户明确的决策

| # | 决策事项 | 决策结果 | 影响 |
|---|---------|---------|------|
| D1 | Tracker status = 9 的含义 | SPT 惯导状态良好, 但 SPT 仅使用惯导的 GPS 授时 | 仅检查 ≠ 0 作为可用性验证, 不做语义解释 |
| D2 | 不完整 Run (通道 < 400 列) | **直接剔除**, 不做运动学反推 | 简化处理流程, 移除 CCFT_RELATIVE_ONLY 模式 |
| D3 | 无 .spec 文件 = 不完整测试 | **已确认**: 完整 Run 100% 有对应 .spec (GAC S9 208/208, E8 195/195, P7+ 235/235) | 无 .spec → 阶段 A 直接排除 |
| D4 | C2C 场景数据源优先级 | **Object 1** (GVT 机器人直接测量, 语义准确) | C2C → Object 1; VRU → Head tracker |
| D5 | FalseReaction 场景 | **分流处理（2026-09-01 修订）**：生成/统一评估侧剔除（无统一 T0）；作为"不应制动"负样本保留给 surrogate 边界建模 | 从生成侧数据表移除；surrogate 负样本池保留 |
| D6 | SCP vs SCPO | **两个独立场景** (SCPO ≠ 含于 SCP) | 场景类型清单增加 SCPO 独立条目 |
| D7 | Mass 参数 | **重要** — 用户将提供每辆实际车辆整备质量 | Layer 5 KFR 摩擦圆约束需要真实质量 |
| D8 | SteerRatio 参数 | **不重要** — PICASO 使用 GPS/IMU 轨迹, 不依赖转向比 | 不纳入处理流程 |

### 7.2 仍需确认或由程序输出锁定的事项

当前无需用户再做方向性决策，但以下事项必须在进入正式训练前由数据或用户补齐：

| 事项 | 来源 | 对实验的影响 |
|------|------|--------------|
| `all_runs_inventory.csv` 最终有效 run 数 | ABD 解析器输出 | 锁定训练/验证/holdout 数据量 |
| 每品牌/每车型实际测试质量 | 用户或测试记录 | 摩擦圆与 KFR 计算需要 |
| A66 CNCAP 与企标目录边界 | inventory + 人工抽查 | 避免企标数据混入 CNCAP 主实验 |
| E8 Cal/企标目录是否可映射 CNCAP 规程 | 人工抽查 `.spec` 与路径 | 决定是否纳入 VRU/ADAS 统计 |
| 专有数据匿名化规则 | 公司合规要求 | 决定论文表格、图例、开源样例可披露范围 |

---

## 附录 A: 场景规程缩写与关键参数 (从 C-NCAP 2024 原文提取)

### ADAS (Appendix L)

| 缩写 | 规程名称 | 章节 | VUT 速度 (km/h) | GVT 速度 (km/h) | 重叠率 | T0 定义 |
|------|---------|------|----------------|----------------|--------|--------|
| CCRs | 静止车尾 AEB/FCW | L.6.1.5 | 20/30/40(AEB), 50/60/70/80(FCW) | 0 | -50%/+50% | TTC=3s |
| CCRH | 高速跟车 AEB/FCW | L.6.1.6 | 80/120(FCW) | 0 (VT后) | 100% | TTC=3s |
| SCP | 横向穿车 AEB/FCW | L.6.1.7 | 30/40(AEB), 50/60(FCW) | 20/30/40/50 | 90°交叉 | TTC=3s |
| SCPO | 对向穿车 FCW | L.6.1.8 | 50/60(FCW) | 40/50 | 含3台遮挡车 | TTC=3s |
| CCFT | 穿车假目标 AEB | L.6.1.9 | 10/20/30(左转) | 20/40/50 | 特定弧形路径 | TTC=3s |
| BSD | 盲区监测 | L.6.5.4 | VUT 50/VUT 30 | GVT 60/TW 40 | 多种 | 33m/变道起点 |

### VRU (Appendix O)

| 缩写 | 规程名称 | 章节 | VUT (km/h) | 目标速度 (km/h) | 重叠 | 日/夜 |
|------|---------|------|-----------|---------------|------|------|
| CPLA-25 | 近端成人纵向 | O.6.1.6 | 20/40/60/80 | PT 5 | 25% | Day+Night |
| CPNCO-25 | 近端儿童纵向遮挡 | O.6.1.7 | 20/40/60 | PT 5 | 25% | Day |
| CPFAO-25 | 远端成人纵向遮挡 | O.6.1.8 | 20/40/60 | PT 6.5 | 25% | Day+Night |
| CPTA-LN-50 | 近端成人转向 | O.6.1.9 | 10/20/30 | PT 5 | 50% | Day |
| CPTA-LF-50 | 远端成人转向(道) | O.6.1.10 | 10/20/30 | PT 6.5 | 50% | Day |
| CPTA-RF-50 | 远端成人转向(路) | O.6.1.11 | 10/20 | PT 6.5 | 50% | Day |
| CBNAO-50 | 近端自行车纵向遮挡 | O.6.1.12 | 20/40/60 | Bicycle 15 | 50% | Day |
| CBLA-25 | 远端自行车纵向 | O.6.1.13 | 20/40/60/80 | Bicycle 15 | 25% | Day |
| CSFAO-50 | 近端踏板车纵向遮挡 | O.6.1.14 | 20/40/60 | Scooter 20 | 50% | Day |
| CSTA-LN-50 | 远端踏板车转向近 | O.6.1.15 | 10/20/30 | Scooter 20 | 50% | Day |
| CSTA-RN | 远端踏板车转向右 | O.6.1.15 | 10/20 | Scooter 20 | — | Day |

> 所有 AEB VRU 场景 T0 = TTC=3s。VRU 目标速度精度 ±0.01 km/h。

---

## 附录 B: ABD Synchro 同步机制详解 (回答用户问题)

### Q: "Synchro trigger 是啥, 我具体不清楚"

**答**: 需要区分两个不同概念:

**概念 1: "Synchro object X trigger" (同步对象触发器)**
- **文档位置**: RC Software Manual §6.1.4.1 (p.63-64) + Triggers 1 CAN message (p.193, bits 48-63)
- **是什么**: 预定义的 1-bit 数字信号, 在 Triggers 1 CAN 消息中传输
- **用途**: 作为 **Advanced Channel 启动触发** 的触发源选项 (在 Advanced Trigger Choice dialog 中选择 `Synchro object X`)
- **不是用于**: 结束同步模式
- **为什么导出数据中始终为 0**: 只有当一个测试**明确配置为使用 Synchro object X 作为启动触发**时, 对应 bit 才会脉冲为 HIGH。在大多数配置中, 这些 bit 保持为 0

**概念 2: "Use trigger to end synchronization mode" (使用触发器结束同步)**
- **文档位置**: RC Software Manual §6.1.4.7 Synchro (p.74)
- **是什么**: Synchro 配置中的一个**复选框选项**
- **工作方式**: 
  1. Subject 上配置一个触发器 (Speed / Time-tolerance / X/Y position 等)
  2. 当触发器条件满足时, 触发信号通过 **Ethernet** 发送给所有 Tracker
  3. Tracker 收到信号后, 退出同步模式, 开始执行独立路径
- **在导出数据中的体现**: 使用的触发器 (如 Time-tolerance trigger 4) **确实会在导出数据中变化**。实测 CCRs 中 Time-tolerance trigger 1 在 TTC=3.0s 时精确触发
- **不是**: 通过 Synchro object X trigger bit 来标记的

### Q: "请你在给出数据处理方案时, 同时给出处理的思路和依据, 你是根据哪份文件是怎么处理该数据的"

**答**: 每个数据处理决策的文档依据已在方案各节中标出。主要参考文档:

1. **ABD RC Software Manual (RM-S-01 Issue 23)** — 354 页
   - §2.2: 测试概念概述
   - §6.1.4.1: SR 触发器类型 (含 Synchro object 0-15 trigger 定义)
   - §6.1.4.5: Time-tolerance parameters (核心 T0 定位依据)
   - §6.1.4.7: Synchro 同步机制 (Full Synchro, 条件触发退出, Broadcast)
   - §8.1.4: 数据采集通道完整列表 (~150 通道)
   - §8.3: ASCII 导出格式规范 (.txt 文件格式定义)
   - §9.3.4.2: CAN 消息定义 (全部 CAN 信号)
   - Triggers 1 CAN (p.193-194): 所有数字触发器信号
   - Triggers 2 CAN (p.194-195): 所有 Time-tolerance 触发器信号

2. **ABD Path Following User Manual (RM-S-02 Issue 17)** — 81 页
   - §8 (p.77-78, Table 8.1): Path Phase 完整状态机 (核心状态检测依据)
   - §6.1 (p.51-53): PF Standard 测试设置 (StartTrigger 类型)

3. **C-NCAP 2024 规程**
   - Appendix L (ADAS): T0 定义, 测试有效性规则, 精度要求
   - Appendix O (VRU): T0 定义, VRU 速度精度 ±0.01 km/h

---

## 附录 C: 数据抽查关键发现汇总

| # | 发现 | 依据 | 影响 | 处理方案 |
|---|------|------|------|---------|
| 1 | Time-tolerance trigger 可精确捕获 T0 (CCRs 实测 T=14.120s, TTC=3.000s) | RC Manual §6.1.4.5 + 实测 | T0 有可靠数据驱动定位方法 | Strategy 1 优先使用 Time-tolerance trigger |
| 2 | `UseSynchronizationMode` 因测试类型而异 (CCRs=False, ELK=True, BSD 部分=False) | .spec 实测 | 不能一概而论 | 必须从 .spec 读取, 动态选择 T0 策略 |
| 3 | Synchro object X trigger 始终为 0 | RC Manual §6.1.4.1 + 实测 | 不可用作 T0 定位 (用于启动测试, 非结束同步) | 不纳入 T0 检测策略 |
| 4 | Object 1/Head tracker 数据完全一致 | 实测 (多个场景交叉验证) | 两者冗余 | C2C → Object 1; VRU → Head tracker (用户决策) |
| 5 | CCFT 完整 Run 有 415 列 (14/22), R1 <400 列直接排除 | 实测 22 个文件 | 修正了之前"CCFT 仅 384 列"的错误 | ≥400 列 FULL 模式; <400 列直接剔除 |
| 6 | GAC A66 CNCAP 数据分布在 5 个目录 (2/3/8/9/11), 企标在 5 个目录 (12-16) | 实测目录结构 | 仅排除企标, CNCAP 完整保留 | A66 作为第四品牌独立验证数据 |
| 7 | Tracker status = 9: SPT 惯导状态良好, 仅使用 GPS 授时 | 用户确认 | 含义已明确 | 仅检查 ≠ 0, 不做语义解释 |
| 8 | Path Phase 不包含 PF Standard 测试的同步状态 (仅 SPT 有 -5/-25) | Path Following Manual §8 Table 8.1 | PF Standard 不能用 Path Phase 检测同步结束 | 对 PF Standard 使用 Time-tolerance trigger 或 TTC |
| 9 | Pedestrian articulation/Finger 通道始终为 0 (含 VRU) | 实测 | VRU 假人铰接未启用 ARS | 不纳入 PICASO 特征 |
| 10 | 路径跟随通道 = 机器人控制精度, ≠ VUT 运动学 | Path Following Manual §8 | 不能作为 PICASO 特征 | 仅用于异常检测 |
| 11 | 单变体 1~3 runs 是正常现象 | C-NCAP 2024 L.1.57-1.60 | 1 run ≠ 数据缺失 | SINGLE vs COMPLETE 覆盖标记 |
| 12 | Mass 为默认值 1300 kg (未实际填写) | .spec 实测 | Layer 5 摩擦圆约束需要真实质量 | 用户已提供全部: E8=2410, S9=2600, A66=2535, P7+=2395 kg |
| 13 | SteerRatio 为默认值 15.4 (未实际填写) | .spec 实测 | PICASO 使用 GPS/IMU 轨迹, 不依赖转向比 | 不纳入处理流程 |
| 14 | 完整测试 100% 有对应 .spec 文件 (GAC S9 208/208, E8 195/195, P7+ 235/235) | 实测 | 无 .spec = 不完整测试 | 阶段 A 直接排除 |
| 15 | FalseReaction 分流（2026-09-01 修订） | 用户决策 + 方案评审 | 生成侧剔除（无统一 T0）；surrogate 负样本池保留 | 生成侧移除，surrogate 侧保留 |
| 16 | SCPO 与 SCP 是两个独立场景 | 用户纠正 + 实测目录结构 | 不能将 SCPO 含于 SCP | 场景类型清单独立列出 |

---

## 附录 D: ABD 数据采集链路与 Synchro 信号流

```
                      Full Synchro 测试流程
                      ===================

Phase 1: 准备阶段
  ┌─────────────────────┐            ┌──────────────────────┐
  │ VUT (Subject)       │            │ Target (Tracker)      │
  │ 驾驶机器人控制       │  Ethernet  │ SPT/LaunchPad 控制   │
  │ Path Phase = -99/0  │◄──────────►│ Path Phase = -5/-25  │
  │ (系统待命/Lead-in)   │   Synchro  │ (Fully/Longitudinal  │
  │                     │   耦合      │  Synchronized)       │
  └─────────────────────┘            └──────────────────────┘

Phase 2: VUT 启动 → 同步进行
  - VUT 先出发 (Standing start / Closed Loop)
  - Tracker 在 Full Synchro 模式下与 VUT 保持几何同步
  - 如果 VUT 快 10%, Tracker 也快 10%
  - Tracker 的 Path Phase = -5 (Fully synchronized)

Phase 3: 触发 → 退出同步
  - Subject 上的 Time-tolerance trigger X 条件满足
    (例如: Time to POI 在 0-3s 内 → TTC 约 3.0s)
  - Time tolerance X 通道: 0 → 1  (在导出数据中可观测!)
  - 触发信号通过 Ethernet 广播给 Tracker
  - Tracker Path Phase: -5 → 1 (Unsynchronized)
  - Tracker 开始执行独立路径 (碰撞路径)

Phase 4: 测试结束
  - 无碰撞: 目标完成完整路径, Path Phase → -99
  - 碰撞: 目标在碰撞点停止, SR path abort = 1

关键可观测信号:
  Subject 端 (.txt 数据):
    Time tolerance X: 0 → 1  ← T0!
    TTC (longitudinal): ≈ 3.0s (交叉验证)
    Path phase: 0 → 1 (PF Standard) 或 1 → ... (SPT)

  Tracker 端 (Object 1 / Head tracker 数据, 在 Subject 的 .txt 中):
    Object 1 forward velocity: 0 → ~5 km/h (目标开始运动)
    Object 1 actual X/Y: 从静止开始变化
    Tracker status: = 9 (恒定)
```

---

## 附录 E: .spec 文件读取器规范

```python
class SpecReader:
    """
    .spec 文件是 key=value 格式的文本文件, 不同测试类型有不同字段。
    
    关键提取字段:
    - Type: 测试类型 (PF Standard, SR/AR Combination, Learn, etc.)
    - UseSynchronizationMode: True/False
    - SynchronizationMode: "Subject vehicle" | "Tracker vehicle"
    - SyncEndTriggerType: "Time-tolerance trigger X" | "Speed" | etc.
    - StartTrigger: "Closed Loop" | "Standing start" | "Time-tolerance trigger X"
    - 各种 TimeToleranceX 参数 (Channel, Min, Max, TriggerTime, DelayTime)
    - 车辆参数 (WheelBase, Mass, VehicleLength, VehicleWidth, SteerRatio)
    - 坐标基准 (CurrentDatumLatitude, CurrentDatumLongitude, CurrentDatumBearing)
    - 速度控制参数 (Speed1, Speed2, SpeedMultiplier)
    - 数据库标识 (DBVID, DBTID, DBRID)
    
    get_trigger_number():
        优先从 SyncEndTriggerType 解析 (如果 UseSynchronizationMode=True)
        回退到 StartTrigger 解析
        返回 trigger 编号 (1-16) 或 None
    
    get_scenario_type():
        从目录路径 + .spec Description 字段推断
    """
```

---

## 附录 F: ABD 实车补测协议 (2026-09-01 新增, Stage D 依据)

### F.1 目的

对 surrogate 判定为高危险且 C-NCAP 标准矩阵未覆盖的生成场景，执行真实车辆 ABD 机器人场地补测，量化 sim-to-real 一致性（碰撞判定一致率 / minTTC MAE / AEB 触发时刻 MAE），并将结果反哺 surrogate（主动学习闭环）。

### F.2 批次设计（首批 ≥20 场景，分两组）

| 组别 | 规模 | 选点方式 | 目的 |
|------|------|---------|------|
| G1 规程内复现组 | ≥8 场景 | 从已有标准矩阵工况中抽样（surrogate 应高置信复现已知结果） | 校准：验证 surrogate 在已知域的一致性 |
| G2 规程外新点组 | ≥12 场景 | surrogate 判定高危险 + 规程矩阵未覆盖 + 外推幅度 ≤20% 参数空间 + 不确定度低于阈值 | 探索：验证 surrogate 外推与生成场景的真实有效性 |

### F.3 执行前置条件

- **车辆准备**：与历史测试同配置（整备质量、轮胎、制动系统磨合按 C-NCAP L.5.3/L.5.4 Conditioning 流程执行）
- **传感器/同步**：Motion Pack + 相对运动链路 + Time-tolerance 触发通道与历史数据同链路；Synchro 校准先行（参照既有 Turning/Synchro Calibration 流程）
- **参数化转换**：生成场景参数 → ABD .spec（速度/路径/触发容差），转换脚本需经 1 个规程内场景试跑验证
- **安全中止条件**：按场地安全规程（试验驾驶员接管条件、设备急停、目标载体保护）

### F.4 记录与判定

- 每场景 ≥2 次有效重复；记录通道与历史 .txt 同 schema（100 Hz）
- 结果判定：碰撞 / AEB 触发时刻 / FCW 触发时刻 / minTTC / 最小间距；与 surrogate 预测逐一配对
- 一致性报告：G1/G2 分组报告碰撞判定一致率与各项 MAE + 95% CI

### F.5 反哺与迭代

- 新实测点并入 surrogate 训练池（标注采集批次），重训并比较前后 LOBO 与外推误差
- 每轮补测后更新不确定度模型；G2 组中 surrogate 失配点优先进入下一轮送测

### F.6 风险与降级

- 若场地/车辆窗口不足：首批缩减至 G1+部分 G2，并在论文中明确标注验证规模限制
- 兜底方案：以 holdout ABD 历史数据做"准实车验证"，论文中改述为"基于实测数据的外部验证"，不声称新补测闭环
