# ABD AEB/FCW 与目标平台补采协议

## 1. 已核实的 ABD 软件行为

1. `Post Processor User Guide.pdf` PDF 第 19 页说明：Post Processor 先寻找 VUT 纵向加速度低于
   −1 m/s²的减速段，再回溯到同一减速段首次越过 −0.3 m/s²的位置作为显示的 AEB event。
   同页明确列出 vehicle pre-brake、driver intervention 和其他外部因素都可能满足该规则。
   因此这个 event 是**减速响应起点检测量**，不是 ECU AEB request/status 的直接观测。
2. `AN-6092 Euro NCAP AEB Car-to-Car 2020 Application Guide.pdf` PDF 第 10–11 页说明：FCW
   receiver 信号可由 DBC 导入 RC Software，映射到 `CAN User Defined`；Time Tolerance Trigger
   使用 0.5–1.5 捕捉二值 true，100 Hz 下 true time 为 0.01 s。AEB 测试也可配置该 receiver 以
   取得附加数据。
3. `AN-6157.01 China NCAP 2024 VRU (VUT & LP) Application Guide.pdf` PDF 第 14–15 页说明：
   C-NCAP Special Group 用 TTT3 检测 FCW；从 Bus 0 导入 DBC，将相应二值消息设为
   `CAN User Defined 1` 并设为 Live。第 27 页说明 VUT 与 LaunchPad 应在同一无线网络上分别
   Activate/Connect Synchro。

## 2. 当前历史数据状态

现有 24 条导出没有 AEB/FCW、驾驶员制动或目标平台执行通道。实际测试允许驾驶员在 AEB 过晚时
人工踩刹车，因此 `BR Command == 0` 只能排除制动机器人命令，不能区分车辆 AEB 与驾驶员制动。
此前 10 条 `AEB_by_elimination` 记录全部改为 `unknown_AEB_or_driver_brake`，在逐 run 找到人工介入
记录或补充信号前不得用于 AEB 制动参数校准。

## 3. VUT 必采通道

所有通道采用同一 RC/Synchro 时间轴，目标采样率至少 100 Hz；同时保存原始 CAN frame、DBC 版本、
RC group/spec 和最终导出表。

| 类别 | 最低必采通道 | 用途 |
|---|---|---|
| AEB/FCW 状态 | FCW active；AEB request/active/state；requested deceleration 或 brake request（若 ECU 提供） | 直接确定系统请求与状态跃迁；信号名和枚举值必须按车型 DBC 单独登记 |
| VUT 响应 | 时间、纵向速度、纵向加速度、位置/姿态、相对距离或 TTC | 检测 −0.3/−1 m/s²响应点并计算停车距离、碰撞余量 |
| 驾驶员输入 | 制动踏板位置或力、制动灯开关、主缸/轮缸压力；另加人工接管按钮/标记 | 确定 `T_driver`；按钮只作辅助，不能代替踏板或压力实测 |
| 机器人输入 | BR Command/Position/Force、BR test/abort 状态；AR Command/Position | 确定 `T_BR_abort`，排除机器人与油门控制贡献 |
| 同步与配置 | RC Time、Synchro/PoI 时间、run ID、车辆/软件/DBC 版本 | 跨 VUT、LaunchPad/SPT 和外部采集器对齐 |

配置步骤：在 `Setup > Data Capture & Export > CAN Input` 导入该车型 DBC，把 FCW 二值信号映射到
`CAN User Defined 1` 并设为 Live；同时把 AEB request/state/decel request 和驾驶员制动相关信号
作为普通 CAN 数据通道导出。Bus 0 用于机器人触发。若车型 CAN 不提供 AEB request/status，只能把
该车型标记为“响应起点可观测、请求时刻不可辨识”，不能估计 trigger-to-response delay。

### 无车辆 CAN 的实际边界

当前及计划测试均不连接车辆 CAN，所以 `T_AEB_req`、AEB state 和 requested deceleration 不可获取。
项目不再把 AEB request-to-response delay 设为可校准量。可行替代是：

1. 使用 Post Processor 的 −1/−0.3 m/s²规则定义 `observed_braking_onset`；
2. 通过独立模拟量采集制动踏板位移/力或制动压力，并记录安全接管标记，得到 `T_driver`；
3. 只对确认没有人工/BR 接管的完整减速事件，或 `T_driver` 之前足够长的未污染窗口，拟合响应曲线；
4. 若可增加外置 AVAD audio/light receiver，可独立记录 FCW 并计算 FCW-to-braking interval，
   但仍不得将其命名为 AEB request delay。

在该条件下，“完整校准”应重新定义为**观测制动响应 + 人工接管删失 + 目标平台执行误差校准**，
而不是 ECU AEB 内部请求时延校准。

## 4. NPC/目标平台必采通道

VUT 和 LaunchPad/SPT 通过 Synchro 建立共同时间基准，并分别保留各自的原始 run/export：

- 命令侧：Synchro Start/PoI、目标路径、目标速度/加速度命令、平台启动/停止状态；
- 实测侧：motion-pack/RTK 的时间、x/y、速度、纵横向加速度、航向、定位有效性；
- 质量侧：通信状态、跟踪误差、abort/安全状态、目标型号和控制软件版本。

根据命令与实测序列估计 `T_target_actual - T_target_command`、速度/加速度增益、稳态误差和条件内
抖动。当前 env 的 `action_delay_steps` 可由命令到实测响应的延迟换算为 20 ms step；
`target_accel_scale` 可由实际与命令加速度的稳健回归斜率估计。行人平台与遮挡车平台必须分开拟合，
不能继续共用一个未经验证的缩放分布。

## 5. 统一时刻与裁决规则

每条 run 至少输出：

- `T_FCW`：FCW 原始二值信号有效跃迁；
- `T_AEB_req`：AEB request/active 的车型定义跃迁；
- `T_dec_03`：属于同一显著减速段的首次 −0.3 m/s²越界；
- `T_dec_1`：首次 −1 m/s²越界；
- `T_driver`：人工制动踏板/压力首次越过车辆静息噪声阈值；
- `T_BR_abort`：BR abort/command 首次有效；
- `T_target_cmd`、`T_target_actual`：目标命令和实测运动起点。

优先级裁决：

| 条件 | 用途 |
|---|---|
| AEB request/status 明确，且整个制动事件无 driver/BR intervention | 可用于 AEB 时延、建立过程、减速度和停车距离 |
| AEB 明确，驾驶员稍后介入 | 只使用 `[T_AEB_req, min(T_driver,T_BR_abort))`；窗口不足以识别参数时按右删失处理，只保留“触发过晚/安全接管”结局 |
| 驾驶员早于或同时于 AEB，或时刻不可区分 | 不进入 AEB 动力学校准 |
| 只有 −0.3/−1 m/s²加速度事件，没有 AEB request/status 和可靠人工介入标记 | 仅记为 observed braking event，不标 AEB |
| FCW 有直接信号而 AEB 无直接信号 | 可校准 FCW 时刻与 FCW→减速响应间隔；不得命名为 AEB request delay |

派生量须分别保存，不合并成一个“response delay”：

- sensing/decision delay：`T_AEB_req -` 场景几何触发参考时刻；
- actuation delay：`T_dec_03 - T_AEB_req`；
- build-up time：`T_dec_1 - T_dec_03` 或到达稳态减速度的时间；
- FCW lead：`T_AEB_req - T_FCW`；
- driver takeover latency：`T_driver - T_AEB_req`，只作安全与删失信息。

当前 `scenario_lab.response_delay` 同时进入停车触发距离补偿和制动队列，混合了控制器预判与执行延迟。
在完整校准前，应把它拆为 trigger policy/margin、actuation delay 和 brake build-up；否则即使补到直接
AEB 信号，也不能一一映射。

## 6. 最小补采矩阵

每车型至少覆盖两个速度档、single/dual 对应的关键规程场景，并重复足够次数以同时获得：

1. 无人工介入且 AEB 完整制动的有效 run；
2. AEB 过晚后人工/自动安全接管的删失 run；
3. 无 AEB 响应 run；
4. 目标平台 command/actual 同步日志。

先做每车型 3–5 条通道完整的 smoke run，核对上述七个时刻能否从导出文件自动恢复，再扩大采集。
正式数据表必须包含 `driver_intervention={none,manual,robot,unknown}`、各时刻、信号来源、DBC hash、
同步误差和逐 run 采用/排除理由。
