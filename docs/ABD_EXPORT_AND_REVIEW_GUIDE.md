# ABD 同步目标通道导出与无接管复核

## 1. 现有 TXT 是否漏导了目标 command/actual

没有。Robot Controller 中从 `Results` 选择运行并执行 `Export Data`，通道列表勾选“全部”就是正确
操作。`RC Software Manual.pdf` 的 Export 章节（PDF 第 143–148、183–186 页）说明，ASCII/TXT 只会
导出本次运行已经采集并可用的通道；勾选全部不能补回测试时没有采集的信号。

现有 14-BZ3X smoke 导出已经在相同 `Time` 数据行中包含：

- 参考侧：`Object 1 reference X position`、`Object 1 reference Y position`；
- 实测侧：`Object 1 actual X (front axle)`、`Object 1 actual Y (front axle)`、
  `Object 1 forward velocity (ref point)`、`Object 1 forward acceleration`；
- 同步质量侧：Object/Subject time、status、incoming extrapolation time、相对位置和 TTC 等。

因此不需要重新导出这些历史运行。reference 与 actual 的物理参考点可能不同，算法必须先估计固定二维
偏置，再把去偏置后的残差作为动态轨迹跟踪误差；不能把固定偏置直接解释为平台跟踪误差。

## 2. 新试验如何保证目标通道被采集

1. 在 `Setup > Transducer Connections > Synchro Setup` 配置 Subject 与 Tracker/Object；Subject 端
   Activate Synchro，Tracker 端 Connect。多目标试验使用 Advanced multi-object 配置。
2. 在 `Setup > Data Capture & Export > Data Capture` 检查采样频率和需要采集的 Object/Tracker
   通道；推荐保持 100 Hz。
3. 在 `Setup > Data Capture & Export > Export` 保留上述 reference、actual、status、time-error 和
   extrapolation 通道。测试后在 `Results > Export Data` 勾选全部即可。
4. 如果需要 LaunchPad 电机/DAC 等低层命令，应在对应 LaunchPad/Tracker 的 Robot Controller 中
   预先启用并保存这些通道。Full Synchronization 与 Subject/Tracker FTP 可用于测试后传输各系统
   文件，但不能生成当时未采集的低层信号。

车辆未连接 CAN 时，TXT 中不会出现车辆 ECU 的 AEB request/active/state。当前目录的 4,121 份数据
导出中没有原始列名直接标识 AEB/FCW 或 `CAN User Defined`，但 companion `.spec` 中存在 201 条
Time Tolerance Trigger 到 `CAN User Defined 1/2` 的映射。结合测试团队关于 AVAD3 的通道约定，
可从对应的 `Time tolerance X (within tolerances)` 首次 0→1 恢复
`T_FCW_audio_observed`；全目录有 188 条有效上升沿，覆盖 10 个车型目录。这个时刻是 AVAD3 对车内
声音报警的外部观测，不是 ECU 内部 FCW request。历史数据仍不能恢复 ECU AEB 触发时刻。

## 3. 无接管候选筛选算法

运行：

```bash
/home/shuai/.venvs/scenario-gpu/bin/python scripts/screen_abd_no_takeover.py
```

算法先在实测纵向加速度中寻找持续低于 −1 m/s²的主减速段，再回溯至同一段首次低于
−0.3 m/s²的时刻，并将其保存为 `observed_braking_onset`。事件窗内 `BR Command` 非零的运行归入
机器人制动；BR Command 为零的运行再与四条人工确认无接管记录比较以下运动特征：

- 制动前后横向速度增量、偏航角速度增量和航向变化；
- 纵向速度回升、非单调速度步比例、减速脉冲数和 jerk；
- 目标 reference/actual 是否动态可用及去固定偏置后的轨迹残差。

四条样本不足以训练或验证监督分类器，因此 `prototype_similarity_0_1` 只是人工复核优先级，不是
无接管概率。候选按照车型优先、车型×场景优先、相似度补齐的顺序组成 40 条队列。

## 4. 现场复核只填哪一列

打开 [人工复核表](../runs/20260912_abd_no_takeover_screen/manual_intervention_review.csv)，只修改最后一列
`driver_intervention`：

- `none_confirmed`：Check Paths、Forward velocity 和 Lateral velocity 综合判断无人工接管；
- `manual`：存在明显人工制动或避让接管；
- `unknown`：无法可靠判断，保持不变。

`none_confirmed` 仍表示“没有观察到人工接管特征”。在没有独立踏板/压力通道的情况下，直线人工制动
可能与 AEB 刹停相似，因此不能把该值升级为 ECU AEB 真值。人工确认后，可以把
`observed_braking_onset` 用作实测制动响应起点。按测试团队提供的 0.15–0.35 s 工程先验，算法同时
给出 `T_AEB_proxy ∈ [T_dec_03-0.35, T_dec_03-0.15]` 和名义值 `T_dec_03-0.25`。这些列是倒算的
代理时刻，不能命名为实测 `T_AEB_req`。
