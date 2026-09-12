# ABD AEB 数据集踏板签名全量筛选（892 BR-zero + 268 robot-flagged AEB-path run）

日期：2026-09-12。脚本：`scripts/screen_abd_aeb_pedal_signature.py`（判据复用
`scripts/validate_abd_pedal_signature.py` 的 classify，阈值单一来源）。
输入：`runs/20260912_abd_no_takeover_screen/screening.csv` 1,258 行中
`br_zero_observed_braking` 892 条 + `robot_channel_active` 268 条（98 条无事件跳过）；
逐 run 流式解析（EXACT_CHANNELS）、sha256、事件窗重算（−1 m/s² sustained + 回溯 −0.3）、
窗前 2 s 基线、D4F4 附加窗前 5 s cmd 检查。**不改任何 driver_intervention 标签**；
foot/裁决/中途接管只进 REVIEW_QUEUE.csv。

## 1. 总结果（1,160 run）

| screen_class | n | 说明 |
|---|---:|---|
| aeb_pedal_dragged | 881 | 耦合型 AEB：行程>2 mm + 压缩≤50 N（张力为硬证据加分） |
| aeb_pedal_static | 16 | 解耦型 AEB：行程≤2 mm |
| **AEB 签名合计** | **897** | 77.3% of 1,160 |
| foot_compression | 33 | 人脚：压缩>50 N 且无张力 |
| drag_then_foot_adjudicate | 6 | 张力+大压缩同窗 → 操作员裁决 |
| robot_braked_main_window | 217 | 主减速段 BR Command 活跃（机器人制动，排除） |
| robot_joined_mid_window | 7 | cmd 在 onset +0.5 s 后才出现（见 §5） |

## 2. 两项对 screen 分类的修正（本轮实证）

1. **BR Command 噪声误标**：268 条 robot_channel_active 中 44 条的事件窗+窗前 5 s
   cmd 均≤0.5 EU，但其 `robot_max_abs_json` 显示 Codex 以 BR Command event_max_abs
   0.01–0.06 EU（传感器噪声）判活跃。这 44 条已按 BR-quiet 规则重分类
   （`screen_flag_noisy=1` 留痕）：25 dragged + 3 static + 15 foot + 1 裁决。
   → **真 BR Command 零阈值不可用，须 ≥0.5 EU**。
2. **cmd 中途加入拆分**：原 robot_braked_main_window 中 7 条的 cmd 在减速已进行
   0.5 s 以上后才出现，单独成类并逐条做了 0.1 s 时序核验（§5）。

## 3. 车型踏板架构（AEB 签名集内）

| 车型 | dragged | static | 张力出现 | 耦合形态 |
|---|---:|---:|---:|---|
| 14-BZ3X | 238 | 2 | 217/240 | 耦合，张力显著（−16~−22 N 量级） |
| 9-BZ7 | 374 | 6 | 0 | 耦合，无张力 |
| 8-huajingS | 108 | 1 | 5 | 耦合，偶发张力 |
| 15-TANG | 30 | 0 | 0 | 耦合，无张力 |
| 13-xiaopengG9 | 25 | 4 | 25 | 耦合，张力最强（−66~−85 N） |
| Guang_Qi_A66 | 35 | 0 | 13 | 耦合，部分张力 |
| Guang_Qi_S9 | 34 | 0 | 0 | 耦合，无张力 |
| XPengP7+ | 27 | 3 | 1 | 耦合为主 |
| Guang_Qi_E8 / 6-Bao5 | 5 / 5 | 0 | 2 / 0 | 耦合（n 小） |

张力（载荷计被拉着走的硬证据）按车型二值化清晰：G9/BZ3X/A66 有、BZ7/TANG/S9 无——
与"载荷计安装几何因车而异"一致；**static（解耦型）总共只有 16 条**，
本车队以耦合型为主（踏板行程不能当驾驶员检测器的直接原因）。

## 4. operator 标签一致性（44 条已复核 run）

| operator intervention_class | 签名分类 | n |
|---|---|---:|
| none_confirmed | aeb_pedal_dragged | 41 |
| none_confirmed | aeb_pedal_static | 1 |
| none_confirmed | drag_then_foot_adjudicate | 1（V13_T332_R2，已知裁决项） |
| manual_after_aeb_stop | aeb_pedal_dragged | 1（AEB 段签名，与截尾规则一致） |

**44/44 与操作员复核结论无矛盾**（唯一分歧即已知的 V13_T332_R2 裁决项）。

## 5. D4F4 扫描结论：0 条命中；7 条"中途接管"逐条定性

- 判据：onset 前 5 s 内小幅 cmd 轻踩脉冲（0.5<|cmd|≤30 EU）+ 主减速段 cmd≡0 +
  主段 AEB 签名。**全语料 0 条命中**——本批历史数据不含 BR 轻踩→AEB 的 D4F4 模式
  （D4F4 规则保留在脚本中，未来 C-NCAP 2024 补测可直接复用）。
- 7 条 robot_joined_mid_window（按 per_run_signature.csv：V3_T116/T119/T140/T143、
  V14_T494/T1804、V15_T5261_R9）逐条时序核验（全通道时间轴 + .spec + §8 碰撞结局）：
  - **5 条 = 尾部小 cmd 无力学效应（运行收尾段伺服空闲/保持 trim，非干预）**：
    V3_T116_R1（干净 AEB 停车，min 距离 1.45 m）、V3_T119_R1（同型，2.48 m）、
    V3_T140_R1、V3_T143_R1（AEB 中途完全释放后**碰撞**，见下方修正记录）、
    V14_T494_R1（制动中距离过零 = 目标越过，v@cross 32.5 kph，非停车压入）。
    共同证据：cmd ≤3.9 EU 期间踏板行程恒定（40.75/45.27/20.41/26.33 mm 不动）、
    载荷计 −0.66~+0.83 N ≈0、.spec `UseBrakeRobot=False`、全程 Motion Going BR=0 /
    BR start=0 / BR test=0、AR Command 同期归零。机器人真推踏板必有压缩+行程
    （对照 V15_T5258_R1：cmd −51 + 106 N）。主减速段仍为干净 AEB（拖拽签名+主段
    cmd≡0），**截尾到 cmd 起点后可用**（保守规则，待操作员确认）。
  - **2 条 = 机器人施力/异常，排除待操作员确认**：
    V15_T5261_R9（cmd 43.3 与压缩 17–70 N 同现、行程近零 0.004 mm——cmd 与力同量级
    增长更像机器人施力，但结局窗内距离在 onset 前已过零、疑 tracker 换目标，交裁决）；
    V14_T1804_R4（onset 前踏板已在 −88 mm 且车辆在加速、+1.15 s cmd −88.1 与行程数值
    相同疑 cmd 通道记位置，结局为 4.3 kph 低速接触——单独裁决）。
  - 注：V15_T5258_R1（cmd −51.1 + 压缩 98.4 N = 机器人推踏板标准形态）属
    robot_braked_main_window（cmd +0.2 s 即活跃），不在本类 7 条之内——初版 §5
    误将其列入，以 per_run_signature.csv 为准。
- **V3_T140/T143 修正记录（2026-09-12，操作员质询触发）**：初版曾把这两条写成
  "AEB 释放后机器人尾部小指令收尾停完"。操作员指出"机器人不会接手刹停，只有司机会"。
  全通道复核证实**机器人未接手**（上述三重证据）、**驾驶员也未踩**（载荷计 ≈0）；
  把车停住的是**碰撞**——相对纵向距离分别于 +2.81 s（接触时 6.0 kph，压入 −0.339 m）
  与 +1.91 s（接触时 19.8 kph，压入 −1.394 m）过零，T143 的 −12 m/s² 与距离转负严格
  同步 = 碰撞动力学。真实形态：AEB 触发→制动→完全释放（减速 ≈0 滑行逼近）→接触
  停车。这两条是 **AEB 中途释放致碰撞的直接观测**（CBNAO 60 kph、CSFAO 60 kph
  各一），比初版解释更有价值；尾部 −3.05/−3.94 EU cmd 与接触几乎同时出现，为运行
  收尾段伺服空闲信号。

## 6. 去重（sha256）

- 1,160 扫描 run → 914 unique；246 组完全重复（492 run 涉及）。
- AEB 签名 897 run → **707 unique**（190 条冗余副本）。
  unique 按车型：BZ3X 240、BZ7 190、huajingS 109、A66 35、S9 34、TANG 30、
  XPengP7+ 30、G9 29、E8 5、Bao5 5。
- foot+裁决 39 run → 27 unique。9-BZ7 冗余最重（360→183，约半数为重复导出）。
  **下游校准必须按 sha256 去重**（否则伪重复）。

## 7. AEB 筛选数据集（aeb_dataset_v0）

定义：`per_run_signature.csv` 中 screen_class ∈ {aeb_pedal_dragged,
aeb_pedal_static}，按 sha256 去重（同组保留路径字典序第一条，组内明细另列）。
**707 unique run / 10 车型 / 14 场景标签**；onset 速度 8.4–80.5 kph
(p25 20.4 / 中位 30.5 / p75 40.5)；场景分布 AEB 314、CCRS 111、CPTA 111、CPLA 79、
CCFT 46、SCP 45、CBLA 44、CPNCO 44、CBNA/CPNA/CCRM 各 30、CPFA 6、CBFA 5、OTHER 2
（去重前计数）。派生规则：per_run_signature.csv 过滤 screen_class ∈
{aeb_pedal_dragged, aeb_pedal_static} → 按 sha256 去重（同组保留路径字典序第一条；
190 条冗余副本列于 aeb_dataset_v0_duplicates.csv）→ aeb_dataset_v0.csv（707 行，
列：run/vehicle/scenario/sha256/signature/onset_s/onset_speed_kph/travel_win_mm/
force_min_win/force_max_win/cmd_abs_max_win/tension/screen_flag_noisy/operator_label）。

**边界**：签名 = 力学事实（BR 无指令 + 载荷计无主动施力 + 踏板被拖/不动），不是
AEB ECU 信号观测；无车辆 CAN 边界不变。foot/裁决/中途接管未入集。张力+压缩同窗、
阈值边缘（如 V15_T5222_R1 fmax 50.88 N、fmin −8.28 N）共 6 条在队列中等操作员。

## 8. 碰撞结局扫描（scripts/scan_abd_contact_outcome.py，1,160 run 全量）

事件窗 ±2 s 内 Relative longitudinal distance 最小值分类；**过零类再按"过零时车速
≤25 kph 且 |min|≤3 m"分为真接触与越过**（纵向代理判据，横向几何未查，已在脚本
docstring 标注边界）：

| screen_class | contact_and_stopped | passed_or_swept | near≤0.5 m | clear |
|---|---:|---:|---:|---:|
| AEB 签名 897 | **66 (7.4%)** | 105 | 30 | 696 |
| foot 33 | 8 | 11 | 1 | 13 |
| robot_braked 217 | 7 | 48 | 2 | 160 |
| mid_window 7 | 3 | 2 | 0 | 2 |

- contact_and_stopped 压入深度 −0.018~−2.683 m（物理压溃/推挤量级）；passed_or_swept
  |min| 3.1~632 m、过零时车速中位 0.1 kph = 目标被甩到车后（车继续行驶/爬行越过）或
  目标走过停住的车，**不是压溃**。若把全部过零当 contact（初版口径 171 条）会高估
  碰撞率近 3 倍——深度分布双峰（−0.5 m 与 −6 m 两个量级）是判据依据。
- AEB 签名集内 66 条真接触 + 30 条 ≤0.5 m 近接触 = "踏板签名成立但 AEB 未能避免
  接触"的难例子集，是 observed-braking-response 分析的重点素材（下游须 sha256 去重
  后重算分布）。
- V3_T140/T143 落在 contact_and_stopped（§5 修正记录）。

## 9. 操作员复核队列（REVIEW_QUEUE.csv，46 条）

1. **foot_compression 33**（unique ~20；24 条在 9-BZ7）：压缩 50–301 N、
   foot_pre_event 全 0（脚为事件中突然加入，符合"AEB 过晚、驾驶员补刹"形态）；
   其中 15 条属 §2.1 噪声误标组（从未进过 892 池）。
2. **drag_then_foot_adjudicate 6**：13-G9 × 5（V13_T518_R3/T308_R2/T332_R2/T341_R1/
   T365_R1，张力 −66~−85 N 与 66~290 N 压缩同窗，疑 AEB 先作动驾驶员后补脚）+
   V15_T5222_R1（阈值边缘）。
3. **robot_joined_mid_window 7**：§5 已修正定性——5 条尾部 cmd 无力学效应（截尾后
   可用）+ 2 条机器人施力/异常排除（V15_T5261_R9、V14_T1804_R4）。

## 10. 未解决 / 下一步

- 46 条队列待操作员裁决（机器只给签名，不改标签）。
- 阈值（2 mm/50 N/−5 N/0.5 EU/30 EU/0.5 s）跨速度档未再标定；载荷计几何因车而异。
- 碰撞结局为纵向代理：横向几何未查，passed_or_swept 不能排除擦边接触，
  contact_and_stopped 不能排除目标先被撞开后车再停；如需精判用目标 lateral 通道。
- V14_T1804_R4 的 BR Command 通道语义疑为位置记录（cmd −88.1 ≈ 行程 −88.09）待查。
- test_id 非唯一 + 246 组 sha256 重复：所有下游消费必须 sha256 去重。
- D4F4：历史数据 0 命中；若 C-NCAP 2024 补测做 D4F4，直接用脚本内嵌规则扫。
- aeb_dataset_v0 进入 observed-braking-response 分析的资格仍受 P3 归因撤回边界约束
  （无 AEB request 信号；response_delay 拆分未完成）。
