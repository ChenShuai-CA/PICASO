# aeb_dataset_v1 + 倒推 AEB request/active 时间戳 + 时延校准配置

日期：2026-09-13。操作员三项决策（本报告即执行记录）：
1. 46 条复核队列（foot 33 / 裁决 6 / mid-window 7）**永久排除**，不再等待人工复核；
   mid-window 5 条"截尾后可用"的提议撤销。
2. 机器分类条目的轻脚力残余风险**直接剔除**（"我也不能完全排除是否有人干预"）。
3. 无车辆 CAN 不构成阻塞：从观测 onset **倒推 0.15~0.35 s** 合成 AEB request 时刻，
   跑通 request-to-response 时延校准方法。

脚本：`scripts/build_aeb_dataset_v1.py`、`scripts/derive_aeb_request_timing.py`。

## 1. aeb_dataset_v1 = 680 unique

v0（707）基础上剔除 27 条，规则与验证依据：

| 规则 | n | 依据 |
|---|---:|---|
| foot_pre_event=1 | 12 | 窗前 2 s 踏板在动；操作员确认 AEB 集 44/44 全为 0 |
| 无张力且 force_max_win ≥30 N（无操作员确认标签） | 15 | 确认集主力带中位 12.7 / p90 ~25 N 之上；**操作员确认条目豁免**（剔除目的即"无法人工排除"，已人工排除者不剔；确认集内该区间 2 条反例 V2_T76_R1 37.6 N / V2_T1170_R2 32.5 N 因此保留） |

- v1 构成：dragged 672 + static 8；操作员确认 43 条；sha256 全唯一。
- 结局分布（join contact_outcome.csv）：clear 516 / passed 84 / **contact 56** / near 24。

## 2. 倒推时间戳（timing.csv，680 行）

每条 run 重算事件窗后输出：
- `active_s` = 观测减速 onset（日志可观测的作动生效点）；
- `request_s_d150/d250/d350` = onset − {0.15, 0.25, 0.35}（三档敏感性带，非采样隐藏）；
- 诊断列：peak 减速度、10 s 内停车判定（670/680 停稳）、等效恒定减速度
  a_eff=(v₀²−v_end²)/(2∫v dt)、onset TTC（覆盖 677/680）、margin 代理
  = onset_TTC − v₀/|peak|。

**边界（必须随数据携带）**：request 列是**合成值**（注入先验倒推），不是测量；
margin 中位 0.066 s、p5 −0.349 s（部分速度档触发晚于理想制动点），只作描述。

## 3. 校准配置（abd_derived_v2_sensitivity.json，通过 load_perturb_config 校验）

| 参数 | 域 | 来源 |
|---|---|---|
| brake_deceleration | **U(5.772, 8.952)** m/s² | 612 条干净停车（非接触）等效恒定减速度 p5–p95；contact 剔除（减速度含碰撞贡献） |
| response_delay | **U(0.15, 0.35)** s | 操作员倒推先验（合成，非测量） |
| action_delay_steps | integers 0–2 | 保留假定（≤40 ms 低于 100 Hz 分辨） |
| target_accel_scale | U(0.85, 1.15) | 保留假定（NPC 侧，无执行误差映射） |

- 旧 abd_supported_v1（10 条，U(5.455,8.174)）与新域对比：612 条域整体上移且更窄
  （p5 5.77 vs 5.46 下界、p95 8.95 vs 8.17 上界）——更大样本下制动强度包络收紧。
- **下尾处置留痕**：10 条 equiv<4.5 m/s²（占 612 的 1.6%，CPLA/CPTA 为主）为分段
  间歇制动摊薄形态（peak −12~−16 但全程均摊低），不是弱执行器；完整极值 1.405
  记录于 JSON，不入采样域。10 条样本时用极值、612 条时用分位数，理由：极值包络
  在大样本下被单个离群主导。
- peak 减速度（全量 680，含接触）：−5.67~−20.33、中位 −11.78——与 10 条时代的
  −9.1~−11.4 相比，宽尾来自高速档与接触类，仅诊断用。

## 4. 使用边界

- 本配置解锁 request-to-response 校准**方法闭环**（observed onset → 合成 request/
  active → response_delay 域 → perturb_spec 消费），response_delay 数值本身是
  注入先验，**任何下游报告不得把它写成 AEB 时延测量**。
- v1 数据集边界继承 v0：踏板签名是力学事实非 AEB ECU 观测；680 条中仅 43 条有
  操作员确认标签，其余为机器分类（剔除规则已按保守原则收紧）。
- 46 条排除队列明细保留于 REVIEW_QUEUE.csv 与 aeb_dataset_v1_removed.csv（27 条）
  供追溯；排除状态不因未来复看而自动恢复，如恢复须操作员逐条确认。
