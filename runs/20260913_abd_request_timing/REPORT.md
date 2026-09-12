# aeb_dataset_v1 + AEB request/active 代理 + ABD 支持的敏感性配置

日期：2026-09-13。操作员三项决策（本报告即执行记录）：
1. 46 条复核队列记录（34 个唯一文件哈希；foot 33 / 裁决 6 / mid-window 7）
   **永久排除**，且在 v0=707 形成前已经排除，不再等待人工复核；
   mid-window 5 条"截尾后可用"的提议撤销。
2. 机器分类条目的轻脚力残余风险**直接剔除**（"我也不能完全排除是否有人干预"）。
3. 无车辆 CAN 不阻塞敏感性分析：从观测 onset **倒推 0.15~0.35 s** 合成
   AEB request/active 代理。该先验不能验证真实 ECU 时延或 sim-real 偏差。

脚本：`scripts/build_aeb_dataset_v1.py`、`scripts/derive_aeb_request_timing.py`。
独立核对与修正记录：`CODEX_AUDIT.md`。

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
- `observed_response_onset_s` = 观测减速 onset；
- `aeb_request_proxy_s_d150/d250/d350` 与对应的 `aeb_active_proxy_*`
  = onset − {0.15, 0.25, 0.35}。request 与 ECU active 因无 CAN 无法分开，故代理列
  显式相同；它们不是观测列；
- 诊断列：peak 减速度、10 s 内停车判定（670/680 停稳）、等效恒定减速度
  a_eff=(v₀²−v_end²)/(2∫v dt)、onset TTC（覆盖 677/680）、margin 代理
  = onset_TTC − v₀/|peak|。

**边界（必须随数据携带）**：request/active proxy 列是**合成值**，不是测量；
其中 1 条在 0.35 s 档的代理时间早于日志起点，保留为显式外推而未截断。margin 中位
0.066 s、p5 −0.348 s，只作描述。

## 3. 校准配置（abd_derived_v2_sensitivity.json，通过 load_perturb_config 校验）

| 参数 | 域 | 来源 |
|---|---|---|
| brake_deceleration | **U(5.801, 8.951)** m/s² | 612 条按纵向距离代理排除 contact 的停稳响应，标准线性 p5–p95 |
| aeb_actuation_delay | **U(0.15, 0.35)** s | 操作员工程先验（合成，非测量） |
| controller_preview_delay | **0.25 s 固定** | 冻结仿真控制器的名义预瞄假定，与执行时延分开 |
| action_delay_steps | integers 0–2 | NPC 动作延迟假定；每步 0.1 s，即 0–200 ms |
| target_accel_scale | U(0.85, 1.15) | 保留假定（NPC 侧，无执行误差映射） |

- 旧 abd_supported_v1（10 条，U(5.455,8.174)）与新域不采用同一种统计规则；新域
  使用 612 条的标准线性 p5–p95。它可作为更大样本支持的数值敏感性范围，不能据此
  宣称概率分布或总体制动能力得到精确估计。
- **下尾处置留痕**：10 条 equiv<4.5 m/s²（占 612 的 1.6%，CPLA/CPTA 为主）为分段
  间歇制动摊薄形态（peak −12~−16 但全程均摊低），不是弱执行器；完整极值 1.405
  记录于 JSON，不入采样域。10 条样本时用极值、612 条时用分位数，理由：极值包络
  在大样本下被单个离群主导。
- peak 减速度（全量 680，含接触）：−5.67~−20.33、中位 −11.78——与 10 条时代的
  −9.1~−11.4 相比，宽尾来自高速档与接触类，仅诊断用。

## 4. 使用边界

- 仿真已把冻结控制器的 `controller_preview_delay` 与执行侧
  `aeb_actuation_delay` 分开；旧实验缺少新字段时仍回退到原 `response_delay`，从而
  保持回放语义。新配置可被 `perturb_spec` 消费，但 **0.15–0.35 s 仍是注入先验**。
- 612 条池化样本来自 10 个车辆目录，且最大两个目录占 388/612；U(5.801,8.951)
  是敏感性包络，不是按车型均衡的车队概率分布。
- v1 数据集边界继承 v0：踏板签名是力学事实非 AEB ECU 观测；680 条中仅 43 条有
  操作员确认标签，其余 637 条为机器筛选的响应候选。论文不得写成“680 条 ECU
  确认 AEB”。
- `REVIEW_QUEUE.csv` 保存预先排除的 46 条队列记录；`aeb_dataset_v1_removed.csv`
  保存从 v0 追加剔除的 27 条。两者不是同一个集合，也不能相加后再次从 707 扣除。
