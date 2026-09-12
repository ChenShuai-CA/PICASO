# P3.1/P3.2 复核与 ABD-supported 扰动域（2026-09-12）

本报告替代同目录中早先将峰值减速度直接映射到仿真恒定制动、并将 TTC 余量映射到
`response_delay` 的解释。历史产物 `abd_calibrated_v1.json` 保留用于追溯，但已废止；正式实验只使用
`abd_supported_v1.json`。

## P3.1 证据结论

`manual_review.csv` 24 条均已裁决并给出手册页码。10 条入选运行的原始文件 SHA256 与校准表一致，
test_id 和 SHA256 均为 10/10 唯一；事件窗内 BR Command 的绝对值为零。结合 ABD 规程中的 AR 油门
保持和驾驶员不干预要求，这 10 条可标记为 **AEB-attributed by elimination**。由于日志没有直接
AEB request/status 通道，结论不是直接激活证明；驾驶员违规介入仍是残余风险。

## 可用于 scenario_lab 的参数映射

| 参数 | P3.2 采用域 | 证据等级与边界 |
|---|---:|---|
| `brake_deceleration` | **U(5.455, 8.174) m/s²** | ABD 支持的经验敏感性包络。按 `a_eff=(v_onset²-v_end²)/(2∫vdt)` 换算，匹配 env 的恒定减速度及停车距离语义；不是车型总体概率分布。 |
| `response_delay` | U(0.1, 0.4) s | 保留原假定。观测 margin-time 为 0.055–0.386 s，但混合了触发策略、几何和制动建立过程，不能当成触发到输出的时延。 |
| `action_delay_steps` | {0,1,2} | 保留原假定；没有 AEB request 通道，100 Hz VUT 日志不能辨识该执行抖动。 |
| `target_accel_scale` | U(0.85,1.15) | 保留原假定；这是 NPC 侧参数。更正（2026-09-12）：VUT 日志经 Synchro 含目标端 Head tracker reference/actual 执行通道（CCFT/CPTA 有值、CCRs 零值与规程一致，见 MANUAL_EVIDENCE.md E9）；v1 仍保留假定（运动目标 run 仅 2/10，位置差→加速度缩放需专门建模），该通道组列为未来版本候选数据源。 |

原峰值减速度范围 9.128–11.419 m/s² 仍作为事件平台诊断量保存在表内，但峰值不能直接替代 env
从触发后持续施加的恒定减速度。修正后的等效范围与先前假定 U(5.5,8.0) 基本重合，因此数据不支持
“先前扰动整体低估 ego 制动强度”的说法。

## 数据、稳健性与限制

- 入选 10 条、8 车型：CCRs 8 条（约 20 km/h），turning 2 条（10.5 km/h）；15-TANG 贡献 3 条。
- 替代检测阈值下 CCRs 峰值稳定，turning 起点不稳。该检查是事件检测敏感性分析。
- leave-one-out 对等效恒定减速度端点的最大移动为 0.226/0.456 m/s²。它是内部端点敏感性，
  不是独立验证，也不能证明跨车型泛化。
- 只有 4 个仿真扰动参数中的 1 个获得 ABD 支持的 env 映射，所以运行标签为
  `abd_supported_v1_partial`。

产物：`abd_supported_v1.json`、`calibration_table.csv`、`distributions.json`、
`verification.json`。生成脚本为 `scripts/calibrate_abd_v1.py`；默认仿真路径保持不变，正式实验须显式
传入该配置。

更正记录：2026-09-12 修正 `target_accel_scale` 行——原表述"VUT 日志没有对应执行通道"不实
（Head tracker reference/actual 通道存在，MANUAL_EVIDENCE.md E9）。`abd_supported_v1.json` 内的
同句旧措辞保留原样：该 JSON 属归因撤回后的历史复现产物，脚本已按 driver-intervention 确认门
拒绝在无 `none_confirmed` 记录时重新生成，故不重生成、不手改生成物。
