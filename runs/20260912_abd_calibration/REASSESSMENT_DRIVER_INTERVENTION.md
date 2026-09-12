# P3.1/P3.2 归因重评：人工安全制动

2026-09-12 用户补充：实际测试中，AEB 已触发但过晚时，驾驶员会人工踩刹车以避免碰撞。

这条事实否定了 P3.1 的关键排除前提“规程要求驾驶员不干预，所以观测减速可归因于 AEB”。
Post Processor User Guide PDF 第 19 页也明确指出，driver intervention 可能被其基于纵向加速度的
AEB event 检测误识别。现有导出没有直接 AEB/FCW 通道，也没有独立驾驶员制动通道；BR Command
为零只能排除机器人命令。

因此：

- 10 条原 `AEB_by_elimination` 全部降级为 `unknown_AEB_or_driver_brake`；
- `abd_supported_v1.json` 的数值保留用于历史复现，但其 ABD/AEB 校准证据声明撤回；
- P2.8 使用的范围与原 assumed U(5.5,8.0) 很接近，可继续解释为敏感性实验，不能解释为
  ABD 校准后的验证；
- 在逐 run 补齐 `driver_intervention` 记录，或新采 AEB request/status 与驾驶员踏板/压力通道前，
  不生成新的 ABD-supported 制动参数配置。

补采与裁决规范见 `docs/ABD_RECALIBRATION_PROTOCOL.md`。
