# 14-BZ3X 人工制动介入复核说明

请编辑同目录的 `manual_intervention_review.csv`。只需要修改 `driver_intervention`，不要修改
`run`、`vehicle` 或 `scenario`，也不需要逐行填写其他证据字段。

本批次的统一判读方法已经固化在 `manifest.json`：进入 Robot Controller，结合 Results 中的
Check Paths，以及 Motion Pack 中的 Forward velocity [m/s] 和 Lateral velocity [m/s] 判断。
人工避让通常会出现明显横向速度，纵向速度收敛到 0 的形态也会与正常 AEB 刹停不同。

## `driver_intervention` 允许值

- `none_confirmed`：综合上述路径和速度曲线，未观察到人为介入接管特征；
- `manual`：综合上述路径和速度曲线，确认存在人为介入接管；
- `unknown`：曲线含糊、无法打开对应结果，或无法可靠判断。

该方法属于试验员基于运动学曲线的间接复核。由于没有独立踏板/制动压力标记，若有人只做直线制动，
且速度曲线恰好与 AEB 刹停高度相似，仍可能无法识别；因此 `none_confirmed` 只表示“未观察到接管
特征”，不表示通过传感器证明驾驶员没有踩踏板。

填写后在 WSL 中运行：

```bash
/home/shuai/.venvs/scenario-gpu/bin/python scripts/select_abd_smoke_runs.py
```

脚本会校验四条路径及枚举值，然后更新 `manifest.json` 和 `REPORT.md`，不会覆盖 CSV。

即使填为 `none_confirmed`，由于没有车辆 CAN，该条也只能进入 observed braking response 拟合，不能
用于 AEB request-to-response delay 校准。
