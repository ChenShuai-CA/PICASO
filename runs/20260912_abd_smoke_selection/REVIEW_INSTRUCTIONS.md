# 14-BZ3X 人工制动介入复核说明

请编辑同目录的 `manual_intervention_review.csv`。四条 BR-zero run 已预填为 `unknown`；只修改人工复核
字段，不修改 `run`、`vehicle` 或 `scenario`。

## `driver_intervention` 允许值

- `none_confirmed`：现场运行记录、试验员记录或同步视频明确证明该 run 没有人工踩刹车；
- `manual`：记录明确说明驾驶员进行了安全制动；
- `unknown`：没有记录、记录含糊、只能看到车辆减速，或不能对应到该具体 run。

“记录没有写人工介入”不等于 `none_confirmed`，应填 `unknown`。

## 证据字段

- `evidence_source`：例如纸质试验记录、电子 run sheet、同步视频、试验日报；
- `evidence_locator`：文件名、表格行号、视频时间码或纸质页码；
- `intervention_time_s`：能与 ABD `Time` 对齐时才填；不能可靠对齐则留空；
- `reviewer`、`review_date`：复核人和日期；
- `notes`：记录“为什么确认无介入”或“在哪个阶段人工踩下”等信息。

## 示例

```csv
driver_intervention,evidence_source,evidence_locator,intervention_time_s,reviewer,review_date,notes
none_confirmed,试验日报,2024-05-18 第12行,,张三,2026-09-12,试验员明确记录全程未接管
manual,同步视频,CAM02 00:01:24.530,24.53,张三,2026-09-12,AEB过晚后人工制动
unknown,,,,张三,2026-09-12,没有能对应到该run的现场记录
```

填写后在 WSL 中运行：

```bash
/home/shuai/.venvs/scenario-gpu/bin/python scripts/select_abd_smoke_runs.py
```

脚本会校验四条路径及枚举值，然后更新 `manifest.json` 和 `REPORT.md`，不会覆盖 CSV。

即使填为 `none_confirmed`，由于没有车辆 CAN，该条也只能进入 observed braking response 拟合，不能
用于 AEB request-to-response delay 校准。
