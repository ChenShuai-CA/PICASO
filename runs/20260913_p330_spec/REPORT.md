# P3.3.0 冻结报告

_2026-09-13；任务规范和最小模型设计_

---

P3.3.0 已完成任务、数据、张量、模型、评价和确认协议的实现级冻结。状态为
`design_frozen_not_trained`；本阶段没有转换数据内容、训练模型或读取新的 final-confirmation 轨迹。

## 交付结果

| 工件 | 结果 |
|---|---|
| 任务规范 | `docs/p33_task_spec.md` |
| 冻结配置 | `configs/p33/ar_scene_v1.json` |
| 样本 schema | `schemas/p33_scene_v1.schema.json` |
| 可执行校验器 | `scenario_lab/p33_spec.py` |
| 文件 inventory | `data_inventory.json`，1,523 个文件 |
| 冻结标记 | `FROZEN.json` |
| 规范校验 | `SPEC_VALIDATION.json`，全部 pass |
| 测试 | 定向 5 passed；全套 138 passed |

## 文件角色

| 来源 | development | final confirmation |
|---|---:|---:|
| Waymo | 1,000 training shard + 2 个已读 validation shard | 148 个 validation shard |
| INTERACTION | 338 个 CSV（294 train、44 dev） | 35 个 CSV、4 个未见地点 |

inventory 为 metadata-only：只读取目录项和文件大小，`dataset_content_bytes_read=0`。development 文件
的内容哈希由 P3.3.1 在流式转换时计算；final 文件的内容哈希推迟到一次性 confirmation。

## 冻结指纹

`a1cca5157c9c6963769f733bd61e95300b52ac07f49095e2a76871adea2b2728`

下一阶段为 `P3.3.1_streaming_data_pipeline`。
