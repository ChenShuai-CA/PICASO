# P3.3.1 公开数据流式转换与运动码本

_版本 p33.1-pipeline-v1，2026-09-13；状态：smoke G0 通过，尚未训练模型_

P3.3.1 将 P3.3.0 冻结的 `scene-shard-v1` 契约落实为可断点恢复的数据管线。正式 smoke run 位于
`runs/20260913_p331_data_pipeline/smoke`。本阶段只证明数据工程边界成立，不构成 G1 真实度结果或
模型优越性证据。

## 1. 输入与解析

- Waymo：读取 smoke inventory 固定的 10 个 training TFRecord shard，共解析 4,982 个独立
  Scenario。解析字段遵循官方 `Scenario`、`Track`、`ObjectState` 与 `MapFeature` protobuf 编号；
  不依赖 TensorFlow，也不读取 validation/final-confirmation 轨迹内容。
- INTERACTION：读取 8 个 development 地点各 2 个记录 case。同一地点、同一末尾编号的
  `vehicle_tracks` 和 `pedestrian_tracks` 组成一个不可拆分单元，共 16 个独立 case、26,233 个重叠窗
  样本。地图从对应 Lanelet2 OSM 文件读取并单独记录内容哈希。
- INTERACTION 的 `pedestrian/bicycle` 文件无法从列中可靠拆分两种类型，统一编码为 `other`；缺失
  尺寸使用 1.8 m × 0.8 m，缺失朝向仅在速度大于 0.5 m/s 时由速度方向推导并前向填充。它们不能
  用作 pedestrian-only 或 cyclist-only 类型监督。

Waymo protobuf 定义：

- <https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto>
- <https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto>

## 2. 工件与恢复边界

每个源 TFRecord 或 INTERACTION 记录 case 是一个恢复单元。单元完成后原子写入：

- `part-00000.npz`：不超过 4,096 个样本的定长数组；
- `part-00000.jsonl`：逐样本来源、split、独立组、坐标逆变换、截断量和数组索引；
- `UNIT.json`：输入内容哈希、样本数、错误和 shard 清单。

再次执行相同命令时，只有输入路径、输入哈希、配置哈希和完成标志全部一致的单元才会复用。运动码本
只读取 train 样本，并用每个 `source × agent_type` 最多 8,192 条向量的 reservoir 限制内存。转换和
token 标注按 shard 处理，不构造全局样本数组。

## 3. Smoke 结果

| 统计 | train | dev | 合计 |
|---|---:|---:|---:|
| Waymo 样本/Scenario | 4,501 | 481 | 4,982 |
| INTERACTION 重叠窗样本 | 22,298 | 3,935 | 26,233 |
| 全部样本 | 26,799 | 4,416 | 31,215 |

独立统计单位为 4,982 个 Waymo Scenario 和 16 个 INTERACTION `(location, case_id)`，合计 4,998；
INTERACTION 的 26,233 个重叠窗不能作为 26,233 个独立统计样本。dev 只有 481 个 Waymo Scenario
和 4 个 INTERACTION case，低于 G1 要求的 5,000/100，因此 smoke 只允许作工程验证。

码本使用随机种子 7、128 个质心、200 个 mini-batch step。train reservoir 实际看到 2,847,950 个
有效运动块，保留 39,633 个分层样本。标注后的 3,371,044 个有效目标覆盖 128/128 个 token；最小
计数 14，最大 token 占比 21.58%，归一化熵 0.840。长尾和静止/近静止模态占比须在 P3.3.2 训练
诊断中继续报告，不能用“词表全占用”代替预测质量。

## 4. G0 验证

`G0_VALIDATION.json` 和独立的 `VERIFICATION.json` 共同确认：

- 26/26 单元可断点恢复，最大驻留样本数上限为 4,096；
- 两个来源及 train/dev 均存在，heldout 轨迹内容未读；
- shard 内容哈希、行数、metadata 和数组契约一致；
- 31,215 个 sample ID 无重复，4,998 个独立组零跨 split；
- 全部样本通过 finite、padding、mask、visibility 和 token 契约；
- 坐标正反变换最大绝对误差为 `4.56e-12 m`，低于 `1e-5 m` 门槛；
- Waymo 与 INTERACTION 可视化预览中的地图和轨迹方向一致。

## 5. 已知限制

- 几何可见性代理只处理 80 m 距离与第三方动态包围盒遮挡，不包含建筑物遮挡；不能称为传感器真值。
- 7,423/31,215 个样本发生 agent 截断，30,685/31,215 个样本发生 map-polyline 截断。P3.3.2 应按
  截断与否分层报告 dev 误差，并把 `max_agents`/`max_map_polylines` 纳入 architecture-stage 消融。
- smoke 的样本量和地点数不足以执行 G1 bootstrap，也未实现或训练 AR-Scene-v1。
- 本阶段没有使用 ABD 训练 NPC；ABD 仍只在后续冻结闭环评价中提供响应参数敏感性域。

## 6. 复现命令

```bash
/home/shuai/.venvs/scenario-gpu/bin/python research_tasks/convert_p33_data.py \
  --output runs/20260913_p331_data_pipeline/smoke

/home/shuai/.venvs/scenario-gpu/bin/python research_tasks/verify_p33_data.py \
  --output runs/20260913_p331_data_pipeline/smoke
```

下一阶段 P3.3.2 应先实现数据加载器、constant-velocity 基线和最小 AR-Scene-v1 单 batch
overfit/smoke training；通过数值、mask、显存和开放环 rollout 检查后，才进入 100-shard architecture
训练。
