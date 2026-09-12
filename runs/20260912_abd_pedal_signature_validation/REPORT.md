# AEB 制动源踏板签名验证（操作员力学模型 → 判据 → 已标注 run 验证）

日期：2026-09-12。脚本：`scripts/validate_abd_pedal_signature.py`；
输入标签：`runs/20260912_abd_braking_source_analysis/labelled_event_features.csv`
（44 条 operator 确认 AEB + 33 条 FCW 后"manual"，均 BR Command≡0）。

## 1. 操作员提供的力学事实（2026-09-12）

1. ABD BR 与车辆制动踏板**刚性连接**；载荷计只在**机器人主动施力**或**人脚主动下踩**时
   读到明显的力（压缩）。
2. **耦合车型**：AEB 触发时踏板被执行器带着下压，载荷计是**被拉着走**的——行程大、
   力很小（可出现负值=张力）；**解耦车型**：AEB 触发时踏板**完全不动**。
3. C-NCAP 2024 部分 AEB 场景（D4F4）：听 FCW 报警后 **BR 先轻踩一下**（模拟人看到
   目标先轻带刹车），随后 **AEB 刮停**（主减速段 BR Command 回 0）。

## 2. 签名判据（阈值暂定，待操作员裁决后固化）

| 制动源 | BR Command | 踏板行程（事件窗 vs 窗前 2 s 基线） | 载荷计（带符号） |
|---|---|---|---|
| 机器人 | ≠0 | 跟随指令 | 大压缩 |
| 人脚 | ≡0 | 动或不动（脚压刚性保持的踏板时几乎不动） | **大压缩 >50 N** |
| AEB·耦合 | ≡0 | **动（14–50 mm）** | **小（\|F\| 小；负值=张力=硬证据）** |
| AEB·解耦 | ≡0 | **不动（≤2 mm）** | 小 |
| D4F4 | 主段≡0（前有轻踩脉冲） | 主段同 AEB·耦合/解耦 | 主段小 |

辅助：窗前 2 s 踏板已有 >5 mm 活动 = **脚在踏板上**（人工倾向先验）。

## 3. 验证结果（77 条已标注 run）

**operator 确认 AEB（44）**：`aeb_pedal_dragged` 42 + `aeb_pedal_static` 1 = **43/44 与
AEB 签名一致**；`drag_then_foot_adjudicate` 1（V13_T332_R2：张力 −84.5 N 与 75 N 压缩
同窗——疑 AEB 先作动、驾驶员后补脚，留操作员裁决）；纯 foot 0；ambiguous 0。
张力（fmin≤−5 N）19/44，车型相关：13-G9 −73~−84、14-BZ3X −16.7、E8 −4.7、9-BZ7 无。
foot_pre_event 0/44（规程驾驶员脚不在踏板上，符合预期）。

**FCW 后"manual"（33，标签为规程推定非力学证据）**：仅 **3 条 foot_compression**
（V9_T69_R3：行程 0.04 mm + 51.5 N——脚压在被机器人刚性保持的踏板上；
V2_T60_R1/R2：306.7/117 N）；**30 条呈 AEB 签名**（18 dragged + 12 static）；
张力 0；foot_pre_event 14/33。

**9-BZ7 同车对比（最硬的结构证据）**：30 条"manual" vs 9 条确认 AEB——
travel 25.8 vs 25.2 mm、fmin +1.1 vs +0.7 N、fmax 17.7 vs 17.7 N。同车同签名，
支持这 30 条实际由 AEB 制动（与操作员"FCW 场景有时让 AEB 制动"的说明一致）。

## 4. 结论与边界

- 踏板签名判据在 AEB 侧验证通过（43/44，唯一例外进入裁决而非误判为人工）。
- 上轮 braking_source_analysis 的"特征无分离"结论被推翻：不是特征无判别力，而是
  **manual 标签被污染**（规程推定 vs 力学事实）。"签名"（数据事实）与"标签"（推断）
  必须分开维护。
- 局限：阈值（2 mm/50 N/−5 N/5 mm）暂定；载荷计几何因车而异（张力仅部分车型出现）；
  manual 类无力学真值；test_id 非唯一（15 个同名不同路径文件，去重须 sha256+test_id）；
  D4F4 模式尚未扫描（需扫 BR-active AEB-path run 找"轻踩脉冲+主段 cmd≡0+AEB 签名"）。
- FCW 场景按用户决定不进本项目范围；9-BZ7 的 30 条与 V13_T332_R2 留档待裁决。

## 5. 下一步（待确认）

1. 用本判据扫 892 条 BR-zero AEB-path run → 直接产出 AEB 筛选数据集
   （foot/ambiguous/adjudicate 进操作员队列）；
2. D4F4 扫描规则：onset 前小幅 BR Command 脉冲（轻踩）+ 主减速段 cmd≡0 + 主段
   AEB 签名 → `aeb_after_light_touch`（轻踩段单独记录，主段动力学可用于 AEB 分析）；
3. V13_T332_R2 与 9-BZ7 30 条交操作员复核。
