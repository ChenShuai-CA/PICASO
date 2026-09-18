# GLM 接续计划：P3.3.4 审计闭环 → 修正版机制实验 → scale

日期：2026-09-16。此文件是下一轮工作方案；列出的训练尚未启动，时间为排期预算而非实验结果。
先读 `runs/20260915_p334_kinematics/CODEX_REVIEW.md` 和 `codex_review/EVIDENCE.json`。

## 0. 必须保留的研究边界

- 原 SPEC、权重、数据与 A/B/C 原始读数封存为 v1；不得覆写来制造通过。
- 采用原冻结 §4：macro minADE、分源 minFDE、joint 各≤+5%；分源 minADE补充披露。
- 修复实现不回改门槛；新增物理有效性定义、含锚点诊断、数据异常策略要在新评估前冻结。
- 单/双目标正式分支均保留，按分支、来源、agent_type分别评价；Waymo/INTERACTION为公开轨迹角色，
  ABD只在其已验证证据范围用于响应标定/执行诊断，不混成同一训练或验证来源。
- 不读取heldout、不用未来GT做推理修复、不按结果删样本；保持actor可见性与训练critic权限边界。
- 所有命令用WSL Ubuntu和指定Python；无关编辑/删除保持；不自动push。

## 1. P3.3.4-R0：补齐证据与共享评估（建议9/16–9/17，先不重训）

1. 保存git diff/HEAD、环境、SPEC与所有配置、数据manifest/码本、脚本、检查点、指标的SHA-256清单。
   标明历史预注册顺序哪些只有声明、哪些有可核查日志，不补造历史时间。
2. 修复candidate_diversity时间平均；保留legacy计算用于核对；测试1m固定间距、不同有效长度、全无效。
3. 显式输出numeric_valid/kinematic_valid/solver_failure，不以minADE有限代替候选有效。
   输出原冻结差分与含P0起步差分两套指标；统计非有限值为失败，测试NaN/Inf/缺历史/部分future mask。
4. 追溯EVIDENCE中的613m/s样本：sample→shard→原始scenario/agent/time；检查坐标系、身份、单位、时间间隔、
   valid mask、记录vx/vy与位置差分。基于历史设计异常策略，保留原集合和异常清单。
5. 用相同dev、同采样对orig/B/C-v1补评；原始异常样本必须计入主分母，额外洁净分层只作诊断。
   若修复数据转换则版本化并对所有方案同步处理，旧结果与新数据结果分表。
6. 同环境量测orig、orig+B、C完整推理成本；区分加载/生成/后处理/统计，记录batch、硬件、精度、warmup、重复次数。

产物目录：`runs/20260916_p334_review_closure/`。
交付：`ARTIFACT_MANIFEST.json`、`INPUT_OUTLIERS.jsonl`、`EVAL_PROTOCOL_V2.md`、`METRICS_V2.json`、`REVIEW_CLOSURE.md`。

每条异常建议字段：sample_id、source、group_id、agent_id/slot、shard、raw_reference、history_valid、
dt、xy差分速度、记录速度、异常原因、history_only处理、是否不可行、是否仍计入分母。
每个候选建议字段：sample/group/source/type/branch、candidate_id、numeric_valid、speed/accel/jerk/start violations、
solver_status、failure_reason；主表保留attempted数量，不能只统计成功者。

验收：指标单测通过；orig/B/C的样本ID与分母一致；原错误复现后被回归测试捕获；
规范多样性比例及候选有效性补齐；异常源头或尚未解决的依赖明确列出。此时重评B是否可作为scale备用。

## 2. P3.3.4-R1：修正C机制（建议9/17–9/19）

1. 新variant/config版本明确当前token条件输入（embedding或centroid），统一teacher-forced与rollout接口。
   干预当前token时验证同chunk响应；最后token必须有到轨迹的有效计算路径；固定可见token历史验证隐藏信息不泄漏。
2. 推导并实现起点约束联合可行构造。测试饱和a0、方向相反、不同agent_type、极端初速、缺历史、chunk边界；
   包含锚点P0，不能以输出索引排除最早违例来宣称完整轨迹保证。
3. 明确是笛卡尔运动学限制还是更完整车辆运动学；如果没有转向/横向加速度模型，就收缩方法表述。
4. penalty分支与bounded分支除构造/对应罚项外保持一致；罚项按有效agent及连续时间掩码加权，避免pad污染。
5. 先CPU回归与≤30 updates GPU探针，检查梯度有限、token干预、可见性、resume、显存与真实吞吐。
6. 新模型全新初始化，以100-shard、同seed7、同30epoch/早停规则训练；不把新头强塞旧权重后当已训练C。
   若仅做无重训修补，也须单列版本、重新eval，不复用旧通过结论。

交付：新SPEC/config、版本化模型、反例回归、训练启动清单、日志/检查点/哈希、全dev orig/B/C-v2复评、联合判据表。
验收：冻结误差项及规范多样性通过；显式失败率和起步检查齐全；声明的构造保证有分析和对抗测试支持。
若不过：保留负结果，停在architecture；不加epoch或开heldout救结果。

## 3. penalty决策（R1之后，预计一轮训练预算）

建议跑修正后的对照，前提是R1通过且排期允许；**现在不启动现版penalty**。
问题限定为“同等预算下，硬构造相对软罚项的误差、有效性、多样性与成本差异”。
冻结配对配置、同初始化规则、相同train/dev与采样、相同checkpoint选择；报告失败与全部候选。
它不再是追溯改变v1门槛的手段，也不是多种子统计结论。
原17.5h仅是恢复后计时，排期预留约一整天并在探针后修订，不能当已核准GPU成本。
若窗口不足，推迟此项并收缩论文中的硬约束优于软罚项主张。

## 4. P3.3.5：受控scale（R0/R1闭环后再启动）

按`configs/p33/ar_scene_v1.json::scale_ladder`：**500 Waymo shards + INTERACTION development locations的100%**。
先做train/dev来源组隔离、已有数据重复/时间窗去重、各来源/类型曝光量及内存审计。固定验证集用于与100-shard比较；
若新增dev，独立列出且不得混淆数据扩展与模型收益。

- 先有界吞吐探针，据实估计训练总时间/磁盘/显存；不要简单承诺500-shard在同一天结束。
- 冻结数据、参数量、训练预算、早停、选择规则与失败停止条件；至少保持orig+B这个零重训修复参照。
- 数据规模效应与架构效应分开：同模型100→500测scale，同500预算orig/B/C测方法差异；资源不足明确不能分离的因素。
- 报告source/group聚合与single/dual分支；INTERACTION group数量小，不拿大量窗口当独立重复。
- 逐来源组进行配对不确定性分析；最终三种子正式训练属于1000-shard full档，单独排期，不冒充已完成。
- 检查最终重建/执行轨迹的运动学与闭环可见性，再开展危险场景生成、可避让性、稳健性及同预算强基线比较。
  名义预测过关不等于G1/G3、危险生成能力或真实车辆可执行性验收。
- heldout仍封存；只有最终模型/协议冻结并达到既有授权流程后才进入正式测试。

## 5. 建议排期与9/30初稿

9/16–17：R0与追源；9/17–19：R1和100-shard复评；随后按吞吐安排penalty与500-shard实验。
初稿可同步整理已验证方法、v1负面发现与限制，9/27起集中结果审阅；若训练未按期完成，缩小初稿主张与实验范围，
不把计划、smoke、单seed dev结果填成正式结论。此排期是建议，实际以修复复杂度和探针吞吐更新。

## 可复制给GLM的启动指令

> 先读 AGENTS.md、runs/20260915_p334_kinematics/CODEX_REVIEW.md、本计划与codex_review/EVIDENCE.json。
> 保留v1所有产物，先执行R0共享评估与输入异常追源，再做R1机制修复。不要先跑现版penalty或500-shard训练。
> 用历史信息处理异常；不删不利样本；不读heldout。每阶段提交可复核的配置、代码diff、测试、哈希和实际运行命令，
> 不以208个既有测试通过掩盖新反例。R1训练及scale按本计划的前置验收执行，失败如实报告。

PowerShell中复核现有审计：

```powershell
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- /home/shuai/.venvs/scenario-gpu/bin/python research_tasks/p334_codex_review.py
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- /home/shuai/.venvs/scenario-gpu/bin/python -m pytest -q
```
