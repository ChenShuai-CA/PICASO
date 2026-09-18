# GLM 5.3 接续方案：P3.3.4-R1 审计修订 → R2 起步递推修复 → 有条件 scale

日期：2026-09-17。状态：**提交用户审核的工作方案，尚未授权或启动下一轮训练**。
依据：`runs/20260917_p334_r1_codex_audit/CODEX_REVIEW.md` 与同目录的机器可读证据。

## 1. 决策建议

- 接受 R1 已完成一次 100-shard、seed 7、30 epoch 训练和 V2 dev 复评；保留其误差改善和多样性结果。
- 不接受“四门槛全过”“anchored 全零”“纯口径分歧、非机制缺陷”的结论。
- 500-shard 暂缓。C-v2 保留为有价值的候选与对照，先修复起步递推；不能把有缺陷的同一机制放大训练。
- penalty 训练本阶段暂缓。其用途仍是硬构造与软罚项的机制对照，并未因误差改善而消失；资源不足可以延期，同时收缩论文主张。
- R1 按冻结 SPEC 的 diff 主口径记录 3b 未达；未来若选择 recorded-clamped 作为主物理参考，必须在新协议中明确，所有臂使用同一参考并保留 diff 诊断。禁止追溯把 R1 改判为通过。

## 2. 保留边界

1. 仅用本工作区；所有 Python、开发和实验在 WSL Ubuntu，用 `/home/shuai/.venvs/scenario-gpu/bin/python`。
2. R0、R1 原 SPEC、报告、指标、权重、代码证据封存。用新目录、新文件和附加勘误说明历史问题，不覆写旧 JSON 来“修复”结论。
3. 不读取 heldout 轨迹，不为调参打开 heldout，不按结果删样本；异常仍进主分母；推理策略只能使用历史信息。
4. 保存无关编辑和删除，不使用 `git add .`，不自动 push。单/双目标保持正式分支；Waymo、INTERACTION、ABD 的数据角色保持分离。
5. 对外结论限定为单 seed、dev 上的工程结果。名义预测、离散运动学约束不能替代危险场景能力、闭环可见性、G1/G3 或真实车辆执行验收。

## 3. R2-A：先修订证据与冻结协议，不训练

建议目录：`runs/20260917_p334_r2/`；若实际执行日期不同，可改目录日期但全链路保持一致。

### 3.1 复现 CODEX 证据并形成勘误

运行审计目录的 `probe_start_jerk.py`；应先确认旧 C-v2 出现实际 jerk 38 m/s³，同时旧 `kinematic_valid` 为真。
阅读 `ARTIFACT_AUDIT.json`、`C2_FULL_DEV_REEVAL.json`，核对以下事实：

- R1 SPEC §2.3 明定 diff 为主、recorded 为诊断；diff 比例 600756/600804，小于 C-v1 的 600786/600804。
- 旧 `candidate_validity` 只检查 frozen 内部项和一个起步加速度，不包含完整 anchored 的所有最早加速度/jerk 项。
- R1 现存 anchored 表包含显著 jerk 违规；`validity_detail.txt` 中有未处理的 `AttributeError`，不得将脚本中断误读为“后面无非零项”。
- 速度违规准确计数是 210/6719502，不是报告中的 18/约58万步。
- 35 m/s直接钳制的是v0；后续速度只有罚项，没有硬构造保证，不将速度与加速度/jerk的界混写。
- “完全 flag 隔离”应限于已核验的旧 rollout 结果不变；SPEC 附录 A 已披露一个对 flags 全组合生效的 teacher-forced 修复。
- 39.15 ms 是 C-v2 生成全阶段，不是头单独耗时；1.83× 是生成与合成投影分相位相加的比值，不是实测完整管线加速比。
- 本次审计环境实际 GPU 为 RTX 4060 Ti；历史报告硬件应从当时日志/环境快照核实，不写“4090Ti 级”。
- resume 日志中四位小数 loss 重复，不等于权重/optimizer/RNG 全状态逐位验证；训练函数计时在 resume 时重置，不能称 17.9 h 包含所有中断与重放成本。

交付 `R1_ERRATA.md`，将“已完成实验”与“机制验收未通过”分开。

### 3.2 预注册 EVAL_PROTOCOL_V3.md 与 SPEC_R2.md

保持历史 V2 读数原样。新增指标必须在新一轮评估/训练前冻结，禁止因结果不佳临时选择速度参考或阈值。

推荐并列三种速度参考：

| 名称 | 定义 | 用途 |
|---|---|---|
| `diff_raw` | 相邻有效历史位置差分 | 复现历史 gate；测量位置链连续性 |
| `recorded_raw` | 末帧有效记录 vx/vy | 测量相对于原始观测的连续性与钳制代价 |
| `recorded_clamped` | 记录 vx/vy 圆盘钳制到 35 | 构造内部一致性；新工程主参考候选 |

新主参考一旦选定，orig、B、C-v1、C-v2、修复版 C-v3 均按该同一参考评价，不能各臂自行选择有利“原生口径”。
每一种参考都显式处理缺末帧、仅一帧、前帧无效、非有限速度、钳制前后速度差；跳过与失败计数分开。

建议的新工程验收（供用户审定，不能回改 R1）：

- 保留误差 ≤ orig+5% 的原冻结各项及 diversity_v2/orig ≥80%；分源 minADE 继续补充披露。
- `full_anchored_valid` 在指定共同参考下检查 P0→P1 起的每一步速度、每一步可定义加速度，以及最早可定义 jerk；不再仅依赖 legacy `kinematic_valid`。
- 对 bounded 构造声明的加速度/jerk：合成对抗测试必须符合各 agent_type 的内部界（预定数值容差）；全量 dev 使用既定物理评估界，完整 anchored accel/jerk 不得出现无法解释的超限。数值误差同时报告原始量和容差量，不能事后增大容差救结果。
- `full_anchored_valid` 候选率建议 ≥99.997%，按 agent 无有效候选数建议不超过原 C-v1 的 3 个；速度失效保留在总分母并逐例记账。这是新的前瞻工程要求，不宣称 R1 曾按此预注册。
- 场景同时报告：(a) 没有任何有效 agent 的场景；(b) 至少一个必需目标无有效候选的场景；(c) 没有一个共同 k 能让全部必需目标同时有效的场景。旧 `scenes_no_valid_agent=0` 只对应 (a)。后两项的阻断阈值须在新 SPEC 明确，不能用 (a) 代替。
- single/dual、source、agent_type 分层；若当前名义数据不存在可核查 single/dual 标签，记 `not_available` 并列出与场景任务的映射工作，不伪造覆盖。

### 3.3 候选级诊断账本

推荐 `candidate_validity_v3.jsonl`，每个评估目标的每个候选一行；至少字段：

```json
{
  "sample_id": "...", "group_id": "...", "source": "waymo",
  "agent_slot": 1, "agent_type": 1, "branch": "unknown", "candidate_id": 0,
  "arm": "C-v3", "velocity_reference": "recorded_clamped",
  "numeric_valid": true, "legacy_valid": true, "full_anchored_valid": false,
  "history_usable": true, "v0_over_limit": false,
  "failure_reasons": ["jerk"], "first_violation_step": 0,
  "speed_max_mps": 0.0, "accel_max_mps2": 0.0, "jerk_max_mps3": 38.0,
  "speed_bad_steps": 0, "accel_bad_steps": 0, "jerk_bad_steps": 1
}
```

示例数值仅说明格式。输出每个 timestep 的超限数/分母及 p50/p95/p99/max 超限幅值；对全部失败保持追踪，而不是仅抽取最终最佳候选。
专项核对旧 8 个 diff 全无效 agent、2 个 recorded 全无效 agent、613 族与3个原始超速输入；其集合可能重叠，逐例按实测原因归类。

R2-A 验收：纠正报告；冻结新定义；V2 可逐位复现；所有新增诊断脚本正常结束且遇错误返回非零。此阶段不需要重训。

## 4. R2-B：修复起步递推与评估覆盖

### 4.1 根因与候选修复

`scenario_lab/p33_model.py::_kinematic_pass` 在 `joint and step == 0` 保存 `a_first=candidate` 后直接 `continue`，没有推进加速度状态 `a`。
因此旧版计算的是 `a1=proj(a0+h*j1)` 和 `a2=proj(a0+h*j2)`，不是 `a2=proj(a1+h*j2)`。
即使两项统一缩放，未触发缩放的反向 jerk 也可使 `|a2-a1|/h` 接近 `2J`。

修复应恢复真正的连续递推：暂缓提交 v1，但不能暂缓 a1 对 a2 的状态依赖。先算原始 a1，再从原始 a1 算 a2，然后同一比例缩放 a1/a2、提交 v1/v2，并把缩放后的 a2 传给后续步。
实现细节由 GLM 自检；不要只修改测试容差。补充证明：圆盘投影的非扩张性与 joint scale≤1 如何共同保持 `|a2-a1|≤hJ`、各步 accel 界及起点对和界。

以显式新版本/flag（例如 `start_recurrence_version:2`）保存旧 C-v2 重放路径，或使用冻结代码快照；不能修改通用默认路径后仍声称旧 checkpoint 已逐位复现。

### 4.2 必需回归

1. 旧反例 `a0=0, j1=(19,0), j2=(-19,0)`：旧版实际 jerk=38，新版≤19；两者都直接从输出位置含 P0 还原，不能只看模型输出的 commanded jerk。
2. 饱和初始加速度、同向与反向 jerk、斜向向量、各 agent_type；联合起点和界、a1→a2、a2→a3 与 chunk 边界全部检查。
3. 起步与各内部 timestep 的人工单点违规：分别验证 speed、accel、jerk 必被新完整判据抓获；同时保留 legacy 的旧定义测试。
4. 非有限位置/速度、缺历史、部分 future mask、mask 中断、pad agent；不可用计数与可行性结论分开。
5. 当前 token/最后 token 干预、相同 token 下 rollout/teacher-forced 对齐；需要逐位声明就用 `torch.equal`，否则准确报告容差和最大差值。
6. 固定可见历史与已生成可见动作，扰动隐藏 agent 状态，验证可见性；不能把纯 decoder 固定 token 不变误称闭环可见性。
7. CPU 与 GPU fp32 头；训练 AMP 下梯度有限。用最坏方向与边界覆盖新机制，不依赖随机小网络恰好不触界。

R2-B 验收：新反例确实红→绿，全套测试通过，旧 R0/R1 可重放；先记录代码快照/哈希再跑探针。

## 5. R2-C：先做旧权重修复诊断，再决定一次受控训练

1. 先把冻结 C-v2 checkpoint 经修正版执行，标记为 **C-v2 weights + R2 decoder，无重训诊断**。全 dev 与旧 C-v2 配对评估，量化真实 jerk 修复、误差、多样性和速度变化。不冒称它是训练好的 C-v3。
2. 此结果用于判断修复影响和制定资源预算。若仅需解释当前失败，完成此步即可提交；不能为“找回1.3002”不断调整修复。
3. 若用户审核通过本阶段训练预算，训练与推理统一的新 C-v3 从头初始化，以同100-shard、seed7、最多30 epoch、patience5、同选 checkpoint 指标运行一次；不自动增加 epoch/seed，不开 heldout。新训练是针对已证明实现缺陷的机制实验，不是重试同一失败配方。
4. GPU ≤30 updates 探针先于长训：实际 CUDA 可用、模型干预成立、梯度有限、吞吐/内存达预算、resume 的权重和 optimizer 等状态校验。单独记录训练、验证、加载和中断时间，ETA 由实测更新。
5. 完整复评至少包含 orig、原 B、C-v1、C-v2、C-v3；B 若更换锚点/速度参考作为推理输入，须另列 `B-v2`，不能沿用旧 B 的误差或通过状态。
6. 某项不过则保留负结果并停在 architecture，给出失败账本；不得自动进入 penalty 或 scale 作为救结果手段。

交付：`SPEC_R2.md`、`EVAL_PROTOCOL_V3.md`、新配置、代码 diff、测试记录、探针、修复诊断、训练日志/检查点（若获批）、`METRICS_V3.json`、失败账本、R2 报告。
清单必须覆盖实际命令、环境、train/eval/test代码、数据manifest/码本、配置、权重、日志、指标、报告；封存前逐项核验，不能仅有 git HEAD+dirty diff stat。

## 6. R2 通过后的条件工作

### 6.1 penalty

默认继续延期，优先完成正确性与 scale 前置准备。若论文要主张“硬构造优于软罚项”，另行冻结同预算配对实验后再运行；否则明确没有该消融证据。
不要引用“C过门槛所以 penalty 原动机消失”；误差验收与机制归因是两个问题。

### 6.2 500-shard scale

只有 R2 验收闭环且用户接受新协议后再启动：

1. 定向提交本课题必要源码、配置、测试和文档；大权重以哈希与本地路径管理。无关改动/删除保持，不全仓打包提交，不 push。
2. 按既有 scale ladder 使用500 Waymo shards + INTERACTION development locations 100%。审计 source/group 切分、重复时间窗、分源曝光与峰值内存；heldout 仍不读。
3. 锁定100-shard阶段 dev 用作 paired comparison；新增 dev 单独列出，不能混淆数据变化与模型改进。
4. 在正式启动前冻结训练预算、停止规则、存储需求和按实测吞吐得到的 ETA。500-shard 不能简单沿用100-shard的17.9h。
5. 同模型100→500只回答规模效应；若要主张架构优越，需要同500预算的 orig/B/C 对照。资源不够就收缩主张。
6. 同机、同样本、同真实掩码和 v0，分别 warmup 所有臂，实测 `orig+B` 与 C 的完整轨迹输出管线；包含必要数据转移与组装，报告均值/中位数/尾延迟和重复数。CPU离线统计独立计时。
7. 来源 group 级配对不确定性分析；INTERACTION只有少量独立 group，不能把窗口数当重复数。保留 single/dual 分支与最终执行轨迹验证。

## 7. 可复制给 GLM 5.3 的启动指令

> 先读 AGENTS.md、runs/20260917_p334_r1_codex_audit/CODEX_REVIEW.md、本文件，以及 R1 的 SPEC_R1.md。
> 本轮按用户审核范围执行 R2-A→R2-B，先复现38 m/s³起步 jerk 反例并修正报告；封存 R0/R1，冻结 V3 共同速度参考及完整 anchored 判据。
> 修复 joint step0 未推进 a 状态的递推，补足输出位置级的最早 jerk 测试与可见性测试；保留旧版本重放。
> 先做旧权重+修复解码器的无重训配对诊断，再依用户批准的预算进行一次100-shard C-v3训练。
> 未经本计划前置验收，不启动500-shard或penalty，不读heldout，不回改R1通过状态，不自动追加epoch/seed。
> 交付可复核代码/配置/命令、测试、完整指标和失败账本；“原生口径通过”不能替代完整轨迹约束。

已存在审计的复现命令（这些命令只做核查，不开始新训练）：

```powershell
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- /home/shuai/.venvs/scenario-gpu/bin/python runs/20260917_p334_r1_codex_audit/probe_start_jerk.py
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- /home/shuai/.venvs/scenario-gpu/bin/python -m pytest -q
```

完整 C-v2 dev 复核脚本 `runs/20260917_p334_r1_codex_audit/reeval_c2.py` 已在本次审计执行；可直接查证其 JSON，后续复跑应使用新的输出目录以保留本次审计证据。
