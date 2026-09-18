# GLM 5.3 接续计划：R2验收补齐 → 可复现封存 → 500-shard受控扩展

日期：2026-09-18。状态：供用户审核的计划；本文件不代表已启动训练、提交或scale。
先读 `runs/20260918_p334_r2_codex_audit/CODEX_REVIEW.md` 和该目录机器证据。

## 1. 建议决策

- 接受C-v3完成一次100-shard重训、修复2J起步缺陷，以及其在共同recorded-clamped参考下的主要误差/有效率读数。
- C-v3可作为500-shard候选；不再默认重跑100-shard训练。先完成下面不需训练的验收补齐。
- 不保留“全门槛无遗漏、任何解码器不可修复快车、误差全面优于orig、penalty动机消失”等表述。
- 原C-v2的低minADE是约束失效条件下的真实观测，保留为负面机制证据，不称其为伪造或虚假读数。
- penalty继续延期：当前优先级是规模效应和可靠工程管线。若不做配对消融，论文不宣称硬构造优于软罚项。

建议本轮用户批准范围：R2-E补齐、完成后的定向git提交、500-shard数据/验证设计与短探针。
正式500-shard长训练须已有明确时间或update预算授权；若本轮没有给定预算，交付实测ETA与SPEC后停在启动前，供用户审核，不把“同意计划”解释为无限算力授权。

## 2. R2-E：无需训练的验收补齐

所有历史文件封存；建议新目录 `runs/20260918_p334_r2_closure/`，输出补充报告，不能覆写SPEC_R2/EVAL_PROTOCOL_V3改变历史判据。

### A. 采用真实C-v1场景对照

本次CODEX已独立重跑C-v1和C-v3的全量dev，三参考和主参考场景值见
`runs/20260918_p334_r2_codex_audit/FULL_DEV_REEVAL.json`。
将实测值写入SPEC_R2门槛4核对，引用其权重、代码、sample-id哈希。
不要把C-v1的a2-only缺陷与C-v2的未推进a1缺陷混为同一个，也不要以“必然大面积失败”代替计量。

**注意适用集合**：上述实测复用当前评估器主目标mask（present且role为anchor/prediction_target且有future），不是协议§5写的全部present。
数据审计确认154074个present里只有100134个进入该mask；排除53940个context、涉及8583个场景（其中53520个context有future）。
因此必须另列并补测C-v1/C-v3的全部present场景(a)(b)(c)，冻结对无future/缺历史context的skip规则；主候选分母600804保持原样。
如果“全部present”是协议笔误，应明确提交用户裁决并写新版本修订，不能静默把主目标统计当作原文字面要求已通过。

### B. 构造界与数值精度报告

V3有两层界，必须分表：全局评估界35/10/20；模型各类型内部界A/J（vehicle9.5/19、ped3/5、cyclist/other5/10）。
全局accel/jerk零超限不代表内部界在1e-3容差下零超限。

按现有完整账本重算所有内部界超限行，并与本次 `FULL_DEV_REEVAL.json::type_checks` 交叉核对。
输出每个候选的原始位置链、内部v_seq/commanded jerk、从相同v_seq以float64重新积分得到的位置链三种结果：

```json
{
  "sample_id": "...", "agent_slot": 0, "candidate_id": 0,
  "source": "interaction", "agent_type": 1,
  "quantity": "jerk", "limit": 19.0, "frozen_tol": 0.001,
  "output_fp32_max": 19.01, "internal_state_max": 19.0,
  "fp64_reintegration_max": 19.0,
  "violation_steps": [10], "classification": "output_quantization"
}
```

数字是格式示例，执行时逐行计算。必须区分真实递推违规、内部状态数值误差、最终坐标存储/差分放大三种原因。
本次审计已提供全量精度归因证据；GLM应引用并补齐候选级归因，不必无目的重复长评估。

验收要求：每一项超限有可复核归因；不得把未达固定容差的输出称零超限，也不得事后放大容差。
若保留fp32输出，报告明确“解析递推有界，最终fp32位置差分有量化残差；全局评估界仍通过”，由用户接受该工程边界。
若下游必须严格满足原类型界及容差，另开输出精度修订（优先高精度位置积分/存储），不更换权重、不追加训练；新输出臂与原C-v3分表复评。
不要直接对速度逐点clip后宣称修复，因为这可能破坏accel/jerk连续性。

### C. 两个剩余速度失败的正确归因

逐例记录原始v0、钳制v0、预测v(t)、a(t)、首次越限时刻和max幅值。
“历史越界”与“预测第7步再越界”是不同事实；初速度已钳到35后，可构造恒速35的主参考可行轨迹，故不能说任何解码器都无法修复。
本次审计的可行反例见 `remaining_speed_failure_counterexamples`。

保留12个失败候选/2个agent/276个超速步进主分母；当前门槛容许它们，不因纠正归因就擅自增加新的零速度违规门槛。
本阶段不默认新增速度约束或重训；若研究目标改为严格限速，单独前瞻设计速度/加速度/jerk联合可行方案。

### D. 正式评估函数的边界测试

新增测试必须直接调用 `research_tasks/p334_r2_diagnosis.py::full_anchored_validity`，不能只测试测试文件里另写的同名参考函数。
本次已复现两类问题：

- 非有限anchor、但未来位置有限时，函数可能返回valid=true；NaN比较不能静默通过。
- future mask有中断时，当前跨被掩帧差分仍影响判定。先明确mask语义：若表示位置不可用，则按各阶完整依赖链传播有效性；若仅选择评价终点、全部预测位置仍须有效，则可以保留跨帧判定并写清定义。不能为了降低违规数临时改变语义。

NaN-anchor漏洞按新评估实现版本修正，mask语义明确后补对应测试；保留原V3结果。对已有dev做差异审计：若分母、判定变化，逐例说明，不能直接覆盖原JSON。
无参考速度的skip规则与无效anchor规则分开。三参考结果、逐timestep分母/违规数、原始与容差后统计、超限分位数均按协议补齐。
字段 `branch` 无可核查标签时明确not_available；不能从agent数自动推断single/dual。

### E. 补真实延迟测量

协议§6的报告义务未由单臂41.47ms满足。实现独立benchmark：

- 同机、同dev固定batch列表、同batch16/K6/seed与真实目标mask；明确每个臂实际输入v0。
- orig、orig+B、C-v1、C-v2、C-v3分别warmup；每臂至少2个warmup batch、8个计时batch，建议按固定顺序轮转多轮，记录原始时间。
- 输出管线包含必要输入转移、生成、真实投影/解码、输出转移及组装；CUDA前后同步；CPU离线指标另计。
- 报mean/median/p95、样本数与batch数、硬件/软件/精度；如果只有批级数据就标注批级分摊，不能称独立单样本尾延迟。
- B保留已有真实投影定义，不为了加速改成全valid或v0=0；如修复B的约束，另列B-v2。

不要求证明加速；测到慢也如实交付，不改变方法选择的冻结规则。

### F. 真正补全封存清单

当前33项哈希一致，但列表遗漏正式C-v3候选账本、训练入口/epoch日志、数据manifest/码本及共享指标依赖；`required=[]`只表示脚本自己列出的文件齐全。
新清单覆盖完整依赖闭包：

- 训练/评估/数据转换/加载/指标/投影源码、配置、实际训练/探针/评估/benchmark命令；
- 数据manifest、码本、shard/index内容哈希清单与split身份；
- best/last checkpoint、训练epoch日志、结果、探针、所有最终指标与候选账本、报告和测试记录；
- Python/PyTorch/CUDA、GPU、确定性/SDPA设置；源码git提交与必要补充diff。

冻结哈希清单先核验再发布；退出码失败传播，临时文件完成校验后原子改名，避免部分JSON/账本被识别为正式完成。

R2-E输出：`R2_CLOSURE_REPORT.md`、修订评估器/回归测试、构造数值账本、`LATENCY_RAW.jsonl`/`LATENCY_SUMMARY.json`、完整新清单。
将每项明确标为PASS/NOT_MET/NOT_MEASURED；仅有解释文字的条目不得自动PASS。

## 3. 定向git提交

用户批准本计划后，可把相关源码、配置、协议、审计与闭环文档定向提交；本次审计没有执行提交或push。
先 `git diff --name-status` 与 `git diff --cached` 检查范围，不使用 `git add .`。
已有大量ABD等无关删除和修改必须保留原状。脚本用显式文件列表分批stage，检查暂存diff后提交。
大权重/候选账本/数据用完整manifest和归档位置管理，不默认塞进git；源码snapshot必须包含新增untracked依赖文件。
提交完成记录commit、manifest哈希；不自动push。

## 4. P3.3.5准备：500-shard与验证设计

R2-E完成后，先做数据与预算准备，不直接套100-shard训练命令。

1. **数据规模**：按现有scale_ladder，500 Waymo shards + INTERACTION development locations 100%。保持数据角色：ABD不并入轨迹训练/验证。
2. **切分与去重**：按来源group追踪训练/dev隔离、原始记录重叠、相邻时间窗去重；禁止heldout进入任何转换/训练/评估内容读取。
3. **固定比较集**：锁定原11,586个dev及SHA用于100→500配对比较；新增dev独立列表。明确早停使用哪一套，另一套只作额外评价，不能事后择优。
4. **码本/架构**：优先冻结现有100-shard码本、网络、loss、C-v3机制与输出精度，以隔离训练数据规模效应；若重拟码本或改输出精度，作为独立版本/因素披露。
5. **加载与曝光**：保持有界shard驻留；本loader epoch由较小来源一遍定义，扩数据后必须按实际pool重算epoch/update、分源曝光及独立group覆盖，不能简单把样本或耗时乘5。
6. **验证入口**：现训练器validation跟随训练manifest，若需固定旧dev而训练用500-shard，先增加明确的独立验证manifest/样本列表接口及最小隔离测试；不得通过临时改全局常量偷换数据。
7. **统计单位**：Waymo按scenario、INTERACTION按location/case组报告；保留group配对差及预定cluster-bootstrap。旧dev的INTERACTION只有6个group，不把数万窗口当独立重复，也不冒称达到配置中正式G1的最低覆盖。

交付 `SPEC_P335_SCALE.md`、训练/固定dev/扩展dev manifest、split与曝光审计、SHA列表、可复现命令与失败停止规则。

## 5. 探针后的一次scale实验

- 先CPU数据检查和≤30updates GPU探针，验证梯度/实际CUDA/吞吐/峰值内存/恢复一致性。
- ETA分训练、验证、I/O和归档；按预计updates与实际吞吐给预算，不承诺18.4h或简单五倍。
- 目标限定为C-v3同机制100→500规模效应。建议从头初始化、seed7、沿已冻结max30epoch/patience5，任何预算变化在启动前写入SPEC。
- 若时间预算不足，可前瞻冻结有限update实验并准确标注；不得中途改预算后继续叫原同预算实验。
- 不自动追加seed/epoch、不对失败挑最佳重试、不读heldout。
- 500同预算orig/B对照用于架构比较；如果只训练C-v3并引用100-shard orig/B，则只能讨论规模变化，不能宣称同规模方法优势。
- 结果沿冻结误差、多样性、三参考完整有效性、类型构造界、场景(a)(b)(c)、最终输出精度与延迟联合报告，负结果有效。

scale工程验收不等于G1/G3：原配置G1要求两来源对最强基线各至少5%改善及组级配对区间；当前macro改善0.55%不能替代。
三seed正式训练属于后续full档；危险生成、可避让性、闭环可见性、single/dual执行验证另列阶段。heldout继续封存。

## 6. 可复制给GLM的启动指令

> 先读AGENTS.md、runs/20260918_p334_r2_codex_audit/CODEX_REVIEW.md和本计划。
> 保留C-v3训练与V3原始结果，不默认重新训练100-shard。先执行R2-E：用CODEX实测C-v1场景对照补门槛4，补类型界数值归因，纠正快车不可修复的归因，直接测试正式评估函数的NaN-anchor及mask中断，补真实配对延迟和完整封存清单。
> 所有协议变更/评估修订另行版本化，不覆盖历史PASS/NOT_MET。R2闭环确认后定向提交相关成果，保留无关dirty修改/删除，不push。
> 然后准备500-shard数据、固定dev、码本和实际曝光/时间预算；探针通过且达到本计划前置条件后，按用户批准的预算启动一次C-v3 scale。penalty暂缓，heldout不读，不自动加seed/epoch。
> 交付实际测量与失败账本，不再用“必然”“动机消失”“任何解码器不可修”替代证据。

现有审计产物可直接核查，不必重复全量推理。必要时复跑请使用新输出目录以保留本次证据。

```powershell
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- /home/shuai/.venvs/scenario-gpu/bin/python -m pytest -q
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- /home/shuai/.venvs/scenario-gpu/bin/python runs/20260918_p334_r2_codex_audit/probe_evaluator.py
```
