# HANDOFF — GLM 5.3 接续实施与 Codex 回审

> 更新：2026-09-11。接手人：Claude-glm / GLM 5.3。后续审查人：Codex。  
> 用户明确要求：当前 D 盘目录是唯一工作区；每次通过 WSL2 Ubuntu 工作；不要再次迁移项目。  
> **当前状态：环境与小预算端到端流程已跑通，研究尚未完成，尚无“优于强基线”的证据。**

## 1. 先读这些文件

1. 本文件及 [AGENTS.md](AGENTS.md)：用户决策、执行方式和接续顺序。
2. [README.md](README.md)：真实可运行入口。
3. [当前实施状态](docs/CURRENT_STATUS.md)、[预实验报告](runs/20260911_gpu_pilot/REPORT.md)。
4. [规程映射](docs/protocol_mapping.md)、[研究主张](docs/research_claims.md)：需要继续核验，模型撰写的文字不是权威证据。
5. `scenario_lab/`、`tests/` 和 `scripts/` 的当前代码。

根目录 Stage1–Stage4、stage2/stage3 和 Word 文件是历史调研与旧方案，部分来自 Gemini/Copilot/Kimi，存在相互冲突的路线。**不得据此恢复已经放弃的性能预测、VAE、Flow Matching 或庞大 Mamba/因果/域适配主线。** 最新用户决策和本交接中的当前实现优先。

## 2. 已锁定的目标与范围

- 目标：生成危险、安全关键场景；不是根据少量测试点预测新车型成绩，也不是预测全部 NCAP ADAS 表现。
- 主结构：公共数据行为先验＋ABD 支持的车辆响应/执行误差校准＋闭环强化学习场景生成。
- **单目标和双目标均为正式研究对象**，同一个角色条件模型支持两者。单目标不能只作为双目标的对照基线。
- 首篇范围：主车＋横穿行人；以及主车＋横穿行人＋动态遮挡车。最多两个学习目标是研究选择，不是规程统一上限。
- 同一共享策略采用角色编码、注意力、GRU、类型动作头和存在掩码。集中训练/分散执行，主车控制器冻结。
- PPO/MAPPO/IPPO 是训练基础和对照，不因名称或网络拼接就构成创新。检验角色/观测约束与执行扰动稳定性机制的真实增益。
- 规程脚本复现、允许范围内参数变化、规程关联的研究扩展必须区分；自适应目标策略不能自动称为 NCAP 合规。
- 状态级几何遮挡＋最后可见状态跟踪，不宣称验证真实摄像头/雷达；本月不要求试验场硬件控制。
- 月底目标：2026-09-30 完整英文论文初稿＋主要实验和可复现代码；录用或完成实车部署不是可承诺的月底成果。
- 投稿目标：中科院大类一区、允许传统订阅发表；分区与出版政策需投稿前核查，不承诺录用。

## 3. 环境与每次开工命令

### 唯一工作区

- Windows：`D:\Projects\Scenario_Generation_Research`
- Ubuntu：`/mnt/d/Projects/Scenario_Generation_Research`
- WSL 发行版：**`Ubuntu`，实际是 Ubuntu 24.04 LTS**；不要切到另一个 `Ubuntu-22.04`。
- 虚拟环境：`/home/shuai/.venvs/scenario-gpu`
- 解释器：`/home/shuai/.venvs/scenario-gpu/bin/python`
- Python 3.12.3；PyTorch 2.14.0+cu130；CUDA runtime 13.0；RTX 4060 Ti 16 GB。
- Windows 物理内存约 32 GB，当前 WSL 可见约 15 GB；大数据必须有界读取，不能把全部数据载入内存。

PowerShell 仅作为进入 Ubuntu 的入口：

```powershell
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research
```

然后所有工作在 Ubuntu 内进行：

```bash
source /home/shuai/.venvs/scenario-gpu/bin/activate
pwd
python scripts/check_environment.py
python -m pytest -q
bash scripts/run.sh --help
```

也可以逐条通过 `wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research -- <Linux命令>` 调用。

**禁止重新建立 `/home/shuai/projects/Scenario_Generation_Research` 工作副本。** 此目录曾误建，用户要求删除后已经删除；13 个独有修改/结果已校验并合回当前目录。保留 GPU 虚拟环境是有意的，它不是第二工作区。`migration/` 仅为历史记录，其中的旧 active_workspace 不是现行指令。

如果从 PowerShell 管道传 Python/Bash 源文本给 WSL，先在该 PowerShell 进程设置：

```powershell
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
```

否则中文可能变成问号。优先直接在 Ubuntu 编辑/运行文件。复杂命令不要多层拼接引号；不要读取或打印 PowerShell profile、服务 token 或完整模型包装器定义。

GPU 已验证，普通 PyTorch 训练不需要另装 Linux NVIDIA 驱动。不要为此修改全局驱动、WSL 内存设置或 Windows Python。当前不是完整 nvcc 编译环境；如确实需要自定义 CUDA 扩展，再单独处理。

## 4. 工作区状态：保留用户已有改动

接续前运行 `git status --short` 并记录。

交接前已有旧 `abd_parser/` 和 `inventory/` 多个文件删除记录；Stage3 文档已有独立修改及 CRLF/尾随空白差异。**不要 git reset、恢复旧解析器或顺手格式化整份 Stage3 文档。** 全仓 `git diff --check` 会被这些既存文档差异影响，需区分本次新增问题。

本轮大量代码和结果仍是未提交文件；不要以 untracked 为理由清理。当前不存在需要等待的本轮训练任务，`completed.json` 已记录 25 个任务完成。不要终止其他用户服务或其他会话进程。

`HANDOFF.md` 曾处于旧文件删除状态；本文件是用户要求新建的当前交接，不是恢复旧交接内容。

## 5. 代码地图与真实实现

| 文件 | 当前能力与注意事项 |
|---|---|
| `scenario_lab/schema.py` | ScenarioSpec、Body、EpisodeRecord；单位 m/s/rad；双目标固定两个槽位 |
| `env.py` / `geometry.py` | 50 Hz 物理、10 Hz 决策、二维几何遮挡、观测年龄、冻结 TTC/停车距离 AEB、角色和目标间碰撞检查 |
| `policy.py` | 注意力＋GRU、两个角色输出头；actor 不接收 critic_state；bundle schema 0.1 |
| `train.py` | 完整 episode 的循环 PPO/MAPPO/IPPO；混合模式前 20% 单目标热身；后 20% 可选成组扰动后训练 |
| `data.py` | INTERACTION 轨迹、ABD 有界审计；配置候选与真正可用于 AEB 校准严格分开 |
| `waymo.py` | 手写的有界 Motion TFRecord/局部 protobuf 读取，包含 CRC；未解码地图和感知字段 |
| `pretrain.py` | **自身运动先验**，不是完整场景/交互预训练；来源/文件/类型配额、分组隔离、运动窗口指纹、角色频率权重 |
| `evaluate.py` | 脚本、参数 CEM、固定动作序列 CEM、共同条件评价、按初始场景聚类的区间、回放、延迟 |
| `runtime.py` | 设备选择与实际 Linux/Python/CUDA 运行记录 |
| `scripts/run.sh` | 固定本工作区和 Ubuntu 解释器的启动入口 |
| `scripts/run_pilot.py` | 顺序执行小预算流程，任务失败即停，保存 jobs.json 与日志 |
| `scripts/summarize_pilot.py` | 从真实结果生成比较 CSV、图和报告；当前写死 v4 语料路径及部分 pilot 文案，复用前应参数化 |
| `scripts/prepare_abd_review.py` | 24 条代表记录的有界字段/控制来源核对材料 |

当前“固定轨迹”基线实际上输出固定动作节点，经动力学积分形成轨迹；写论文时准确命名。不要把它说成已经复现某个外部 SOTA。

当前只有可见静止对象制动回归测试，**没有完成真实车辆形状和完整条款下的 CCRs 规程复现**。角色任务完成、完整可避让性判定、超时受控终止等计划内容也不能因文档写过就算已实现。

## 6. 已验证成果与位置

### 工程检查

- 26 个 Python 文件语法检查通过。
- 40 项 pytest 通过，包含实际 CUDA 更新、单双目标存在掩码、循环序列回放、遮挡信息边界、数据来源配额和跨集合组隔离。
- 证据：`runs/20260911_compatibility/environment.json`、`verification.json`、`source_audit.json`、`requirements-lock.txt`。
- 依赖中补充了 python-docx；旧 `stage2_extract.py` 改为项目相对路径。历史盘点脚本没有自动全量重跑。

### 当前使用的公共语料

路径：`runs/20260911_public_v4/`

- 6,601 个运动片段，131 个来源组。
- INTERACTION 4,519；Waymo 2,082。
- 训练 5,331；验证 1,122；测试 148。
- 车辆 6,410；行人 191，明显不均衡；训练使用角色频率权重。
- 来源只有有界子集：各 4 个文件，Waymo 每文件最多 32 条 Scenario。
- INTERACTION 优先选择原始 `vehicle_tracks_*`，不混用重切分挑战赛版本；明确行人来自 Waymo。不能把 `pedestrian/bicycle` 混合类型硬解释为行人。
- split 的总体非空不等于每个来源×角色都有独立验证覆盖。下一步必须输出此交叉表，尤其当前 INTERACTION 验证覆盖不足。
- `manifest.json` 含 group/track/time/content fingerprint 与来源；NPZ 是当前预训练输入。
- CRC 检查的是实际读取的记录，未验证所有 1,150 个 Waymo 文件。
- `wsl_public_pilot`、`20260911_public_v2`、`20260911_public_v3` 是排错产物，**不要用于下一轮正式训练**：早期存在来源被挤占、错误文件优先级等问题。

### GPU 预实验

路径：`runs/20260911_gpu_pilot/`。`completed.json`：25 个任务完成。

- 5 轮预训练；7 组 RL，每组 12 updates、4 episodes/update。
- 混合训练 seed 7/17/27；独立 single/dual seed7；无先验、无稳健性后训练 seed7。
- 每方法每分支 20 个共同测试初始条件；另一主车控制器；10 条初始条件×10 次扰动/分支；4 个单案例 CEM；延迟；动作轨迹回放。
- `jobs.json` 保留所有真实命令和耗时；子目录 `runtime.json` 记录 GPU 与 Linux。
- 主策略：`mixed_seed7/policy.pt`；先验：`prior/prior.pt`。
- 单目标 p99=3.003 ms，双目标 p99=4.919 ms；每分支 10,000 步，超过 100 ms 的比例为零，无渲染。完整仿真实时因子约 48.1 / 22.3。仅本机软实时测量。

部分实测值（有效危险率，分母包含无效尝试）：

| 方法 | 单目标 | 双目标 |
|---|---:|---:|
| 脚本 | 15% | 25% |
| 纯先验 | 15% | 25% |
| mixed seed7 | 20% | 25% |
| mixed seed17 | 25% | 15% |
| mixed seed27 | 15% | 15% |
| mixed 无先验 seed7 | 25% | 15% |
| mixed 无稳健性后训练 seed7 | 20% | 20% |

**不能从这些小差异得出方法优越性。** 预训练验证 MSE 约 0.233，5 轮没有明显改善。样本少、更新少、假设扰动、交互预算不完全一致。正式结果还缺强对照。

故障证据：`failure_reasons.json`。脚本双目标 20 次中 9 次目标间碰撞；mixed seed17 有 10 次目标间碰撞、7 次遮挡车角色违规。不要简单剔除失败案例。

### ABD

原始目录 `Data/ABD_Data`；全目录旧清点 4,121 个匹配运行文件，不等于有效独立实验数。目录名也不自动等于独立品牌/车型/软件版本。

- `runs/wsl_data_audit/`：8 个有界抽样审计。
- `runs/20260911_abd_review/manual_review.csv` 和 `evidence.json`：24 条 CCRs/CPTA/CCFT 记录，每条只读 300 行进行初步字段审查。
- 不把 abort 当碰撞；不把 UseBrakeRobot=False 当已确认 AEB 制动；不把 spec 默认质量当实车质量。
- `calibration_candidate` 只表示可继续审查，`suitable_for_aeb_calibration` 仍为 False。
- **尚未完成 ABD 响应分布校准。** 当前 `perturb_spec` 的参数范围均标为 `assumed_sensitivity_not_abd_calibrated`，不能改标签来伪装校准。

## 7. GLM 接续顺序与验收要求

### P0：先复现与诊断，不急着扩大训练

1. 记录 git 状态；运行环境检查和 40 项测试；在新目录复现一小段训练与一条回放。不要重跑整轮只为证明启动成功。
2. 从固定 pilot 初始条件生成失败分层报告：参考脚本目标间冲突、策略新增冲突、车辆/行人角色违规分别计数；给出至少 3 条代表轨迹的平面图与时间曲线。
3. 检查初始化几何、行人启动前状态、遮挡/跟踪刷新、制动响应延迟、动作限幅和奖励终止逻辑。特别核对实际位置未动时速度是否仍按运动状态上报等物理一致性问题。
4. 新采样器建立有版本的场景集合，区分参考可执行场景与刻意压力场景；原始集合继续保留。不要靠事后按模型结果筛样本来提高成功率。

验收：新旧采样器能追溯版本；同一条件各方法共享；无效尝试不消失；测试覆盖真正发现的问题；旧轨迹若因物理修正无法回放，应记录兼容边界而不是悄悄更改日志。

### P1：把预训练从自身运动推进到交互上下文

1. 先实现 `source × split × kind` 的样本/来源组统计及零动作预测基线、分角色验证误差，明确自身运动先验到底学到了什么。
2. 以同一 Scenario/录制片段内的共同时间戳对齐邻居，输入只含当前与历史状态；未来状态只能用于监督标签，不能混入输入。
3. 首先加入可追溯的邻居相对位置/速度，验证效果后再接道路上下文；当前 Waymo 未解码地图，需要遵循官方 schema。
4. 公共轨迹中的完整可见状态不是传感器可见性真值。若合成遮挡，标注为合成模型；不要将公共全局真值当作实际可观测状态。
5. 扩大明确行人数据并记录筛选原因；不要从含混 pedestrian/bicycle 标签中臆造分类。
6. schema/网络输入改变时升级语料、模型版本；加载旧权重应明确适配或拒绝，不能 silent mismatch。

验收：无跨集合来源组泄漏；同一录制片段重复导出不跨集合；输入无未来泄漏；分角色指标与零动作基线齐全；自身运动版与交互版在同一固定验证集合比较。不能只报总 MSE。

### P2：公平预算与有效的场景生成评价

1. 增加实际环境交互步数预算。仅用相同 update 数或相同 episode 数不等价，因为失败轨迹更短。
2. 分开报告：训练总成本、固定策略生成成本，以及每测试场景再搜索的成本。避免拿 CEM 的 best-of-search 与策略一次采样直接比。
3. 让参数/固定动作序列搜索覆盖多个初始条件，而非当前单案例；无效尝试计入预算与分母。
4. 增加独立自然性/角色任务完成指标、可避让性诊断、角色机制消融。避免奖励函数自己证明自然性。
5. 敏感性对照至少比较有/无稳健性后训练在相同扰动种子上的结果；现有只有主模型的扰动评价，不足以证明该机制。
6. 现有 seed1000 的 20 条 pilot 条件已经被查看，是诊断集合。后续调参只能用开发集，**正式保留集合必须重新冻结且不得用来挑模型**。

验收：至少 3 个种子；预先冻结数据划分、危险阈值和去重规则；按独立初始场景聚类计算区间；报告负结果。未经这些检查不形成“显著优于基线”的措辞。

### P3：ABD 可追溯校准

1. 根据 manual_review.csv 组织字段和事件证据，优先确认车辆配置、AEB/人工/机器人控制来源、时间同步及事件区间。
2. 无法从文件确认的含义向用户提出具体问题，附记录相对路径和候选通道；别让用户重新说明整套数据。
3. 只有有依据的记录用于估计响应时延、制动减速度或跟踪误差；按车辆/配置/采集组分开并保留独立验证记录。
4. 样本不足的参数继续作为假设敏感性范围，不外推整个车队，也不训练“新车型成绩预测器”。

验收：每个估计参数能追到原始运行、通道、单位、事件窗口、筛选和拟合方法；训练/验证隔离；给出误差与不确定性。若控制来源仍不清楚，先完成其他独立任务，不捏造校准结果。

### P4：通过前述检查后再扩大实验与写作

9 月 30 日目标保持为完整英文初稿与主要实验。先建立清晰实验结论，再组织方法、结果、失败案例和局限；不保证一区录用。方法机制若没有增益，应收缩主张而非更换报告口径。

优先交付 P0/P1 的可审查成果，再逐步扩大 P2/P3；不要同时引入一批新模型而失去可归因性。

## 8. 可直接复用的命令

在 Ubuntu 工作区中：

```bash
source /home/shuai/.venvs/scenario-gpu/bin/activate
python -m pytest -q

# 当前检查点回放，不会训练
bash scripts/run.sh replay runs/20260911_gpu_pilot/eval_mixed_seed7/dual_0000_00.json

# 小规模接续核验：使用新目录，避免覆盖旧模型
bash scripts/run.sh train --output runs/glm_smoke_v1 --updates 2 --episodes-per-update 4 --mode mixed --algorithm mappo --device cuda --pretrained runs/20260911_gpu_pilot/prior/prior.pt

# 评价到新目录
bash scripts/run.sh evaluate --policy runs/glm_smoke_v1/policy.pt --output runs/glm_smoke_eval_v1 --count 5 --device cuda
```

完整 pilot 脚本不是断点恢复系统，输出目录存在会报错；训练检查点目前不是完整优化器/RNG 恢复点。不要用旧目录“续训”后拼接日志冒充同一次训练。新模型或新数据使用新版本目录。

## 9. 交回 Codex 时必须留下的材料

请维护本 HANDOFF 的状态摘要，并创建 `docs/GLM_CHANGELOG.md`，每轮至少记录：

- 日期、目标、修改文件和原因；哪些属于缺陷修复，哪些属于新研究机制。
- 数据/场景/模型版本及兼容性；哪些旧结果因修正需要重跑。
- 真实执行命令、Python/平台/GPU、随机种子、交互预算、运行时间、日志与产物路径。
- 测试结果、失败测试、修复过程；不要只写“全部通过”而无命令和证据。
- 按相同评价集合组织的旧/新对照表，包含失败率、负结果和不确定性。
- 未完成事项、依赖用户确认的精确字段、仍属假设的参数。

每个实验保留配置、来源清单、权重、随机种子、结果和至少一条可回放轨迹。不要提交原始 Data、凭据或巨大二进制文件；不要自动 push/发布。是否本地提交按用户后续指示处理，不用为了交接重置现有 Git 状态。

Codex 回审重点：

1. WSL/唯一工作区要求是否遵守，是否触碰无关文件。
2. 新结果能否按记录重现；旧场景/语料是否被悄悄替换。
3. actor 是否使用隐藏真值或未来信息；预训练和测试是否泄漏。
4. 无效样本和搜索预算是否被完整计数，正式测试是否被用于调参。
5. 角色/交互机制是否真正带来增益，还是数据/预算变化造成的表面提升。
6. ABD 标定有没有原始依据，是否越界声称真实感知、规程合规或论文水平。

## 10. 一句话给接手模型

**先读本交接和代码，复现当前结果，优先解决双目标无效场景及交互先验不足；在当前 D 盘工作区使用 WSL2 Ubuntu 继续实施，逐步留下能被 Codex 独立复核的证据。不要从零重写项目，也不要将目前的工程 pilot 包装成已完成论文。**
