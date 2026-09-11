# GLM_CHANGELOG — GLM 5.3 接续实施记录（供 Codex 独立复核）

> 依据 HANDOFF §9 维护。每轮记录：日期/目标、修改与原因（缺陷修复 vs 新机制）、版本与重跑标记、
> 真实命令与环境、种子与预算、测试证据、按相同评价集合组织的对照表、未解决问题。
> 措辞边界：本项目的 pilot 结果是工程检查，不构成论文证据；"优于基线"的结论在 P2 检查通过前禁止出现。

---

## 2026-09-11 · P0：复现、诊断、physics v2 修复、sampler v2、条件清单显式化

**执行环境**：WSL2 Ubuntu 24.04（`wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research`），
Python 3.12.3（`/home/shuai/.venvs/scenario-gpu/bin/python`），PyTorch 2.14.0+cu130，RTX 4060 Ti 16 GB。
说明：从 Git Bash 调用 wsl.exe 需 `MSYS_NO_PATHCONV=1` 防止 `/mnt/d/...` 被 MSYS 路径转换改写。

### 1. 基线留档（改代码之前）

- `git status --short` 与 HANDOFF §4 记载一致（`M HANDOFF.md`、`M stage2_extract.py`、abd_parser/ 与
  inventory/ 删除、scenario_lab/scripts/tests/runs/docs 未跟踪）。未做任何 reset/checkout/清理。
- `python scripts/check_environment.py` 通过（22 个源文件、CUDA 可用、Data 在位）。
- `python -m pytest -q` → **40 passed in 6.66s**（改动前基线）。
- smoke 复现（新目录，未覆盖任何旧产物）：
  ```bash
  bash scripts/run.sh train --output runs/20260911_p0_smoke --updates 2 --episodes-per-update 4 \
    --mode mixed --algorithm mappo --device cuda --pretrained runs/20260911_gpu_pilot/prior/prior.pt
  bash scripts/run.sh evaluate --policy runs/20260911_p0_smoke/policy.pt \
    --output runs/20260911_p0_smoke_eval --count 5 --device cuda
  bash scripts/run.sh replay runs/20260911_gpu_pilot/eval_mixed_seed7/dual_0000_00.json   # matched（改动前）
  ```

### 2. 缺陷修复（physics v2）：行人启动前状态物理不一致

- **缺陷定位**（env.py `_integrate`，20260911_p0 之前的代码）：`time < pedestrian_delay` 期间行人
  speed/heading 按动作演化、位置被 `continue` 冻结，但初始 speed 即 pedestrian_speed →
  (a) `_refresh_tracks` 上报非零 vx/vy；(b) ego 冻结 AEB 的恒速外推把静止行人预测成北向移动（提前制动）；
  (c) `critic_state` 真值自相矛盾；(d) 等待期转向动作可把 heading 拉出 [0.8, 2.35] 造成假 pedestrian_role。
  诊断图 `runs/20260911_p0_diagnosis/figures/*.png` 右子图可直接看到该现象（灰色等待期内 ped speed 非零）。
- **修复语义（用户 2026-09-11 确认"完全静止等待"）**：physics v2 下等待期行人 speed=acceleration=0、
  位置/heading 全冻结、动作不生效；跨过 delay 的首个物理子步 speed 置为 pedestrian_speed（**已知近似**：
  0.02 s 内瞬时起步，未做 jerk 受限起步；若评审质疑再升 v2.1）。delay=0 时 reset 即以 pedestrian_speed 起步。
- **兼容边界（不悄悄改日志）**：`ScenarioSpec` 新增 `physics_version/sampler_version/condition_role`
  字段（默认 2/2/'unspecified'，旧 dict 构造兼容）；`replay()` 对无版本字段的旧 trace `setdefault('physics_version', 1)`
  并打印 `warning: legacy physics v1 trace; replaying under v1 semantics for compatibility`，env 保留 v1 积分
  分支（标注 legacy，仅回放用）；`EpisodeRecord.schema_version` → '0.2'。
- 验证：`bash scripts/run.sh replay runs/20260911_gpu_pilot/eval_mixed_seed7/dual_0000_00.json`
  → `matched: true` + 上述 warning + `physics_version: 1`、`schema_version: "0.1"`。

### 3. 新研究机制（sampler v2）：构造性参考可行性 + 显式条件清单

- 新增 `scenario_lab/sampling.py`：
  - `sample_spec_v1`：env.sample_spec 原样搬入（rng 消费顺序逐 draw 一致），版本钉 1，**永不修改**；
  - `sample_spec_v2`：先按 v1 分布采样，dual+reference 不可行时重采样
    (occluder_speed, occluder_x∈crossing_x×U(.5,.75), pedestrian_delay, pedestrian_y, pedestrian_speed)
    五元组至多 40 次（crossing_x/ego_speed 保持首抽），仍不可行标记 `condition_role='reference_infeasible'`
    保留并计数（不静默丢弃）；stress 角色完全不加约束；
  - `reference_feasible`：解析时间窗判定——行人占据遮挡车走廊 y=occluder_y±1.2 m 的时间窗
    与遮挡车占据 crossing_x±2.55 m 的时间窗间隔 ≥0.5 s。**只读 spec 参数、不运行任何被评策略**，
    属采样时构造性设计规则而非按模型结果筛选（区分依据：判定输入不含任何策略输出；规则先于样本存在；
    重采样次数全部留痕于 manifest）。
  - `export_conditions/load_conditions`：条件清单 JSON（含 condition_set_version、purpose∈
    {diagnostic,development,heldout}、resample_counts、逐条 spec）。
- `evaluate()` 新增 `conditions/condition_set_version`：从清单加载时逐条执行，消除旧版 rng 顺序消费
  导致的"只评 single 与连评时 dual spec 不同"的耦合；header 记录三版本字段；清单版本与 spec 不一致时拒绝执行。
- seed1000×20（已被查看）永久标记 `purpose=diagnostic`；P2 冻结保留集必须用新 seed 且 purpose=heldout。
- 导出产物：
  ```bash
  python scripts/export_conditions.py --output runs/20260911_p0_conditions/pilot_seed1000_v1.json \
    --seed 1000 --count 20 --branches single dual --sampler-version 1 --purpose diagnostic
  python scripts/export_conditions.py --output runs/20260911_p0_conditions/ref_v2_seed1000.json \
    --seed 1000 --count 20 --branches single dual --sampler-version 2 --role reference --purpose diagnostic
  # v2: resamples dual=[15,2,1,0,0,7,0,3,0,18,0,0,2,4,28,0,0,0,...], reference_infeasible=0
  ```
- v1 清单与 pilot 逐字段对齐验证：40 条 spec 与 `runs/20260911_gpu_pilot/eval_script/episodes.jsonl`
  的 spec 全字段一致（0 mismatch）；锚点值固化在 `tests/test_sampling.py::PILOT_ANCHOR`。

### 4. 失败分层诊断（scripts/diagnose_failures.py，新增）

```bash
python scripts/diagnose_failures.py runs/20260911_gpu_pilot \
  --conditions runs/20260911_p0_conditions/pilot_seed1000_v1.json \
  --output runs/20260911_p0_diagnosis --device cuda --max-figures 10
```

- 分层：同一 scenario_id 上 script 基线 vs 各策略 → `script_conflict:<reason>` / `policy_added_conflict:<reason>` /
  `script_dangerous_policy_valid` / `both_valid`（互斥，计数总和=attempts，无效尝试不消失）。
- 轨迹再生确定性：10 条代表轨迹全部从（policy bundle + spec + seed）完整复现，
  min_clearance 与存储值一致到小数 4 位、结局（invalid_reasons/collision）全同（`regeneration_mismatches: []`）。
- 图：左 x-y 平面（1 Hz oriented footprint、碰撞红星、ped-occluder 接触 X）；右时间曲线（速度、
  ego-ped/ped-occluder clearance、等待期灰带、遮挡黄纹）。挑图规则固定（每层 scenario_id 升序第一条 +
  collision_speed 最大一条），禁止目测挑选。颜色 Okabe-Ito 三色经 CVD 验证器通过。
- **主要发现（physics v1 / sampler v1 下的故障定位，非方法优劣）**：
  1. dual 分支所有方法 `script_conflict:target_target_collision` 恒为 9/20——结构性初始化缺陷（行人北上
     必穿遮挡车走廊，时间窗高度重叠），与策略无关 → 由 sampler v2 修正；
  2. `dual_seed7` 在 **single 分支有 13/20 pedestrian_role 违规**（REPORT.md 只写 35% 有效率，未分层原因）；
     其余所有方法 single 分支零策略新增冲突——dual→single 迁移失败的模式是角色违规而非几何；
  3. `mixed_seed17` dual 分支 7 次 occluder_role + 1 次策略新增碰撞（仅 1 条 both_valid）——该种子病理化；
  4. `dual_seed7` 在 dual-00003 上把 script-valid 场景打成 min_clearance=0（script_dangerous_policy_valid 层）。
- 完整表格见 `runs/20260911_p0_diagnosis/DIAGNOSIS.md`、`layers.csv`、`per_condition.csv`。

### 5. 测试

- 改后全量：**53 passed in 9.67s**（40 既有 + 13 新增）。
- 新增 `tests/test_physics_compat.py`（6）：等待期完全静止/上报为零/到期起步；等待期转向无效且无假
  pedestrian_role；v1 legacy 语义保持（含 quirk 本身）；双版本 trace 回放往返；无版本字段旧 trace 默认 v1
  +warning；v1 trace 强制 v2 语义必须 mismatch（防 v1 分支被删）。
- 新增 `tests/test_sampling.py`（7）：v1 快照锚点（pilot spec 逐字段）；v2 200 条 reference 全部可行且
  script rollout 20 条 dual 零 target_target_collision；stress 无约束且保留难例；dispatch 等价；清单往返
  （含"同清单评 dual-only 与连评 spec 相同"直击 rng 耦合）；版本错配/缺字段拒绝。
- 过程中的失败与修复（保留证据）：v2 初版只重采样三元组 → 29/200 不可行（行人 y/速度是时间窗关键维度，
  且 v1 本就不做跨分支参数配对）→ 扩为五元组；`sample_spec` 分发默认 max_resamples 与 v2 不一致（12 vs 40）
  导致 dispatch 测试失败 → 统一；物理冻结测试初版用 dual 分支因遮挡 KeyError → 改 single。
- 既有测试在 v2 代码路径下全部通过（默认 spec delay=0，行为不变；无既有测试因 v2 失效需改写）。

### 6. v2 条件下的评价（跨版本对照，仅诊断意义）

```bash
bash scripts/run.sh evaluate --output runs/20260911_p0_eval_v2_script \
  --conditions runs/20260911_p0_conditions/ref_v2_seed1000.json
bash scripts/run.sh evaluate --policy runs/20260911_gpu_pilot/mixed_seed7/policy.pt \
  --output runs/20260911_p0_eval_v2_mixedseed7 \
  --conditions runs/20260911_p0_conditions/ref_v2_seed1000.json --device cuda
```

（结果见 §8 对照表。）

### 7. 版本与重跑标记

| 旧产物 | 状态 |
|---|---|
| `runs/20260911_gpu_pilot/eval_*` 全部指标 | **physics_v1/sampler_v1 legacy，仅历史参考**（不只 dual——single 中 delay>0 条目的 AEB 触发同样依赖假 vx/vy 上报）。诊断结论（§4）仍有效，因为它描述的就是 legacy 行为 |
| `runs/20260911_gpu_pilot/mixed_*/policy.pt` 等检查点 | 可加载（bundle schema 0.1 与 actor 结构未变）；正式重训待 P2 等预算方案 |
| `search_*`、`perturbations`、`latency` | 不受行人物理影响（latency 纯推理计时），保留 |
| `runs/20260911_p0_smoke*` | 改动前 smoke，physics v1 |
| 交叉版本对照 | 一切 v1/v2 数字并列必须标注 physics/sampler 版本，禁止混排成趋势 |

### 8. 对照表（同一 v2 条件清单 `diagnostic_seed1000_samplerv2`，physics v2）

| 方法 | 训练物理/采样 | 分支 | attempts | 有效率 | 危险率(分母含无效) | 碰撞率 | mean_clearance | invalid 明细 |
|---|---|---|---:|---:|---:|---:|---:|---|
| script | — | single | 20 | 1.00 | 0.15 | 0.10 | 1.31 m | 无 |
| script | — | dual | 20 | **1.00** | 0.25 | 0.10 | 0.76 m | **无**（v1 同 seed 为 0.55 有效/9 次目标间碰撞） |
| mixed_seed7 (旧 ckpt) | v1/v1 | single | 20 | 1.00 | 0.15 | 0.15 | 1.30 m | 无 |
| mixed_seed7 (旧 ckpt) | v1/v1 | dual | 20 | **1.00** | 0.30 | 0.10 | 0.74 m | **无**（v1 同 seed 为 0.60 有效/8 次目标间碰撞） |

- 危险率簇 bootstrap 95% CI：script single [0.05,0.30]、dual [0.10,0.45]；mixed_seed7 single [0.05,0.30]、dual [0.10,0.50]。
- **解读边界**：跨版本对照（v1 训练的 checkpoint 在 v2 条件下评价）只证明几何修正消除了无效性来源，
  不构成方法比较；n=20 的差异（0.25 vs 0.30）在 CI 内不可区分。危险率与 v1 相近说明 v2 采样在消除
  结构性冲突的同时保留了危险场景——"真实的可执行性修正而非难度下降"。
- 产物：`runs/20260911_p0_eval_v2_script/`、`runs/20260911_p0_eval_v2_mixedseed7/`
  （summary.json 含 physics_version=2/sampler_version=2/condition_set_version）。

### 9. 未解决问题 / 依赖用户确认 / 仍属假设

- `perturb_spec` 全部参数仍为 `assumed_sensitivity_not_abd_calibrated`（P3 ABD 校准未开始）。
- v2 起步瞬时跳变近似已文档化；是否升 v2.1（jerk 受限起步）待评审意见。
- invalid 提前终止导致各方法交互步数不等（公平性缺陷）——P2 处理，本轮只记录不改动。
- `summarize_pilot.py` 写死 v4 语料路径与 pilot 文案，复用前需参数化（P1 顺手处理）。
- ABD `manual_review.csv` 后 7 列仍空（依赖用户/人工核对，P3）。
- seed17 病理化（7 次 occluder_role）原因未查——属 P2 种子敏感性分析范围。

### 10. 本轮修改文件清单

- 新增：`scenario_lab/sampling.py`、`scripts/export_conditions.py`、`scripts/diagnose_failures.py`、
  `tests/test_sampling.py`、`tests/test_physics_compat.py`、`docs/GLM_CHANGELOG.md`（本文件）。
- 修改：`scenario_lab/schema.py`（3 版本字段 + EpisodeRecord 0.2）、`scenario_lab/env.py`
  （v2 行人物理 + v1 legacy 分支 + reset 静止初速 + sample_spec 改由 sampling re-export）、
  `scenario_lab/evaluate.py`（conditions 加载、replay 版本兼容、summary 版本字段）、
  `scenario_lab/__main__.py`（evaluate --conditions）。
- 未触碰：用户既有未提交改动（HANDOFF.md、stage2_extract.py、abd_parser/inventory 删除记录）、
  runs/20260911_gpu_pilot 全部旧产物、Data/。

---

## 2026-09-11 · P1.1：语料交叉统计 + 零动作基线 + 分角色验证误差（只量化现状，不改模型）

**执行环境**：同上（WSL2 Ubuntu，scenario-gpu venv）。产物：`runs/20260911_p1_corpus_stats/`
（cross_table.csv、zero_action_vs_prior.csv、CORPUS_STATS.md、summary.json）。

```bash
python scripts/corpus_stats.py   # 新脚本；corpus=runs/20260911_public_v4, prior=runs/20260911_gpu_pilot/prior/prior.pt
```

### 1. 覆盖交叉表（source × split × kind，12 格中 5 格为空，每格注明原因）

| 发现 | 数值 |
|---|---|
| 行人占比 | 191/6601（2.9%），**全部来自 Waymo**；INTERACTION×pedestrian 为结构性零（vehicle_tracks 导出无行人行） |
| val split 构成 | 1122 条 = Waymo vehicle 1030 + Waymo pedestrian 92，**INTERACTION 贡献为零**（其 4 个 location 组全部 hash 进 train 桶；recorded_trackfiles 文件名无官方 split token） |
| test split | 148 条（Waymo training tfrecord 的 per-scenario hash 桶 9；validation 目录被 official token 强制 val） |

比 HANDOFF 指出的"INTERACTION 验证集来源×角色覆盖缺口"更严重的一层：**prior 从未在 INTERACTION
分布上被验证过**（不只行人缺失——车辆角色同样缺失），train/val 的来源构成完全不同
（train 68% INTERACTION，val 0% INTERACTION）。

### 2. 零动作基线 vs 训练后 prior（val 1122，pretrain.py 同口径）

| 口径 | 零动作 | trained prior |
|---|---:|---:|
| 全角色无加权（val_mse 口径） | **0.237336** | 0.237515（劣 0.08%） |
| 全角色 train 角色频率加权 | 0.319599 | 0.319932 |
| role 0 pedestrian | 0.337935 | 0.338304 |
| role 1 vehicle | 0.228351 | 0.228512 |

- **所有口径上 prior 均不优于零动作**（pretraining.json 5 epochs 的 val_mse 0.2331→0.2333 也几乎
  不降）。这量化解释了 pilot 中 prior 与 script 评价几乎一致的现象：5 epoch 的自身运动先验
  收敛到"近似零动作"——动作目标已归一化且多数窗口近匀速，零预测是强基线。
- 口径自检 PASS：本脚本在 logged val[:1024] 子集复算 prior 得 0.23327850，与 pretraining.json
  末轮 val_mse 0.23327853 差 3e-08——上表全部数字与 pretrain.py 口径逐位一致。
- role 0（行人）零动作误差 0.338 > role 1（车）0.228：行人窗口动作目标方差更大，而行人样本仅
  191 条——先验最薄弱处恰是本项目场景（横穿行人）最需要的角色。

### 3. 对 P1.2 的输入（不改任何模型的纯诊断结论）

P1.2 邻居对齐交互先验的设计必须同时回应：val 零 INTERACTION 构成（换 split 策略或在 Waymo 内
做来源平衡）；行人样本量（扩样并记录筛选依据，HANDOFF 已列）；超越零动作的最低门槛
（新先验 val 报告必须并列零动作基线，这是本脚本固化下来的对照协议）。

### 4. 本轮修改文件

- 新增：`scripts/corpus_stats.py`（审计脚本，含 pretraining.json 口径自检，自检不过即报错退出）。
- 产物：`runs/20260911_p1_corpus_stats/`（未提交，属 runs/ 大产物）。
