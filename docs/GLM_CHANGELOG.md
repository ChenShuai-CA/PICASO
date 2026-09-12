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

---

## 2026-09-11 · P1.2 预注册（先于任何 v5 语料构建与训练落笔；用户已确认范围与判据）

### R1. 动机（来自 P1.1 审计的三个量化事实）

1. val split 100% 来自 Waymo（INTERACTION 12 location hash 出 11 train/1 test/0 val）——prior 从未
   在 INTERACTION 分布上验证；2. prior val MSE 0.2375 ≥ 零动作 0.2373（所有口径同向）；3. 行人
   2.9% 全 Waymo。本轮加入邻居对齐交互上下文与 INTERACTION 行人导出（166 个
   pedestrian_tracks CSV，v4 被 `startswith('vehicle_tracks_')` 过滤排除）。

### R2. 语料 v5 构成规则（固定，不依结果调整）

- INTERACTION split 改为**显式 location 映射**（split_group 函数不动，prepare 层覆盖）：
  规则 = 按 `sha256('interaction_holdout_v5\x1f{location}')` 十六进制升序取末 2 个为 val。
  计算结果（2026-09-11 落笔前算定）：**val = DR_USA_Roundabout_SR、DR_DEU_Roundabout_OF**；
  train = DR_USA_Roundabout_EP、DR_USA_Intersection_EP1、DR_USA_Intersection_GL、
  DR_USA_Roundabout_FT、DR_USA_Intersection_MA、DR_USA_Intersection_EP0。INTERACTION 不设
  test 桶；test 保持 Waymo-only。官方 validation-set-list 文件为空（官方 val 划分不可得），
  故用本规则替代 location 级 hash（12 location hash 出 0 个 val 的既成事实）。
- 邻居对齐：同 group（INTERACTION=location、Waymo=sid）共同时间戳（整数帧键 round(t*10)，
  100ms 网格精确匹配不插值）≥8 帧的候选里，按窗口内平均距离取每 kind top-1，>40m 不填；
  窗口内邻居身份固定（禁止逐帧切换）。特征：旋转到 anchor 朝向系，/40,/10,/15 + kind one-hot
  （OBS_DIM 布局同 env.observe）；anchor 自身槽编码与 v4 完全一致；行人 anchor 的车邻居填
  slot2、车 anchor 的行人邻居填 slot1、slot0 恒空。
- 预算：max_examples=20000，INTERACTION 每 location 文件上限 4，行人目标占比 ≥20%，
  Waymo 沿用 4 tfrecord（2 train+2 val）。产物 runs/20260911_p1_corpus_v5/。
- **行人筛选依据（首轮构建时发现并记录）**：INTERACTION 行人导出的 `agent_type` 唯一取值是
  官方混合类 `pedestrian/bicycle`（首轮语料 0 个 INTERACTION 例子的根因：映射表未收录 →
  kind='other' 被丢弃）。v5 接受该混合类映射为 'pedestrian'（官方格式事实，非我们自己的
  判别；Waymo 侧维持既有 unambiguous 规则），桶内文件排序改为 location 主序（否则名字序让
  pedestrian 文件占满全部选择名额）。首轮 548 例全 Waymo 的废语料已删除重建，此段即留痕。

### R3. 训练变体（超参钉死：epochs=10、hidden=64、seed=7、lr=3e-4，不做任何调参）

1. `prior_v5_self`：v5 语料，self-only 视图（邻居槽 token_mask 置 False，同一份 npz——
   消融不引入语料差异）；
2. `prior_v5_nb`：v5 语料，neighbor 视图。
对照：零动作基线（解析）；旧 prior_v4（v4 语料 5 epochs 训练）在 v5 val self-only 视图上推理，
仅作跨语料参考行，不进判据。

### R4. 判据与统计（用户 2026-09-11 确认）

- **主判据**：prior_v5_nb 的 val MSE 低于零动作基线，且两者 95% CI 不重叠；
- **次判据**：prior_v5_nb 低于 prior_v5_self，且 95% CI 不重叠；
- bootstrap：聚类单位 (source, group_id)，B=2000，seed=2026；全角色 + 分角色（role0/role1）
  两版；INTERACTION 侧组粒度为 location 级（val 仅 2 组），另出 per-source 分层数字并如实
  标注统计力限制。
- 判据结果 PASS/FAIL 均如实落盘；FAIL 即负结果（P4 论文的合法内容），禁止事后换标准或
  按结果回调超参/规则。

---

## 2026-09-11 · P1.2 实施：语料 v5 + 邻居对齐 + 两变体训练 + 预注册判据判定（结果：双 FAIL 负结果）

**执行环境**：同 P0。产物：`runs/20260911_p1_corpus_v5/`（语料）、`runs/20260911_p1_prior_self/`、
`runs/20260911_p1_prior_nb/`（训练）、`runs/20260911_p1_eval/`（正式对照与判定）。

### 1. 语料 v5 构建：五轮失败与修复（全部留痕，最终构成见 §2）

| 轮次 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | 548 例全 Waymo，INTERACTION 0 例 | 行人导出 `agent_type` 唯一值是官方混合类 `pedestrian/bicycle`，不在 `AGENT_TYPE_KIND` → kind='other' 被丢 | data.py 加映射（官方格式事实，非自判别；预注册 R2 已补记） |
| 2 | 32 个入选文件全 pedestrian_tracks | 桶内按文件名排序，pedestrian_tracks 字母序占满预算 | 桶排序改 (location, name)——location 主序交叉两族文件 |
| 3 | 混入 DR_CHN_*/DR_DEU_Merging 等映射外 location（走 hash 进 train 桶挤名额） | 显式映射只覆盖参与名单，未排除名单外文件 | `location_splits` 兼作参与名单过滤 |
| 4 | vehicle 仅 12.6% | per_location=4 在行人文件 ≥4 的 location 把 vehicle 文件截光 | 每 location 文件族前缀平分（各 per_location//2） |
| 5 | Waymo 548 例（v4 为 2082） | records_per_file=8 → 每 tfrecord 只解 8 个 scenario | `--records-per-file 32` |

每轮失败语料目录已删除重建（脚本 FileExistsError 防重入设计使然）；最终语料构成与第 5 轮修复后一致。

### 2. 最终语料构成（runs/20260911_p1_corpus_v5，交叉表全达标）

**10770 例**：INTERACTION 8750 + Waymo 2020；train 7404 / val 3228 / test 138；
vehicle 7833 / pedestrian **2937 = 27.3%**（预注册 ≥20% ✓）；独立组 135；
`cross_split_groups_quarantined=0` ✓；`windows_with_neighbors=10757`（99.9%）。
val 构成：INTERACTION vehicle 1500 + pedestrian 654（SR+OF 两 location、两 kind 均非零 ✓）+
Waymo vehicle 986 + pedestrian 88——**val 67% 来自 INTERACTION，v4 的"val 零 INTERACTION"缺口消除**。
INTERACTION×test 为零（预注册 R2：INTERACTION 不设 test 桶，test 保持 Waymo-only，如实记录）。

构建命令（参数由 report.json 与终端历史复原，report.json 为准）：
```bash
python -m scenario_lab prepare-public --output runs/20260911_p1_corpus_v5 --max-files 4 \
  --max-files-interaction 32 --records-per-file 32 --max-examples 20000 \
  --include-pedestrians --per-location 4 --location-splits-v5
```

### 3. NaN 训练崩溃与修复（语料构成不变）

首轮训练 prior_v5_nb 在 epoch 1 内崩于 `Normal()` 断言（log_prob 全 NaN）：npz 内 6 个非有限
token。根因：INTERACTION 行人无 psi_rad（heading 速度推导，低速 NaN）；anchor 自身窗口有
isfinite 检查，但**作为邻居被编码时无检查**——低速邻居把 NaN 注入 token。修复（双保险）：
`_neighbor_candidates` 对候选帧 (xy, velocity, heading) 加与 anchor 相同的有限性检查；
`pretrain()` 对 tokens 加 finite 硬拒绝（非有限即 raise，防再训）。语料原样重建后构成数字逐项
一致，训练正常完成。教训留痕：v4 无邻居路径故从未暴露；邻居编码引入了第二类数据质量入口。

### 4. 训练两变体（超参 = 预注册 R3：epochs=10、hidden=64、seed=7、lr=3e-4，不调参）

```bash
python -m scenario_lab pretrain --corpus runs/20260911_p1_corpus_v5/motion_prior.npz \
  --output runs/20260911_p1_prior_self --epochs 10 --device cuda --no-neighbors
python -m scenario_lab pretrain --corpus runs/20260911_p1_corpus_v5/motion_prior.npz \
  --output runs/20260911_p1_prior_nb --epochs 10 --device cuda
```

| 变体 | ep1 train/val | ep10 train/val | 最佳 val |
|---|---|---|---|
| prior_v5_self | 0.17982 / 0.19147 | 0.17687 / 0.19221 | **0.19085**（ep9） |
| prior_v5_nb | 0.17972 / 0.19172 | 0.17615 / 0.19282 | **0.19137**（ep9） |

train_mse 持续下降而 val_mse 微升（gap ≈ 0.016）——记忆化而非泛化；两变体 val 曲线几乎重合。

### 5. 正式对照与预注册判据判定（runs/20260911_p1_eval）

```bash
python scripts/corpus_stats.py --corpus runs/20260911_p1_corpus_v5 --output runs/20260911_p1_eval \
  --priors prior_v5_self=runs/20260911_p1_prior_self/prior.pt prior_v5_nb=runs/20260911_p1_prior_nb/prior.pt \
  --reference prior_v4_cross=runs/20260911_gpu_pilot/prior/prior.pt \
  --verdict "prior_v5_nb,zero-action baseline" --verdict "prior_v5_nb,prior_v5_self" \
  --bootstrap 2000 --seed 2026
```

- 工程留痕：首轮跑用下划线名 `zero-action_baseline` 与默认空格名不匹配，主判据对未生成
  verdict；用引号传空格名重跑（同 seed 同数据，点估计与 CI 完全一致）后正式落盘。
- 本轮脚本升级：每个 prior 按 bundle 内保存的 `use_neighbors` 恢复自身训练视图评估
  （prior_v5_self 若在全量 mask 下评估会与其 logged val_mse 口径不符）；新增 `--reference`
  通道（跨语料参考行：v4 prior 在 v5 val self-only 视图推理，自检不适用、不进判据）；
  `--verdict` 改可重复多对。
- 口径自检：两 v5 变体在 val[:1024] 复算与 pretraining.json 末轮 val_mse 差 ~1e-7，PASS。

**判定结果：主判据、次判据全部切片 FAIL（预注册 R4 口径，负结果如实落盘）**

| 判据 | 切片 | below MSE [95% CI] | above MSE [95% CI] | 不重叠 | 判定 |
|---|---|---|---|---|---|
| 主：nb < 零动作 | all roles | 0.196315 [0.1746, 0.2496] | 0.193648 [0.1706, 0.2476] | 否 | **FAIL** |
| 主：nb < 零动作 | role 0 ped | 0.176483 [0.1319, 0.3588] | 0.175638 [0.1315, 0.3582] | 否 | **FAIL** |
| 主：nb < 零动作 | role 1 veh | 0.202235 [0.1900, 0.2419] | 0.199024 [0.1852, 0.2390] | 否 | **FAIL** |
| 次：nb < self | all roles | 0.196315 [0.1746, 0.2496] | 0.195709 [0.1733, 0.2503] | 否 | **FAIL** |
| 次：nb < self | role 0 ped | 0.176483 [0.1319, 0.3588] | 0.176123 [0.1316, 0.3595] | 否 | **FAIL** |
| 次：nb < self | role 1 veh | 0.202235 [0.1900, 0.2419] | 0.201555 [0.1886, 0.2425] | 否 | **FAIL** |

完整 4 模型 × 6 切片表（含 role 加权口径、per-source 分层、v4 跨语料参考行）见
`runs/20260911_p1_eval/CORPUS_STATS.md` 与 `mse_table.csv`。

### 6. 解读（工程诊断边界内，非论文证据）

1. **负结果与 P1.1 同向且更干净**：v5 val 全角色 MSE 排序 零动作 0.19365 < prior_v4_cross
   0.19385 < prior_v5_self 0.19571 < prior_v5_nb 0.19632——所有学习变体点估计不低于零动作，
   且 v4 在 v5 语料上同样≈零动作。"动作目标已归一化 + 多数窗口近匀速 → 零预测强基线"的解释
   在扩大语料（+行人 27%）、显式 INTERACTION 留出、加邻居上下文后依然成立。
2. **邻居特征零净增益**（次判据方向为负）：self 略优于 nb。在 hidden=64 / 10 epochs 公平预算下，
   top-1 邻居 token 未提供可用的预测信息；结合 §4 的 train/val gap，模型容量或预算不足以利用
   交互上下文——但按预注册超参不回调，此备择解释如实保留。
3. **来源间差异主导**：零动作口径下 INTERACTION val 0.1709 vs Waymo val 0.2392（差 0.068），
   远大于任何模型间差（<0.003）；4 个模型在两 source 上排序一致。val 指标被来源构成
   （67% INTERACTION）主导——后续一切 val 比较必须分层报告（本表已含 per-source 切片）。
4. **CI 宽度 = 统计力限制的量化**：role0 CI [0.1315, 0.3582] 极宽，因 INTERACTION 行人 654 例
   聚在 2 个 location 组、val 共仅 ~88 组（INTERACTION 2 + Waymo 86）——组级聚类 bootstrap 的
   方差主导 CI。这是预注册风险清单写明的限制，非新发现；如需有统计力的 CI 判定，未来需更多
   独立 val 组（INTERACTION case 级组是候选项，但须先论证 case 级无泄漏）。
5. **对 P2 的输入**：开环 val MSE 无法区分先验价值——P2 公平预算评价中先验应作为策略训练的
   初始化/正则来检验，零动作与 prior_v5_* 均作对照行保留。负结果本身是 P4 可写的预注册内容。

### 7. 测试

- 全量 **60 passed**（53 既有 + 7 新增 tests/test_neighbor_alignment.py）；预算断言随 per-group
  3:2 配额更新为 30（tests/test_pretrain.py，带注释）。
- 新增测试覆盖：邻居未来帧篡改→窗口输入不变（构造性无泄漏）；anchor 未来帧篡改→仅 target 变；
  旋转几何（anchor π/4、邻居正北 5m → (5/√2, 5/√2)）；窗口级 top-1 选择与 >40m 拒绝；
  heldout location 组绝不进 train；location split 映射尊重。

### 8. 版本与重跑标记

| 产物 | 状态 |
|---|---|
| `runs/20260911_public_v4`（v4 语料）与旧 prior | 不变，仅历史参考（P1.1 已标） |
| `runs/20260911_p1_corpus_v5` | 当前唯一语料（feature_version=neighbor-v1；corpus_sha256 已存入两 prior bundle） |
| prior_v5_self / prior_v5_nb | 判定 FAIL，**不得作"改进"宣称**；保留作 P2 对照行 |
| `runs/20260911_p1_eval` | 预注册判定正式落盘（summary.json 含 6 条 verdict 全记录） |
| 交叉版本并列 | 任何 v4/v5 数字并列必须标注 corpus_version，禁混排成趋势 |

### 9. 本轮修改文件

- 修改：`scenario_lab/pretrain.py`（v5 显式留出+邻居对齐窗口+per-group 配额+use_neighbors 消融
  +finite 硬检查）、`scenario_lab/data.py`（AGENT_TYPE_KIND 加 pedestrian/bicycle）、
  `scenario_lab/__main__.py`（prepare-public/pretrain 新参数）、`scripts/corpus_stats.py`
  （多 prior+聚类 bootstrap+多 verdict+训练视图恢复+--reference）、`tests/test_pretrain.py`。
- 新增：`tests/test_neighbor_alignment.py`。
- 未触碰：用户既有未提交改动（HANDOFF.md、stage2_extract.py、abd_parser/inventory 删除记录）、
  runs/ 既有产物、Data/。

### 10. 未解决问题

- CI 统计力不足（§6.4）——不改聚类单位（预注册钉死），P2 若需更强判定再预注册新规则。
- hidden=64/10 epochs 欠拟合与容量不足的备择解释无法排除（超参不回调，§6.2）。
- Waymo 地图解码后置（P1.2d，未开始）。
- 语料 token 为 anchor 旋转系编码，与 env.observe 全局相对编码存在迁移 gap（v4 起即如此，
  P2 前不解决）。
- 行人 heading 低速 NaN：anchor 侧靠 speed≥0.3 过滤兜底，未做系统性抽查统计（本轮仅修复
  邻居侧注入路径）。

---

## 2026-09-11 · P2 预注册（先于任何 P2 代码与实验落笔；用户已确认先验/规模/预算三项）

### R1. 动机（HANDOFF §7 P2 五条 + P0 遗留两项）

1. 等交互步数公平性：invalid 提前终止使各方法 episode 长度不等，相同 update/episode 数不等价
   （P0 §9 遗留）；2. 三类成本（训练总成本/固定策略生成成本/每场景再搜索成本）须分账并列，
   CEM best-of-search 不得与策略一次采样直接比；3. 参数/动作序列搜索目前各只优化一个固定案例，
   须覆盖多初始条件且无效尝试计入预算与分母；4. 缺独立自然性指标与可避让性诊断（奖励函数不能
   自证自然性）；5. 敏感性对照须有/无稳健性后训练在相同扰动种子上比较（pilot 只评过主模型
   扰动）；6. seed1000×20 已被查看=诊断集，正式保留集必须重新冻结且不得用于挑模型；
   7. pilot 全部 checkpoint 为 physics v1/sampler v1 训练，正式矩阵须在 v2 下重训。

### R2. 条件集冻结（生成即固定，不依结果调整）

- **dev**：seed 31000、40 条件/分支（single+dual 共 80）、sampler v2、role=reference、
  physics v2、controller=stopping、purpose=development。用于本轮全部评价与迭代。
- **heldout**：seed 41000、同规格、purpose=heldout。**生成即封存，本轮任何命令不得读取**；
  待 dev 侧结论稳定、判据口径不变更后作一次性最终评价（独立后续步骤）。
- 危险阈值：`dangerous` 实现定义（valid 且 collision 或 min_clearance<0.5 m）；有效率分母含
  无效尝试（无效拉低危险率，禁止靠制造无效刷高）；去重：`evaluate.signature()` 参数+结局粗 bin
  （实现既有，冻结沿用）；聚类单位：scenario_id；bootstrap：组率 cluster bootstrap，
  **B=2000、seed=2026**（evaluate.summarize 的 b_rounds 由硬编码 1000 参数化为 2000，属口径
  升级，pilot 旧 summary 不重算，新表注明 B）。

### R3. 训练矩阵（physics v2 / sampler v2；名义预算 = 24 updates × 4 episodes = pilot 的 2 倍，与 pilot 数字不可比，如实标注）

超参全部 TrainConfig 默认（epochs=3、hidden=64、lr=3e-4、mappo、mixed、robust=True、
role_constraints=True），只变 seed/先验/robust/sampler_version（新字段，默认 1 兼容旧路径）：

| 组 | 配置 | 对应判据 |
|---|---|---|
| A7/A17/A27 | prior=prior_v5_nb、robust、sampler v2、seed 7/17/27 | 主判据（3 种子） |
| B7 | 同 A7 但无 --pretrained | 次判据 1（先验价值） |
| C7 | 同 A7 但 --no-robust | 次判据 2（稳健性机制） |

总环境步数逐 update 落盘（training.jsonl steps），组间偏差如实报告。

### R4. 评价与判据（全部在 dev 集；CEM 与可避让性为参考/诊断行）

- **固定方法一次采样**：script（零动作基线）+ A7/A17/A27/B7/C7 × dev 80 条件 × 1 扰动。
- **主判据**：A 组 3 seed 合并——每 scenario 的组率 = 同 scenario 3 次尝试（3 个 seed 模型各一）
  的 dangerous 均值，对 80×2 分支内 scenario 聚类 bootstrap（B=2000 seed=2026）；PASS 条件 =
  合并危险率点估计高于 script 且 95% CI 不重叠（A 上界侧在 script 之上）。任一方向 FAIL 如实报。
- **次判据 1（先验）**：B7 vs A7（同 seed、同条件集），同口径 CI 判定。
- **次判据 2（稳健性）**：A7 与 C7 × dev × 10 扰动/条件（相同扰动种子序列：seed+i*100+j）；
  指标 = 每 scenario 跨扰动 dangerous 率的组内 std（中位数）+ 合并危险率；PASS 条件 = A7 的
  std 中位数低于 C7 且合并危险率点估计不低于 C7（CI 并列报告）。**预注册修正留痕
  （2026-09-11，任何 P2 实验运行前）**：初稿把"危险率不劣"写成 CI 分离（下界≥C7 上界），
  这要求稳健性机制同时显著*提高*危险率，与该机制宣称的稳定化语义不符；改为点估计不低于，
  CI 并列报告不进 PASS 条件。此为设计修正而非按结果回调（时点：训练矩阵未启动）。
- **CEM 多条件参考行**（不进判据）：dual 40 条件 × {parameters, trajectory} × budget 40/条件；
  落盘 total_evaluations / total_interaction_steps / valid_attempts（无效尝试计入）；输出标注
  best-of-search，禁止与一次采样行直接排序。
- **可避让性诊断**（参考行）：A 组 3 seed 在 dev 上 controller=ttc 反事实重评；
  avoidable_fraction = stopping 下危险且 ttc 下不危险的条件比例；标注为近似（仅替换 ego 制动律，
  非完整避让可行性判定）。
- **成本三列分账**（每方法×分支并列）：训练总步数、固定策略评价步数（total_steps）、
  CEM 每条件平均步数+总评估数。

### R5. 措辞边界

dev 集一切数字 = 开发诊断，不是论文证据；"优于基线"表述必须等 heldout 一次性评价通过主判据后
才可进入论文主张（本轮不评 heldout）。判据 PASS/FAIL 均如实落盘；FAIL 即负结果，禁止事后换
标准、按结果回调超参/条件集/阈值。

---

## 2026-09-11 · P2 实施：训练矩阵 + dev 评价矩阵 + 预注册判据判定（结果：主判据与两个次判据全 FAIL）

**执行环境**：同 P0。产物：`runs/20260911_p2_train_{a7,a17,a27,b7,c7}/`、
`runs/20260911_p2_eval_{script,a7,a17,a27,b7,c7,a7_perturb,c7_perturb,a7_ttc,a17_ttc,a27_ttc}/`、
`runs/20260911_p2_search_{param,traj}/`、`runs/20260911_p2_summary/`（comparison.csv、verdicts.csv、
sensitivity.csv、avoidability.csv、REPORT.md、comparison.png/svg）。

### 1. 条件集冻结（先于训练）

- dev：seed 31000、80 条件（single 40 + dual 40）、sampler v2 reference 重采样最多 35 次、
  `reference_infeasible=0`、version=`development_seed31000_samplerv2`。
- heldout：seed 41000、80 条件、dual 重采样最多 40 次、`reference_infeasible=1`（保留并计数）、
  version=`heldout_seed41000_samplerv2`。**生成后封存，本段任何命令未读取。**

### 2. 训练矩阵（预注册 R3 逐字执行，无调参）

```bash
python -m scenario_lab train --output runs/20260911_p2_train_a7 --updates 24 \
  --episodes-per-update 4 --seed 7 --sampler-version 2 --device cuda \
  --pretrained runs/20260911_p1_prior_nb/prior.pt        # a17/a27 同式换 seed
python -m scenario_lab train --output runs/20260911_p2_train_b7 ...（无 --pretrained）
python -m scenario_lab train --output runs/20260911_p2_train_c7 ... --no-robust
```

| 组 | bundle 校验 | 训练总步数 | 末 update valid_rate | 备注 |
|---|---|---:|---:|---|
| a7 | pretrained=T, robust=T, sampler=2 | 5752 | 0.5 | |
| a17 | pretrained=T, robust=T, sampler=2 | 6085 | **0.0**（robust_utility=-1.0） | **reference_kl 0.0017→0.35 单调上漂**（a7/a27 稳定在 <0.05） |
| a27 | pretrained=T, robust=T, sampler=2 | 5716 | 0.5 | |
| b7 | pretrained=F, robust=T, sampler=2 | 6190 | 0.5 | |
| c7 | pretrained=T, robust=F, sampler=2 | 5665 | 0.5 | |

- 组间步数偏差 ±4.5%（5665–6190）——sampler v2 重采样使 episode 长度本身变化，如实分账。
- **训练期红旗（先于评价即已可见）**：a17 是唯一 reference_kl 大幅上漂且末段 valid_rate 塌到 0
  的组——策略在训练中即坍缩，不是评价期偶发。

### 3. 评价矩阵（全部 dev 80 条件；执行种子 1000=CLI 默认，全方法共享同一扰动序列）

13 个 run：script + 5 模型（×1 扰动）、a7/c7 ×10 扰动（敏感性）、CEM dual×{parameters,
trajectory} budget 40、A 组 ×controller=ttc。全部成功，无重跑。

### 4. 对照表（完整表见 `runs/20260911_p2_summary/REPORT.md`，B=2000 seed=2026 cluster bootstrap）

| 方法 | 分支 | 危险率 [95% CI] | 有效率 | 尝试 | 独立场景 | 评价步数 | 训练步数 |
|---|---|---|---:|---:|---:|---:|---:|
| script | single | 0.2500 [0.125,0.375] | 1.000 | 40 | 40 | 2976 | — |
| script | dual | 0.3500 [0.200,0.500] | 1.000 | 40 | 40 | 2737 | — |
| a7 | single | 0.2500 [0.125,0.375] | 1.000 | 40 | 40 | 2975 | 5752 |
| a7 | dual | 0.1000 [0.025,0.200] | **0.100** | 40 | 40 | 1438 | 5752 |
| a17 | single | 0.0500 [0.000,0.125] | **0.050** | 40 | 40 | 1989 | 6085 |
| a17 | dual | 0.1250 [0.025,0.225] | **0.125** | 40 | 40 | 2046 | 6085 |
| a27 | single | 0.3000 [0.175,0.450] | 1.000 | 40 | 40 | 2830 | 5716 |
| a27 | dual | 0.3000 [0.150,0.450] | 0.975 | 40 | 40 | 2793 | 5716 |
| b7 | single | 0.2500 [0.125,0.375] | 1.000 | 40 | 40 | 2974 | 6190 |
| b7 | dual | 0.3250 [0.175,0.475] | 0.800 | 40 | 40 | 2744 | 6190 |
| c7 | single | 0.2500 [0.125,0.375] | 1.000 | 40 | 40 | 2880 | 5665 |
| c7 | dual | 0.3250 [0.175,0.475] | 0.625 | 40 | 40 | 2336 | 5665 |
| **A_3seed 合并** | single | 0.2000 [0.108,0.300] | | 120 | 40 | 7794 | 17553 |
| **A_3seed 合并** | dual | 0.1750 [0.100,0.258] | | 120 | 40 | 6277 | 17553 |

无效尝试明细（分母含无效，预注册 R2）：a17 single 38/40、dual 35/40 全为 **pedestrian_role**
违规；a7 dual 36/40 = occluder_role 22 + target_target_collision 14；a27 仅 1（dual）；
b7 dual 8（occluder_role）；c7 dual 15（occluder_role 9 + target_target 6）。

### 5. 预注册判据判定（全部落盘 verdicts.csv；FAIL 如实报）

| 判据 | 分支 | 数字 | 判定 |
|---|---|---|---|
| 主：A_3seed > script（CI 不重叠且更高） | single | 0.2000 [0.108,0.300] vs 0.2500 [0.125,0.375] | **FAIL**（点估计反低且重叠） |
| 主：A_3seed > script | dual | 0.1750 [0.100,0.258] vs 0.3500 [0.200,0.500] | **FAIL**（同上） |
| 次 1（先验）：a7 > b7 | single | 0.2500 vs 0.2500（CI 全同） | **FAIL**（打平） |
| 次 1（先验）：a7 > b7 | dual | 0.1000 vs 0.3250 | **FAIL**（方向反） |
| 次 2（稳健性）：A7 std<C7 且危险率不低于 | single | std 0.0000 vs 0.0000；率 0.2525 vs 0.2500 | **FAIL**（std 不严格更低） |
| 次 2（稳健性） | dual | std 0.0000 vs 0.0000；率 0.1225 vs 0.3300 | **FAIL**（两条件均不满足） |

### 6. 失效模式诊断（判据之外的结构性发现）

1. **种子方差主导，且以"无效坍缩"形态出现**：a27 干净（dual 0.975 有效、危险率 0.300 ≥
   script 0.350 的 CI 内），a7 只塌 dual（0.10 有效），a17 双分支塌（0.05/0.125）。合并判据被
   无效尝试稀释——这正是 R2"分母含无效、禁止靠制造无效刷高"设计的镜像惩罚：坍缩组把 A 组
   合并危险率拉到 script 之下。
2. **a17 与 P0 的 seed17 病理跨版本再现**：P0（physics v1/sampler v1）dual_seed7 评价 13/20
   pedestrian_role；本轮（v2/v2 重训）a17 训练期 reference_kl 单调漂移到 0.35 + 评价期 73/80
   pedestrian_role。先验锚被挣脱 → 策略滑出角色约束包络。根因未查（候选：KL 权重 0.02 不足、
   robust 阶段 mean-std 效用放大越界动作），如实列为未解决问题。
3. **先验的 dual 负效应不能归因（单 seed）**：a7（先验）dual 0.10 有效 vs b7（无先验）0.80——
   同 seed 同预算下唯一差异是先验，但无重复，不能宣称"先验有害"；single 上两者完全打平
   （含 CI），与 P1.2"先验≈零动作"的开环结论在闭环评价中复现。
4. **扰动敏感性对照无区分力**：a7/c7 的 400 次扰动尝试（40 条件×10）within-scenario std 全为
   0——预注册扰动尺度下结局从不翻转，判据设计对此无区分力（本身是发现：需更大扰动尺度或
   初始条件级扰动才能测稳健性）。次判据 2 的 FAIL 属"无区分力型 FAIL"而非"稳健性更差"。
5. **可避让性 0%**：A 组全部 45 个 stopping 危险条件在 ttc 反事实下无一转安全——危险结局不是
   ego 制动律可挽回的（结构性危险而非控制器失误）。近似边界：只换 ego 制动律，非完整
   避让可行性判定。
6. **CEM 参考行（best-of-search，禁与一次采样排序）**：parameters 28/40（70%）、trajectory
   36/40（90%）条件找到危险解；成本 1600 评估/条件均值 2487 步（param）、1395 步（traj），
   对比固定策略 ~70 步/episode——35×/20× 的每条件搜索成本换取覆盖率（vs script 一次采样
   dual 35% 条件危险）。三列分账齐备（训练 5665–6190 / 评价 1438–2976 / 搜索 1395–2487 每条件）。
7. **自然性原材料**：effort_mean script=0（解析零动作恒等）、b7 ~4e-5（近零动作）、a7/a27
   0.001–0.014、c7 dual 0.0095、a17 0.036–0.047——坍缩组动作不平滑度最高，与失效模式一致。

### 7. 测试与验证

- 全量 **65 passed**（60 既有 + 5 新增 tests/test_p2_fairness.py：search_conditions 预算精确性
  与无效计入、单 spec 模式不变、total_effort 累计与零动作恒零、summarize 新字段、sampler v2
  训练路径记录）。
- WSL 转发缺陷（第三次遇到，留痕）：`bash -lc` 多行命令里的 shell 变量赋值/循环变量在
  wsl.exe 转发中丢失（`$P` 展开为空 → exit 127；`for f in ...` 的 `$f` 同样）——训练矩阵首启
  与本轮两次检查脚本均中招；处置 = 全展开单行命令 / 临时 .py 文件。无半成品目录残留
  （FileExistsError 防重入设计使然，已验证）。

### 8. 版本与重跑标记

| 产物 | 状态 |
|---|---|
| `runs/20260911_gpu_pilot/*`、P0/P1 全部旧产物 | 不变，pilot=physics v1/sampler v1 legacy |
| `runs/20260911_p2_conditions/heldout_seed41000.json` | **仍封存**（本段未读取；是否执行一次性评价待用户决策——主判据 dev 已 FAIL，跑 heldout 不会改变 FAIL 事实本身） |
| `runs/20260911_p2_*` | 本轮正式产物（v2/v2） |
| a17/a7 checkpoint | 坍缩组，保留作诊断证据，不得删除或重训替换（预注册禁止事后筛样本） |

### 9. 本轮修改文件

- 修改：`scenario_lab/env.py`（effort_total 累计 + summary total_effort）、`scenario_lab/train.py`
  （TrainConfig.sampler_version + 采样调用改 sampling.sample_spec dispatch）、
  `scenario_lab/evaluate.py`（summarize b_rounds/自然性聚合字段、_cem 提取、search_conditions
  多条件 CEM）、`scenario_lab/__main__.py`（train --sampler-version、search --conditions）、
  `docs/GLM_CHANGELOG.md`（本段）。
- 新增：`scripts/summarize_p2.py`、`tests/test_p2_fairness.py`。
- 未触碰：用户既有未提交改动（HANDOFF.md、stage2_extract.py、abd_parser/inventory 删除记录）、
  runs/ 既有产物、Data/、heldout 条件集。

### 10. 未解决问题 / 依赖用户决策

- heldout 一次性评价是否执行：主判据 dev 已 FAIL；按 R5 措辞边界，"优于基线"主张在本轮
  任何情况下不可用。heldout 评价的唯一作用是给论文留一个干净的"预注册全流程执行完毕"记录，
  是否花这次评价由用户定。
- a17 reference_kl 漂移根因未查（KL 权重 / robust 效用交互两个候选假设未区分）。
- 扰动尺度无区分力：下轮若重设敏感性实验，需预注册更大的扰动幅度或初始条件级扰动。
- 24×4 名义预算下三种子即出现两例坍缩：预算-稳定性关系（12×4 pilot 无此现象）本身值得
  系统研究，但属新实验、须重新预注册。
- CEM trajectory 有效尝试率仅 42%（679/1600）——动作序列参数化在中budget 下大量无效，
  参数化方式待改进（下轮预注册范围）。

---

## 2026-09-11 · P2.1 稳定性救援预注册（Codex 回审后；正式实验前）

### R1. 本阶段回答的问题

P2 的主要失败不是危险度不足，而是训练种子以角色违规形式坍缩。当前阶段只判断：

1. 将正式角色的横向动作投影到固定运动轴后，`pedestrian_role`/`occluder_role` 是否按构造归零；
2. prior 与 robust 后训练的 2×2 组合中，哪一项与训练不稳定和目标间碰撞相关；
3. 在严格相同的环境决策步预算下，坍缩是否仍跨种子出现。

本阶段不验证 ABD 稳健性，不运行 heldout，不形成“优于基线”主张。

### R2. 固定实现与预算

- `role_action_mode=lane_locked`：行人保持初始横穿方向、遮挡车保持初始车道方向；纵向加速度仍由
  策略学习。投影发生在动作限幅之后、执行延迟队列之前，并记录介入事件与 L1 修正量。
- 每次训练严格收集 **6000 decision steps**；最后一个 episode 如被预算截断，使用下一状态 critic
  值 bootstrap，禁止把非终止截断当作终止。`updates=40` 仅为允许达到预算的上限。
- sampler v2 / physics v2、mixed MAPPO、episodes/update=4、hidden=64、lr=3e-4、其余沿用
  TrainConfig。开发条件继续使用已查看的 `development_seed31000_samplerv2`。
- 训练种子固定为 **7/17/27/37/47**。

### R3. 2×2 因子矩阵

| 因子组 | prior_v5_nb | robust 后训练 |
|---|---:|---:|
| PR | 是 | 是 |
| PN | 是 | 否 |
| NR | 否 | 是 |
| NN | 否 | 否 |

共 4×5=20 个训练 run；每个模型在同一 dev 80 条件上确定性评价一次。prior_v5_nb 仅作为既有
机制的因果诊断项：其 INTERACTION `pedestrian/bicycle` 混合标签问题尚未修复，不能据本阶段结果
建立公共“行人先验”贡献。

### R4. 统计与 go/no-go

- 每个方法先逐 seed 报告 single/dual 的 valid、dangerous、target-target collision；不得只报五种子合并值。
- 共同场景比较使用 scenario 配对的 dangerous-rate 差值 bootstrap，B=2000、seed=2026；不再用
  两个边际 CI 是否重叠代替配对检验。
- 训练种子只报告五个独立结果的分布，不把 scenario bootstrap CI 解释为训练种子不确定性。
- **稳定性门槛**：每个 seed、每个分支 valid_rate ≥0.80；`pedestrian_role` 与
  `occluder_role` 必须为 0。任一 seed 低于门槛即判该因子组不稳定。
- `target_target_collision` 单列；lane lock 不保证目标间避碰。若它成为主要无效来源，下一阶段
  再预注册 target-separation guard，不在本阶段按 dev 结果临时修改。

### R5. 后续路由

- 若至少一个因子组五个种子全部通过稳定性门槛，再进入连续风险指标、ABD 扰动校准和强基线实验。
- 若四组均不稳定，停止扩大当前 PPO/MAPPO 配置，转向受约束策略优化或策略+CEM 混合生成。
- 原 `heldout_seed41000.json` 继续封存；最终方法冻结后另建 seed 对开发代理不可见的独立保留集。

---

## 2026-09-11 · P2.1 实施结果：角色坍缩解除，既有 prior 仍导致种子级时序坍缩

**实现与验证**：`ScenarioSpec.role_action_mode=lane_locked`；训练增加精确
`interaction_budget`、预算截断 critic bootstrap、逐 update invalid/action/projection/grad 诊断；评价从
policy bundle 自动恢复投影模式；`summarize_p2.py` 增加共同场景配对差值；新增
`run_p21_rescue.py` 与 `summarize_p21_rescue.py`。正式运行前全量 **69 passed**。

执行命令：

```bash
python scripts/run_p21_rescue.py --output runs/20260911_p21_rescue \
  --conditions runs/20260911_p2_conditions/dev_seed31000.json \
  --prior runs/20260911_p1_prior_nb/prior.pt --budget 6000 --device cuda
python scripts/summarize_p21_rescue.py runs/20260911_p21_rescue --bootstrap 2000 --seed 2026
```

41/41 jobs 通过；20 个训练 run 各精确 6000 decision steps；总任务耗时 1213.13 s（训练
750.31 s，评价 462.82 s）。heldout 未读。逐 seed 结果、配对双层 bootstrap 和完整日志见
`runs/20260911_p21_rescue/{REPORT.md,per_seed.csv,paired_effects.csv,jobs.json}`。

### 1. 预注册稳定性门槛

| 组 | 最低 valid_rate（任一 seed/branch） | role invalid | 结论 |
|---|---:|---:|---|
| PR（prior+robust） | 0.10（s47 dual） | 0 | **不稳定** |
| PN（prior、无 robust） | 0.20（s47 dual） | 0 | **不稳定** |
| NR（无 prior+robust） | 1.00 | 0 | **稳定** |
| NN（无 prior、无 robust） | 1.00 | 0 | **稳定** |

lane lock 将四组、五种子、单双分支的 `pedestrian_role`/`occluder_role` 全部归零，证明 P2 的
角色违规主要来自无约束横向动作，而不是条件集几何。它不保证目标间分离：PR/PN 的 seed47 dual
分别出现 36/32 个 `target_target_collision`，因此仍未通过稳定性门槛。

### 2. 因子结论（双层配对 bootstrap：seed×scenario，B=2000）

- 无 prior 的 NR/NN 五种子全部有效，危险率与 script 基本相同：NR-script single delta=0，
  dual=0.000 [-0.020,0.030]；NN-script single=0，dual=-0.005 [-0.030,0.000]。当前策略稳定但
  **没有学出优于零动作脚本的场景生成行为**。
- prior 在 robust 条件下 dual dangerous delta=-0.085 [-0.245,0.020]、valid delta=-0.205
  [-0.570,0.000]；无 robust 时 dual dangerous delta=-0.080 [-0.220,0.015]、valid delta=-0.180
  [-0.525,0.000]。既有 prior 不仅无正增益，还带来 seed47 目标间时序坍缩。
- robust 的效应接近零：有 prior 时 dual dangerous delta=0.000 [-0.065,0.100]；无 prior 时
  0.005 [-0.015,0.035]。当前未校准扰动和四个末段 robust update 不支持稳健性贡献。
- seed47 PR 的 max reference KL=0.331、PN=0.364；但 lane lock 后均无角色违规。KL 上升是策略
  漂移信号，不是角色坍缩的充分条件；纵向动作与目标占用时序需要单独约束/诊断。

### 3. 路由决定

按 R5，NR/NN 已通过稳定性门槛，可以进入下一阶段；主线暂时移除 prior，NN 作为最小稳定基线，
NR 仅保留为 robust 诊断对照。下一步先进行 ABD 控制来源与事件窗口核验，再预注册连续扰动和
target-separation 机制。P1 prior 降级为负结果；混合 `pedestrian/bicycle` 标签不得继续称纯行人证据。

---

## 2026-09-11 · P3 ABD 可追溯证据审计：AEB 专属校准 NO-GO

新增 `scripts/audit_abd_calibration.py`，对既有 24 条 CCRs/CPTA/CCFT 核对记录逐文件流式读取
完整时序，只保留命名通道的小数组，并记录原文件 SHA-256。`.spec` 解析结果拆为
`motion_control` 与 `brake_control`：SR/AR/PF 只能证明试验运动由机器人辅助，不能证明制动事件
由机器人触发。未修改原始数据和空白人工核对表。

结果位于 `runs/20260911_abd_calibration_audit/`：

- 24/24 条的 `Points` 与可解析行数一致，无坏行；三类场景各 8 条；
- 所有通道表中均无直接 AEB/FCW 激活状态；自动生成的 AEB 标签和碰撞标签均为 0；
- `UseBrakeRobot=False` 的 8 条存在可检测减速响应；其中 7 条 CCRs 在事件窗口内
  `BR Command`、`BR start`、`BR test`、`Motion Going BR` 均无变化，检测到的起始 TTC 为
  0.654–0.915 s、峰值减速度为 -11.419 至 -9.128 m/s²；这些仅作为描述性证据包络；
- 余下 1 条 E8 CPTA 虽标记 `UseBrakeRobot=False`，事件期 `BR Command` 变化约 16.06，单独列为
  语义冲突项；
- 11 条 `UseBrakeRobot=True` 记录因制动来源无法分离而排除，其中 1 条没有清晰减速响应；另有
  5 条 `UseBrakeRobot=False` 记录没有清晰减速响应。

结论为 **NO-GO for AEB-specific perturbation calibration**。7 条 CCRs 的一致模式是强候选证据，
但目录名、TTC 触发和减速轨迹都不能替代 AEB 因果来源确认，因此未改动 `perturb_spec`，标签继续是
`assumed_sensitivity_not_abd_calibrated`。`confirmation_queue.csv` 已逐条给出事件窗口、候选通道和
唯一需要外部确认的问题；满足 P3“先完成其他独立任务，不捏造校准结果”的路由条件。

---

## 2026-09-11 · P2.2 共享/专用架构开发集筛查预注册（运行前）

### R1. 问题与边界

P2.1 已确定无 prior、无 robust 的 NN 是稳定最小配置，但它与零动作脚本几乎相同。本阶段在不引入
新机制的情况下回答一个更窄问题：相同 actor 参数量、相同总交互预算时，mixed 共享 actor 在
single/dual 正式分支上是否劣于对应 single-only/dual-only 专用训练。该结果只是开发集筛查，
不能直接把 RQ1–RQ3 标为已支持。

### R2. 固定矩阵

- 模式：`mixed`、`single`、`dual`；种子 7/17/27/37/47，共 15 个训练 run；
- 每个模型严格 6000 decision steps，MAPPO、sampler v2 / physics v2、`lane_locked`、无 prior、
  无 robust；actor hidden=64，参数量必须完全相同；
- mixed 的 `mixed_warmup_fraction=0`，从第一轮起 single/dual 交替，不使用原有 single-only
  warmup；逐 update 和汇总均记录实际 branch steps/episodes；
- 所有模型评价相同的 80 条 `development_seed31000_samplerv2` 条件；heldout 继续不读。

这是**等总预算**比较：mixed 每个分支得到的训练步数少于专用模型，差额必须报告。等每分支预算
比较需另行预注册，不能与本轮混报。

### R3. 判据

- 主比较：mixed-single-only 在 single 分支、mixed-dual-only 在 dual 分支的有效危险率差值；
- 双层配对 bootstrap 同时重采样 5 个训练种子和 40 个共同场景，B=2000、seed=2026；
- 开发集非劣容差预先固定为绝对危险率 0.05；95% 双侧区间下界 ≥ -0.05，且两侧所有相关
  seed 的 valid_rate ≥0.80、角色违规为 0，才计 noninferiority pass；
- 增设 activity gate：相关策略相对 script 至少改变一个 valid/dangerous 结局，或 mean absolute
  applied action ≥0.01。非劣通过但 activity gate 失败，结论记为“低活动/无区分力”，不能支持
  有用的共享学习；
- target-target collision、mean absolute applied action、effort 和训练分支步数全部单列。

### R4. 实施结果

执行：

```bash
python scripts/run_p22_architecture.py \
  --output runs/20260911_p22_architecture \
  --conditions runs/20260911_p2_conditions/dev_seed31000.json \
  --budget 6000 --device cuda
python scripts/summarize_p22_architecture.py runs/20260911_p22_architecture \
  --bootstrap 2000 --seed 2026
```

31/31 jobs 通过（15 train + script + 15 eval），总任务耗时 919.66 s，其中训练 553.57 s、评价
366.09 s；15 个模型均为 43,016 个 actor 参数并精确训练 6000 步。所有评价 seed/branch 的
valid_rate=1.0，角色违规和评价期 target-target collision 均为 0；heldout 未读。

| 比较（mixed - specialized） | 分支 | 有效危险率差 [95% CI] | 非劣 | activity gate |
|---|---|---|---:|---:|
| mixed - single-only | single | +0.040 [-0.020, +0.125] | PASS | PASS |
| mixed - dual-only | dual | +0.025 [+0.000, +0.070] | PASS | PASS |

mixed 每种子的实际分支训练步数为 single 2983–3216、dual 2784–3017，约为专用模型 6000 步的
一半，却通过两个开发集架构非劣判据。策略并非全为零动作：mixed 的 mean absolute applied action
按 seed/branch 为 0.011–0.069，相对 script 共改变 single 10/200、dual 13/200 个结局。

这项结果只支持一个窄结论：**在当前开发集、等总预算和受约束角色动作下，没有观察到共享训练相对
专用训练的负迁移**。它不支持方法优于脚本：mixed-script 的 single 差为 +0.040
[0.000,+0.105]、dual 为 -0.005 [-0.075,+0.060]，预注册的严格 superiority 均未通过；
single-only 与 script 打平，dual-only 点估计低 0.030。下一项独立实验应做等每分支预算的 mixed
比较，随后才考虑最终封存集；不能用本开发集结果直接把 RQ1–RQ3 标为“已支持”。

---

## 2026-09-11 · P2.3 等每分支预算预注册（运行前）

P2.2 是等总预算比较，mixed 的每分支实际只得到约 3000 步。为完成 `research_claims.md` 要求的
第二种预算口径，本阶段只训练 mixed seed 7/17/27/37/47，每个模型 single **精确 6000 步**、
dual **精确 6000 步**，总计 12000 步；复用 P2.2 中同种子、同 43016 参数、对应分支 6000 步的
single-only/dual-only checkpoint 与评价结果。

其余配置固定：sampler v2 / physics v2、MAPPO、`lane_locked`、无 prior、无 robust、
`mixed_warmup_fraction=0`，开发条件仍为 `development_seed31000_samplerv2`，heldout 不读。
实现必须在 episode 层分别截断并验证两个分支的累计步数，不能以 12000 总步近似替代。

主判据沿用 P2.2：mixed-single-only 和 mixed-dual-only 的有效危险率差做 seed×scenario 双层配对
bootstrap（B=2000、seed=2026），95% CI 下界 ≥ -0.05，且 mixed 各 seed/branch valid_rate
≥0.80、角色违规为 0。另报 12000-step mixed 相对 6000-step mixed 的预算扩展效应，以及相对
script 的方法效应；后两项不改变主判据。

### P2.3 实施结果

10/10 jobs 通过（5 train + 5 eval），总任务耗时 457.13 s，其中训练 342.34 s、评价 114.78 s。
每个 checkpoint 的 single/dual 累计步数均精确为 6000；五个 seed 两分支评价 valid_rate 均为
1.0，角色违规和评价期 target-target collision 均为 0，heldout 未读。

| 比较 | 分支 | 有效危险率差 [95% CI] | 判定 |
|---|---|---|---|
| 12k mixed - 6k single-only | single | 0.000 [-0.100, +0.090] | **非劣 FAIL** |
| 12k mixed - 6k dual-only | dual | +0.035 [-0.060, +0.120] | **非劣 FAIL** |
| 12k mixed - 6k mixed | single | -0.040 [-0.130, +0.030] | 预算扩展无正证据 |
| 12k mixed - 6k mixed | dual | +0.010 [-0.075, +0.095] | 预算扩展无正证据 |
| 12k mixed - script | single | 0.000 [-0.100, +0.090] | superiority FAIL |
| 12k mixed - script | dual | +0.005 [-0.090, +0.095] | superiority FAIL |

P2.2 的等总预算开发集非劣没有在等每分支预算下变得更稳；点估计没有显示 shared 负迁移，但
训练种子不确定性越过 -0.05 容差。继续增加同一 PPO/MAPPO 配置的预算没有依据，当前最准确的
结论是“共享架构可运行且稳定，是否非劣仍未定；生成方法没有超过脚本”。下一阶段应先验证受
`lane_locked` 约束的纵向动作搜索是否仍有可利用空间：若约束 CEM 明显高于策略，则优化器/学习
信号是瓶颈；若 CEM 也无增益，则需要改场景参数化，而不是继续堆 PPO 步数。

同时修正未来 P1 语料路径：INTERACTION 官方 `pedestrian/bicycle` 混合类现在映射为 `other` 并
带明确原因，不再进入 pedestrian-only 监督。既有 v5 corpus/prior 产物保持不变并继续作为负结果
证据，不以重建覆盖。

---

## 2026-09-11 · P2.4 受约束 CEM 可行性预注册（运行前）

### 目的与边界

本阶段只回答优化瓶颈问题：在与 P2.2/P2.3 相同的 sampler v2 / physics v2、角色约束和
`lane_locked` 执行空间内，是否存在明显多于 script 和当前稳定 MAPPO 策略的有效危险解。
CEM 是多次仿真的诊断 oracle，不作为与一次前向策略同成本的最终方法，也不据此声称方法优越。
继续使用 80 条 `development_seed31000_samplerv2` 条件，single/dual 各 40 条；heldout 不读。

### 搜索矩阵与预算

- 搜索类型：低维 acceleration-pulse `parameters` 与 8-knot `trajectory`；
- 分支：single、dual 分开运行和判定；
- 搜索随机种子：7、17、27、37、47；population=8；
- 每个 kind×branch×seed×condition **精确 2500 decision steps**，所有有效、无效及末尾预算截断
  episode 都落盘并计入成本；末尾未自然终止的 episode 不得成为 best/success；
- `lane_locked` 后只搜索实际生效的纵向动作：parameters 维度 single=3、dual=6，trajectory 维度
  single=8、dual=16。旧 P2 CEM 未使用该执行投影且按 episode 数预算，只作历史线索，不进判定。

### 指标与判据

每个条件的成功定义为在预算内至少找到一个**完整、有效且 dangerous** 的 episode。对每种搜索、
每个分支，将五个搜索种子的条件成功率与同条件 script 做 seed×scenario 双层配对 bootstrap
（B=2000、seed=2026）。可行性 PASS 需同时满足：危险条件覆盖率差点估计 ≥0.10、95% CI 下界
>0、角色违规总数为 0。另报相对 P2.3 12k mixed 的诊断差、搜索种子复现率、全尝试有效率、
target-target collision、成功所需累计步数、动作 effort 与预算截断数。

判定路径预先固定：parameters 在两个分支均 PASS，归为低维搜索可行且当前学习/优化信号受限；
只有 trajectory 两分支均 PASS，归为低维参数化不足；trajectory 任一分支不 PASS，则按分支定位
场景空间、目标或搜索预算不足。任何 CEM best-of-search 结果均不得直接写成同计算成本的方法
superiority，heldout 继续封存。

### P2.4 实施结果

20/20 搜索任务通过（2 kind × 2 branch × 5 seed），每个任务覆盖 40 个开发条件，每条件精确
2500 步，总计 2,000,000 decision steps；4 路并行墙钟约 690 s。全部任务的条件文件 SHA-256、
condition set version、搜索维度、`lane_locked` 和逐条件预算均经运行器复核，heldout 未读。

| 搜索 | 分支 | 成功率（5 seed×40 条件） | 相对 script 覆盖差 [95% CI] | 角色违规 | 判定 |
|---|---|---:|---|---:|---:|
| parameters | single | 0.690 | +0.440 [+0.305,+0.575] | 0 | **PASS** |
| parameters | dual | 0.685 | +0.335 [+0.225,+0.450] | 0 | **PASS** |
| trajectory | single | 0.845 | +0.595 [+0.450,+0.735] | 0 | **PASS** |
| trajectory | dual | 0.880 | +0.530 [+0.375,+0.690] | 0 | **PASS** |

parameters 各种子成功率为 single 0.650–0.725、dual 0.650–0.700；trajectory 为 single
0.825–0.875、dual 0.800–0.925。五个搜索种子全部成功的条件比例依次为 0.500、0.375、0.750、
0.675。各 run 成功条件中的首次成功中位累计成本为 92–640 步，说明 2500 步上限足以诊断可行性，
但仍比策略单次前向贵得多。

`lane_locked` 将 pedestrian/occluder 角色轴违规降为 0；dual 搜索仍产生 target-target collision：
parameters 274/9726 attempts，trajectory 699/10238 attempts。它们全部保留在成本和分母中，且
无效 episode 不可能成为成功解。完整有效 attempt 占比按种子为 parameters-dual 0.946–0.957、
trajectory-dual 0.898–0.922，结果不是通过删除无效尝试得到。

trajectory 相对 parameters 的条件覆盖差为 single +0.155 [+0.050,+0.265]、dual +0.195
[+0.075,+0.315]。因此更丰富的时序动作仍有额外价值，但低维 pulse 参数搜索本身已在两个正式
分支通过预注册可行性判据。按预注册决策树，诊断为
`low_dimensional_search_feasible_learning_or_optimization_bottleneck`：当前主要瓶颈位于 MAPPO 的
探索、目标塑形或从搜索解到闭环策略的学习过程，而不是 `lane_locked` 场景空间没有危险解。

下一步不继续增加同配置 PPO 步数，也不读取 heldout。P2.4 解来自当前 dev80，只能用于诊断和
teacher schema 验证，禁止把它们用于训练后再在同一 dev80 评价。P2.5 必须先冻结现有 dev80 为
验证集，再从独立 training-only 条件种子运行 parameter-CEM，建立带 provenance 的 teacher corpus；
随后以固定总交互预算比较 script、纯 MAPPO、BC-only 和 BC→MAPPO。trajectory 解只作为可达
上界和后续残差动作扩展依据。

---

## 2026-09-11 · P2.5 独立 CEM teacher 与策略迁移预注册（运行前）

### 目的与数据边界

本阶段检验 P2.4 定位出的学习/优化瓶颈能否通过低维 parameter-CEM teacher 缓解。现有
`development_seed31000_samplerv2` 的 80 条条件冻结为验证集，P2.4 的 dev CEM 解禁止进入监督。
另以 seed=51000、sampler v2 / physics v2、reference role 生成 purpose=`training` 的条件集，
single/dual 各 120 条。构建 corpus 前按除 scenario_id 外的完整物理/控制参数指纹检查 training/dev
交集，任何重叠直接终止。

### Teacher 与行为克隆

- training-only 条件上运行 parameter-CEM，search seed=107、population=8、`lane_locked`，每条件
  精确 1000 decision steps；所有无效和末尾截断尝试继续计入搜索成本；
- 只纳入能以原搜索 seed 确定性重放为 complete、valid、dangerous 的 best pulse 解；不成功条件
  保留在搜索分母但不伪造监督标签；
- 每个分支按 condition_index % 5 == 0 固定为 teacher-val，其余 teacher-train；
- Actor 保持与 P2.3 相同的 43,016 参数结构，监督输入只含 actor 可见 observation；整 episode
  recurrent replay，20 epochs、batch=16、learning rate 3e-4，使用 inverse-role-frequency 和
  `1 + 2*abs(longitudinal_action)` 活动加权 MSE；策略种子 7/17/27/37/47；
- BC-only checkpoint 直接评价；BC→MAPPO 从对应 BC checkpoint 初始化，reference KL 固定为 0，
  无 robust、无 public prior、mixed warmup=0，每个分支精确 6000 在线训练步，总计 12000。

### 冻结对照与判据

对照复用 P2.3 同种子纯 MAPPO 12k checkpoint/evaluation 和 P2.2 script，全部在同一 dev80、
`lane_locked` 条件上评价。主判据：BC→MAPPO 相对 script 的有效危险率差在 single、dual 两分支
分别做 seed×scenario 双层配对 bootstrap（B=2000、seed=2026），点估计 >0、95% CI 下界严格
>0，同时 BC→MAPPO 每 seed/branch valid_rate ≥0.80、角色违规总数为 0。两分支均满足才允许进入
heldout 决策。

机制指标预先固定但不替代主判据：BC-only-script、BC→MAPPO-pure MAPPO、BC→MAPPO-BC-only
的危险率/风险差；teacher train/val 数、重放拒绝数、BC val MSE、动作活动度、target-target
collision 和训练期无效原因全部报告。若 BC-only 提升而 BC→MAPPO 退化，归因于在线微调遗忘；
若二者都无提升，归因于 teacher 蒸馏或 actor 可观测性不足；任何 dev 结果仍不作为 heldout 结论。

### P2.5 实施结果

23/23 个正式 job 通过，总计 873.67 s。training-only parameter-CEM 对 single/dual 各执行
120 条件 × 1000 decision steps，得到 151 个 complete、valid、dangerous 且重放一致的 teacher
episode；其中 single train/val 为 70/13，dual 为 53/15。training/dev 完整条件指纹交集为 0，
teacher corpus SHA-256 为
`db310f0df743c5065045688724a8987be787213667cca0ec51d19fa85a6b23dd`。五个 BC 和五个
BC→MAPPO 均完成，后者逐模型 single/dual 各精确训练 6000 步。全部 dev 评价 valid_rate=1.0、
角色违规为 0；heldout 未读。

| 比较 | 分支 | 有效危险率差 [95% CI] | 判定 |
|---|---|---:|---:|
| BC-only - script | single | +0.030 [-0.150,+0.210] | 无可靠提升 |
| BC-only - script | dual | -0.020 [-0.125,+0.085] | 无可靠提升 |
| BC→MAPPO - script | single | +0.055 [-0.110,+0.205] | **主判据 FAIL** |
| BC→MAPPO - script | dual | -0.010 [-0.110,+0.090] | **主判据 FAIL** |
| BC→MAPPO - pure MAPPO | single | +0.055 [-0.065,+0.175] | 无可靠迁移增益 |
| BC→MAPPO - pure MAPPO | dual | -0.015 [-0.120,+0.075] | 无可靠迁移增益 |

五个 BC 的最佳 val MSE 均出现在 epoch 1（0.05854–0.05906），到预注册的 epoch 20 上升为
0.06593–0.07348；这一现象只能作为事后过拟合线索，不能改用 epoch 1 重算正式判据。BC 的 dev
输出跨种子高度一致，而 BC→MAPPO 增加了动作幅度和种子方差，却没有形成稳定正效应，因此没有
证据把失败主要归为在线微调遗忘；BC-only 在进入在线训练前已经未超过 script。

为缩小失败原因，另做不改变正式判据的 training-domain 事后诊断。151 个被选 teacher 中有 75 个
（49.7%）在零动作 script 下已经 dangerous，说明“只要求 CEM dangerous”会混入大量无需学习
增量干预的监督。更关键的是，在 script-safe 而 CEM 成功的 teacher 条件上，BC 对 single 的闭环
危险复现率为 train 0.50–0.575、val 0.50，而 dual 仅为 train 0.10–0.20、val 0.125–0.25；teacher
重放本身为 1.0。teacher-forced active-action 符号准确率也只有 single-val 0.533、dual-val 0.608。
因此当前最具体的失败定位是：**低维开环 pulse 解存在，但逐时刻动作回归没有可靠转化为闭环策略，
尤其 dual 分支存在严重的条件动作混叠或闭环分布偏移**。现有观测不能再细分这两种机制，不能把
其中任一种写成已经证明的根因。

P2.5 不满足进入 heldout 的预注册门槛，heldout 继续封存。下一阶段不再重复同一 BC→MAPPO 配方；
应先用独立 training-only 数据预注册一个小型机制实验：只保留 script-safe 的增量 teacher，把
监督目标从逐步 pulse 动作改为 condition/early-history 到 pulse 参数的预测，使用 teacher-val 选择
停止点，并分别报告参数重放、teacher 条件闭环复现和冻结 dev 效应。只有 dual 的训练域闭环复现
先显著高于本次 0.10–0.25，才值得再次投入正式 dev 多种子评价。

---

## 2026-09-11 · P2.6 增量 pulse 参数预测机制筛选预注册（运行前）

### 目的与证据边界

P2.6 只检验 P2.5 的逐时刻行为克隆失败能否通过“先识别条件、再预测完整低维 pulse”缓解，
不在本阶段继续叠加 PPO。训练监督只来自已完成的 P2.5 training-only CEM；重新执行零动作 script，
仅保留 script-safe、CEM 重放 complete/valid/dangerous 的增量 teacher。现有 dev80 不参与模型选择。

另生成 `training_seed61000_samplerv2`，single/dual 各 120 条，运行 parameter-CEM，search seed=127、
population=8、`lane_locked`、每条件精确 1000 decision steps。该全新集合只作一次机制 screen，不进
训练和早停。机制筛选前先做 training-screen 指纹审计；只有机制门槛通过时，才在 dev 评价前按
除 scenario_id 外的完整物理/控制字段补做 dev 与二者的指纹审计；任一交集直接终止。heldout 不读。

### 模型、输入与监督

- 每个角色只使用自身 actor-visible observation；不接收 critic state、其他角色 token 或隐藏真值；
- 收集 episode 最初 5 个 decision observation（0.0–0.4 s），期间输出零动作；只纳入所有活跃角色
  pulse onset ≥0.5 s 的 teacher，保证这段 history 与 teacher 重放的前缀一致；
- 共享 token encoder、role embedding、attention 和 GRU，但每个角色独立输出其 amplitude、start、
  duration 三个归一化参数；执行从 t=0.5 s 起沿绝对 episode 时钟恢复 pulse，steering 固定为 0；
- 策略种子 7/17/27/37/47，最多 50 epochs、batch=16、Adam 3e-4；只用 P2.5 原固定 split 的
  teacher-train 更新，以 single/dual 分支等权的 teacher-val 参数 MSE 选择最佳 epoch，平局取较早者；
- 不得以 P2.6 screen 或 dev 指标选 epoch、改输入长度、改筛选阈值或挑种子。

### 机制门槛与条件执行路径

P2.6 screen 中每个入选条件按原 CEM replay seed 闭环运行一次预测策略；因入选条件的 script 均安全，
危险率就是新增覆盖率。对每个分支做 seed×scenario 双层 bootstrap（B=2000、seed=2026）。进入 dev
评价须同时满足：每分支至少 15 个合格 screen 条件；single 危险率点估计 >0.50 且 95% CI 下界
>0.50；dual 点估计 >0.25 且 95% CI 下界 >0.25；每 seed/branch valid_rate ≥0.80，角色违规总数
为 0。0.50/0.25 是 P2.5 在 script-safe teacher-val 上观察到的对应上界，不从 P2.6 数据调整。

只有上述两分支机制门槛均通过，运行器才允许读取冻结 dev80 并评价五个 checkpoint。dev 主判据为
预测策略相对 script 的危险率差在 single、dual 均点估计 >0、双层 bootstrap 95% CI 下界 >0，
同时保持 valid_rate 和角色约束门槛。若机制门槛未通过，P2.6 以训练域机制负结果结束，dev 和
heldout 均不读取；若通过但 dev 不通过，只说明训练域 pulse 复现没有转化成未见开发条件优势。

### P2.6 实施结果

14/14 个正式 job 通过，总计 347.15 s。全新 screen 的 parameter-CEM 对 single/dual 各执行
120 条件 × 1000 decision steps；training-screen 完整条件指纹交集为 0。按预注册筛选后，P2.5
训练 corpus 留下 60 个增量 teacher（single train/val 35/6，dual 15/4），独立 P2.6 screen 留下
45 个（single 26，dual 19）。训练源的 240 条条件中，89 条无有效危险 teacher、69 条 script
已危险、22 条 pulse 早于 0.5 s；screen 对应为 90、75、30 条。所有拒绝原因和未入选条件均保留
在审计分母中。

| 分支 | 独立 screen 条件 | 五种子危险复现率 [95% CI] | P2.5 固定门槛 | valid / 角色违规 | 判定 |
|---|---:|---:|---:|---:|---:|
| single | 26 | 0.269 [0.108,0.454] | >0.50 且 CI 下界 >0.50 | 1.0 / 0 | **FAIL** |
| dual | 19 | 0.358 [0.158,0.579] | >0.25 且 CI 下界 >0.25 | 1.0 / 0 | **FAIL** |

single 各 seed 为 0.192–0.346，dual 为 0.263–0.474。dual 点估计高于 P2.5 的 0.25 上界，
但区间很宽且下界低于门槛，不能写成可靠改进；single 点估计和区间都没有达到预设水平。五个
checkpoint 由 P2.5 teacher-val 选择在 epoch 8–44，分支等权 val 参数 MSE 为 0.817–0.824；模型均为
43,142 参数。按条件执行路径，P2.6 没有读取或评价 dev，heldout 也未读取。

事后参数诊断进一步显示，失败不能只归因于 screen 样本少。single 的预测参数在不同条件间标准差
仅 0.000–0.004，而 teacher amplitude/start/duration 标准差在 screen 为 0.512/0.257/0.383；模型
screen MSE 0.481–0.510，与训练角色均值常数基线 0.486 接近。dual 每个角色的预测标准差也均
≤0.002，而 teacher 各维标准差约 0.20–0.62；模型 MSE 0.677–0.699，没有稳定优于角色均值基线
0.677。screen 中 single 26 条有 17 条对五个 seed 全部失败、5 条全部成功；dual 19 条对应为
10 条和 5 条，符合策略输出近似固定 pulse、只覆盖固定子集的表现。

因此 P2.6 的可支持结论是：**把单个 CEM best 解改成整段 pulse 参数回归，仍未解决监督标签的
可识别性问题**。每个条件可能存在多组功能等价的 CEM 解，单个随机最优参数并不是规范化真值；
MSE 会把多模态解平均成近常数 pulse。0.5 s actor-visible history 是否还缺少足够条件信息仍是候选
解释，但本次没有单独操纵 history 长度，不能把它确认为唯一根因。

后续若继续推进，应先使用已落盘的全部 CEM attempts 量化“同条件成功解内部方差”与“跨条件方差”，
再决定 P2.7：只有确认同条件多解性主导，才构建多解集合监督（例如多头候选加 set-min loss 或
候选排序），并把多候选仿真成本显式计入方法比较；若 actor-visible history 对条件区分本身不足，
则应调整可观测信息或任务定义，而不是继续增加 MSE 训练轮数。

---

## 2026-09-11 · 全部 CEM attempts 方差审计与 P2.5/P2.6 失效归因

这是一项事后描述性诊断，不改变 P2.4–P2.6 的正式判据。共盘点 32 个 `attempts.jsonl`；其中
P2.4 的 20 个 `lane_locked` 正式多种子搜索及 P2.5/P2.6 的 4 个正式 parameter 搜索进入分层
方差分析。早期无约束 P2、GPU pilot 和 P2.4 smoke 共 8 个文件只登记，不与正式层合并。分析分别
在归一化搜索参数空间和实际 80-step 纵向动作序列空间执行；CEM 候选随迭代自适应，以下方差只作
描述，不当作独立样本推断。heldout 未读。

在全部 complete、valid、dangerous attempts 中，同一条件内部动作方差占总方差的比例为：P2.4
parameters single 0.641、dual 0.812，trajectory single 0.733、dual 0.811；P2.5 parameters
single 0.734、dual 0.751；P2.6 为 single 0.747、dual 0.749。换言之，条件身份通常只解释约
19%–36% 的成功动作差异。只保留每条件得分最高的四分之一后，P2.5/P2.6 dual 的条件内动作
方差占比仍为 0.701/0.706，结果不是由低分危险解单独造成。

P2.4 提供了更直接的标签稳定性检查：同一 dev 条件由五个独立搜索 seed 选出的 best 解之间，
parameters 的条件内动作方差占 single 0.486、dual 0.618；trajectory 为 single 0.699、dual
0.749。parameters-dual 的 32 个可分析条件中有 20 个同时包含正、负 amplitude 的成功 pulse。
因此“每个条件只有一组正确 CEM 参数”与现有 attempts 明显不符，单个 best 解不能作为规范化回归
真值。

对 P2.6 实际采用的 script-safe、onset≥0.5 s 增量子集单独分析后，多解性仍存在但没有完全淹没
条件差异。P2.5 来源的 single/dual 条件内动作方差占 0.526/0.500，top-quartile 降至
0.312/0.422；P2.6 screen 来源为 0.364/0.555，top-quartile 为 0.167/0.325。每个条件的成功
attempt 中位数只有 2–2.5 条，说明过滤一方面提高了标签可区分性，另一方面也使可用监督非常稀疏。

为区分观测与样本问题，使用 P2.5 train/val 选择 ridge 正则强度，再一次性预测 P2.6 screen；该
方法和结果均为事后诊断。角色 0 有 50 train / 10 val / 45 screen，actor-visible history 的参数
MSE 为 0.416（训练均值基线 0.540），完整 ScenarioSpec 诊断上界为 0.275；对应闭环危险率为
0.654 和 0.808。角色 1 仅有 15/4/19 个样本，history MSE 0.756、完整 spec MSE 0.799，均只接近
训练均值基线 0.812；但完整 spec 联合预测的 dual 闭环危险率仍达到 0.789。精确 teacher 参数重放
在两个分支均为 1.0。完整 ScenarioSpec 含 actor 执行期不应获得的信息，只用于诊断上界，不能作为
当前方法结果。

现阶段对“为什么效果仍不明显”的证据排序如下：

1. **高置信度：监督目标不适定。** 同一条件存在大量相距很远、甚至 amplitude 符号相反的成功解；
   单标签 MSE 把功能等价的多模态解平均成近常数 pulse。
2. **高置信度：过滤后的有效监督太少且模型过大。** P2.6 实际只有 50 个 train condition，角色 1
   仅 15 个，却拟合 43,142 参数；五个神经模型的条件间输出标准差几乎为 0。简单 ridge 在 single
   上反而明显更好，符合小样本高方差/强正则不足，而非增加 epoch 能解决的问题。
3. **中高置信度：dual 的局部可观测性和角色样本不平衡共同限制。** 完整 spec 的闭环上界远高于
   actor-visible history，但角色 1 样本太少，尚不能把差额完全归因于隐藏信息。
4. **高置信度：参数 MSE 与危险结局错配。** dual 的训练角色均值常数 pulse 闭环危险率 0.474，
   高于 history ridge 的 0.368，尽管后者参数 MSE更低；降低任意 best 参数误差并不保证跨过危险
   结局阈值。

因此问题包含“训练数据集构造”因素，但不是 sampler v2 场景完全没有可学习危险解：P2.4 CEM
覆盖率、P2.6 精确 teacher 重放和完整 spec 诊断上界都反驳了这一解释。更准确的说法是：当前从
搜索轨迹抽取出的监督集太小、角色失衡、标签非唯一，且监督损失与最终危险覆盖目标不一致；ABD
校准尚未完成也意味着这些合成条件仍不能代表最终目标分布。

---

## 2026-09-11 · P2.7 多成功解集合监督预注册（运行前）

### 数据与边界

P2.7 检验“单个 CEM best + 参数 MSE”是否是 P2.6 模式平均的主要原因。训练源复用互不重叠的
`training_seed51000_samplerv2` 与 `training_seed61000_samplerv2` attempts，并新增 dual-only
`training_seed62000_samplerv2` 160 条条件，使用 search seed=137/157、population=8、每条件精确
1000 decision steps，以增加角色1条件数和同条件多解。最终一次性机制 screen 使用全新的
`training_seed71000_samplerv2`，single/dual 各120条，search seed=177、相同 CEM 预算。

所有源继续限定 `lane_locked`。每个条件重新以对应 replay seed 执行零动作 script，只保留 script
安全、CEM attempt complete/valid/dangerous、所有活跃 pulse onset≥0.5 s 的条件和候选。候选先取
得分上半部，再从最高分解开始按80-step实际动作 RMS 距离作 farthest-first，最多8解；少于2解的
条件仍保留并如实报告。training 与 screen 及现有 dev 做条件指纹隔离；dev 指纹只在 screen 门槛
通过后读取和检查。heldout 不读。

### 模型与损失

- 每个角色只输入自身最初5个 actor-visible observation；0.0–0.4 s 输出零动作；
- 共享 token encoder、role embedding、attention、GRU，hidden=16；每角色输出 K=4 组
  amplitude/start/duration，候选 head 编号在角色间对齐，但角色特征不交叉；
- 集合损失为 teacher→head coverage Chamfer 项加 0.25×head→teacher precision 项，距离在所有
  活跃角色及三个参数维度上平均；训练时按分支逆频率加权；
- policy seed=7/17/27/37/47，最多100 epochs、batch=16、Adam 3e-4；condition_index%5==0
  为固定 val，以 single/dual 等权 val set loss 选择 epoch，screen 不参与选择；
- 同时冻结一个强正则 ridge 单候选基线：每角色以成功动作 medoid 为目标，正则强度只由 train-val
  在预定网格 {0.01,0.1,1,10,100} 选择。

### 评价、成本与门槛

每个 learned-set condition 执行4个候选 rollout，条件成功定义为至少一个 complete、valid、dangerous
候选；所有候选步数、无效原因和成功前累计步数均记录。对照为 ridge 单候选和按固定 seed=2026
生成、同样满足 onset≥0.5 s 的 uniform random 4候选；screen 条件本身均为 script-safe，精确 teacher
候选重放只作100%可达性校验。

进入 dev 须在 single、dual 分别同时满足：screen 合格条件≥15；learned-4 危险覆盖率的 seed×condition
bootstrap（B=2000、seed=2026）95% CI 下界分别严格高于 P2.6 固定阈值0.50/0.25；learned-4 相对
ridge-1 的配对覆盖差95% CI下界>0；相对 random-4 的配对差95% CI下界≥-0.05；全部 learned 候选
attempt valid_rate≥0.80且角色违规为0。两分支均通过才读取dev80；否则以训练域机制结果结束。

若进入 dev，正式报告 learned-4、ridge-1、random-4、script 的条件覆盖和总交互成本。learned-4
相对 script 的 superiority 只有在两分支配对95% CI下界>0、有效性门槛满足时成立，并必须明确限定
为“四候选 rollout 预算”，不能写成单次前向策略优越或等成本于 script/CEM。

---

## 2026-09-11 · P2.7 多成功解集合监督结果

P2.7 已按上述预注册执行。新增 CEM 搜索在 `training_seed62000_samplerv2` dual 160 条件、两个
search seed，以及 `training_seed71000_samplerv2` single/dual 各120条独立 screen 上共消耗
560,000 decision steps；所有搜索均为 `lane_locked`、population=8、每条件精确1000步。复用的
seed51000/61000 attempts 与新增 seed62000 组成训练源，所有训练、screen、dev 条件指纹重叠均为0；
heldout 未读。

过滤后训练集为174个条件（single 76、dual 98；train/val 分别为64/12和75/23），独立 screen
为68个条件（single 48、dual 20）。精确 selected-teacher 重放在两分支条件覆盖率均为1.0。候选
密度低于设计预期：训练174条中100条最终只有1个候选，候选数中位数为1；screen 68条中44条
只有1个候选。主要损耗来自无合格危险解、script 本身已危险及 onset<0.5 s 过滤。

五个3,416参数的 hidden=16、K=4 模型分别在 epoch 91/16/16/28/20 选择 checkpoint。预注册
screen 机制门双分支均通过：

- single learned-4 0.721 [0.596, 0.838]，ridge-1 0.479，差0.242
  [0.108, 0.383]；random-4 0.417，差0.304 [0.163, 0.438]；候选有效率1.000；
- dual learned-4 0.700 [0.500, 0.890]，ridge-1 0.450，差0.250
  [0.080, 0.450]；random-4 0.350，差0.350 [0.150, 0.550]；候选有效率0.932，
  角色违规0。

因此按预注册规则解封 dev80。learned-4 在 dev single 为0.520、script 0.250，配对差0.270
[0.100, 0.440]；dual 为0.550、script 0.350，配对差0.200 [0.095, 0.325]。候选有效率分别
1.000/0.958，角色违规均为0，双分支均通过本阶段 dev superiority 判据。这一结果严格对应每条件
最多4次 rollout；screen 中 learned 的全部候选交互步为 single 68,075、dual 27,690（包含5个
policy seed），每 condition×seed 到首次成功或耗尽的平均步数为180.6/181.3，不能解释为与单次
script 或 ridge 等成本。

事后 head 诊断同时限制了结论。每个 seed 的 any-4 覆盖比其最佳固定 head 在 screen 高
single 0.208–0.271、dual 固定0.150，在 dev 高 single 0.100–0.150、dual 0.125–0.225；四个
head 的动作确实互异。但同一 head 的跨条件动作 RMS 仅为 screen single 0.0002、dual 0.0012，
而 head 质心间 RMS 为0.1859/0.1379；条件相关方差占比分别只有0.0000/0.0003，dev 也近乎为0。
所以 P2.7 实际学到的是四个全局 pulse 原型库，尚未学到随 actor-visible history 改变的条件化候选集。

当前可支持的结论是：**在当前合成 lane-locked 分布和四次闭环尝试预算下，训练数据得到的四候选
原型库在一次独立 training-domain screen 及解封后的 dev 上优于 script，并在 screen 上优于等候选数
uniform random 与单候选 ridge。** 这部分首次满足了预算限定的“学习结果优于脚本”判据，但不能
扩写为单次闭环策略优越、条件化生成已解决、heldout 已确认或 ABD 扰动已校准。P2.7 也说明单标签
模式平均不是唯一瓶颈；下一阶段若继续学习主线，应把强固定四原型库列为正式基线，并预注册最低
条件依赖/相对原型库增益门槛，否则继续扩大 K 只是在增加搜索式 portfolio 成本。

## 2026-09-12 · P3.1 ABD 说明书证据解锁（manual_review.csv 24/24 填毕）

**动机**：audit（runs/20260911_abd_calibration_audit）判 NO-GO 的根因是"制动归因链未文档化"；
`C:\Program Files (x86)\ABD` 安装目录含全部官方说明书，可解锁人工核对。

**做法**：pymupdf 提取 8 份关键 PDF 共 572 页为页标记文本（runs/20260911_abd_review/manuals_txt/，
引用一律给文件+页码）；通道语义逐条对照 RC Software Manual（RM-S-01 Iss.23）§6.12.11.11；
规程配方对照 AN-6092（Euro NCAP C2C 2020）与 AN-6157.01/.02（C-NCAP 2024 VRU）；
数据侧用 audit evidence.json 窗口绝对值复核（delta 判据不充分——V14_T503 基线 9.95 EU 证明
"恒非零保持"存在，必须查 base=min=max 绝对值）。

**关键发现**（证据全文：runs/20260911_abd_review/MANUAL_EVIDENCE.md E1–E7）：
1. `UseBrakeRobot` spec 标志真实语义 = "Use BR for speed control"速度控制选项
   （RC §11.4.5.7 p.339-340；AN-6157.01 p.13 n.b. "will not change BR use for turning tests"），
   不是"制动事件由机器人执行"的声明。audit 以 flag=True 整体排除 11 条属保守政策。
2. CCRs AEB 配方 = VUT SR/AR 组合，AR 于 TTC=3s 转 hold filtered throttle"确保不干扰 AEB"，
   AR 结束=速度低于测试速度 5 kph（防油门覆盖 AEB，协议 §8.4.5）；BR 仅 FCW 变体；
   驾驶员仅 deadman（AN-6092 p.14-15）。CPTA = Car-to-Pedestrian **Turning** Adult（转弯），
   BR 使用不受 flag 控制（AN-6157.02 p.45-46）。
3. 消除法归因链成立：窗内 BR Command≡0（绝对值）+ 触发标志无变化 + AR 油门保持不可能
   −9~−11 m/s² + 规程驾驶员不干预 ⇒ AEB；驾驶员违规踩踏板为不可区分残余风险（已文档化）。
4. 绝对值复核把 3 条被 flag 误杀的运行恢复为 AEB 归因（V15_T3400 CCFT、V15_T5195 CPTA、
   V4_T27 S9 CCRs）——audit 原产物未改动，升级在此留痕。

**结果**：manual_review.csv 24/24 行填毕（utf-8-sig，每格证据引用）：eligible_for_abd_calibrated_v1=10、
excluded_brake_robot_sourced=7、excluded_no_braking_response=3、
excluded_calibration_run_no_response=3、excluded_calibration_run_robot_command_step=1。
audit 的 NO-GO 解除条件（归因链文档化）已满足。

## 2026-09-12 · P3.2 正式校准 abd_calibrated_v1（10 条 AEB 归因记录）

**脚本**：`scripts/calibrate_abd_v1.py`（新；复用 audit 解析器/事件窗，audit 产物不动）。
**产物**：runs/20260912_abd_calibration/{abd_calibrated_v1.json, calibration_table.csv,
distributions.json, verification.json, REPORT.md}。

**四项交付**：
1. 去重/划分：sha256+test_id 双唯一（10/10）；CCRs 8（19.2–21.0 kph）+ turning 2（10.5 kph）
   分速度制度；8 车型，15-TANG 贡献 3 条。
2. 分布：`brake_deceleration` **U(9.128, 11.419)** m/s²（峰减速 pooled 中位 10.74、std 0.76；
   p05 与峰差 ≤0.15=平台期）——假定域 U(5.5,8.0) 整体低于实测，此前扰动低估 ego 制动强度。
   `response_delay` **U(0.055, 0.386)** s（CCRs margin-time=onset_TTC−v/|peak| 代理，中位 0.227；
   env 语义=触发→输出延迟，AEB 请求信号未记录故为上界代理，已如实标注）。
   `action_delay_steps`/`target_accel_scale` 保留假定并标注不可辨识原因（≤40 ms 无通道可辨；
   后者为 NPC 侧缩放、VUT 日志无 NPC 通道——曾考虑用峰减速散布映射，读 env.py:165 后纠正）。
3. 独立验证：LOO 重拟合（peak 域端点最大移 0.31/0.40=极值定义效应；margin 下界由两独立车型
   支撑）；替代阈值 −0.5 m/s²：CCRs 峰差全部 0.000、onset 差 ≤0.11 s（7/8），turning 2 条
   不稳（软阈值锁到转弯早期轻微减速）⇒ response_delay 只用 CCRs 子集有据。
4. 版本化配置：abd_calibrated_v1.json（版本/provenance/每参数证据/不可辨识标注）；
   scenario_lab 集成：`perturb_spec(spec, rng, calibrated=None)` + `load_perturb_config()` +
   evaluate `--perturb-config`（默认路径行为不变，source 标签 abd_calibrated_v1_partial）。

**测试**：新增 tests/test_perturb_config.py（4 项：校准域内采样/默认回归/缺字段拒绝/交付配置
可消费含"实测域整体高于假定域"断言）；全套 pytest 93 passed。

**未解决/边界**：n=10 跨 8 车型，per-vehicle 分布不可辨识；归因仍为消除法（无直接 AEB CAN 通道，
驾驶员违规踩踏板不可区分）；用户 Robot Controller 抽查未做（reviewer 列已注明 pending）；
margin-time 为代理量非直接时延测量。C-NCAP C2C 无 ABD 专册，引 Euro NCAP 2020 为最近同构配方。

## 2026-09-12 · P3 补充：BR 硬件存在性证据链（MANUAL_EVIDENCE.md E8）

用户质询"若未装 BR，txt 通道是否可信"后补充核查。手册侧：BR Position 数据源仅两种
（外置编码器/执行器电机编码器，RC p.107），Brake force 为载荷计 "if connected"，
BR Command 为实际指令信号——不存在"无硬件仍输出"的路径。数据侧（audit evidence.json
全 run 统计）：24/24 条 BR Position nonzero=1.000 且静息位各车互异（−16.6~−114.3 mm）、
BR Velocity/载荷计全程活跃、BR Command 全部有非零指令历史；7/8 车型同车存在 BR 主动
制动闭环联动（cmd→pos→force 三方一致，如 V14_T503 55.5 EU/61.2 mm/149 N；唯一例外
8-huajingS 亦有编码器全程读数+载荷计 65 N）；run 目录 .spec 含 BR 专属选项字段
（CorrectForBrakeNonLinearity）。结论：装了 BR 使"事件窗内 cmd≡0"成为有效检验；
若未装 BR 则 AEB 归因仍成立（−9~−11 m/s² 减速只能来自车辆自身或驾驶员违规）。
仪表级确认途径（RC Setup > Transducer Connections）已列入用户抽查清单。

## 2026-09-12 · P3.2 复核修正：abd_supported_v1

对原始手册、事件窗与 `scenario_lab` 参数语义复核后，P3.1 的 10 条入选记录保留，但证据等级明确为
`AEB-attributed by elimination`，不是直接 AEB 激活信号。P3.2 的两项映射作如下更正：

1. env 的 `brake_deceleration` 是触发后施加的恒定减速度。原配置使用瞬时峰值
   U(9.128,11.419)，会高估整个停车过程。新配置按事件窗速度积分与停车距离换算等效恒定减速度，
   得到 U(5.455,8.174) m/s²；这是 10 条观测极值构成的敏感性包络，不是总体概率分布。
2. `onset_TTC-v/|peak|` 是触发策略、几何、制动建立过程的混合余量，不能识别
   trigger-to-output delay。`response_delay` 因此恢复假定 U(0.1,0.4)，观测 0.055–0.386 仅作描述。

`action_delay_steps` 与 `target_accel_scale` 继续保留假定。新交付为
`runs/20260912_abd_calibration/abd_supported_v1.json`，source 标签
`abd_supported_v1_partial`。原 `abd_calibrated_v1.json` 保留作历史记录但废止，不得用于正式 P2.8。
配置加载器同时增加版本、分布、有限数值、上下界和整数范围校验；evaluate 报告的 source 改为从
配置版本生成。LOO 更名为内部端点敏感性，不再称为独立验证。

## 2026-09-12 · P2.8 条件化原型路由（正式 screen）

P2.8 冻结 P2.7 中仅按既有 validation set loss 选出的 seed47 四原型库，不再优化 pulse 参数。
新的 branch-specific ridge router 以最初 5 帧 actor-visible history、可见性 mask 和 presence mask
为输入，学习每个原型在 3 个 `abd_supported_v1_partial` 扰动下的 dangerous-and-valid 比例。
训练条件 seed72000，独立 screen seed73000，均为 sampler/physics v2、`lane_locked`；候选结果不参与
条件筛选，只保留 nominal 下 complete、valid 且 script-safe 的条件。训练 eligible 为 single 110、
dual 100；screen 为 single 103、dual 101。固定排序、ridge alpha 和所有门槛均在 screen 前冻结。

正式 screen 每条件使用 5 个成对扰动。single：router-2 0.365、fixed-2 0.324，配对差 0.041
[0.012,0.078]；相对 shuffled-input router-2 差 0.052 [0.014,0.099]，通过。dual：router-2
0.301、fixed-2 0.232，配对差 0.069 [0.034,0.113]；相对 shuffled-input 差 0.038
[-0.006,0.083]，置信区间跨零，未通过预注册负控门。两分支候选有效率分别 1.000/0.987，角色违规
均为 0，排序也确实随条件变化。

因此 P2.8 总 gate 为 FAIL，fresh development 未评价，heldout 未读取。结果支持“可见历史相对固定
全局原型排序有增益”的有限结论，但 dual 的输入对应关系尚不能排除由有限样本或排序边际分布造成；
不能宣称条件化机制已双分支确认。router 是集中式场景级候选选择器，不是分散式闭环 actor policy。

## 2026-09-12 · P3 归因撤回：用户确认存在人工安全制动

用户确认：AEB 已触发但过晚时，驾驶员会人工踩刹车避免碰撞。Post Processor User Guide PDF
第 19 页也明确说明其 −1/−0.3 m/s²加速度阈值 AEB event 检测可能把 driver intervention 误识别
为 AEB。现有 24 条导出没有直接 AEB/FCW 或驾驶员制动通道，因此 BR Command=0 只能排除机器人，
不能排除驾驶员。

原 10 条 `AEB_by_elimination` 全部降级为 `unknown_AEB_or_driver_brake`，`abd_supported_v1.json`
仅保留历史复现用途，ABD/AEB 校准证据声明撤回。`scripts/calibrate_abd_v1.py` 现在要求每条入选 run
显式具有 `driver_intervention=none_confirmed`，否则拒绝生成。P2.8 数值仍可作为与原 assumed 范围
近似的敏感性实验复现，但不再称为 ABD 校准验证。完整补采字段、同步、删失裁决和 env 参数拆分见
`docs/ABD_RECALIBRATION_PROTOCOL.md`。

## 2026-09-12 · 无车辆 CAN 边界与历史 smoke 车型筛选

用户确认全部测试均未连接车辆 CAN，因此 AEB request/active/state 和 requested deceleration 无法补采。
后续校准范围改为 observed braking onset、驾驶员安全接管删失和目标平台 command/actual 执行误差；
不再以 AEB request-to-response delay 为目标。FCW 只有在新增外置 AVAD audio/light receiver 时才可
直接记录。

对 `Data/ABD_Data` 约 4,130 条导出进行路径和表头筛查后，选择 14-BZ3X 做 5 条历史 smoke：
V14_T56_R1/R2（CCRs 重复）、V14_T133_R2（CCFT、BR-zero）、V14_T503_R4（CPTA、BR-zero）及
V14_T133_R1（BR-active 负控）。五条均为 415 通道、约 100 Hz，配套 `.spec/.log/.CRUN` 齐全；
四条 BR-zero 仍为 `unknown_aeb_or_driver_brake`，不能用于 AEB 校准。清单、哈希和事件诊断见
`runs/20260912_abd_smoke_selection/`。

## 2026-09-12 · 14-BZ3X 人工介入复核完成

项目试验员已在 Robot Controller 中逐条检查四条 BR-zero run，综合 Results > Check Paths 与
Motion Pack 的 Forward velocity/Lateral velocity 曲线，四条均填写为 `none_confirmed`。CSV 已精简为
只需维护 `driver_intervention`；统一判读方法、日期和局限由脚本写入 manifest，不要求逐行重复填写。

四条记录现标为 `observed_braking_no_takeover_signature_aeb_unconfirmed`，可进入 observed braking
response 分析。该复核属于运动学曲线的间接证据：它可以识别明显避让和异常停车形态，但没有独立
踏板/制动压力标记，可能漏掉与 AEB 曲线相似的纯直线人工制动。由于车辆 AEB request/status 仍不可得，
四条记录继续禁止用于 AEB request-to-response timing 校准；BR-active 的 V14_T133_R1 保持机器人制动负控。

## 2026-09-12 · P2.9 路由上限与可观测性诊断

P2.9 在计算新指标前冻结输入哈希、门槛和决策树，仅复用 P2.8 的 training/screen attempt 矩阵；
没有新增 rollout，没有读取 fresh development 或 heldout。每个条件计算事后最优 top-2 上限，并用
leave-one-perturbation-out（其余四个扰动选原型、留出扰动计分）检验原型偏好是否跨扰动稳定；另以
5000 次条件置乱检验 actor-visible history 与原型排序的对应关系，并用固定网格 RBF kernel ridge
作为非线性容量探针。

single 的 fixed-2 / optimistic oracle-2 / LOO oracle-2 为 0.324 / 0.417 / 0.406；LOO 相对 fixed-2
差 0.082 [0.037,0.134]。dual 分别为 0.232 / 0.343 / 0.333；LOO 差 0.101
[0.053,0.154]。有限扰动噪声扣除后的平均条件信号可靠度为 single 0.966、dual 0.947。
所有 LOO 折都存在并列最优 pair，但取并列候选中的最差 held-draw 结果时，single/dual 仍为
0.404/0.325，相对 fixed-2 的 bootstrap CI 下界均大于零，因此稳定性结论不依赖字典序 tie-break。

P2.8 ridge 的 single top-2 为 0.365，相对置乱均值 +0.049，p=0.0058；dual 为 0.301，
相对置乱均值 +0.037，p=0.0108。RBF 为 0.357/0.307，均没有显著超过 ridge（dual 差 +0.006，
CI [-0.016,0.032]）。因此两分支诊断均为 `aligned_routing_signal_requires_new_screen`：现有合法可见
特征含有弱但可检出的条件排序信号，暂不支持“原型无互补”“扰动偏好不稳定”“可观测性完全不足”
或“必须换非线性模型”。P2.8 的预注册总 gate 仍保持 FAIL；P2.9 是对已消费 screen 的诊断，下一步
应冻结原 P2.8 ridge 与原型库，使用新的预注册 screen 和多置换负控重新检验，不能据此解封 heldout。

## 2026-09-12 · P2.10 冻结 ridge 独立确认

在生成任何新条件和 outcome 前冻结 P2.7 四原型库、P2.8 branch-specific ridge、固定 top-2 顺序、
数值敏感性配置、seed 和全部门槛。新的 confirmatory screen 使用 seed75000、sampler v2、
`lane_locked`、每分支生成160条件、每个合格条件5个配对扰动；负控为5000次跨条件排序置换。
screen 与 P2.8 training/screen 的物理条件指纹交集为0。两分支 screen 全门槛通过后，才读取此前
未评价的 `development_seed74000_samplerv2`；它与全部先前条件的指纹交集也为0。heldout 未检查、未读。

confirmatory screen：single N=121，router-2 0.388、fixed-2 0.357，差0.031
[0.008,0.060]，置乱对齐差0.043、p=0.0008；dual N=104，router-2 0.373、fixed-2 0.290，
差0.083 [0.035,0.135]，置乱对齐差0.051、p=0.0018。候选有效率1.000/0.969，角色违规均为0。

条件式 fresh development：single N=107，router-2 0.325、fixed-2 0.279，差0.047
[0.021,0.077]，置乱差0.043、p=0.0034；dual N=116，router-2 0.345、fixed-2 0.267，
差0.078 [0.038,0.124]，置乱差0.042、p=0.0066。候选有效率1.000/0.963，角色违规均为0。
router-2 相对扰动下 script 的 bootstrap CI 下界在四个分支×阶段组合中也全部大于0。

因此冻结的两候选 ridge router 通过 P2.10 双阶段判据，具备之后一次性 heldout 评价资格。本结论限定为
每条件最多两次候选 rollout 的 dangerous-and-valid 覆盖率，并且当前数值扰动只作为可复现敏感性域，
不是 AEB 校准分布；不能表述为单次闭环策略优于脚本。P2.10 本身不执行 heldout。

## 2026-09-12 · P2.11 冻结单候选路由独立确认

在生成新条件与 outcome 前，冻结 P2.7 四原型库、P2.8 branch-specific ridge、训练域最强单原型
head 1、seed76000、5 个配对数值扰动以及全部判据。每分支一次性生成 360 个条件，只按 nominal
script 的 complete、valid、safe 状态筛选；得到 single 239、dual 232 个合格条件。新 screen 与
P2.8 training/screen、P2.10 confirmatory screen 和 fresh development 的物理条件指纹交集为 0。

正式方法每条件只执行 router 排名第一的一个原型。为实施预注册的条件置换负控，实验后端评估了完整
四原型 outcome 矩阵；其余三个 outcome 仅用于统计负控，不计入正式方法覆盖率或候选预算。single 的
router-1 / fixed-1 / script 为 0.352 / 0.308 / 0.000，router-1 相对 fixed-1 的配对差为 0.044
[0.012, 0.078]，相对 script 的差为 0.352 [0.295, 0.410]；相对 5000 次条件置换均值的差为
0.110，单侧 p=0.0002。dual 分别为 0.253 / 0.216 / 0.023，相对 fixed-1 的差为 0.037
[0.003, 0.073]，相对 script 的差为 0.230 [0.178, 0.284]；相对置换均值的差为 0.068，
单侧 p=0.0002。

候选有效率 single/dual 为 1.000/0.959，角色违规均为 0；top-1 原型众数占比为 0.653/0.608，
满足条件依赖多样性门槛。P2.11 因此双分支 PASS，按预注册决策以 router-1 取代 router-2，成为首选
最终方法并具备之后一次性 heldout 评价资格。该结论仍限定为一次场景级条件选择加一次 rollout 的
dangerous-and-valid 覆盖率；router 不是持续反应式 actor，当前扰动是数值敏感性域而非 AEB 校准分布。
heldout 在 P2.11 中未检查、未读取。

## 2026-09-12 · P2.12 一次性 final confirmation 预注册与授权门

P2.11 通过并提交后，冻结一次性最终确认协议与执行脚本，但未生成、检查或读取 heldout 条件。
P2.12 固定使用 seed77000、sampler/physics v2、`lane_locked`、每分支一次性生成 360 条条件、每条件
5 个配对数值扰动，并要求每分支至少 220 个 nominal-script-safe 合格条件。最终方法保持 P2.11
router-1；等预算基线为训练域最强 fixed-1，script 为次基线，5000 次条件置换为负控。统计门槛与
P2.11 完全一致，任一分支失败都如实记为最终确认 FAIL，不允许改模型、改 K、改 seed、改分母或
回退到 P2.10 router-2 重新解释。

执行器在无参数时只验证冻结模型、P2.11 结果和仿真源文件哈希，并确认 heldout 尚未创建；只有收到
明确授权并传入 `--authorize-one-time-heldout` 才会写入 attempt marker、生成条件和开始 rollout。
完成标记存在后拒绝重跑；中断时只允许在同一预注册哈希与确定性缓存上恢复。旧
`heldout_seed41000` 因较早 P2 设计的规模与筛选协议不匹配，继续封存且 P2.12 明确不读取。

## 2026-09-12 · P2.12 一次性 heldout final confirmation 结果

用户明确授权后，执行冻结提交 `0dcc3a0` 中的唯一 P2.12 attempt。seed77000 每分支各生成 360 条条件，
只按 nominal script 的 complete、valid、safe 状态筛选，得到 single 222、dual 227 个合格条件；两支
均达到预注册最小 220 条要求。720 个 heldout 条件与 P2.8 training/screen、fresh development、
P2.10 screen 和 P2.11 screen 的物理指纹交集为 0。旧 `heldout_seed41000` 未读取。

single 的 router-1 / fixed-1 / script 为 0.368 / 0.327 / 0.000；router-1 相对 fixed-1 的配对差
为 0.041 [0.012, 0.071]，相对 script 为 0.368 [0.310, 0.428]，相对 5000 次条件置换均值的差为
0.114，单侧 p=0.0002。dual 分别为 0.204 / 0.174 / 0.022；相对 fixed-1 的差为 0.029
[0.001, 0.058]，相对 script 为 0.181 [0.135, 0.230]，相对置换均值的差为 0.057，单侧
p=0.0002。candidate valid rate 为 1.000/0.942，角色违规均为 0；所有预注册门槛双分支通过。

独立 post-run 审计从 449 条条件矩阵重算配对 bootstrap 与置换检验，核对 8,980 条候选记录、
2,245 条 script 记录、全部 raw artifact 哈希和完成标记哈希，结果一致。fingerprint audit 原字段保存
的是完成前 attempt marker 哈希；复核时将其明确重命名并补充完成态 marker 哈希，没有改变条件、
outcome、统计或门槛。P2.12 最终结论为 PASS，C3/C5/C6 可作为论文主结果；dual 的 fixed-1 优势
CI 下界 0.0009、置换增益 0.057，虽过预注册门槛但余量较窄，必须按数值如实呈现。

最终结论限定为：冻结的场景级条件路由在每条件一个候选 rollout 的等预算下，提高 single/dual 的
dangerous-and-valid 覆盖率。它不是持续反应式 actor；扰动仍是数值敏感性域，不是 AEB 校准分布，
也不构成车辆/车型级验证。完成标记设为 `rerun_forbidden=true`，不得再运行或替换本 heldout。

## 2026-09-12 · 目标端（GST/LaunchPad）数据调查与 target_accel_scale 论断修正

**动机**：用户质询"ABD 数据集里面只有主车端（机器人）的数据吗？目标端 GST 和 LaunchPad 的
数据都不在里面吗"。

**方法**：临时脚本（runs/tmp_chans.py，用后已删）逐车型抽前 40 个 txt 稳健解析得到通道并集；
对 4 个代表 run（V14_T56/V9_T52 CCRs、V15_T3400 CCFT、V15_T5195 CPTA）验证 tracker 通道
数据存在性；盘点 BYD_Bao5/C-NCAP_2024 文件性质。

**发现（证据全文：runs/20260911_abd_review/MANUAL_EVIDENCE.md E9）**：
1. 5,543 个 txt 全部为 VUT 端单控制器 100 Hz 导出（13 文件夹 txt 计数见 E9）；LaunchPad80
   专属通道组（Servo Brakes 等，RC §6.12.11.11.6）在全部车型缺席——GST/LaunchPad **自身**的
   导出文件不在数据集内。
2. 但 VUT 日志经 Synchro 中继含目标端实时状态与**指令**通道（Head tracker reference/actual
   X/Y、forward velocity/acceleration、lateral error、Pedestrian articulation、Tracker
   status/time error；RC p.112-113）：CCFT V15_T3400 actual+reference 全程有值
   （lateral error ≤0.105 m）、CPTA V15_T5195 lateral error 0.009–0.043 m；CCRs（GVT 静止）
   tracker 通道全零，与 AN-6092 p.15 "no test is run in RC on the GVT" 一致；Tracker
   status=9、Tracker time error 全零（CCFtap 非 full-sync 模式，AN-6092 Appendix 2）。
3. BYD_Bao5 文件夹 = 129 .pmc（路径表 Distance/Time/X/Y/Curvature）+129 .tem（测试方法）
   +21 .spf（速度剖面 T/V/D），无 txt；C-NCAP_2024 = 3 份 PDF。

**修正**：P3.2 `target_accel_scale` 证据中 "VUT-side ABD exports contain no NPC execution
channels" 论断**错误**。`scripts/calibrate_abd_v1.py` evidence 文字与 status 已更正为
`retained_assumed_not_fitted_v1_tracker_channels_exist`（v1 仍保留假定域：10 条入选 run 中
运动目标仅 2 条、ref-actual 位置差映射到加速度缩放需专门建模；标注为未来版本候选数据源）。
`abd_supported_v1.json` 属归因撤回后的历史复现产物，脚本现按 driver-intervention 确认门拒绝在
无 `none_confirmed` 记录时重新生成（本次重跑确认报错 "driver intervention ... not confirmed-none
for 10 eligible runs"，为 P3 归因撤回的设计行为，不绕过），JSON 内旧措辞不改动，以
`runs/20260912_abd_calibration/REPORT.md` 同日更正为准。

**测试**：pytest 全量 **106 passed**（本轮无 scenario_lab 代码改动，仅证据文字修正）。

## 2026-09-12 · ABD 数据现状核验 + BR 踏板通道判别力补分析（回应"AEB 数据有多少/Codex 为何不自行判别"）

**背景**：runs/ 下存在四个未在 CHANGELOG 留痕的 ABD 目录（20260912_abd_fcw_audio_audit /
braking_source_analysis / no_takeover_screen / proxy_calibration_v2），本轮逐一核验其产物并
回答用户质询。

**ABD 数据漏斗（no_takeover_screen/screening.csv + proxy_calibration_v2）**：5,543 txt →
4,121 通道导出 → 1,258 条 AEB-path → 892 条 BR-zero 且有观测制动事件 → **44 条 operator
复核 AEB 响应 run**（43 条 none_confirmed 全窗 + 1 条 AEB 停止后人工接管、截尾到首次停止；
10 车型 14 场景标签）。FCW-only 188 条（T_FCW_audio 已恢复，全部可用于报警时刻分析；其中
33 条 BR-zero 经确认为报警后人工制动，禁止进入 AEB 响应/代理校准）。proxy_calibration_v2
主动封存：在 response_delay 拆分为 trigger-policy 与 actuation delay 前不直接进仿真，防止
0.15–0.35 s 先验被重复计入。

**BR 踏板通道判别力（本轮新算，labelled_event_features.csv 44 AEB vs 33 manual）**：
braking_source_analysis 提取了 br_position_travel_mm/brake_force_max_n 但 REPORT 未给出
该对照，本轮补齐——
- `br_position_travel_mm`：AEB 中位 22.1（0–46.2）vs manual 中位 25.8（0.01–45.1）mm，
  **完全重叠无分离**。即这些车型的 AEB 液压/助力执行会反驱踏板（cmd=0 时踏板仍移动
  14–50 mm，载荷计仅 3–20 N）——踏板位移不能作为驾驶员检测器。
- `brake_force_max_n`：AEB 中位 12.7（max 75.0）vs manual 中位 18.9（min 15.3，max 306.7）N，
  核心区间重叠；**仅 manual 高力尾（>75 N）具单向证据力**（"有脚"可判，"≤75 N"不能证 AEB）。
- 同速度子集（50.3–64.2 kph）：peak decel AEB 11.10 vs manual 11.01 m/s²、onset-to-peak
  1.37 vs 1.27 s——**"人工接管踩得轻"的假设在该车队被数据否定**（FCW 接管为避撞全力踩，
  ~1g）。有分离的特征（speed rebound AUC 0.924 / duration 0.916 / time-to-peak 0.915）被
  起始速度混淆（AEB 中位 20.8 vs manual 59.3 kph），不满足自动打标条件——Codex 拒绝自动
  标注、改出 40 条排序人工复核队列的处置与该证据一致。

**待办建议（未执行）**：1) 以 force>75 N 作单向 triage 扫剩余 ~848 条未复核 BR-zero run；
2) 在 892 条中寻找 AEB 测试速度档的疑似人工接管样本，补齐 matched-speed manual 类
（现有 33 条 manual 全来自高速 FCW 测试，是自动判别器标定的真实瓶颈）；3) 本段四个目录
的原始工作应补 CHANGELOG 留痕（本轮只补核验记录）。

## 2026-09-12 · AEB 踏板签名判据：操作员力学模型 → 已标注 run 验证（43/44 通过，manual 标签被推翻）

**用户补充的力学事实**：1) BR 与踏板刚性连接，载荷计仅在机器人主动施力或人脚主动下踩时
读大值（压缩）；2) 耦合车型 AEB 拖踏板下压、载荷计被拉着走（力小，可为负=张力），解耦
车型踏板不动；3) C-NCAP 2024 D4F4 = BR 先轻踩（cmd 小脉冲）再 AEB 刮停（主段 cmd≡0）；
4) FCW 场景有时驾驶员踩刹车、有时让 AEB 制动，且 FCW 场景不进本项目范围。

**产物**：`scripts/validate_abd_pedal_signature.py` +
`runs/20260912_abd_pedal_signature_validation/{per_run_signature.csv, REPORT.md}`。

**验证结果（77 条已标注 run，事件窗 + 窗前 2 s 基线，带符号载荷计）**：
- 44 条 operator 确认 AEB：42 `aeb_pedal_dragged`（行程 14–50 mm + 力小/张力）+
  1 `aeb_pedal_static`（解耦型）= **43/44 签名一致**；1 `drag_then_foot_adjudicate`
  （V13_T332_R2，张力 −84.5 N 与 75 N 压缩同窗，疑 AEB 先作动驾驶员后补脚）；
  张力 19/44 且车型相关（13-G9 −73~−84 / 14-BZ3X −16.7 / E8 −4.7 / 9-BZ7 无）；
  foot_pre_event 0/44。
- 33 条 FCW 后"manual"（规程推定标签）：仅 3 条 `foot_compression`（V9_T69_R3
  0.04 mm+51.5 N；V2_T60_R1/R2 306.7/117 N）；**30 条呈 AEB 签名**（18 dragged +
  12 static）；张力 0；foot_pre_event 14/33。
- **9-BZ7 同车对比**：30 manual vs 9 AEB 的 travel 25.8/25.2 mm、fmin +1.1/+0.7、
  fmax 17.7/17.7 N——同车同签名，支持 30 条实际为 AEB 制动。
- 由此**推翻上一段"特征无分离"结论**：根因是 manual 标签污染（规程推定 ≠ 力学事实），
  并非通道无判别力。

**AEB 筛选规则（回答"怎么筛"）**：BR Command≡0 前提下，(a) 窗内踏板行程≤2 mm →
解耦型 AEB；(b) 行程>2 mm 且压缩力≤50 N（负值张力加分）→ 耦合型 AEB；(c) 压缩力
>50 N 且无张力 → 人脚；(d) 张力+大压缩同窗 → 裁决；辅证：窗前 2 s 踏板>5 mm 活动=
脚在踏板上。D4F4 识别（待扫）：轻踩脉冲 + 主段 cmd≡0 + 主段 AEB 签名。

**未解决**：阈值暂定未跨速度档标定；载荷计几何因车而异；test_id 非唯一（15 个同名
不同路径，去重须 sha256+test_id）；892 条 BR-zero 全量扫描与 D4F4 扫描待用户确认后
执行；V13_T332_R2 与 9-BZ7 30 条待操作员复核。

## 2026-09-12 · AEB 踏板签名全量筛选：897/1160 AEB 签名、去重后 707 条 aeb_dataset_v0、D4F4 零命中

**执行**（用户批准"开始下一步工作"）：`scripts/screen_abd_aeb_pedal_signature.py`
（新；classify 与阈值从 validate 脚本 import，单一来源）×
`runs/20260912_abd_no_takeover_screen/screening.csv` 的 892 条 BR-zero +
268 条 robot_channel_active（98 条无事件跳过），4 进程 165 s，逐 run 流式解析 +
sha256 + 事件窗重算。产物：`runs/20260912_abd_aeb_pedal_screen/`
（per_run_signature.csv 1,160 行、summary.json、REVIEW_QUEUE.csv 46 行、
aeb_dataset_v0.csv 707 行 + duplicates 明细、REPORT.md）。

**总分类**：aeb_pedal_dragged 881 + aeb_pedal_static 16 = **AEB 签名 897**（77.3%）；
foot_compression 33；drag_then_foot_adjudicate 6；robot_braked_main_window 217；
robot_joined_mid_window 7。operator 44 条复核 run 与签名**零矛盾**
（41 dragged + 1 static + 1 已知裁决 + 1 截尾 run 的 AEB 段签名）。

**两项对 Codex screen 分类的实证修正**：
1. robot_channel_active 有 44 条系 **BR Command 噪声 0.01–0.06 EU 触发零阈值误标**
   （事件窗+前 5 s cmd 实际≤0.5 EU）；已按 BR-quiet 规则重分类（screen_flag_noisy=1
   留痕：25 dragged + 3 static + 15 foot + 1 裁决）。**下游判 BR 活跃须 |cmd|≥0.5 EU**。
2. 217 条主段 cmd 活跃中拆出 7 条 cmd 于 onset +0.5 s 后才出现者
   （robot_joined_mid_window），逐条 0.1 s 时序核验：**5 条为 AEB 主减速在先、
   机器人尾部 −2.4~−3.9 EU 小指令收尾**（V3_T116/119/140/143、V14_T494；
   截尾到 cmd 起点后可用，待操作员确认）；2 条为机器人制动
   （V15_T5258 cmd −51+98 N 压缩；V14_T1804 前相异常单独裁决）。

**D4F4 扫描结论：全语料 0 命中**（判据：前 5 s 内 0.5<|cmd|≤30 EU 轻踩脉冲 +
主段 cmd≡0 + 主段 AEB 签名）。历史数据不含 BR 轻踩→AEB 模式；规则保留在脚本中
供 C-NCAP 2024 补测复用。注：D4F4 字面量在路径中本就 0 命中，此为模式级扫描。

**车型踏板架构**（AEB 签名集）：张力出现按车型二值化清晰——13-G9 25/29、
14-BZ3X 217/240、A66 13/35 有张力；9-BZ7、15-TANG、S9 为 0；static（解耦型）全队
仅 16 条 → 本车队以耦合型为主。onset 速度 8.4–80.5 kph（中位 30.5）；
场景 AEB 314 / CCRS 111 / CPTA 111 / CPLA 79 / CCFT 46 / SCP 45 / CBLA 44 /
CPNCO 44 / CBNA·CPNA·CCRM 各 30 / CPFA 6 / CBFA 5 / OTHER 2（去重前）。

**去重**：1,160 → 914 unique（246 组 sha256 完全重复）；AEB 签名 897 → **707
unique**（9-BZ7 冗余最重 360→183）。**所有下游消费必须 sha256 去重**。

**aeb_dataset_v0**（707 unique，10 车型）：screen_class ∈ {dragged, static} 按
sha256 去重（同组留字典序第一条，190 条冗余副本在 aeb_dataset_v0_duplicates.csv）。
边界不变：签名是力学事实非 AEB ECU 观测；foot/裁决/中途接管未入集；
response_delay 拆分前不进仿真。

**操作员队列（46 条，机器不改标签）**：foot 33（24 条 9-BZ7，压缩 50–301 N，
foot_pre_event 全 0——事件中突然补脚，符合"AEB 过晚驾驶员接管"形态；其中 15 条
来自噪声误标组）+ 裁决 6（13-G9 ×5 张力 −66~−85 与压缩 66~290 同窗 +
V15_T5222 阈值边缘）+ 中途接管 7（§上）。

**本轮文件**：新增 scripts/screen_abd_aeb_pedal_signature.py；产物
runs/20260912_abd_aeb_pedal_screen/（CSV 与 REPORT 提交，大文件不提交）；
诊断用 scripts/tmp_d4f4_detail.py 用后删除（证据数字已录入 REPORT §5）。

## 2026-09-12 · 修正：V3_T140/T143"机器人尾部收尾"解释错误（操作员质询触发）+ 碰撞结局全量扫描

**触发**：操作员指出"机器人不会在 AEB 释放后接手刹停，只有司机会接手"，
质疑 T140/T143 两条的定性。

**复核证据**（scripts/tmp_v3t140_probe.py 全通道时间轴 + .spec，用后删除，
数字录入 REPORT §5）：
1. 力学：cmd −3.05/−3.94 EU 期间踏板行程恒定（40.75/45.27 mm）、载荷计
   −0.66~+1.09 N ≈0——机器人真推踏板必有压缩+行程（同日对照 V15_T5258_R1：
   cmd −51 + 106 N 压缩）；驾驶员踩踏同样会显示压缩力，亦为 ≈0。
2. RC 状态：两条 .spec 均 `UseBrakeRobot=False`，全程 Motion Going BR=0 /
   BR start=0 / BR test=0，AR Command 同期归零。
3. 结局：相对纵向距离过零——T140 +2.81 s（接触时 6.0 kph，压入 −0.339 m）、
   T143 +1.91 s（接触时 19.8 kph，压入 −1.394 m）；T143 二段 −12 m/s² 与距离
   转负严格同步 = 碰撞动力学。

**结论**：机器人未接手（用户判断正确），驾驶员也未踩——把车停住的是碰撞。
两条的真实形态 = AEB 触发→制动→**完全释放**（减速 ≈0 滑行逼近）→接触：
这是 **AEB 中途释放致碰撞的直接观测**（A66 CBNAO/CSFAO 60 kph 各一）。
初版 REPORT §5/CHANGELOG"5 条 AEB 主减速在先、机器人尾部小指令收尾（截尾后
可用）"中的"机器人收尾"解释作废；5 条（T116/T119/T140/T143/T494）的尾部
−2.4~−3.9 EU cmd 均无力学效应，为运行收尾段伺服空闲/保持 trim 信号。截尾规则
本身不变（保守处理，待操作员确认）。另修正一处笔误：mid-window 第 7 条按
per_run_signature.csv 实为 **V15_T5261_R9**（cmd 43.3+压缩 17–70 N，机器人施力/
tracker 疑换目标，交裁决）；V15_T5258_R1 属 robot_braked_main_window
（cmd +0.2 s 即活跃），初版 §5 误列。

**碰撞结局全量扫描**（新增 scripts/scan_abd_contact_outcome.py，1,160 run，
87 s；产物 runs/20260912_abd_aeb_pedal_screen/contact_outcome.csv）：事件窗 ±2 s
内 Relative longitudinal distance 最小值分类；过零类按"过零时车速 ≤25 kph 且
|min|≤3 m"再分（纵向代理，横向几何未查）：
- AEB 签名 897 = **contact_and_stopped 66（7.4%）** + passed_or_swept 105 +
  near≤0.5 m 30 + clear 696；foot 33 中 contact 8；robot_braked 217 中 7；
  mid_window 7 中 3（T140/T143/T1804）。
- contact_and_stopped 压入深度 −0.018~−2.683 m（物理量级）；passed_or_swept
  |min| 3.1~632 m、过零时车速中位 0.1 kph = 目标被甩到车后/走过停住的车，
  **非压溃**。初版把全部过零当 contact（171 条、19.1%）属高估，已修正。
- 深度分布双峰（−0.5 m 与 −6 m 两个量级）是过零二分判据的依据；边界与限制
  已写入脚本 docstring 与 REPORT §8/§10。

**对下游的影响**：per_run_signature.csv 分类不变（踏板签名是力学事实，碰撞结局
是独立维度列于 contact_outcome.csv）；aeb_dataset_v0 不变。AEB 签名集中 66 条
真接触 + 30 条近接触构成"签名成立但未能避免接触"难例子集，供 observed-braking-
response 分析优先使用（须 sha256 去重）。
