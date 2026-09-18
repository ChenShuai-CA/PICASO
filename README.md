# 安全关键场景生成研究

唯一工作区：`D:\Projects\Scenario_Generation_Research`。所有开发、测试和训练通过 **WSL2 Ubuntu** 执行，文件保留在当前工作区。

## 进入与检查

Windows PowerShell 只用于启动 Ubuntu：

```powershell
wsl -d Ubuntu --cd /mnt/d/Projects/Scenario_Generation_Research
```

随后在 Ubuntu 中运行：

```bash
source /home/shuai/.venvs/scenario-gpu/bin/activate
python scripts/check_environment.py
python -m pytest -q
bash scripts/run.sh --help
```

`bash scripts/run.sh ...` 自动使用 Ubuntu 的 GPU 虚拟环境。`--device auto` 优先 CUDA；`--device cuda` 在 GPU 不可用时明确报错；CPU 选项也在 Ubuntu 内运行。

## 主流程

输出路径应使用尚未存在的新实验目录，避免混淆旧结果：

```bash
bash scripts/run.sh prepare-public --output runs/new_public --max-files 4 --records-per-file 32 --max-examples 16000
bash scripts/run.sh pretrain --corpus runs/new_public/motion_prior.npz --output runs/new_prior --device cuda
bash scripts/run.sh train --mode mixed --algorithm mappo --pretrained runs/new_prior/prior.pt --output runs/new_mixed --updates 12 --device cuda
bash scripts/run.sh evaluate --policy runs/new_mixed/policy.pt --output runs/new_eval --count 20 --device cuda
bash scripts/run.sh benchmark --policy runs/new_mixed/policy.pt --output runs/new_benchmark/latency.json --steps 10000 --device cuda
```

单目标使用 `--mode single --algorithm ppo`；双目标使用 `--mode dual --algorithm mappo`。

完整的小预算流程：

```bash
python scripts/run_pilot.py --corpus runs/20260911_public_v4/motion_prior.npz --output runs/new_gpu_pilot
python scripts/summarize_pilot.py runs/new_gpu_pilot
```

## 模块

- `schema.py`、`env.py`、`geometry.py`：二维运动、遮挡跟踪、冻结 AEB 和角色检查。
- `data.py`、`waymo.py`：有界审计、公共轨迹与带 CRC 校验的 Motion TFRecord 读取。
- `pretrain.py`：双来源自身运动先验，数据配额、分组隔离和角色权重。
- `policy.py`、`train.py`：注意力＋GRU、共享角色策略、集中式 critic、混合训练。
- `evaluate.py`、`runtime.py`：评价、搜索、回放、延迟及平台记录。

## 当前边界

目前不是 NCAP 认证系统，也不是已经完成的论文算法。先验尚未加入对齐邻居与地图；当前执行扰动仍是敏感性假设，不能标为 ABD 实测分布。

`python scripts/prepare_abd_review.py` 生成控制来源核对表。abort 不作为碰撞标签，`UseBrakeRobot=False` 也不等于已经确认 AEB 制动。

INTERACTION 使用原始车辆轨迹；明确的行人样本由 Waymo 提供，混合 pedestrian/bicycle 类型不强行当作行人。正式跨来源泛化仍需更丰富的验证数据。

当前有效联合预实验数据是 `runs/20260911_public_v4`。旧的 `wsl_public_pilot`、`20260911_public_v2`、`20260911_public_v3` 仅保留作排错记录。

历史 Stage1–Stage4 和 Word 文档仅供参考，部分路线已替换。旧 `abd_inventory.py` 默认四个车辆目录；全目录索引在 `research_audit_20260910/expanded_inventory/`。文档提取脚本已使用项目相对路径，不会随训练自动运行。
