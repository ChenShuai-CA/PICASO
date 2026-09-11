"""Summarize measured pilot artifacts without inventing publication conclusions."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    a=p.parse_args()
    root=a.output
    completed=json.loads((root/'completed.json').read_text())
    rows=[]
    for path in sorted(root.glob('eval_*/summary.json')):
        name=path.parent.name.removeprefix('eval_')
        training=root/name/'training.jsonl'
        steps=0
        if training.exists():
            steps=json.loads(training.read_text().splitlines()[-1])['steps']
        for branch,metrics in json.loads(path.read_text())['metrics'].items():
            rows.append(dict(method=name,branch=branch,training_steps=steps,
                attempts=metrics['attempts'],valid_rate=metrics['valid_rate'],
                dangerous_rate=metrics['dangerous_rate'],collision_rate=metrics['collision_rate'],
                unique_valid_dangerous=metrics['unique_valid_dangerous'],
                ci_low=metrics['dangerous_rate_cluster_ci95'][0],ci_high=metrics['dangerous_rate_cluster_ci95'][1]))
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    methods=list(dict.fromkeys(r['method'] for r in rows))
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    x=np.arange(len(methods))
    for j,branch in enumerate(('single','dual')):
        values=[next(r['dangerous_rate'] for r in rows if r['method']==m and r['branch']==branch) for m in methods]
        ax.barh(x+(j-.5)*.36,values,height=.34,label=branch)
    ax.set(yticks=x,yticklabels=methods,xlim=(0,1),xlabel='Valid dangerous episode fraction',
           title='Small-budget pilot: 20 matched initial conditions per branch')
    ax.invert_yaxis();ax.legend();ax.grid(axis='x',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=160);fig.savefig(root/'comparison.svg');plt.close(fig)
    latency=json.loads((root/'latency.json').read_text())
    corpus=json.loads(Path('runs/20260911_public_v4/report.json').read_text())
    lines=['# Ubuntu / GPU 预实验记录',
        '\n所有命令在 WSL2 Ubuntu 中执行，工作区保持在 D 盘。以下是小预算工程预实验，不是正式论文结论。\n',
        '## 公共数据\n',
        json.dumps({k:corpus[k] for k in ('examples','sources','splits','kinds','independent_groups')},ensure_ascii=False),
        '\n## 实测结果\n',
        '| 方法 | 场景 | 训练交互步数 | 有效率 | 有效危险率 |',
        '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['branch']} | {row['training_steps']} | {row['valid_rate']:.1%} | {row['dangerous_rate']:.1%} |")
    lines += ['\n![预实验比较](comparison.png)\n','## 实时性\n']
    for branch,metrics in latency['branches'].items():
        lines.append(f"- {branch}: {metrics['decision_steps']} 步，p99={metrics['p99_ms']:.3f} ms，超时率={metrics['deadline_miss_rate']:.3%}，完整仿真实时因子={metrics['realtime_factor']:.2f}。")
    lines += ['\n## 结论边界与下一阶段\n',
        '- 先验目前只学自身运动，尚未完成对齐邻居、地图及真实交互预训练。',
        '- 行人样本较少；采用训练角色频率权重，但不能替代扩大有效数据和分角色验证。',
        '- ABD 仅完成数据审计和人工核对表，当前执行扰动仍是敏感性假设。',
        '- 每次 RL 只有 12 次更新；不同策略终止时刻不同，交互步数并不完全相同。正式比较须改用等交互预算。',
        '- 独立单/双目标模型在另一分支的结果仅为迁移诊断。',
        '- 参数和轨迹 CEM 当前各优化一个固定案例，不能将其最优值与固定分布评价率直接比较。',
        '- 尚需避让可行性判定、独立自然性指标、更多消融和更大的预注册测试集。',
        '- 低延迟仅为本机软件测量，未验证真实传感器或试验场执行。',
        '\n原始配置、日志、模型、轨迹与统计位于本目录。']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(completed_jobs=completed['jobs'],rows=len(rows),latency=latency['branches'],report=str(root/'REPORT.md')),indent=2))

if __name__=='__main__':main()
