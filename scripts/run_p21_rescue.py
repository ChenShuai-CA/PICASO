"""Run the preregistered P2.1 2x2 stability rescue matrix in WSL Ubuntu.

The runner is resumable only from verified artifacts: a train job is complete
when its saved config matches the requested exact interaction budget and its
last history row reaches that budget; an evaluation is complete when its
summary and 80 episode rows exist. No heldout path is accepted or read.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (7, 17, 27, 37, 47)
FACTORS = {
    'pr': dict(prior=True, robust=True),
    'pn': dict(prior=True, robust=False),
    'nr': dict(prior=False, robust=True),
    'nn': dict(prior=False, robust=False),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train_complete(directory, budget, factor, seed):
    directory = Path(directory)
    needed = (directory / 'config.json', directory / 'training.jsonl', directory / 'policy.pt')
    if not all(p.exists() for p in needed):
        return False
    cfg = read_json(needed[0])
    lines = needed[1].read_text(encoding='utf-8').splitlines()
    if not lines:
        return False
    last = json.loads(lines[-1])
    expected = FACTORS[factor]
    import torch
    bundle = torch.load(needed[2], map_location='cpu', weights_only=True)
    return (cfg.get('seed') == seed and cfg.get('interaction_budget') == budget
            and cfg.get('role_action_mode') == 'lane_locked'
            and cfg.get('sampler_version') == 2
            and cfg.get('robust') == expected['robust']
            and bool(bundle.get('extra', {}).get('pretrained')) == expected['prior']
            and last.get('steps') == budget)


def eval_complete(directory, condition_version):
    directory = Path(directory)
    summary_path, episodes_path = directory / 'summary.json', directory / 'episodes.jsonl'
    if not summary_path.exists() or not episodes_path.exists():
        return False
    summary = read_json(summary_path)
    rows = episodes_path.read_text(encoding='utf-8').splitlines()
    return (summary.get('condition_set_version') == condition_version
            and summary.get('role_action_mode') == 'lane_locked' and len(rows) == 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--prior', required=True)
    parser.add_argument('--budget', type=int, default=6000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if 'heldout' in str(args.conditions).casefold():
        raise ValueError('P2.1 must not read a heldout condition set')

    root = Path(__file__).resolve().parents[1]
    conditions = (root / args.conditions).resolve()
    prior = (root / args.prior).resolve()
    manifest = read_json(conditions)
    if manifest.get('purpose') != 'development':
        raise ValueError('P2.1 requires a development condition manifest')
    condition_version = manifest['condition_set_version']
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs_path = output / 'jobs.json'
    jobs = read_json(jobs_path) if jobs_path.exists() else []

    def run(name, command, complete):
        if complete():
            print(f'SKIP verified {name}', flush=True)
            return
        job = dict(name=name, command=[sys.executable, '-m', 'scenario_lab', *map(str, command)],
                   status='running', started_at=time.time())
        jobs.append(job)
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
        print(f'START {name}', flush=True)
        started = time.perf_counter()
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(job['command'], cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT)
        job.update(status='passed' if completed.returncode == 0 else 'failed',
                   returncode=completed.returncode,
                   elapsed_s=time.perf_counter() - started, finished_at=time.time())
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in job.items() if k != 'command'}), flush=True)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')
        if not complete():
            raise RuntimeError(f'{name} exited successfully but artifact verification failed')

    for factor, settings in FACTORS.items():
        for seed in SEEDS:
            name = f'{factor}_s{seed}'
            train_dir = output / f'train_{name}'
            command = ['train', '--output', train_dir, '--updates', 40,
                       '--episodes-per-update', 4, '--seed', seed, '--sampler-version', 2,
                       '--interaction-budget', args.budget, '--role-action-mode', 'lane_locked',
                       '--device', args.device]
            if settings['prior']:
                command += ['--pretrained', prior]
            if not settings['robust']:
                command += ['--no-robust']
            run(f'train_{name}', command,
                lambda d=train_dir, b=args.budget, f=factor, s=seed:
                train_complete(d, b, f, s))

    script_dir = output / 'eval_script'
    run('eval_script', ['evaluate', '--output', script_dir, '--conditions', conditions,
                        '--role-action-mode', 'lane_locked'],
        lambda: eval_complete(script_dir, condition_version))
    for factor in FACTORS:
        for seed in SEEDS:
            name = f'{factor}_s{seed}'
            train_dir = output / f'train_{name}'
            eval_dir = output / f'eval_{name}'
            run(f'eval_{name}', ['evaluate', '--policy', train_dir / 'policy.pt',
                                 '--output', eval_dir, '--conditions', conditions,
                                 '--device', args.device],
                lambda d=eval_dir: eval_complete(d, condition_version))

    (output / 'completed.json').write_text(json.dumps(
        dict(status='completed', factors=FACTORS, seeds=SEEDS,
             interaction_budget=args.budget, condition_set_version=condition_version,
             heldout_read=False, results_kind='development_stability_rescue_not_paper_evidence'),
        indent=2), encoding='utf-8')
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
