"""Run the preregistered P2.2 shared-vs-specialized development matrix."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (7, 17, 27, 37, 47)
MODES = ('mixed', 'single', 'dual')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train_complete(directory, budget, mode, seed):
    directory = Path(directory)
    needed = (directory / 'config.json', directory / 'training.jsonl', directory / 'policy.pt')
    if not all(path.exists() for path in needed):
        return False
    config = read_json(needed[0])
    history = [json.loads(line) for line in needed[1].read_text(encoding='utf-8').splitlines()]
    if not history:
        return False
    import torch
    bundle = torch.load(needed[2], map_location='cpu', weights_only=True)
    branch_steps = sum(sum(row.get('branch_steps', {}).values()) for row in history)
    return (config.get('seed') == seed and config.get('mode') == mode
            and config.get('interaction_budget') == budget
            and config.get('role_action_mode') == 'lane_locked'
            and config.get('sampler_version') == 2 and config.get('robust') is False
            and config.get('mixed_warmup_fraction') == 0.0
            and not bool(bundle.get('extra', {}).get('pretrained'))
            and history[-1].get('steps') == budget and branch_steps == budget)


def eval_complete(directory, condition_version):
    directory = Path(directory)
    summary_path = directory / 'summary.json'
    episodes_path = directory / 'episodes.jsonl'
    if not summary_path.exists() or not episodes_path.exists():
        return False
    summary = read_json(summary_path)
    return (summary.get('condition_set_version') == condition_version
            and summary.get('role_action_mode') == 'lane_locked'
            and len(episodes_path.read_text(encoding='utf-8').splitlines()) == 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--budget', type=int, default=6000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if 'heldout' in str(args.conditions).casefold():
        raise ValueError('P2.2 must not read a heldout condition set')

    root = Path(__file__).resolve().parents[1]
    conditions = (root / args.conditions).resolve()
    manifest = read_json(conditions)
    if manifest.get('purpose') != 'development':
        raise ValueError('P2.2 requires a development condition manifest')
    condition_version = manifest['condition_set_version']
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs_path = output / 'jobs.json'
    jobs = read_json(jobs_path) if jobs_path.exists() else []

    def run(name, command, complete):
        if complete():
            print(f'SKIP verified {name}', flush=True)
            return
        job = {'name': name,
               'command': [sys.executable, '-m', 'scenario_lab', *map(str, command)],
               'status': 'running', 'started_at': time.time()}
        jobs.append(job)
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
        started = time.perf_counter()
        print(f'START {name}', flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(job['command'], cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT)
        job.update(status='passed' if completed.returncode == 0 else 'failed',
                   returncode=completed.returncode,
                   elapsed_s=time.perf_counter() - started, finished_at=time.time())
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in job.items() if key != 'command'}),
              flush=True)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')
        if not complete():
            raise RuntimeError(f'{name} exited successfully but artifact verification failed')

    for mode in MODES:
        for seed in SEEDS:
            name = f'{mode}_s{seed}'
            directory = output / f'train_{name}'
            command = [
                'train', '--output', directory, '--updates', 40,
                '--episodes-per-update', 4, '--seed', seed, '--mode', mode,
                '--sampler-version', 2, '--interaction-budget', args.budget,
                '--role-action-mode', 'lane_locked', '--mixed-warmup-fraction', 0,
                '--no-robust', '--device', args.device,
            ]
            run(f'train_{name}', command,
                lambda d=directory, b=args.budget, m=mode, s=seed:
                train_complete(d, b, m, s))

    script_dir = output / 'eval_script'
    run('eval_script', [
        'evaluate', '--output', script_dir, '--conditions', conditions,
        '--role-action-mode', 'lane_locked'],
        lambda: eval_complete(script_dir, condition_version))
    for mode in MODES:
        for seed in SEEDS:
            name = f'{mode}_s{seed}'
            train_dir = output / f'train_{name}'
            eval_dir = output / f'eval_{name}'
            run(f'eval_{name}', [
                'evaluate', '--policy', train_dir / 'policy.pt', '--output', eval_dir,
                '--conditions', conditions, '--device', args.device],
                lambda d=eval_dir: eval_complete(d, condition_version))

    (output / 'completed.json').write_text(json.dumps({
        'status': 'completed', 'modes': MODES, 'seeds': SEEDS,
        'interaction_budget': args.budget, 'condition_set_version': condition_version,
        'heldout_read': False,
        'results_kind': 'development_architecture_screen_not_confirmatory_evidence',
    }, indent=2), encoding='utf-8')
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
