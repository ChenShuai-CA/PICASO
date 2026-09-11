"""Run P2.3 mixed policies with exact equal per-branch interaction budgets."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (7, 17, 27, 37, 47)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train_complete(directory, branch_budget, seed):
    directory = Path(directory)
    files = (directory / 'config.json', directory / 'training.jsonl', directory / 'policy.pt')
    if not all(path.exists() for path in files):
        return False
    config = read_json(files[0])
    history = [json.loads(line) for line in files[1].read_text(encoding='utf-8').splitlines()]
    if not history:
        return False
    totals = {'single': 0, 'dual': 0}
    for row in history:
        for branch, steps in row.get('branch_steps', {}).items():
            totals[branch] += steps
    return (config.get('seed') == seed and config.get('mode') == 'mixed'
            and config.get('interaction_budget') == 2 * branch_budget
            and config.get('branch_interaction_budget') == branch_budget
            and config.get('role_action_mode') == 'lane_locked'
            and config.get('mixed_warmup_fraction') == 0.0
            and config.get('sampler_version') == 2 and config.get('robust') is False
            and totals == {'single': branch_budget, 'dual': branch_budget})


def eval_complete(directory, version):
    directory = Path(directory)
    if not (directory / 'summary.json').exists() or not (directory / 'episodes.jsonl').exists():
        return False
    summary = read_json(directory / 'summary.json')
    return (summary.get('condition_set_version') == version
            and summary.get('role_action_mode') == 'lane_locked'
            and len((directory / 'episodes.jsonl').read_text().splitlines()) == 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--branch-budget', type=int, default=6000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if 'heldout' in str(args.conditions).casefold():
        raise ValueError('P2.3 must not read heldout conditions')
    root = Path(__file__).resolve().parents[1]
    conditions = (root / args.conditions).resolve()
    manifest = read_json(conditions)
    if manifest.get('purpose') != 'development':
        raise ValueError('development conditions required')
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
        print(json.dumps({k: v for k, v in job.items() if k != 'command'}), flush=True)
        if completed.returncode or not complete():
            raise RuntimeError(f'{name} failed or artifact verification failed')

    for seed in SEEDS:
        train_dir = output / f'train_mixed_s{seed}'
        run(f'train_mixed_s{seed}', [
            'train', '--output', train_dir, '--updates', 80, '--episodes-per-update', 4,
            '--seed', seed, '--mode', 'mixed', '--sampler-version', 2,
            '--interaction-budget', 2 * args.branch_budget,
            '--branch-interaction-budget', args.branch_budget,
            '--mixed-warmup-fraction', 0, '--role-action-mode', 'lane_locked',
            '--no-robust', '--device', args.device],
            lambda d=train_dir, b=args.branch_budget, s=seed: train_complete(d, b, s))
    for seed in SEEDS:
        eval_dir = output / f'eval_mixed_s{seed}'
        run(f'eval_mixed_s{seed}', [
            'evaluate', '--policy', output / f'train_mixed_s{seed}' / 'policy.pt',
            '--output', eval_dir, '--conditions', conditions, '--device', args.device],
            lambda d=eval_dir: eval_complete(d, manifest['condition_set_version']))
    (output / 'completed.json').write_text(json.dumps({
        'status': 'completed', 'seeds': SEEDS,
        'branch_interaction_budget': args.branch_budget,
        'total_interaction_budget': 2 * args.branch_budget,
        'condition_set_version': manifest['condition_set_version'],
        'heldout_read': False,
        'comparison_source': 'runs/20260911_p22_architecture',
        'results_kind': 'development_equal_branch_budget_screen_not_confirmatory_evidence',
    }, indent=2), encoding='utf-8')
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
