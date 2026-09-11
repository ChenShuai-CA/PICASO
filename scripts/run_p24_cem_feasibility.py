"""Run the preregistered P2.4 lane-locked CEM feasibility matrix."""
import argparse
import concurrent.futures
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (7, 17, 27, 37, 47)
KINDS = ('parameters', 'trajectory')
BRANCHES = ('single', 'dual')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def search_complete(directory, version, branch, kind, step_budget):
    directory = Path(directory)
    summary_path = directory / 'conditions_search.json'
    attempts_path = directory / 'attempts.jsonl'
    if not summary_path.exists() or not attempts_path.exists():
        return False
    summary = read_json(summary_path)
    expected_dimension = {'parameters': {'single': 3, 'dual': 6},
                          'trajectory': {'single': 8, 'dual': 16}}[kind][branch]
    per_condition = summary.get('per_condition', [])
    return (
        summary.get('kind') == kind
        and summary.get('branches') == [branch]
        and summary.get('role_action_mode') == 'lane_locked'
        and summary.get('condition_set_version') == version
        and summary.get('interaction_budget_per_condition') == step_budget
        and summary.get('population') == 8
        and summary.get('n_conditions') == 40
        and summary.get('search_dimensions') == [expected_dimension]
        and summary.get('total_interaction_steps') == 40 * step_budget
        and len(per_condition) == 40
        and all(row.get('interaction_steps') == step_budget for row in per_condition)
        and sum(1 for line in attempts_path.read_text(encoding='utf-8').splitlines()
                if line.strip()) == summary.get('total_evaluations')
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--interaction-budget', type=int, default=2500)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if args.interaction_budget < 2 or args.workers < 1:
        raise ValueError('interaction budget must be >=2 and workers must be positive')
    if 'heldout' in str(args.conditions).casefold():
        raise ValueError('P2.4 must not read heldout conditions')

    root = Path(__file__).resolve().parents[1]
    conditions = (root / args.conditions).resolve()
    manifest = read_json(conditions)
    if manifest.get('purpose') != 'development':
        raise ValueError('development conditions required')
    if manifest.get('count_per_branch') != 40:
        raise ValueError('P2.4 preregistration requires 40 conditions per branch')
    version = manifest['condition_set_version']
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preregistration = {
        'protocol': 'P2.4 lane-locked CEM feasibility',
        'created_before_formal_jobs': True,
        'condition_path': str(conditions.relative_to(root)),
        'condition_sha256': hashlib.sha256(conditions.read_bytes()).hexdigest(),
        'condition_set_version': version,
        'purpose': manifest['purpose'],
        'kinds': list(KINDS), 'branches': list(BRANCHES), 'seeds': list(SEEDS),
        'population': 8, 'interaction_budget_per_condition': args.interaction_budget,
        'role_action_mode': 'lane_locked', 'heldout_read': False,
        'success_definition': 'at least one complete valid dangerous episode within budget',
        'feasibility_gate': {
            'dangerous_coverage_delta_min': .10,
            'paired_bootstrap_ci_lower_strictly_above': 0.0,
            'role_invalid_total': 0,
            'bootstrap_rounds': 2000, 'bootstrap_seed': 2026,
        },
        'interpretation_boundary': (
            'best-of-search diagnostic oracle; not equal-cost method superiority'),
    }
    prereg_path = output / 'preregistration.json'
    if prereg_path.exists() and read_json(prereg_path) != preregistration:
        raise ValueError('existing preregistration differs; use a new output directory')
    prereg_path.write_text(json.dumps(preregistration, indent=2), encoding='utf-8')

    tasks = []
    for kind in KINDS:
        for branch in BRANCHES:
            for seed in SEEDS:
                name = f'{kind}_{branch}_s{seed}'
                directory = output / name
                if search_complete(directory, version, branch, kind,
                                   args.interaction_budget):
                    print(f'SKIP verified {name}', flush=True)
                    continue
                command = [
                    sys.executable, '-m', 'scenario_lab', 'search',
                    '--conditions', str(conditions), '--branch', branch,
                    '--kind', kind, '--interaction-budget', str(args.interaction_budget),
                    '--population', '8', '--role-action-mode', 'lane_locked',
                    '--seed', str(seed), '--output', str(directory),
                ]
                tasks.append((name, directory, command, branch, kind, seed))

    jobs = []

    def execute(task):
        name, directory, command, branch, kind, seed = task
        started_at = time.time()
        started = time.perf_counter()
        print(f'START {name}', flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT)
        return {
            'name': name, 'branch': branch, 'kind': kind, 'seed': seed,
            'command': command, 'returncode': completed.returncode,
            'status': 'passed' if completed.returncode == 0 else 'failed',
            'started_at': started_at, 'finished_at': time.time(),
            'elapsed_s': time.perf_counter() - started,
            'artifact_verified': completed.returncode == 0 and search_complete(
                directory, version, branch, kind, args.interaction_budget),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            job = future.result()
            jobs.append(job)
            (output / 'jobs.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in job.items() if k != 'command'}), flush=True)
            if job['status'] != 'passed' or not job['artifact_verified']:
                raise RuntimeError(f"{job['name']} failed or artifact verification failed")

    missing = []
    for kind in KINDS:
        for branch in BRANCHES:
            for seed in SEEDS:
                directory = output / f'{kind}_{branch}_s{seed}'
                if not search_complete(directory, version, branch, kind,
                                       args.interaction_budget):
                    missing.append(directory.name)
    if missing:
        raise RuntimeError(f'incomplete P2.4 jobs: {missing}')
    # The legacy CLI keeps an evaluation-count default for backward compatibility,
    # but P2.4 activates only the exact interaction-step budget. Normalize completed
    # metadata so the unused default cannot be mistaken for a second budget.
    for kind in KINDS:
        for branch in BRANCHES:
            for seed in SEEDS:
                path = output / f'{kind}_{branch}_s{seed}' / 'conditions_search.json'
                summary = read_json(path)
                summary['budget_per_condition'] = None
                path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    completed = {
        'status': 'completed', 'condition_set_version': version,
        'condition_sha256': preregistration['condition_sha256'],
        'kinds': list(KINDS), 'branches': list(BRANCHES), 'seeds': list(SEEDS),
        'population': 8, 'interaction_budget_per_condition': args.interaction_budget,
        'total_formal_search_steps': (len(KINDS) * len(BRANCHES) * len(SEEDS)
                                      * 40 * args.interaction_budget),
        'role_action_mode': 'lane_locked', 'heldout_read': False,
        'results_kind': 'development_feasibility_diagnostic_not_equal_cost_superiority',
    }
    (output / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
