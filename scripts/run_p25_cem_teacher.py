"""Run P2.5 training-only CEM teacher, BC, and BC-to-MAPPO evaluation."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


SEEDS = (7, 17, 27, 37, 47)
TRAINING_SEED = 51000
SEARCH_SEED = 107
TRAINING_COUNT = 120
SEARCH_BUDGET = 1000
BRANCH_BUDGET = 6000


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def search_complete(directory, version, branch):
    directory = Path(directory)
    summary_path, attempts_path = (directory / 'conditions_search.json',
                                   directory / 'attempts.jsonl')
    if not summary_path.exists() or not attempts_path.exists():
        return False
    summary = read_json(summary_path)
    dimension = 3 if branch == 'single' else 6
    return (
        summary.get('kind') == 'parameters' and summary.get('seed') == SEARCH_SEED
        and summary.get('branches') == [branch]
        and summary.get('role_action_mode') == 'lane_locked'
        and summary.get('condition_set_version') == version
        and summary.get('interaction_budget_per_condition') == SEARCH_BUDGET
        and summary.get('budget_per_condition') is None
        and summary.get('search_dimensions') == [dimension]
        and summary.get('n_conditions') == TRAINING_COUNT
        and summary.get('total_interaction_steps') == TRAINING_COUNT * SEARCH_BUDGET
        and all(row.get('interaction_steps') == SEARCH_BUDGET
                for row in summary.get('per_condition', []))
        and len(attempts_path.read_text(encoding='utf-8').splitlines())
        == summary.get('total_evaluations'))


def teacher_complete(directory, training_version, dev_version):
    directory = Path(directory)
    needed = (directory / 'teacher.npz', directory / 'manifest.json',
              directory / 'report.json')
    if not all(path.exists() for path in needed):
        return False
    report = read_json(needed[2])
    return (report.get('training_condition_set_version') == training_version
            and report.get('development_condition_set_version') == dev_version
            and report.get('condition_fingerprint_overlap') == 0
            and report.get('examples', 0) > 0
            and report.get('corpus_sha256') == sha256(needed[0])
            and all(report.get('counts', {}).get(f'{branch}_{split}', 0) > 0
                    for branch in ('single', 'dual') for split in ('train', 'val')))


def bc_complete(directory, seed, corpus_hash):
    directory = Path(directory)
    needed = (directory / 'prior.pt', directory / 'pretraining.json')
    if not all(path.exists() for path in needed):
        return False
    import torch
    bundle = torch.load(needed[0], map_location='cpu', weights_only=True)
    history = read_json(needed[1])
    return (bundle.get('config', {}).get('seed') == seed
            and bundle.get('config', {}).get('epochs') == 20
            and bundle.get('config', {}).get('role_action_mode') == 'lane_locked'
            and bundle.get('extra', {}).get('corpus_sha256') == corpus_hash
            and len(history) == 20)


def train_complete(directory, seed):
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
    totals = Counter()
    for row in history:
        totals.update(row.get('branch_steps', {}))
    return (
        config.get('seed') == seed and config.get('mode') == 'mixed'
        and config.get('interaction_budget') == 2 * BRANCH_BUDGET
        and config.get('branch_interaction_budget') == BRANCH_BUDGET
        and config.get('reference_kl') == 0
        and config.get('role_action_mode') == 'lane_locked'
        and config.get('mixed_warmup_fraction') == 0
        and config.get('sampler_version') == 2 and config.get('robust') is False
        and totals == {'single': BRANCH_BUDGET, 'dual': BRANCH_BUDGET}
        and bundle.get('extra', {}).get('pretrained') is True)


def eval_complete(directory, version):
    directory = Path(directory)
    summary_path, episodes_path = directory / 'summary.json', directory / 'episodes.jsonl'
    if not summary_path.exists() or not episodes_path.exists():
        return False
    summary = read_json(summary_path)
    return (summary.get('condition_set_version') == version
            and summary.get('role_action_mode') == 'lane_locked'
            and len(episodes_path.read_text(encoding='utf-8').splitlines()) == 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dev-conditions', required=True)
    parser.add_argument('--p22', default='runs/20260911_p22_architecture')
    parser.add_argument('--p23', default='runs/20260911_p23_equal_branch')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if 'heldout' in str(args.dev_conditions).casefold():
        raise ValueError('P2.5 must not read heldout conditions')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    output = (root / args.output).resolve()
    dev_conditions = (root / args.dev_conditions).resolve()
    p22, p23 = (root / args.p22).resolve(), (root / args.p23).resolve()
    dev_manifest = read_json(dev_conditions)
    if dev_manifest.get('purpose') != 'development':
        raise ValueError('P2.5 validation requires development conditions')
    dev_version = dev_manifest['condition_set_version']
    for baseline in [p22 / 'eval_script', *[p23 / f'eval_mixed_s{seed}' for seed in SEEDS]]:
        if not eval_complete(baseline, dev_version):
            raise ValueError(f'baseline artifact missing or incompatible: {baseline}')
    output.mkdir(parents=True, exist_ok=True)
    preregistration = {
        'protocol': 'P2.5 independent CEM teacher transfer',
        'created_before_training_conditions': True,
        'training_seed': TRAINING_SEED, 'training_count_per_branch': TRAINING_COUNT,
        'training_purpose': 'training', 'search_seed': SEARCH_SEED,
        'search_kind': 'parameters', 'search_population': 8,
        'search_interaction_budget_per_condition': SEARCH_BUDGET,
        'teacher_selection': 'complete valid dangerous replayable best only',
        'teacher_val_rule': 'condition_index modulo 5 equals 0',
        'bc_epochs': 20, 'bc_batch_size': 16,
        'policy_seeds': list(SEEDS), 'online_branch_budget': BRANCH_BUDGET,
        'reference_kl': 0, 'role_action_mode': 'lane_locked',
        'dev_condition_set_version': dev_version,
        'dev_conditions_sha256': sha256(dev_conditions), 'heldout_read': False,
        'primary_gate': ('BC-to-MAPPO minus script paired 95% CI lower >0 in both '
                         'branches, point delta >0, min valid >=0.8, role invalid=0'),
        'bootstrap_rounds': 2000, 'bootstrap_seed': 2026,
    }
    prereg_path = output / 'preregistration.json'
    if prereg_path.exists() and read_json(prereg_path) != preregistration:
        raise ValueError('existing preregistration differs; use a new output directory')
    prereg_path.write_text(json.dumps(preregistration, indent=2), encoding='utf-8')

    jobs_path = output / 'jobs.json'
    jobs = read_json(jobs_path) if jobs_path.exists() else []

    def run(name, command, complete):
        if complete():
            print(f'SKIP verified {name}', flush=True)
            return
        job = {'name': name, 'command': list(map(str, command)),
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
        job['artifact_verified'] = completed.returncode == 0 and complete()
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in job.items() if key != 'command'}),
              flush=True)
        if job['status'] != 'passed' or not job['artifact_verified']:
            raise RuntimeError(f'{name} failed or artifact verification failed')

    training_conditions = output / 'training_conditions.json'
    if training_conditions.exists():
        training_manifest = read_json(training_conditions)
    else:
        from scenario_lab.sampling import export_conditions
        training_manifest = export_conditions(
            training_conditions, seed=TRAINING_SEED, count=TRAINING_COUNT,
            sampler_version=2, role='reference', purpose='training')
    training_version = f'training_seed{TRAINING_SEED}_samplerv2'
    if (training_manifest.get('condition_set_version') != training_version
            or training_manifest.get('purpose') != 'training'
            or training_manifest.get('count_per_branch') != TRAINING_COUNT):
        raise ValueError('training condition manifest differs from preregistration')

    for branch in ('single', 'dual'):
        directory = output / f'teacher_search_{branch}'
        run(f'teacher_search_{branch}', [
            sys.executable, '-m', 'scenario_lab', 'search', '--conditions', training_conditions,
            '--branch', branch, '--kind', 'parameters', '--interaction-budget', SEARCH_BUDGET,
            '--population', 8, '--role-action-mode', 'lane_locked', '--seed', SEARCH_SEED,
            '--output', directory],
            lambda d=directory, b=branch: search_complete(d, training_version, b))

    teacher_dir = output / 'teacher_corpus'
    run('build_teacher', [
        sys.executable, '-m', 'scenario_lab', 'build-teacher',
        '--conditions', training_conditions, '--dev-conditions', dev_conditions,
        '--search-single', output / 'teacher_search_single',
        '--search-dual', output / 'teacher_search_dual', '--output', teacher_dir],
        lambda: teacher_complete(teacher_dir, training_version, dev_version))
    corpus = teacher_dir / 'teacher.npz'
    corpus_hash = sha256(corpus)

    for seed in SEEDS:
        directory = output / f'bc_s{seed}'
        run(f'bc_s{seed}', [
            sys.executable, '-m', 'scenario_lab', 'pretrain-teacher',
            '--corpus', corpus, '--output', directory, '--epochs', 20,
            '--batch-size', 16, '--hidden', 64, '--seed', seed, '--device', args.device],
            lambda d=directory, s=seed: bc_complete(d, s, corpus_hash))
    for seed in SEEDS:
        directory = output / f'train_bc_mappo_s{seed}'
        run(f'train_bc_mappo_s{seed}', [
            sys.executable, '-m', 'scenario_lab', 'train', '--output', directory,
            '--updates', 80, '--episodes-per-update', 4, '--seed', seed,
            '--mode', 'mixed', '--sampler-version', 2,
            '--interaction-budget', 2 * BRANCH_BUDGET,
            '--branch-interaction-budget', BRANCH_BUDGET,
            '--mixed-warmup-fraction', 0, '--role-action-mode', 'lane_locked',
            '--reference-kl', 0, '--no-robust', '--device', args.device,
            '--pretrained', output / f'bc_s{seed}' / 'prior.pt'],
            lambda d=directory, s=seed: train_complete(d, s))
    for seed in SEEDS:
        for method, policy in (
                ('bc', output / f'bc_s{seed}' / 'prior.pt'),
                ('bc_mappo', output / f'train_bc_mappo_s{seed}' / 'policy.pt')):
            directory = output / f'eval_{method}_s{seed}'
            run(f'eval_{method}_s{seed}', [
                sys.executable, '-m', 'scenario_lab', 'evaluate', '--policy', policy,
                '--output', directory, '--conditions', dev_conditions,
                '--device', args.device],
                lambda d=directory: eval_complete(d, dev_version))

    completed = {
        'status': 'completed', 'policy_seeds': list(SEEDS),
        'training_condition_set_version': training_version,
        'training_conditions_sha256': sha256(training_conditions),
        'dev_condition_set_version': dev_version,
        'dev_conditions_sha256': sha256(dev_conditions),
        'teacher_corpus_sha256': corpus_hash,
        'teacher_examples': read_json(teacher_dir / 'report.json')['examples'],
        'online_branch_budget': BRANCH_BUDGET,
        'heldout_read': False,
        'baseline_pure_mappo': str(p23.relative_to(root)),
        'baseline_script': str((p22 / 'eval_script').relative_to(root)),
        'results_kind': 'development_teacher_transfer_screen_not_confirmatory_evidence',
    }
    (output / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
