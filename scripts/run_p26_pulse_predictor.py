"""Run preregistered P2.6 incremental pulse-parameter mechanism screening."""
import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time


SEEDS = (7, 17, 27, 37, 47)
SCREEN_SEED = 61000
SEARCH_SEED = 127
COUNT = 120
SEARCH_BUDGET = 1000


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
    expected_dimension = 3 if branch == 'single' else 6
    return (
        summary.get('kind') == 'parameters' and summary.get('seed') == SEARCH_SEED
        and summary.get('branches') == [branch]
        and summary.get('condition_set_version') == version
        and summary.get('role_action_mode') == 'lane_locked'
        and summary.get('population') == 8
        and summary.get('interaction_budget_per_condition') == SEARCH_BUDGET
        and summary.get('search_dimensions') == [expected_dimension]
        and summary.get('n_conditions') == COUNT
        and summary.get('total_interaction_steps') == COUNT * SEARCH_BUDGET
        and all(row.get('interaction_steps') == SEARCH_BUDGET
                for row in summary.get('per_condition', []))
        and len(attempts_path.read_text(encoding='utf-8').splitlines())
        == summary.get('total_evaluations'))


def corpus_complete(directory, version, split_kind):
    directory = Path(directory)
    needed = directory / 'pulse.npz', directory / 'manifest.json', directory / 'report.json'
    if not all(path.exists() for path in needed):
        return False
    report = read_json(needed[2])
    return (report.get('schema_version') == 'incremental-pulse-v1'
            and report.get('condition_set_version') == version
            and report.get('split_kind') == split_kind
            and report.get('history_steps') == 5
            and report.get('heldout_read') is False
            and report.get('examples', 0) > 0
            and report.get('corpus_sha256') == sha256(needed[0]))


def model_complete(directory, seed, corpus_hash):
    directory = Path(directory)
    needed = directory / 'pulse.pt', directory / 'training.json'
    if not all(path.exists() for path in needed):
        return False
    import torch
    bundle = torch.load(needed[0], map_location='cpu', weights_only=True)
    history = read_json(needed[1])
    return (bundle.get('schema_version') == 'incremental-pulse-v1'
            and bundle.get('config', {}).get('seed') == seed
            and bundle.get('config', {}).get('epochs') == 50
            and bundle.get('config', {}).get('history_steps') == 5
            and 1 <= bundle.get('config', {}).get('selected_epoch', 0) <= 50
            and bundle.get('extra', {}).get('corpus_sha256') == corpus_hash
            and bundle.get('extra', {}).get('heldout_read') is False
            and len(history) == 50)


def screen_eval_complete(directory, version, count):
    directory = Path(directory)
    summary_path, rows_path = directory / 'summary.json', directory / 'episodes.jsonl'
    if not summary_path.exists() or not rows_path.exists():
        return False
    summary = read_json(summary_path)
    return (summary.get('kind') == 'incremental_pulse_screen'
            and summary.get('condition_set_version') == version
            and summary.get('episodes') == count
            and summary.get('heldout_read') is False
            and len(rows_path.read_text(encoding='utf-8').splitlines()) == count)


def dev_eval_complete(directory, version):
    directory = Path(directory)
    summary_path, rows_path = directory / 'summary.json', directory / 'episodes.jsonl'
    if not summary_path.exists() or not rows_path.exists():
        return False
    summary = read_json(summary_path)
    return (summary.get('condition_set_version') == version
            and summary.get('role_action_mode') == 'lane_locked'
            and len(rows_path.read_text(encoding='utf-8').splitlines()) == 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--p25', default='runs/20260911_p25_cem_teacher')
    parser.add_argument('--dev-conditions', default='runs/20260911_p2_conditions/dev_seed31000.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('run inside WSL2 Ubuntu')
    if 'heldout' in str(args.dev_conditions).casefold():
        raise ValueError('P2.6 must not use heldout conditions')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    output = (project / args.output).resolve()
    p25 = (project / args.p25).resolve()
    dev_conditions = (project / args.dev_conditions).resolve()
    output.mkdir(parents=True, exist_ok=True)

    p25_completed = read_json(p25 / 'completed.json')
    if p25_completed.get('heldout_read') is not False:
        raise ValueError('P2.5 heldout-read certificate missing')
    p25_conditions = p25 / 'training_conditions.json'
    p25_version = p25_completed['training_condition_set_version']
    preregistration = {
        'protocol': 'P2.6 incremental pulse parameter prediction',
        'created_before_screen_conditions': True,
        'training_source_version': p25_version,
        'training_source_sha256': sha256(p25_conditions),
        'screen_seed': SCREEN_SEED, 'screen_count_per_branch': COUNT,
        'screen_purpose': 'training', 'search_seed': SEARCH_SEED,
        'search_kind': 'parameters', 'search_population': 8,
        'search_interaction_budget_per_condition': SEARCH_BUDGET,
        'teacher_selection': ('complete valid dangerous CEM, valid safe script, '
                              'all active onsets >=0.5 seconds'),
        'history_steps': 5, 'policy_seeds': list(SEEDS),
        'epochs': 50, 'batch_size': 16, 'learning_rate': 3e-4,
        'checkpoint_selection': 'minimum branch-balanced P2.5 teacher-val parameter MSE',
        'mechanism_thresholds': {'single': .50, 'dual': .25},
        'minimum_screen_conditions_per_branch': 15,
        'bootstrap_rounds': 2000, 'bootstrap_seed': 2026,
        'conditional_dev_path': str(dev_conditions.relative_to(project)),
        'dev_read_only_if_mechanism_gate_passes': True,
        'heldout_read': False,
    }
    prereg_path = output / 'preregistration.json'
    if prereg_path.exists() and read_json(prereg_path) != preregistration:
        raise ValueError('existing preregistration differs; use a fresh output directory')
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
            completed = subprocess.run(job['command'], cwd=project, stdout=log,
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

    screen_conditions = output / 'screen_conditions.json'
    if screen_conditions.exists():
        screen_manifest = read_json(screen_conditions)
    else:
        from scenario_lab.sampling import export_conditions
        screen_manifest = export_conditions(
            screen_conditions, seed=SCREEN_SEED, count=COUNT,
            sampler_version=2, role='reference', purpose='training')
    screen_version = f'training_seed{SCREEN_SEED}_samplerv2'
    if (screen_manifest.get('condition_set_version') != screen_version
            or screen_manifest.get('count_per_branch') != COUNT
            or screen_manifest.get('purpose') != 'training'):
        raise ValueError('P2.6 screen conditions differ from preregistration')

    for branch in ('single', 'dual'):
        directory = output / f'screen_search_{branch}'
        run(f'screen_search_{branch}', [
            sys.executable, '-m', 'scenario_lab', 'search',
            '--conditions', screen_conditions, '--branch', branch,
            '--kind', 'parameters', '--interaction-budget', SEARCH_BUDGET,
            '--population', 8, '--role-action-mode', 'lane_locked',
            '--seed', SEARCH_SEED, '--output', directory],
            lambda d=directory, b=branch: search_complete(d, screen_version, b))

    train_corpus = output / 'train_corpus'
    run('build_train_corpus', [
        sys.executable, '-m', 'scenario_lab', 'build-pulse-corpus',
        '--conditions', p25_conditions,
        '--search-single', p25 / 'teacher_search_single',
        '--search-dual', p25 / 'teacher_search_dual',
        '--output', train_corpus, '--split-kind', 'train_val',
        '--audit-conditions', screen_conditions],
        lambda: corpus_complete(train_corpus, p25_version, 'train_val'))
    screen_corpus = output / 'screen_corpus'
    run('build_screen_corpus', [
        sys.executable, '-m', 'scenario_lab', 'build-pulse-corpus',
        '--conditions', screen_conditions,
        '--search-single', output / 'screen_search_single',
        '--search-dual', output / 'screen_search_dual',
        '--output', screen_corpus, '--split-kind', 'screen',
        '--audit-conditions', p25_conditions],
        lambda: corpus_complete(screen_corpus, screen_version, 'screen'))

    train_report = read_json(train_corpus / 'report.json')
    screen_report = read_json(screen_corpus / 'report.json')
    train_hash = train_report['corpus_sha256']
    for seed in SEEDS:
        directory = output / f'pulse_s{seed}'
        run(f'pulse_s{seed}', [
            sys.executable, '-m', 'scenario_lab', 'train-pulse',
            '--corpus', train_corpus / 'pulse.npz', '--output', directory,
            '--epochs', 50, '--batch-size', 16, '--hidden', 64,
            '--seed', seed, '--device', args.device],
            lambda d=directory, s=seed: model_complete(d, s, train_hash))
    for seed in SEEDS:
        directory = output / f'eval_screen_s{seed}'
        run(f'eval_screen_s{seed}', [
            sys.executable, '-m', 'scenario_lab', 'evaluate-pulse-corpus',
            '--policy', output / f'pulse_s{seed}' / 'pulse.pt',
            '--corpus', screen_corpus, '--output', directory,
            '--device', args.device],
            lambda d=directory: screen_eval_complete(
                d, screen_version, screen_report['examples']))

    summary_command = [sys.executable, project / 'scripts' / 'summarize_p26_pulse_predictor.py',
                       output, '--bootstrap', 2000, '--seed', 2026]
    subprocess.run(list(map(str, summary_command)), cwd=project, check=True)
    gate = read_json(output / 'mechanism_gate.json')
    dev_evaluated = False
    if gate['eligible_for_dev']:
        from scenario_lab.sampling import load_conditions
        from scenario_lab.teacher import spec_fingerprint
        sets = {}
        for path in (p25_conditions, screen_conditions, dev_conditions):
            specs, manifest = load_conditions(path)
            sets[manifest['condition_set_version']] = {spec_fingerprint(spec) for spec in specs}
        overlaps = {}
        names = list(sets)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlaps[f'{left}__{right}'] = len(sets[left] & sets[right])
        if any(overlaps.values()):
            raise ValueError(f'P2.6 condition fingerprint overlap: {overlaps}')
        (output / 'dev_fingerprint_audit.json').write_text(
            json.dumps({'overlaps': overlaps}, indent=2), encoding='utf-8')
        dev_manifest = read_json(dev_conditions)
        dev_version = dev_manifest['condition_set_version']
        for seed in SEEDS:
            directory = output / f'eval_dev_s{seed}'
            run(f'eval_dev_s{seed}', [
                sys.executable, '-m', 'scenario_lab', 'evaluate-pulse',
                '--policy', output / f'pulse_s{seed}' / 'pulse.pt',
                '--conditions', dev_conditions, '--output', directory,
                '--device', args.device],
                lambda d=directory: dev_eval_complete(d, dev_version))
        subprocess.run(list(map(str, summary_command)), cwd=project, check=True)
        dev_evaluated = True

    completed = {
        'status': 'completed', 'policy_seeds': list(SEEDS),
        'training_condition_set_version': p25_version,
        'screen_condition_set_version': screen_version,
        'screen_conditions_sha256': sha256(screen_conditions),
        'train_teacher_examples': train_report['examples'],
        'screen_teacher_examples': screen_report['examples'],
        'mechanism_gate_passed': gate['eligible_for_dev'],
        'dev_evaluated': dev_evaluated, 'heldout_read': False,
        'results_kind': 'training_domain_mechanism_screen_then_conditional_development',
    }
    (output / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == '__main__':
    main()
