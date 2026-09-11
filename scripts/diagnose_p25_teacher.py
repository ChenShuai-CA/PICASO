"""Post-hoc training-domain diagnostics for the completed P2.5 teacher study.

This script never reads development or heldout conditions.  It measures whether
the behavior-cloned actors can reproduce their replay-verified CEM teachers in
closed loop, and how much of the selected teacher set was already dangerous
under the zero-action script.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch


SEEDS = (7, 17, 27, 37, 47)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rate(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    root = args.root.resolve()
    completed = read_json(root / 'completed.json')
    prereg = read_json(root / 'preregistration.json')
    if completed.get('heldout_read') is not False or prereg.get('heldout_read') is not False:
        raise ValueError('P2.5 heldout-read certificate missing')

    from scenario_lab.evaluate import ScriptPolicy, run_episode
    from scenario_lab.policy import load_bundle
    from scenario_lab.runtime import resolve_device
    from scenario_lab.sampling import load_conditions

    device = resolve_device(args.device)
    torch.set_num_threads(1)
    corpus_dir = root / 'teacher_corpus'
    provenance = read_json(corpus_dir / 'manifest.json')
    specs, training_manifest = load_conditions(root / 'training_conditions.json')
    if training_manifest.get('purpose') != 'training':
        raise ValueError('diagnostic input is not a training condition set')
    if training_manifest['condition_set_version'] != completed['training_condition_set_version']:
        raise ValueError('training condition set differs from completed P2.5 run')
    by_key = {(spec.branch, spec.scenario_id): spec for spec in specs}

    runners = {}
    for seed in SEEDS:
        runner, bundle = load_bundle(root / f'bc_s{seed}' / 'prior.pt', device=device)
        if bundle.get('config', {}).get('seed') != seed:
            raise ValueError(f'BC bundle seed mismatch for {seed}')
        runners[seed] = runner

    rows = []
    for index, source in enumerate(provenance):
        key = (source['branch'], source['scenario_id'])
        spec = deepcopy(by_key[key])
        spec.role_action_mode = 'lane_locked'
        replay_seed = int(source['replay_seed'])
        scripted = run_episode(ScriptPolicy(), spec, replay_seed)
        base = {
            'teacher_index': index, 'branch': source['branch'],
            'scenario_id': source['scenario_id'], 'split': source['split'],
            'replay_seed': replay_seed, 'teacher_dangerous': True,
            'script_valid': scripted['valid'], 'script_dangerous': scripted['dangerous'],
            'script_risk': scripted['risk'],
        }
        for seed, runner in runners.items():
            result = run_episode(runner, spec, replay_seed)
            rows.append({
                **base, 'policy_seed': seed, 'valid': result['valid'],
                'dangerous': result['dangerous'], 'risk': result['risk'],
                'mean_abs_action': result['mean_abs_action'],
                'invalid_reasons': result['invalid_reasons'],
            })
        if (index + 1) % 25 == 0 or index + 1 == len(provenance):
            print(json.dumps({'teacher_conditions_completed': index + 1,
                              'teacher_conditions_total': len(provenance)}), flush=True)

    diagnostics = root / 'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / 'closed_loop.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    with np.load(corpus_dir / 'teacher.npz', allow_pickle=False) as archive:
        corpus = {key: archive[key] for key in archive.files}
    teacher_forced = []
    for seed, runner in runners.items():
        actor = runner.actor
        for branch in ('single', 'dual'):
            for split in ('train', 'val'):
                indices = np.flatnonzero((corpus['branch'] == branch)
                                         & (corpus['split'] == split))
                tokens = torch.as_tensor(corpus['tokens'][indices], device=device)
                token_mask = torch.as_tensor(corpus['token_mask'][indices], device=device)
                actor_mask = torch.as_tensor(corpus['actor_mask'][indices], device=device)
                time_mask = torch.as_tensor(corpus['time_mask'][indices], device=device)
                target = torch.as_tensor(corpus['target'][indices], device=device)
                with torch.no_grad():
                    prediction = actor(tokens, token_mask, actor_mask)[0].mean.tanh()
                mask = actor_mask * time_mask[..., None]
                error = (prediction - target).square()
                denom = mask.sum().clamp_min(1.)
                active = (target[..., 0].abs() > .05) & mask.bool()
                sign_match = ((prediction[..., 0] * target[..., 0]) > 0) & active
                teacher_forced.append({
                    'policy_seed': seed, 'branch': branch, 'split': split,
                    'episodes': int(len(indices)),
                    'mse_both_actions': float((error.sum(-1) * mask).sum() / denom),
                    'longitudinal_mse': float((error[..., 0] * mask).sum() / denom),
                    'target_mean_abs_longitudinal': float(
                        (target[..., 0].abs() * mask).sum() / denom),
                    'prediction_mean_abs_longitudinal': float(
                        (prediction[..., 0].abs() * mask).sum() / denom),
                    'target_active_rate': float(active.sum() / denom),
                    'active_target_sign_accuracy': (
                        float(sign_match.sum() / active.sum()) if active.any() else None),
                })

    closed_loop = []
    for branch in ('single', 'dual'):
        for split in ('train', 'val'):
            for seed in SEEDS:
                subset = [row for row in rows if row['branch'] == branch
                          and row['split'] == split and row['policy_seed'] == seed]
                incremental = [row for row in subset if not row['script_dangerous']]
                closed_loop.append({
                    'policy_seed': seed, 'branch': branch, 'split': split,
                    'conditions': len(subset),
                    'script_dangerous_rate': rate(subset, 'script_dangerous'),
                    'bc_dangerous_rate': rate(subset, 'dangerous'),
                    'bc_valid_rate': rate(subset, 'valid'),
                    'bc_mean_risk': rate(subset, 'risk'),
                    'incremental_teacher_conditions': len(incremental),
                    'bc_dangerous_rate_on_script_safe': rate(incremental, 'dangerous'),
                })

    unique = [row for row in rows if row['policy_seed'] == SEEDS[0]]
    selection = []
    for branch in ('single', 'dual'):
        for split in ('train', 'val'):
            subset = [row for row in unique if row['branch'] == branch and row['split'] == split]
            selection.append({
                'branch': branch, 'split': split, 'teacher_conditions': len(subset),
                'script_dangerous': sum(row['script_dangerous'] for row in subset),
                'script_safe_incremental': sum(not row['script_dangerous'] for row in subset),
                'script_dangerous_rate': rate(subset, 'script_dangerous'),
            })

    summary = {
        'kind': 'post_hoc_training_domain_diagnostic',
        'formal_gate_impact': 'none', 'heldout_read': False,
        'training_condition_set_version': training_manifest['condition_set_version'],
        'teacher_examples': len(provenance), 'selection': selection,
        'teacher_forced': teacher_forced, 'closed_loop': closed_loop,
    }
    (diagnostics / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# P2.5 post-hoc teacher diagnostic', '',
        'Training-only diagnostic; it does not alter the preregistered formal gate. '
        'No heldout data were read.', '',
        '## Teacher selection', '',
        '| branch | split | selected | already dangerous under script | script-safe incremental |',
        '|---|---|---:|---:|---:|',
    ]
    for row in selection:
        lines.append(f"| {row['branch']} | {row['split']} | {row['teacher_conditions']} | "
                     f"{row['script_dangerous']} ({row['script_dangerous_rate']:.3f}) | "
                     f"{row['script_safe_incremental']} |")
    lines += ['', '## Closed-loop BC reproduction', '',
              '| branch | split | seed | BC dangerous | script dangerous | '
              'BC dangerous on script-safe teacher conditions | BC valid |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in closed_loop:
        inc = row['bc_dangerous_rate_on_script_safe']
        lines.append(f"| {row['branch']} | {row['split']} | {row['policy_seed']} | "
                     f"{row['bc_dangerous_rate']:.3f} | {row['script_dangerous_rate']:.3f} | "
                     f"{inc:.3f} | {row['bc_valid_rate']:.3f} |")
    lines += ['', '## Teacher-forced action fit', '',
              '| branch | split | seed | longitudinal MSE | target active rate | '
              'sign accuracy on active target |',
              '|---|---|---:|---:|---:|---:|']
    for row in teacher_forced:
        lines.append(f"| {row['branch']} | {row['split']} | {row['policy_seed']} | "
                     f"{row['longitudinal_mse']:.5f} | {row['target_active_rate']:.3f} | "
                     f"{row['active_target_sign_accuracy']:.3f} |")
    (diagnostics / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(diagnostics), 'heldout_read': False,
                      'teacher_examples': len(provenance)}, indent=2))


if __name__ == '__main__':
    main()
