"""CLI: python -m scenario_lab --help."""
import argparse
import json
from pathlib import Path
from .runtime import resolve_device, record_runtime


def main():
    p = argparse.ArgumentParser(description='Unified single/dual actor research tools')
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--output', required=True)
    train.add_argument('--updates', type=int, default=12)
    train.add_argument('--episodes-per-update', type=int, default=4)
    train.add_argument('--seed', type=int, default=7)
    train.add_argument('--mode', choices=['single', 'dual', 'mixed'], default='mixed')
    train.add_argument('--mixed-warmup-fraction', type=float, default=.2,
                       help='fraction of mixed training reserved for single-only warmup')
    train.add_argument('--algorithm', choices=['ppo', 'ippo', 'mappo'], default='mappo')
    train.add_argument('--reference-kl', type=float, default=.02,
                       help='KL weight to the initial actor when --pretrained is used')
    train.add_argument('--device', default='auto')
    train.add_argument('--pretrained')
    train.add_argument('--no-robust', action='store_true')
    train.add_argument('--no-role-constraints', action='store_true')
    train.add_argument('--role-action-mode', choices=['none', 'lane_locked'], default='none',
                       help='project target actions onto their assigned role axes')
    train.add_argument('--interaction-budget', type=int,
                       help='exact decision-step budget; final episode is critic-bootstrapped')
    train.add_argument('--branch-interaction-budget', type=int,
                       help='exact budget for each branch in non-robust mixed training')
    train.add_argument('--sampler-version', type=int, default=1, choices=[1, 2],
                       help='scenario sampler for on-policy draws (2 pairs physics v2 '
                            'with constructive reference feasibility)')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--policy')
    ev.add_argument('--device', default='auto')
    ev.add_argument('--output', required=True)
    ev.add_argument('--count', type=int, default=20)
    ev.add_argument('--seed', type=int, default=1000)
    ev.add_argument('--conditions')
    ev.add_argument('--perturbations', type=int, default=1)
    ev.add_argument('--controller', choices=['stopping', 'ttc'], default='stopping')
    ev.add_argument('--role-action-mode', choices=['none', 'lane_locked'],
                    help='override execution projection; defaults to the policy bundle setting')
    ev.add_argument('--one-learning-target', action='store_true')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--policy', required=True)
    bench.add_argument('--device', default='auto')
    bench.add_argument('--output', required=True)
    bench.add_argument('--steps', type=int, default=10000)
    search = sub.add_parser('search')
    search.add_argument('--branch', choices=['single', 'dual'], default='dual',
                        help='single-spec mode, or the branch filter in --conditions mode')
    search.add_argument('--conditions',
                        help='frozen condition manifest: per-condition CEM over matching branches')
    search.add_argument('--kind', choices=['parameters', 'trajectory'], default='parameters')
    search.add_argument('--budget', type=int, default=40)
    search.add_argument('--interaction-budget', type=int,
                        help='exact decision-step budget per condition; final partial episode '
                             'is logged but cannot be selected as a solution')
    search.add_argument('--population', type=int, default=8)
    search.add_argument('--role-action-mode', choices=['none', 'lane_locked'], default='none')
    search.add_argument('--seed', type=int, default=7)
    search.add_argument('--output', required=True)
    rep = sub.add_parser('replay')
    rep.add_argument('trace')
    audit = sub.add_parser('audit')
    audit.add_argument('--root', default='Data')
    audit.add_argument('--output', required=True)
    audit.add_argument('--limit', type=int, default=8)
    prepare = sub.add_parser('prepare-public')
    prepare.add_argument('--root', default='Data')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--max-files', type=int, default=2)
    prepare.add_argument('--records-per-file', type=int, default=8)
    prepare.add_argument('--max-examples', type=int, default=10000)
    prepare.add_argument('--include-pedestrians', action='store_true',
                         help='also select INTERACTION pedestrian_tracks exports')
    prepare.add_argument('--per-location', type=int, default=None,
                         help='cap INTERACTION files per location (vehicle+pedestrian combined)')
    prepare.add_argument('--max-files-interaction', type=int, default=None,
                         help='separate INTERACTION file budget when it differs from --max-files')
    prepare.add_argument('--location-splits-v5', action='store_true',
                         help='apply the pre-registered v5 INTERACTION holdout map')
    prior = sub.add_parser('pretrain')
    prior.add_argument('--corpus', required=True)
    prior.add_argument('--output', required=True)
    prior.add_argument('--epochs', type=int, default=5)
    prior.add_argument('--device', default='auto')
    prior.add_argument('--seed', type=int, default=7)
    prior.add_argument('--no-neighbors', action='store_true',
                       help='self-only ablation: mask neighbor slots, same corpus')
    teacher = sub.add_parser('build-teacher')
    teacher.add_argument('--conditions', required=True)
    teacher.add_argument('--dev-conditions', required=True)
    teacher.add_argument('--search-single', required=True)
    teacher.add_argument('--search-dual', required=True)
    teacher.add_argument('--output', required=True)
    teacher_train = sub.add_parser('pretrain-teacher')
    teacher_train.add_argument('--corpus', required=True)
    teacher_train.add_argument('--output', required=True)
    teacher_train.add_argument('--epochs', type=int, default=20)
    teacher_train.add_argument('--batch-size', type=int, default=16)
    teacher_train.add_argument('--hidden', type=int, default=64)
    teacher_train.add_argument('--seed', type=int, default=7)
    teacher_train.add_argument('--device', default='auto')
    pulse_corpus = sub.add_parser('build-pulse-corpus')
    pulse_corpus.add_argument('--conditions', required=True)
    pulse_corpus.add_argument('--search-single', required=True)
    pulse_corpus.add_argument('--search-dual', required=True)
    pulse_corpus.add_argument('--output', required=True)
    pulse_corpus.add_argument('--split-kind', choices=['train_val', 'screen'], required=True)
    pulse_corpus.add_argument('--audit-conditions', nargs='*', default=[])
    pulse_train = sub.add_parser('train-pulse')
    pulse_train.add_argument('--corpus', required=True)
    pulse_train.add_argument('--output', required=True)
    pulse_train.add_argument('--epochs', type=int, default=50)
    pulse_train.add_argument('--batch-size', type=int, default=16)
    pulse_train.add_argument('--hidden', type=int, default=64)
    pulse_train.add_argument('--seed', type=int, default=7)
    pulse_train.add_argument('--device', default='auto')
    pulse_eval = sub.add_parser('evaluate-pulse-corpus')
    pulse_eval.add_argument('--policy', required=True)
    pulse_eval.add_argument('--corpus', required=True)
    pulse_eval.add_argument('--output', required=True)
    pulse_eval.add_argument('--device', default='auto')
    pulse_dev = sub.add_parser('evaluate-pulse')
    pulse_dev.add_argument('--policy', required=True)
    pulse_dev.add_argument('--conditions', required=True)
    pulse_dev.add_argument('--output', required=True)
    pulse_dev.add_argument('--device', default='auto')
    a = p.parse_args()
    if hasattr(a, 'device'):
        a.device = resolve_device(a.device)
    if a.command == 'train':
        from .train import train, TrainConfig
        result = train(a.output, TrainConfig(seed=a.seed, updates=a.updates,
                       episodes_per_update=a.episodes_per_update, mode=a.mode,
                       mixed_warmup_fraction=a.mixed_warmup_fraction,
                       algorithm=a.algorithm, reference_kl=a.reference_kl,
                       device=a.device, robust=not a.no_robust,
                       role_constraints=not a.no_role_constraints,
                       role_action_mode=a.role_action_mode,
                       interaction_budget=a.interaction_budget,
                       branch_interaction_budget=a.branch_interaction_budget,
                       sampler_version=a.sampler_version), a.pretrained)
    elif a.command in ('evaluate', 'benchmark'):
        from .evaluate import evaluate, benchmark, ScriptPolicy, OneLearningPolicy
        from .policy import load_bundle
        import torch
        torch.set_num_threads(1)
        bundle = None
        if a.policy:
            runner, bundle = load_bundle(a.policy, device=a.device)
        else:
            runner = ScriptPolicy()
        runtime_dir = Path(a.output).parent if a.command == 'benchmark' else Path(a.output)
        record_runtime(runtime_dir, a.device if a.policy else 'cpu')
        if a.command == 'evaluate':
            if a.one_learning_target:
                runner = OneLearningPolicy(runner)
            conditions = version = None
            if a.conditions:
                from .sampling import load_conditions
                conditions, manifest = load_conditions(a.conditions)
                version = manifest['condition_set_version']
            role_action_mode = (a.role_action_mode
                                or ((bundle or {}).get('config') or {}).get('role_action_mode', 'none'))
            result = evaluate(runner, a.output, a.count, a.seed, perturbations=a.perturbations,
                              controller=a.controller, conditions=conditions,
                              condition_set_version=version, role_action_mode=role_action_mode)
        else:
            result = benchmark(runner, a.output, a.steps)
    elif a.command == 'search':
        from .evaluate import search, search_conditions
        from .schema import ScenarioSpec
        if a.conditions:
            from .sampling import load_conditions
            conditions, manifest = load_conditions(a.conditions)
            result = search_conditions(conditions, a.output, a.kind, a.budget, a.seed,
                                       population=a.population, branches=(a.branch,),
                                       interaction_budget=a.interaction_budget,
                                       role_action_mode=a.role_action_mode,
                                       condition_set_version=manifest['condition_set_version'])
        else:
            result = search(ScenarioSpec(branch=a.branch), a.output, a.kind, a.budget, a.seed,
                            population=a.population,
                            interaction_budget=a.interaction_budget,
                            role_action_mode=a.role_action_mode)
    elif a.command == 'replay':
        from .evaluate import replay
        result = replay(a.trace)
    elif a.command == 'audit':
        from .data import audit_dataset
        audit_result = audit_dataset(Path(a.root), Path(a.output), a.limit)
        result = dict(outputs=audit_result['outputs'], inventory_rows=audit_result['inventory']['n_rows'],
                      sources={k: {key: value for key, value in audit_result[k].items() if key != 'samples'} for k in ('abd', 'interaction')})
    elif a.command == 'prepare-public':
        from .pretrain import prepare_public, LOCATION_SPLITS_V5
        result = prepare_public(a.root, a.output, a.max_files, a.records_per_file, a.max_examples,
                                include_pedestrians=a.include_pedestrians, per_location=a.per_location,
                                location_splits=LOCATION_SPLITS_V5 if a.location_splits_v5 else None,
                                max_files_interaction=a.max_files_interaction)
    elif a.command == 'pretrain':
        from .pretrain import pretrain
        result = pretrain(a.corpus, a.output, a.epochs, seed=a.seed, device=a.device,
                          use_neighbors=not a.no_neighbors)
    elif a.command == 'build-teacher':
        from .teacher import build_teacher_corpus
        result = build_teacher_corpus(
            a.conditions, a.dev_conditions, (a.search_single, a.search_dual), a.output)
    elif a.command == 'pretrain-teacher':
        from .teacher import pretrain_teacher
        result = pretrain_teacher(a.corpus, a.output, a.epochs, a.hidden, a.seed,
                                  a.device, a.batch_size)
    elif a.command == 'build-pulse-corpus':
        from .pulse import build_incremental_pulse_corpus
        result = build_incremental_pulse_corpus(
            a.conditions, (a.search_single, a.search_dual), a.output,
            a.split_kind, a.audit_conditions)
    elif a.command == 'train-pulse':
        from .pulse import train_pulse_predictor
        result = train_pulse_predictor(a.corpus, a.output, a.epochs, a.hidden,
                                       a.seed, a.device, a.batch_size)
    elif a.command == 'evaluate-pulse-corpus':
        from .pulse import evaluate_pulse_corpus, load_pulse_bundle
        runner, _ = load_pulse_bundle(a.policy, a.device)
        result = evaluate_pulse_corpus(runner, a.corpus, a.output)
    else:
        from .evaluate import evaluate
        from .pulse import load_pulse_bundle
        from .sampling import load_conditions
        runner, bundle = load_pulse_bundle(a.policy, a.device)
        conditions, manifest = load_conditions(a.conditions)
        result = evaluate(runner, a.output, conditions=conditions,
                          condition_set_version=manifest['condition_set_version'],
                          role_action_mode=bundle['config']['role_action_mode'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
