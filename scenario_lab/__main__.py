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
    train.add_argument('--algorithm', choices=['ppo', 'ippo', 'mappo'], default='mappo')
    train.add_argument('--device', default='auto')
    train.add_argument('--pretrained')
    train.add_argument('--no-robust', action='store_true')
    train.add_argument('--no-role-constraints', action='store_true')
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
    a = p.parse_args()
    if hasattr(a, 'device'):
        a.device = resolve_device(a.device)
    if a.command == 'train':
        from .train import train, TrainConfig
        result = train(a.output, TrainConfig(seed=a.seed, updates=a.updates,
                       episodes_per_update=a.episodes_per_update, mode=a.mode,
                       algorithm=a.algorithm, device=a.device, robust=not a.no_robust,
                       role_constraints=not a.no_role_constraints,
                       sampler_version=a.sampler_version), a.pretrained)
    elif a.command in ('evaluate', 'benchmark'):
        from .evaluate import evaluate, benchmark, ScriptPolicy, OneLearningPolicy
        from .policy import load_bundle
        import torch
        torch.set_num_threads(1)
        runner = load_bundle(a.policy, device=a.device)[0] if a.policy else ScriptPolicy()
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
            result = evaluate(runner, a.output, a.count, a.seed, perturbations=a.perturbations,
                              controller=a.controller, conditions=conditions, condition_set_version=version)
        else:
            result = benchmark(runner, a.output, a.steps)
    elif a.command == 'search':
        from .evaluate import search, search_conditions
        from .schema import ScenarioSpec
        if a.conditions:
            from .sampling import load_conditions
            conditions, _ = load_conditions(a.conditions)
            result = search_conditions(conditions, a.output, a.kind, a.budget, a.seed,
                                       branches=(a.branch,))
        else:
            result = search(ScenarioSpec(branch=a.branch), a.output, a.kind, a.budget, a.seed)
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
    else:
        from .pretrain import pretrain
        result = pretrain(a.corpus, a.output, a.epochs, seed=a.seed, device=a.device,
                          use_neighbors=not a.no_neighbors)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
