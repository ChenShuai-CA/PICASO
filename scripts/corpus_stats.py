"""Corpus audit + prior-vs-baseline comparison with cluster bootstrap CIs.

P1.1 (v4) audited coverage and found the prior at the zero-action baseline. This
P1.2 version keeps the cross-table audit and adds the pre-registered comparison
protocol (GLM_CHANGELOG P1.2 R3/R4): several priors vs the zero-action baseline
on the val split, group-clustered bootstrap CIs, per-role and per-source slices,
and the pre-registered non-overlap verdicts. Self-checks recompute each prior's
logged val_mse on its val[:1024] subset to prove the MSE convention is identical
to pretrain.py's.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

CELLS = [(s, k, t) for s in ('interaction', 'waymo') for k in ('train', 'val', 'test')
         for t in ('vehicle', 'pedestrian')]
STRUCTURAL_ZERO = 'structural zero: recorded_trackfiles exports of the other kind have no rows here'
KIND_TO_ROLE = {'pedestrian': 0, 'vehicle': 1}


def load_corpus(corpus_dir):
    manifest = json.loads((corpus_dir / 'manifest.json').read_text(encoding='utf-8'))
    report = json.loads((corpus_dir / 'report.json').read_text(encoding='utf-8')) \
        if (corpus_dir / 'report.json').exists() else {}
    with np.load(corpus_dir / 'motion_prior.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    n = len(manifest)
    if any(v.ndim >= 1 and len(v) != n for v in arrays.values()):
        raise ValueError(f'manifest has {n} rows but npz arrays disagree')
    return manifest, arrays, report


def cross_table(manifest):
    stats = {cell: dict(examples=0, groups=set(), tracks=set()) for cell in CELLS}
    for entry in manifest:
        cell = (entry['source'], entry['split'], entry['kind'])
        stats[cell]['examples'] += 1
        stats[cell]['groups'].add((entry['source'], entry['group_id']))
        stats[cell]['tracks'].add((entry['source'], entry['group_id'], entry['track_id']))
    rows = []
    for source, split, kind in CELLS:
        s = stats[(source, split, kind)]
        if s['examples'] == 0:
            note = STRUCTURAL_ZERO if (source, kind) in _expected_zeros(manifest) \
                else 'zero examples (see report.json selection rules)'
        else:
            note = ''
        rows.append(dict(source=source, split=split, kind=kind, role=KIND_TO_ROLE[kind],
                         examples=s['examples'], independent_groups=len(s['groups']),
                         tracks=len(s['tracks']), note=note))
    return rows


def _expected_zeros(manifest):
    """Kinds with no examples anywhere for a source are structural, not accidental."""
    zeros = set()
    for source in ('interaction', 'waymo'):
        for kind in ('vehicle', 'pedestrian'):
            if not any(e['source'] == source and e['kind'] == kind for e in manifest):
                zeros.add((source, kind))
    return zeros


def role_weights(arrays, train_idx):
    role_totals = arrays['actor_mask'][train_idx].sum(axis=(0, 1))
    return role_totals.sum() / (np.maximum(role_totals, 1.) * max(np.count_nonzero(role_totals), 1))


def mse_convention(error, mask, weights=None):
    w = mask if weights is None else mask * weights
    return float((error * w).sum() / w.sum())


def cluster_bootstrap_mse(error, mask, group_ids, b_rounds=2000, seed=2026, alpha=.05):
    """Group-clustered bootstrap of the pretrain.py MSE convention."""
    group_ids = np.asarray(group_ids)
    unique = np.unique(group_ids)
    members = {g: np.flatnonzero(group_ids == g) for g in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(b_rounds)
    for b in range(b_rounds):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        samples[b] = (error[idx] * mask[idx]).sum() / mask[idx].sum()
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def self_only_mask(token_mask):
    """Same self-only view as pretrain(use_neighbors=False): all slots except each
    role's own (role r lives in slot r+1) are invisible to attention."""
    out = np.zeros_like(token_mask)
    for role in (0, 1):
        out[:, :, role, role + 1] = token_mask[:, :, role, role + 1]
    return out


def evaluate_prior(bundle_path, arrays, val_idx, device='cpu', view=None):
    """Errors of one prior on the val split under its own training view.

    The view comes from the bundle's saved use_neighbors flag (a prior trained
    --no-neighbors must be scored with the self-only mask, else its logged
    val_mse is unreproducible); `view` overrides it for cross-corpus reference
    rows, e.g. the v4 prior under the self-only view its v4 training matches.
    """
    import torch
    from scenario_lab.policy import load_bundle
    runner, bundle = load_bundle(bundle_path, device)
    use_neighbors = (bundle.get('config', {}).get('use_neighbors', True)
                     if view is None else view == 'neighbors')
    masks_all = arrays['token_mask'] if use_neighbors else self_only_mask(arrays['token_mask'])
    actor = runner.actor
    actor.eval()
    errors, masks = [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), 256):
            idx = val_idx[start:start + 256]
            x = {k: torch.as_tensor(v[idx], device=device)
                 for k, v in (('tokens', arrays['tokens']), ('token_mask', masks_all),
                              ('actor_mask', arrays['actor_mask']))}
            dist, _ = actor(x['tokens'], x['token_mask'], x['actor_mask'])
            errors.append(((dist.mean.tanh() - torch.as_tensor(arrays['target'][idx], device=device))
                           .square().sum(-1)).cpu().numpy())
            masks.append(arrays['actor_mask'][idx])
    return np.concatenate(errors), np.concatenate(masks), ('neighbors' if use_neighbors else 'self')


def self_check(name, prior_path, arrays, val_idx, error, mask):
    history_path = Path(prior_path).parent / 'pretraining.json'
    if not history_path.exists():
        return None
    logged = json.loads(history_path.read_text())[-1]['val_mse']
    subset = slice(None, min(len(val_idx), 1024))
    recomputed = mse_convention(error[subset], mask[subset])
    passed = bool(abs(recomputed - logged) < 1e-4)
    if not passed:
        raise AssertionError(f'{name}: convention mismatch vs pretraining.json '
                             f'({recomputed:.8f} vs {logged:.8f})')
    return dict(name=name, logged_val_mse=logged, recomputed_val1024_mse=recomputed, passed=passed)


def slices(error, mask, source_col, val_groups):
    out = [
        ('all roles, unweighted', error, mask, None, val_groups),
        ('all roles, train role-frequency weighted', error, mask, 'weighted', val_groups),
        ('role 0 pedestrian', error[:, :, 0], mask[:, :, 0], None, val_groups),
        ('role 1 vehicle', error[:, :, 1], mask[:, :, 1], None, val_groups),
        ('source=interaction', error[source_col == 'interaction'],
         mask[source_col == 'interaction'], None, val_groups[source_col == 'interaction']),
        ('source=waymo', error[source_col == 'waymo'], mask[source_col == 'waymo'],
         None, val_groups[source_col == 'waymo']),
    ]
    return [s for s in out if s[1].shape[0] > 0]  # drop empty per-source slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='runs/20260911_p1_corpus_v5')
    parser.add_argument('--priors', nargs='+', default=[],
                        help='name=bundle_path entries, e.g. prior_v5_nb=runs/.../prior.pt')
    parser.add_argument('--reference', action='append', default=None,
                        help='name=bundle_path cross-corpus reference row: scored under the '
                             'self-only view, excluded from self-checks and verdicts')
    parser.add_argument('--zero-action-name', default='zero-action baseline')
    parser.add_argument('--output', default='runs/20260911_p1_eval')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--verdict', action='append', default=None,
                        help='nameA,nameB: check non-overlapping CIs with A below B (repeatable)')
    args = parser.parse_args()

    corpus_dir, output = ROOT / args.corpus, ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} already has content; use a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest, arrays, report = load_corpus(corpus_dir)
    splits = arrays['split']
    train_idx, val_idx = np.flatnonzero(splits == 'train'), np.flatnonzero(splits == 'val')
    weights = role_weights(arrays, train_idx)
    val_groups = np.array([f"{manifest[i]['source']}|{manifest[i]['group_id']}" for i in val_idx])
    val_source = arrays['source'][val_idx]

    rows = cross_table(manifest)
    fields = list(rows[0])
    with (output / 'cross_table.csv').open('w', encoding='utf-8') as fh:
        fh.write(','.join(fields) + '\n')
        for r in rows:
            fh.write(','.join(str(r[f]) for f in fields) + '\n')

    zero_error = np.square(arrays['target'][val_idx]).sum(-1)
    val_mask = arrays['actor_mask'][val_idx]
    models = [(args.zero_action_name, zero_error, val_mask, 'analytic')]
    checks = []
    for item in args.priors:
        name, _, path = item.partition('=')
        error, mask, view = evaluate_prior(path, arrays, val_idx, args.device)
        models.append((name, error, mask, view))
        check = self_check(name, path, arrays, val_idx, error, mask)
        if check:
            checks.append(check)
    for item in args.reference or []:
        name, _, path = item.partition('=')
        error, mask, view = evaluate_prior(path, arrays, val_idx, args.device, view='self')
        models.append((name, error, mask, 'self (cross-corpus reference)'))
        checks.append(dict(name=name, logged_val_mse=None, recomputed_val1024_mse=None,
                           passed=None,
                           note='cross-corpus reference: scored on this corpus under the '
                                'self-only view; its own pretraining.json logged the v4 '
                                'corpus val, so the convention self-check does not apply'))

    results, boot = [], {}
    for name, error, mask, view in models:
        for slice_name, e, m, mode, groups in slices(error, mask, val_source, val_groups):
            w = weights if mode == 'weighted' else None
            mse = mse_convention(e, m, w)
            results.append(dict(model=name, slice=slice_name, view=view, mse=mse,
                                n_examples=int(m.shape[0])))
            if mode != 'weighted' and slice_name.startswith(('all roles, un', 'role ', 'source=')):
                lo, hi = cluster_bootstrap_mse(e, m, groups, args.bootstrap, args.seed)
                boot[(name, slice_name)] = (mse, lo, hi)
                results[-1].update(ci_lo=lo, ci_hi=hi)

    with (output / 'mse_table.csv').open('w', encoding='utf-8') as fh:
        fh.write('model,view,slice,mse,ci_lo,ci_hi,n_examples\n')
        for r in results:
            fh.write(f"{r['model']},{r['view']},{r['slice']},{r['mse']:.8f},"
                     f"{r.get('ci_lo', float('nan')):.8f},{r.get('ci_hi', float('nan')):.8f},"
                     f"{r['n_examples']}\n")

    verdicts = []
    for pair in args.verdict or []:
        below, above = pair.split(',')
        for slice_name in ('all roles, unweighted', 'role 0 pedestrian', 'role 1 vehicle'):
            if (below, slice_name) in boot and (above, slice_name) in boot:
                mse_a, lo_a, hi_a = boot[(below, slice_name)]
                mse_b, lo_b, hi_b = boot[(above, slice_name)]
                separated = hi_a < lo_b  # A entirely below B
                verdicts.append(dict(comparison=f'{below} < {above}', slice=slice_name,
                                     a_mse=mse_a, a_ci=[lo_a, hi_a], b_mse=mse_b, b_ci=[lo_b, hi_b],
                                     non_overlap=bool(separated),
                                     verdict='PASS' if separated else 'FAIL'))

    n = len(manifest)
    ped = sum(1 for e in manifest if e['kind'] == 'pedestrian')
    val_cells = defaultdict(int)
    for e in manifest:
        if e['split'] == 'val':
            val_cells[(e['source'], e['kind'])] += 1
    lines = [
        f'# Corpus audit + prior comparison ({report.get("corpus_version", "unknown")} corpus)',
        '',
        f'Corpus: `{args.corpus}` ({n} examples, pedestrians {ped} = {ped / n:.1%}). '
        f'feature_version={arrays.get("feature_version", "n/a")}. Val split: '
        + ', '.join(f'{s}/{k}={c}' for (s, k), c in sorted(val_cells.items()))
        + '. Bootstrap: group-clustered, '
        f'B={args.bootstrap}, seed={args.seed}.',
        '',
        '## Cross table',
        '',
        '| source | split | kind | role | examples | groups | tracks | note |',
        '|---|---|---|---|---:|---:|---:|---|',
    ]
    for r in rows:
        lines.append(f"| {r['source']} | {r['split']} | {r['kind']} | {r['role']} | "
                     f"{r['examples']} | {r['independent_groups']} | {r['tracks']} | {r['note']} |")
    lines += ['', '## Val MSE (pretrain.py convention)', '',
              '| model | view | slice | MSE | 95% CI | n |', '|---|---|---|---:|---|---:|']
    for r in results:
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if 'ci_lo' in r else '—'
        lines.append(f"| {r['model']} | {r['view']} | {r['slice']} | {r['mse']:.6f} | {ci} | {r['n_examples']} |")
    if verdicts:
        lines += ['', '## Pre-registered verdicts', '',
                  '| comparison | slice | below MSE [CI] | above MSE [CI] | non-overlap | verdict |',
                  '|---|---|---|---|---|---|']
        for v in verdicts:
            lines.append(f"| {v['comparison']} | {v['slice']} | {v['a_mse']:.6f} "
                         f"[{v['a_ci'][0]:.4f}, {v['a_ci'][1]:.4f}] | {v['b_mse']:.6f} "
                         f"[{v['b_ci'][0]:.4f}, {v['b_ci'][1]:.4f}] | {v['non_overlap']} | {v['verdict']} |")
    lines += ['', '## Interpretation boundary', '',
        'Engineering diagnostics under the pre-registered protocol (GLM_CHANGELOG P1.2 R3/R4); '
        'not paper evidence. Per-source slices on few groups carry limited statistical power '
        '(INTERACTION val groups are location-level).', '']
    (output / 'CORPUS_STATS.md').write_text('\n'.join(lines), encoding='utf-8')
    summary = dict(corpus=str(args.corpus), examples=n, cross_table=rows,
                   mse_table=results, verdicts=verdicts, self_checks=checks,
                   bootstrap=dict(rounds=args.bootstrap, seed=args.seed, unit='(source, group_id)'))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), verdicts={v['slice']: v['verdict'] for v in verdicts},
                          self_checks=[(c['name'], c['passed']) for c in checks]), indent=2))


if __name__ == '__main__':
    main()
