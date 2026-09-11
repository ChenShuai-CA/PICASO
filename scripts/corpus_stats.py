"""P1.1 corpus audit: coverage cross-table, zero-action baseline, per-role prior error.

Quantifies the current public v4 motion-prior corpus without changing any model:
1. source x split x kind cross table (examples, independent groups, tracks) with
   structurally-missing cells spelled out, not silently omitted;
2. zero-action baseline MSE on the val split under both loss weightings used by
   pretrain.py (unmasked val_mse convention and inverse-role-frequency weights);
3. per-role (pedestrian/vehicle) val error of the trained prior vs that baseline,
   with a self-check that our recomputation reproduces pretraining.json's val_mse.
Outputs CSV + Markdown into a fresh runs/<date>_p1_corpus_stats/ directory.
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
# Empty-cell explanations follow the actual split machinery: INTERACTION files carry no
# official split token, so group_id=location goes through the 80/10/10 hash; Waymo split
# is the directory token when present, else the per-scenario hash.
MISSING_NOTES = {
    ('interaction', 'train', 'pedestrian'):
        'structural zero: recorded_trackfiles vehicle_tracks exports contain no pedestrian rows',
    ('interaction', 'val', 'pedestrian'):
        'structural zero: recorded_trackfiles vehicle_tracks exports contain no pedestrian rows',
    ('interaction', 'test', 'pedestrian'):
        'structural zero: recorded_trackfiles vehicle_tracks exports contain no pedestrian rows',
    ('interaction', 'val', 'vehicle'):
        'all 4 selected locations hashed to the train bucket (no official split token in filenames)',
    ('interaction', 'test', 'vehicle'):
        'all 4 selected locations hashed to the train bucket (no official split token in filenames)',
}
KIND_TO_ROLE = {'pedestrian': 0, 'vehicle': 1}


def load_corpus(corpus_dir):
    manifest = json.loads((corpus_dir / 'manifest.json').read_text(encoding='utf-8'))
    with np.load(corpus_dir / 'motion_prior.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    n = len(manifest)
    if any(len(v) != n for v in arrays.values()):
        raise ValueError(f'manifest has {n} rows but npz arrays disagree')
    return manifest, arrays


def cross_table(manifest, arrays):
    rows = []
    stats = {cell: dict(examples=0, groups=set(), tracks=set()) for cell in CELLS}
    for entry in manifest:
        cell = (entry['source'], entry['split'], entry['kind'])
        stats[cell]['examples'] += 1
        stats[cell]['groups'].add((entry['source'], entry['group_id']))
        stats[cell]['tracks'].add((entry['source'], entry['group_id'], entry['track_id']))
    for source, split, kind in CELLS:
        s = stats[(source, split, kind)]
        note = MISSING_NOTES.get((source, split, kind), '')
        if s['examples'] == 0 and not note:
            note = 'zero examples despite the source being selected'
        rows.append(dict(source=source, split=split, kind=kind, role=KIND_TO_ROLE[kind],
                         examples=s['examples'], independent_groups=len(s['groups']),
                         tracks=len(s['tracks']), note=note))
    return rows


def role_weights(arrays, train_idx):
    # Reproduce pretrain.py's inverse training role frequency weights verbatim.
    role_totals = arrays['actor_mask'][train_idx].sum(axis=(0, 1))
    return role_totals.sum() / (np.maximum(role_totals, 1.) * max(np.count_nonzero(role_totals), 1))


def mse_convention(error, mask, weights=None):
    """pretrain.py convention: error (N,T,2) already summed over the 2 action dims."""
    w = mask if weights is None else mask * weights
    return float((error * w).sum() / w.sum())


def evaluate_prior(bundle_path, arrays, val_idx, device='cpu'):
    import torch
    from scenario_lab.policy import load_bundle
    runner, _ = load_bundle(bundle_path, device)
    actor = runner.actor
    actor.eval()
    errors, masks = [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), 256):
            idx = val_idx[start:start + 256]
            x = {k: torch.as_tensor(arrays[k][idx], device=device)
                 for k in ('tokens', 'token_mask', 'actor_mask')}
            dist, _ = actor(x['tokens'], x['token_mask'], x['actor_mask'])
            errors.append(((dist.mean.tanh() - torch.as_tensor(arrays['target'][idx], device=device))
                           .square().sum(-1)).cpu().numpy())
            masks.append(arrays['actor_mask'][idx])
    return np.concatenate(errors), np.concatenate(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='runs/20260911_public_v4')
    parser.add_argument('--prior', default='runs/20260911_gpu_pilot/prior/prior.pt')
    parser.add_argument('--output', default='runs/20260911_p1_corpus_stats')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    corpus_dir, output = ROOT / args.corpus, ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} already has content; use a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest, arrays = load_corpus(corpus_dir)
    splits = arrays['split']
    train_idx, val_idx = np.flatnonzero(splits == 'train'), np.flatnonzero(splits == 'val')

    # --- 1. coverage cross table -------------------------------------------------
    rows = cross_table(manifest, arrays)
    fields = list(rows[0])
    with (output / 'cross_table.csv').open('w', encoding='utf-8') as fh:
        fh.write(','.join(fields) + '\n')
        for r in rows:
            fh.write(','.join(str(r[f]) for f in fields) + '\n')
    missing = [r for r in rows if r['examples'] == 0]

    # --- 2 + 3. zero-action vs prior on val, both weightings, per role -----------
    weights = role_weights(arrays, train_idx)
    target = arrays['target']
    zero_error = np.square(target).sum(-1)
    prior_error, mask = evaluate_prior(args.prior, arrays, val_idx, args.device)
    zero_val = zero_error[val_idx]

    def table(error):
        out = [('all roles, unweighted (val_mse convention)', mse_convention(error, mask)),
               ('all roles, train role-frequency weighted', mse_convention(error, mask, weights)),
               ('role 0 pedestrian', mse_convention(error[:, :, 0], mask[:, :, 0])),
               ('role 1 vehicle', mse_convention(error[:, :, 1], mask[:, :, 1]))]
        return out

    compare = [('zero-action baseline', table(zero_val)), ('trained prior', table(prior_error))]
    with (output / 'zero_action_vs_prior.csv').open('w', encoding='utf-8') as fh:
        fh.write('model,slice,mse\n')
        for name, entries in compare:
            for slice_name, mse in entries:
                fh.write(f'{name},{slice_name},{mse:.8f}\n')

    # Self-check: reproduce pretraining.json's last val_mse on its val[:1024] subset.
    history = json.loads((ROOT / 'runs/20260911_gpu_pilot/prior/pretraining.json').read_text())
    logged = history[-1]['val_mse']
    subset = slice(None, min(len(val_idx), 1024))
    recomputed = mse_convention(prior_error[subset], mask[subset])
    selfcheck = dict(logged_val_mse=logged, recomputed_val1024_mse=recomputed,
                     abs_diff=abs(recomputed - logged), passed=bool(abs(recomputed - logged) < 1e-4))
    if not selfcheck['passed']:
        raise AssertionError(f'convention mismatch vs pretraining.json: {selfcheck}')

    n = len(manifest)
    ped = sum(1 for e in manifest if e['kind'] == 'pedestrian')
    val_rows = [(e['source'], e['kind']) for e in manifest if e['split'] == 'val']
    val_cells = defaultdict(int)
    for cell in val_rows:
        val_cells[cell] += 1
    zero = compare[0][1][0][1]
    prior_mse = compare[1][1][0][1]
    lines = [
        '# P1.1 corpus audit (public v4 motion-prior corpus)',
        '',
        f'Corpus: `{args.corpus}` ({n} examples), prior: `{args.prior}`. '
        'This is an engineering audit of existing artifacts; no model was retrained.',
        '',
        '## 1. Coverage: source x split x kind',
        '',
        '| source | split | kind | role | examples | groups | tracks | note |',
        '|---|---|---|---|---:|---:|---:|---|',
    ]
    for r in rows:
        lines.append(f"| {r['source']} | {r['split']} | {r['kind']} | {r['role']} | "
                     f"{r['examples']} | {r['independent_groups']} | {r['tracks']} | {r['note']} |")
    lines += [
        '',
        f'- Pedestrian examples: {ped}/{n} ({ped / n:.1%}); all pedestrians come from Waymo; '
        'INTERACTION contributes vehicles only (its vehicle_tracks exports have no pedestrian rows), '
        'so any INTERACTION x pedestrian cell is a **structural zero**, not a sampling accident.',
        f'- Val-split composition (source x kind): '
        + ', '.join(f'{s}/{k}={c}' for (s, k), c in sorted(val_cells.items()))
        + '. The HANDOFF-flagged gap is concrete: the val split contains **zero INTERACTION '
        'examples of either kind** (all 4 selected locations hashed into train), so the prior '
        'has never been validated on the INTERACTION distribution at all, pedestrian or not.',
        f'- {len(missing)} of {len(CELLS)} cells are empty; every empty cell carries a reason above.',
        '',
        '## 2. Zero-action baseline vs trained prior on val',
        '',
        '| model | slice | MSE |',
        '|---|---|---:|',
    ]
    for name, entries in compare:
        for slice_name, mse in entries:
            lines.append(f'| {name} | {slice_name} | {mse:.6f} |')
    lines += [
        '',
        f'- Zero-action val MSE is **{zero:.4f}**; the trained prior reaches **{prior_mse:.4f}** '
        f'({(zero - prior_mse) / zero:+.1%} relative). '
        'Action targets are normalized; most windows are near-constant motion, so a zero predictor '
        'is a strong baseline.',
        f'- Convention self-check: recomputing the prior on the logged val[:1024] subset gives '
        f'{recomputed:.8f} vs pretraining.json last-epoch val_mse {logged:.8f} '
        f'(abs diff {abs(recomputed - logged):.2e}) — {"PASS" if selfcheck["passed"] else "FAIL"}. '
        'All MSEs above therefore use the exact pretrain.py convention.',
        '',
        '## Interpretation boundary',
        '',
        'These numbers quantify why the current self-motion prior buys almost nothing at evaluation '
        'time (it sits at the zero-action baseline) and where the corpus is thin (pedestrians, '
        'INTERACTION-side roles). They are engineering diagnostics, not paper evidence; P1.2 '
        '(neighbor-aligned interaction pretraining) is the response, not a re-weighted rerun of this one.',
        '',
    ]
    (output / 'CORPUS_STATS.md').write_text('\n'.join(lines), encoding='utf-8')
    summary = dict(corpus=str(args.corpus), prior=str(args.prior), examples=n,
                   cross_table=rows, comparison=[dict(model=m, slices=dict(e)) for m, e in compare],
                   self_check=selfcheck)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), cells=len(rows), empty_cells=len(missing),
                          zero_action_mse=zero, prior_mse=prior_mse,
                          self_check='PASS' if selfcheck['passed'] else 'FAIL'), indent=2))


if __name__ == '__main__':
    main()
