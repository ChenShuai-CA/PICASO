"""Stratify pilot failures over fixed conditions; fault localization, not method ranking."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scenario_lab.evaluate import run_episode, ScriptPolicy
from scenario_lab.geometry import clearance, corners, segment_blocked
from scenario_lab.policy import load_bundle
from scenario_lab.runtime import resolve_device
from scenario_lab.sampling import load_conditions
from scenario_lab.schema import Body

# Fixed entity colors (Okabe-Ito; validated CVD-safe, see docs/GLM_CHANGELOG.md).
COLOR = dict(ego='#0072B2', pedestrian='#D55E00', occluder='#009E73')
CLEARANCE_COLOR = dict(ego_ped='#444444', ped_occ='#999999')
LAYER_ORDER = ('script_conflict', 'policy_added_conflict', 'script_dangerous_policy_valid', 'both_valid')


def load_episodes(root):
    episodes = {}
    for path in sorted(root.glob('eval_*/episodes.jsonl')):
        method = path.parent.name.removeprefix('eval_')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        episodes[method] = {r['scenario_id']: r for r in rows}
    return episodes


def classify(script_row, policy_row):
    """Mutually exclusive layers; counts over all conditions sum to attempts."""
    if not script_row['valid']:
        return 'script_conflict:' + '+'.join(script_row['invalid_reasons'])
    if not policy_row['valid']:
        return 'policy_added_conflict:' + '+'.join(policy_row['invalid_reasons'])
    if script_row['dangerous']:
        return 'script_dangerous_policy_valid'
    return 'both_valid'


def make_policy(method, root, device):
    if method == 'script':
        return ScriptPolicy()
    path = root / 'prior' / 'prior.pt' if method == 'prior' else root / method / 'policy.pt'
    return load_bundle(path, device=device)[0]


def _bodies(frame):
    return [Body(**b) for b in frame['bodies']]


def plot_trace(record, title, out_png):
    frames = record['frames']
    spec = record['scenario']
    times = np.array([f['time'] for f in frames])
    pos = {name: np.array([[f['bodies'][i]['x'], f['bodies'][i]['y']] for f in frames])
           for i, name in enumerate(('ego', 'pedestrian', 'occluder')[:len(frames[0]['bodies'])])}
    speed = {name: np.array([f['bodies'][i]['speed'] for f in frames])
             for i, name in enumerate(('ego', 'pedestrian', 'occluder')[:len(frames[0]['bodies'])])}
    clear_ep, clear_po, blocked = [], [], []
    for frame in frames:
        b = _bodies(frame)
        clear_ep.append(clearance(b[0], b[1]))
        clear_po.append(clearance(b[1], b[2]) if len(b) == 3 else np.nan)
        blocked.append(bool(len(b) == 3 and segment_blocked((b[0].x, b[0].y), (b[1].x, b[1].y), b[2])))
    clear_ep, clear_po, blocked = np.array(clear_ep), np.array(clear_po), np.array(blocked)
    collision_t = next((t for t, c in zip(times, clear_ep) if c <= 1e-9), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(title, fontsize=11)
    for name in pos:
        ax1.plot(pos[name][:, 0], pos[name][:, 1], color=COLOR[name], lw=1.6,
                 label=dict(ego='ego', pedestrian='pedestrian', occluder='occluder')[name])
        ax1.text(*pos[name][-1] + [0.4, 0.2], name, color=COLOR[name], fontsize=8)
    for k in range(0, len(frames), 10):  # 1 Hz oriented footprints
        for i, name in enumerate(('ego', 'pedestrian', 'occluder')[:len(frames[k]['bodies'])]):
            ax1.add_patch(MplPolygon(corners(_bodies(frames[k])[i]), closed=True, fill=False,
                                     edgecolor=COLOR[name], lw=.7, alpha=.45))
    if collision_t is not None:
        k = int(np.argmax(times >= collision_t))
        ax1.plot(*pos['ego'][k], marker='*', ms=13, color='#B2182B', mec='white', mew=.6,
                 label='ego-ped collision', zorder=5, linestyle='none')
    if len(pos) == 3 and np.nanmin(clear_po) <= 1e-9:
        k = int(np.nanargmin(clear_po))
        ax1.plot(*pos['pedestrian'][k], marker='X', ms=10, color='black', mfc='none', mew=1.4,
                 label='ped-occluder contact', zorder=5, linestyle='none')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(alpha=.25, lw=.5)
    ax1.legend(fontsize=8, loc='best')

    ax2.plot(times, speed['ego'], color=COLOR['ego'], lw=1.6, label='ego speed')
    ax2.plot(times, speed['pedestrian'], color=COLOR['pedestrian'], lw=1.6, label='ped speed')
    if 'occluder' in speed:
        ax2.plot(times, speed['occluder'], color=COLOR['occluder'], lw=1.6, label='occluder speed')
    ax2.plot(times, clear_ep, color=CLEARANCE_COLOR['ego_ped'], lw=1.2, ls='--', label='ego-ped clearance')
    if len(pos) == 3:
        ax2.plot(times, clear_po, color=CLEARANCE_COLOR['ped_occ'], lw=1.2, ls=':', label='ped-occluder clearance')
    if spec.get('pedestrian_delay', 0) > 0:
        ax2.axvspan(0, spec['pedestrian_delay'], color='#000000', alpha=.07, lw=0)
        ax2.text(spec['pedestrian_delay'] / 2, ax2.get_ylim()[1], ' ped waiting', fontsize=7, va='top')
    for t in times[blocked]:
        ax2.axvline(t, color='#E69F00', alpha=.16, lw=.8)
    if collision_t is not None:
        ax2.axvline(collision_t, color='#B2182B', lw=1.2, ls='-.', alpha=.8)
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('speed [m/s] / clearance [m]')
    ax2.grid(alpha=.25, lw=.5)
    ax2.legend(fontsize=8, loc='best')
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--max-figures', type=int, default=10)
    a = p.parse_args()
    device = resolve_device(a.device)
    specs, manifest = load_conditions(a.conditions)
    spec_by_id = {s.scenario_id: s for s in specs}
    episodes = load_episodes(a.root)
    if 'script' not in episodes:
        raise ValueError('no eval_script episodes under root; script baseline is required for stratification')
    methods = sorted(episodes)
    a.output.mkdir(parents=True, exist_ok=True)

    # Condition provenance check: manifest specs must match what the pilot actually ran.
    drifted = 0
    for method, rows in episodes.items():
        for sid, row in rows.items():
            spec = spec_by_id.get(sid)
            if spec is None:
                continue
            if any(spec.to_dict()[k] != v for k, v in row['spec'].items()):
                drifted += 1
    if drifted:
        raise ValueError(f'{drifted} episode specs disagree with the condition manifest; refusing to stratify')

    # Layer counts.
    counts, conditions_rows = {}, []
    for method in methods:
        for sid in sorted(spec_by_id):
            if sid not in episodes[method] or sid not in episodes['script']:
                continue
            branch = spec_by_id[sid].branch
            layer = classify(episodes['script'][sid], episodes[method][sid])
            counts[(method, branch, layer)] = counts.get((method, branch, layer), 0) + 1
    with (a.output / 'layers.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'branch', 'layer', 'count', 'share_of_branch'])
        for (method, branch, layer), n in sorted(counts.items()):
            total = sum(v for (m, b, _), v in counts.items() if m == method and b == branch)
            w.writerow([method, branch, layer, n, f'{n / total:.3f}'])
    per_method_fields = [f'{m}_{field}' for m in methods for field in ('reasons', 'collision', 'min_clearance', 'dangerous')]
    with (a.output / 'per_condition.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['branch', 'scenario_id'] + per_method_fields)
        for sid in sorted(spec_by_id):
            row = [spec_by_id[sid].branch, sid]
            for m in methods:
                r = episodes[m].get(sid)
                row += ['|'.join(r['invalid_reasons']) if r else 'missing',
                        r['collision'] if r else '', f"{r['min_clearance']:.3f}" if r else '',
                        r['dangerous'] if r else '']
            w.writerow(row)

    # Representative figures: per (branch, layer label) the lowest scenario_id and the
    # highest collision_speed case; never hand-picked.
    by_layer = {}
    for method in methods:
        if method == 'script':
            continue
        for sid in sorted(spec_by_id):
            if sid not in episodes[method] or sid not in episodes['script']:
                continue
            layer = classify(episodes['script'][sid], episodes[method][sid])
            by_layer.setdefault((spec_by_id[sid].branch, layer), []).append((sid, method))
    chosen, used = [], set()
    for (branch, layer) in sorted(by_layer, key=lambda k: (k[0], LAYER_ORDER.index(k[1].split(':')[0]) if k[1].split(':')[0] in LAYER_ORDER else 9, k[1])):
        cands = sorted(by_layer[(branch, layer)])
        picks = [cands[0]]
        with_collision = [c for c in cands if episodes[c[1]][c[0]]['collision']]
        if with_collision:
            worst = max(with_collision, key=lambda c: (episodes[c[1]][c[0]]['collision_speed'], -int(c[0].split('-')[1])))
            if worst != picks[0]:
                picks.append(worst)
        for pick in picks:
            if pick not in used and len(chosen) < a.max_figures:
                used.add(pick)
                chosen.append((branch, layer, *pick))

    figures_dir = a.output / 'figures'
    figures_dir.mkdir(exist_ok=True)
    traces_dir = a.output / 'regenerated'
    traces_dir.mkdir(exist_ok=True)
    regeneration_report = []
    for branch, layer, sid, method in chosen:
        policy = make_policy(method, a.root, device)
        trace_path = traces_dir / f'{method}_{sid}.json'
        fresh = run_episode(policy, spec_by_id[sid], episodes[method][sid]['seed'], trace_path)
        stored = episodes[method][sid]
        drift = abs(fresh['min_clearance'] - stored['min_clearance'])
        same_outcome = (fresh['invalid_reasons'] == stored['invalid_reasons']
                        and fresh['collision'] == stored['collision'])
        regeneration_report.append(dict(method=method, scenario_id=sid, layer=layer,
                                        min_clearance_stored=stored['min_clearance'],
                                        min_clearance_fresh=fresh['min_clearance'],
                                        outcome_match=bool(drift <= 1e-3 and same_outcome)))
        record = json.loads(trace_path.read_text(encoding='utf-8'))
        plot_trace(record, f'{sid} | method={method} | layer={layer}', figures_dir / f'{branch}_{sid}_{method}.png')

    # Report.
    lines = ['# Failure stratification over fixed pilot conditions', '',
             f'- pilot root: `{a.root}`', f'- conditions: `{a.conditions}` '
             f'(version `{manifest["condition_set_version"]}`, sampler v{manifest["sampler_version"]}, '
             f'physics v{manifest["physics_version"]}, purpose `{manifest["purpose"]}`)',
             '- stratification compares each policy against the zero-action script baseline on the '
             'same scenario_id; this is fault localization, not a method ranking',
             '- all episodes below were produced under legacy physics v1 '
             '(pedestrian speed reported while position frozen during pedestrian_delay); '
             'see docs/GLM_CHANGELOG.md', '',
             '## Layer counts by branch', '']
    for branch in sorted({b for (_, b, _) in counts}):
        lines.append(f'### {branch}')
        lines.append('')
        header = ['method'] + sorted({l for (_, b, l) in counts if b == branch})
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('|' + '---|' * len(header))
        for method in methods:
            total = sum(v for (m, b, _), v in counts.items() if m == method and b == branch)
            if not total:
                continue
            cells = [method] + [str(counts.get((method, branch, l), 0)) for l in header[1:]]
            lines.append('| ' + ' | '.join(cells) + f' | (n={total})')
        lines.append('')
    lines += ['## Regeneration verification', '',
              '| method | scenario | stored clearance | fresh clearance | outcome match |', '|---|---|---|---|---|']
    for r in regeneration_report:
        lines.append(f"| {r['method']} | {r['scenario_id']} | {r['min_clearance_stored']:.4f} | "
                     f"{r['min_clearance_fresh']:.4f} | {r['outcome_match']} |")
    lines += ['', '## Figures', '']
    for branch, layer, sid, method in chosen:
        lines.append(f'- `figures/{branch}_{sid}_{method}.png` — layer `{layer}`')
    lines += ['', 'Generated by `scripts/diagnose_failures.py`; selection rule is fixed '
              '(lowest scenario_id + highest collision_speed per layer), never hand-picked.', '']
    (a.output / 'DIAGNOSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    mismatched = [r for r in regeneration_report if not r['outcome_match']]
    print(json.dumps(dict(output=str(a.output), conditions=str(a.conditions), methods=methods,
                          layers=len(counts), figures=len(chosen),
                          regeneration_mismatches=[r['scenario_id'] for r in mismatched],
                          elapsed_note=f'device={device}'), indent=2))


if __name__ == '__main__':
    main()
