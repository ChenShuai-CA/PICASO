"""P3: validate physics-based brake-source signature on labelled ABD runs.

Mechanics supplied by the project test operator (2026-09-12):
- The ABD BR robot is rigidly coupled to the vehicle brake pedal. The load
  cell reads appreciable force only when the robot actively pushes or a human
  foot actively presses the pedal (compression).
- On coupled vehicles AEB actuation drags the pedal down: the load cell is
  pulled along, so travel is large while force stays small (possibly
  negative/tension).
- On decoupled vehicles AEB actuation does not move the pedal at all.
- C-NCAP 2024 D4F4 runs: BR applies a light pedal touch first (small BR
  Command pulse), then AEB performs the full stop.

This script recomputes event windows for the operator-labelled runs in
runs/20260912_abd_braking_source_analysis/labelled_event_features.csv and
scores each event against the resulting signature rule. It changes no labels;
disagreements are output as an adjudication queue for the operator.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_abd_calibration import (  # noqa: E402
    EXACT_CHANNELS, _event_window, _header, _stream_selected)

LABELLED = (ROOT / 'runs/20260912_abd_braking_source_analysis/'
            'labelled_event_features.csv')
OUT_DIR = ROOT / 'runs/20260912_abd_pedal_signature_validation'

STATIC_TRAVEL_MM = 2.0        # pedal considered unmoved
FOOT_FORCE_N = 50.0           # compression beyond this = human/robot push
DRAG_FORCE_BAND_N = 50.0      # |force| within this band = pedal dragged free
TENSION_N = -5.0              # force at/below this = pulled-along hard evidence
PRE_FOOT_TRAVEL_MM = 5.0      # pre-event pedal activity = foot on pedal


def classify(rec):
    travel = rec['travel_win_mm']
    fmin, fmax = rec['force_min_win'], rec['force_max_win']
    fnoise = max(abs(rec['force_min_pre']), abs(rec['force_max_pre']))
    rec['tension'] = fmin <= TENSION_N
    rec['foot_pre_event'] = rec['travel_pre_mm'] > PRE_FOOT_TRAVEL_MM
    foot_push = fmax > max(FOOT_FORCE_N, fnoise + 30.0)
    # Tension (negative force) means the load cell is being pulled along with
    # the pedal: hard evidence of AEB back-drive, incompatible with a pure
    # foot press. Large compression with tension in the same window means the
    # foot joined after the drag started -> operator adjudication.
    if foot_push and rec['tension']:
        rec['signature'] = 'drag_then_foot_adjudicate'
    elif foot_push:
        rec['signature'] = 'foot_compression'
    elif travel <= STATIC_TRAVEL_MM:
        rec['signature'] = 'aeb_pedal_static'
    else:
        # compression small; any tension magnitude is still drag evidence
        rec['signature'] = 'aeb_pedal_dragged'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LABELLED.open(encoding='utf-8-sig') as handle:
        labelled = list(csv.DictReader(handle))

    out = []
    for row in labelled:
        path = ROOT / 'Data/ABD_Data' / row['run']
        _, delimiter, names, _, data_offset = _header(path)
        selected = [i for i, name in enumerate(names) if name in EXACT_CHANNELS]
        arrays, _, _, _ = _stream_selected(path, names, delimiter, data_offset,
                                           selected)
        time, speed = arrays['Time'], arrays['Speed']
        event = _event_window(time, speed,
                              arrays.get('Forward acceleration (body)'))
        if event is None:
            out.append({'run': row['run'], 'vehicle': row['vehicle'],
                        'label': row['label'], 'signature': 'no_event'})
            continue
        onset, end = event['onset_index'], event['end_index']
        dt = float(np.median(np.diff(time)))
        pre0 = max(0, onset - round(2.0 / dt))
        br, force = arrays['BR Position'], arrays['Brake force (unfiltered)'
                                                  ] if 'Brake force (unfiltered)' in arrays else None
        rec = {
            'run': row['run'].rsplit('/', 1)[-1], 'vehicle': row['vehicle'],
            'label': row['label'],
            'onset_s': round(float(time[onset]), 2),
            'travel_pre_mm': round(float(np.ptp(br[pre0:onset])), 3),
            'travel_win_mm': round(float(np.ptp(br[onset:end])), 3),
            'force_min_pre': round(float(force[pre0:onset].min()), 2),
            'force_max_pre': round(float(force[pre0:onset].max()), 2),
            'force_min_win': round(float(force[onset:end].min()), 2),
            'force_max_win': round(float(force[onset:end].max()), 2),
            'cmd_abs_max_win': round(
                float(np.abs(arrays['BR Command'][onset:end]).max()), 3),
        }
        classify(rec)
        out.append(rec)

    fields = list(out[0].keys())
    with (OUT_DIR / 'per_run_signature.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)

    for label in sorted({r['label'] for r in out}):
        sub = [r for r in out if r['label'] == label]
        print(f'===== {label} (n={len(sub)}) =====')
        counts = {}
        for r in sub:
            counts[r['signature']] = counts.get(r['signature'], 0) + 1
        print('  signatures:', dict(sorted(counts.items())))
        print('  tension (fmin<=-5N):',
              sum(1 for r in sub if r.get('tension')))
        print('  foot pre-event (travel_pre>5mm):',
              sum(1 for r in sub if r.get('foot_pre_event')))


if __name__ == '__main__':
    main()
