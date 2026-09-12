"""P3: contact-outcome scan for the pedal-signature corpus.

For every run in runs/20260912_abd_aeb_pedal_screen/per_run_signature.csv,
stream Time + 'Relative longitudinal distance' and classify the outcome of
the observed braking event by the minimum relative distance around the event
window [onset-2s, end+2s]:

- contact_and_stopped : relative distance crossed <= 0 with speed at the
                       crossing <= 25 kph and |min| <= 3 m -- the VUT
                       reached the target and stopped on/against it
                       (crush/push depth bounded; e.g. V3_T143_R1 -1.39 m)
- passed_or_swept     : distance crossed <= 0 but the VUT kept driving past
                       the target (very negative min afterwards) or the
                       target moved past a (near-)stopped VUT (e.g. a
                       pedestrian dummy walking by); longitudinal proxy
                       only, lateral geometry not checked
- near_contact_0_5    : no crossing, 0 < min <= 0.5 m (matches scenario_lab
                       danger margin)
- clear               : min > 0.5 m
- missing_channel     : no relative-distance channel in the export

Motivated by V3_T140_R1 / V3_T143_R1 (Guang_Qi_A66 AEB 60 kph): AEB braking,
full mid-event release (decel -> ~0, coasting toward the target), renewed
braking, then relative distance crossing zero (-0.34 / -1.39 m penetration).
The late '-12 m/s^2' phase in T143 is collision dynamics, not actuation --
the brake robot never applied force (UseBrakeRobot=False, Motion Going BR=0,
load cell ~0 N throughout).

Outputs: runs/20260912_abd_aeb_pedal_screen/contact_outcome.csv
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_abd_calibration import (  # noqa: E402
    _event_window, _header, _stream_selected)

PER_RUN = ROOT / 'runs/20260912_abd_aeb_pedal_screen/per_run_signature.csv'
OUT_CSV = ROOT / 'runs/20260912_abd_aeb_pedal_screen/contact_outcome.csv'

REL_DIST = 'Relative longitudinal distance'
NEED = {'Time', 'Speed', 'Forward acceleration (body)', REL_DIST}
MARGIN_S = 2.0
CONTACT_M = 0.0
NEAR_M = 0.5
CONTACT_CRUSH_M = 3.0   # bounded crush/push depth (vs target passed behind)
CONTACT_SPEED_KPH = 25.0  # VUT slowed onto the target, not driving past


def _score(task):
    rel, = task
    path = ROOT / 'Data/ABD_Data' / rel
    rec = {'run': rel, 'rel_dist_at_onset_m': '', 'min_rel_dist_m': '',
           'min_rel_dist_rel_s': '', 'first_contact_rel_s': '',
           'speed_at_first_contact_kph': '', 'outcome': ''}
    try:
        _, delimiter, names, _, data_offset = _header(path)
        selected = [i for i, name in enumerate(names) if name in NEED]
        arrays, _, _, _ = _stream_selected(path, names, delimiter,
                                           data_offset, selected)
        if REL_DIST not in arrays:
            rec['outcome'] = 'missing_channel'
            return rec
        time, speed = arrays['Time'], arrays['Speed']
        ev = _event_window(time, speed,
                           arrays.get('Forward acceleration (body)'))
        if ev is None:
            rec['outcome'] = 'no_event'
            return rec
        onset, end = ev['onset_index'], ev['end_index']
        dt = float(np.median(np.diff(time)))
        lo = max(0, onset - round(MARGIN_S / dt))
        hi = min(len(time), end + round(MARGIN_S / dt))
        dist = arrays[REL_DIST][lo:hi]
        rec['rel_dist_at_onset_m'] = round(float(arrays[REL_DIST][onset]), 3)
        rec['min_rel_dist_m'] = round(float(dist.min()), 3)
        argmin = lo + int(dist.argmin())
        rec['min_rel_dist_rel_s'] = round(
            float(time[argmin] - time[onset]), 3)
        crossed = np.flatnonzero(arrays[REL_DIST][lo:hi] <= CONTACT_M)
        first_idx = None
        if len(crossed):
            first_idx = lo + int(crossed[0])
            rec['first_contact_rel_s'] = round(
                float(time[first_idx] - time[onset]), 3)
            rec['speed_at_first_contact_kph'] = round(
                float(speed[first_idx]), 2)
        mn = float(dist.min())
        if mn <= CONTACT_M:
            if (first_idx is not None
                    and abs(mn) <= CONTACT_CRUSH_M
                    and speed[first_idx] <= CONTACT_SPEED_KPH):
                rec['outcome'] = 'contact_and_stopped'
            else:
                rec['outcome'] = 'passed_or_swept'
        elif mn <= NEAR_M:
            rec['outcome'] = 'near_contact_0_5'
        else:
            rec['outcome'] = 'clear'
        return rec
    except Exception as exc:
        rec['outcome'] = f'parse_error:{type(exc).__name__}'
        return rec


def main():
    with PER_RUN.open(encoding='utf-8-sig') as handle:
        runs = [r['run'] for r in csv.DictReader(handle)]
    print(f'contact-outcome scan over {len(runs)} runs ...', flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_score, (r,)) for r in runs]
        for i, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if i % 200 == 0:
                print(f'  {i}/{len(runs)} ({time.time()-t0:.0f}s)', flush=True)
    order = {r: i for i, r in enumerate(runs)}
    out.sort(key=lambda r: order[r['run']])
    with OUT_CSV.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    with PER_RUN.open(encoding='utf-8-sig') as handle:
        per_run = {r['run']: r for r in csv.DictReader(handle)}
    cross = defaultdict(Counter)
    for r in out:
        cross[per_run[r['run']]['screen_class']][r['outcome']] += 1
    for cls in sorted(cross):
        print(f'{cls:32s} {dict(cross[cls])}')
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
