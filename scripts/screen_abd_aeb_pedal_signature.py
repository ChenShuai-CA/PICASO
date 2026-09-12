"""P3: full-corpus brake-source pedal-signature screen over ABD AEB-path runs.

Input: runs/20260912_abd_no_takeover_screen/screening.csv (1,258 rows).
Scanned subsets (event_status):
- br_zero_observed_braking (892): apply the operator-mechanics signature rule
  validated in scripts/validate_abd_pedal_signature.py (classify imported
  unchanged; single source of truth for thresholds).
- robot_channel_active (268): D4F4 pool. Look for the C-NCAP 2024 D4F4
  pattern the operator described: a small BR Command "light touch" pulse
  BEFORE the main deceleration, main-window BR Command == 0, and an AEB pedal
  signature (dragged/static) in the main window. Runs whose main window has
  robot command activity are routed robot_braked_main_window (excluded from
  the AEB set).

Post-hoc corrections encoded here after time-series inspection of the first
pass (documented in REPORT.md):
1. The screen's robot_channel_active flag fires on BR Command noise as small
   as 0.01-0.06 EU. Runs whose recomputed window AND 5 s pre-window stay below
   CMD_EPS are genuinely brake-robot-quiet at the event and are re-classified
   with the standard BR-zero rule, flagged screen_flag_noisy.
2. Runs whose robot command only starts well after the decel is underway
   (first active cmd > 0.5 s after onset) are split out as
   robot_joined_mid_window: time-series inspection showed these are AEB-first
   events with a small robot tail brake after the AEB phase (usable with
   tail truncation, pending operator confirmation), not robot-initiated
   braking and not D4F4.
no_observed_braking_event rows (98) are skipped: no event window to score.

No driver_intervention label is written or changed anywhere; foot/adjudicate
classes go to an operator review queue only.

Outputs: runs/20260912_abd_aeb_pedal_screen/
- per_run_signature.csv   one row per scanned run
- summary.json            counts, per-vehicle architecture, operator-label
                         agreement, sha256 duplicates, d4f4 diagnostics
- REVIEW_QUEUE.csv        subset requiring operator adjudication
"""
from __future__ import annotations

import csv
import json
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
    EXACT_CHANNELS, _event_window, _header, _stream_selected)
from validate_abd_pedal_signature import classify  # noqa: E402  (thresholds)

SCREENING = ROOT / 'runs/20260912_abd_no_takeover_screen/screening.csv'
REVIEWED = (ROOT / 'runs/20260912_abd_proxy_calibration_v2/'
            'reviewed_aeb_runs.csv')
LABELLED = (ROOT / 'runs/20260912_abd_braking_source_analysis/'
            'labelled_event_features.csv')
OUT_DIR = ROOT / 'runs/20260912_abd_aeb_pedal_screen'

SCAN_STATUSES = ('br_zero_observed_braking', 'robot_channel_active')

PRE_S = 2.0        # baseline window before onset (validated rule)
PRE_CMD_S = 5.0    # wider look-back for the D4F4 light-touch pulse
CMD_EPS = 0.5      # BR Command considered nonzero above this
LIGHT_TOUCH_MAX = 30.0  # provisional "light touch" upper bound (EU)
MID_JOIN_S = 0.5   # robot cmd first active later than this after onset = joined mid-event


def _score_one(task):
    """Worker: parse one run and return its feature/classification record."""
    rel, vehicle, scenario, status = task
    path = ROOT / 'Data/ABD_Data' / rel
    rec = {
        'run': rel, 'vehicle': vehicle, 'scenario': scenario,
        'event_status': status, 'sha256': '', 'operator_label': '',
        'onset_s': '', 'onset_speed_kph': '',
        'travel_pre_mm': '', 'travel_win_mm': '',
        'force_min_pre': '', 'force_max_pre': '',
        'force_min_win': '', 'force_max_win': '',
        'cmd_abs_max_pre2': '', 'cmd_abs_max_pre5': '',
        'cmd_abs_max_win': '', 'cmd_active_frac_win': '',
        'cmd_first_active_rel_s': '',
        'pre_pulse_gap_s': '', 'tension': '', 'foot_pre_event': '',
        'screen_flag_noisy': '',
        'signature': '', 'screen_class': '',
    }
    try:
        _, delimiter, names, _, data_offset = _header(path)
        selected = [i for i, name in enumerate(names) if name in EXACT_CHANNELS]
        arrays, _, _, digest = _stream_selected(path, names, delimiter,
                                                data_offset, selected)
        rec['sha256'] = digest
        time, speed = arrays['Time'], arrays['Speed']
        event = _event_window(time, speed,
                              arrays.get('Forward acceleration (body)'))
        if event is None:
            rec['signature'] = rec['screen_class'] = 'no_event'
            return rec
        onset, end = event['onset_index'], event['end_index']
        dt = float(np.median(np.diff(time)))
        pre0 = max(0, onset - round(PRE_S / dt))
        pre5 = max(0, onset - round(PRE_CMD_S / dt))
        br = arrays.get('BR Position')
        if br is None:
            rec['signature'] = rec['screen_class'] = 'missing_position_channel'
            return rec
        force = arrays.get('Brake force (unfiltered)')
        cmd = arrays['BR Command']

        rec['onset_s'] = round(float(time[onset]), 2)
        rec['onset_speed_kph'] = round(float(speed[onset]), 2)
        rec['travel_pre_mm'] = round(float(np.ptp(br[pre0:onset])), 3)
        rec['travel_win_mm'] = round(float(np.ptp(br[onset:end])), 3)
        if force is not None:
            rec['force_min_pre'] = round(float(force[pre0:onset].min()), 2)
            rec['force_max_pre'] = round(float(force[pre0:onset].max()), 2)
            rec['force_min_win'] = round(float(force[onset:end].min()), 2)
            rec['force_max_win'] = round(float(force[onset:end].max()), 2)
        cmd_win = np.abs(cmd[onset:end])
        pre_mask = cmd[pre5:onset]
        rec['cmd_abs_max_pre2'] = round(float(np.abs(cmd[pre0:onset]).max()), 3)
        rec['cmd_abs_max_pre5'] = round(float(np.abs(pre_mask).max()), 3)
        rec['cmd_abs_max_win'] = round(float(cmd_win.max()), 3)
        rec['cmd_active_frac_win'] = round(
            float(np.mean(cmd_win > CMD_EPS)), 4)
        active = np.flatnonzero(np.abs(pre_mask) > CMD_EPS)
        if len(active):
            # gap between the last pre-onset pulse sample and the onset
            rec['pre_pulse_gap_s'] = round(
                float(time[onset] - time[pre5 + active[-1]]), 3)
        win_active = np.flatnonzero(cmd_win > CMD_EPS)
        if len(win_active):
            rec['cmd_first_active_rel_s'] = round(
                float(time[onset + win_active[0]] - time[onset]), 3)

        # force-based classify needs a force channel; without one the rule
        # degenerates to the travel-only dragged/static split (flagged)
        if force is None:
            rec['signature'] = ('aeb_pedal_static'
                                if rec['travel_win_mm'] <= 2.0
                                else 'aeb_pedal_dragged_no_force')
            classify_ok = False
        else:
            probe = {k: rec[k] for k in
                     ('travel_pre_mm', 'travel_win_mm', 'force_min_pre',
                      'force_max_pre', 'force_min_win', 'force_max_win')}
            classify(probe)
            rec['signature'] = probe['signature']
            rec['tension'] = int(bool(probe['tension']))
            rec['foot_pre_event'] = int(bool(probe['foot_pre_event']))
            classify_ok = True

        if status == 'br_zero_observed_braking':
            if rec['cmd_abs_max_win'] > CMD_EPS:
                # screening said BR-zero but our window sees command activity
                rec['screen_class'] = 'cmd_nonzero_anomaly'
            elif not classify_ok and rec['signature'].endswith('no_force'):
                rec['screen_class'] = 'aeb_signature_no_force_channel'
            else:
                rec['screen_class'] = rec['signature']
        else:  # robot_channel_active -> D4F4 pool
            if (rec['cmd_abs_max_win'] <= CMD_EPS
                    and rec['cmd_abs_max_pre5'] <= CMD_EPS):
                # screen flag fired on BR Command noise (<=0.06 EU observed);
                # the event itself is brake-robot-quiet -> standard rule
                rec['screen_flag_noisy'] = 1
                if not classify_ok and rec['signature'].endswith('no_force'):
                    rec['screen_class'] = 'aeb_signature_no_force_channel'
                else:
                    rec['screen_class'] = rec['signature']
            elif rec['cmd_abs_max_win'] > CMD_EPS:
                first = rec['cmd_first_active_rel_s']
                if first != '' and first > MID_JOIN_S:
                    rec['screen_class'] = 'robot_joined_mid_window'
                else:
                    rec['screen_class'] = 'robot_braked_main_window'
            elif rec['cmd_abs_max_pre5'] > CMD_EPS:
                if rec['cmd_abs_max_pre5'] <= LIGHT_TOUCH_MAX:
                    rec['screen_class'] = 'd4f4_light_touch_then_aeb'
                else:
                    rec['screen_class'] = 'pre_pulse_large_adjudicate'
                # main-window mechanics: drag vs static (foot logic not
                # applicable while the robot touched the pedal pre-onset)
                rec['signature'] = ('aeb_pedal_static'
                                    if rec['travel_win_mm'] <= 2.0
                                    else 'aeb_pedal_dragged')
            else:
                rec['screen_class'] = 'main_quiet_no_pre_pulse'
        return rec
    except Exception as exc:  # keep the scan alive; record the failure
        rec['signature'] = 'parse_error'
        rec['screen_class'] = f'parse_error:{type(exc).__name__}'
        return rec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SCREENING.open(encoding='utf-8-sig') as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['event_status'] in SCAN_STATUSES]
    operator = {}
    if REVIEWED.exists():
        with REVIEWED.open(encoding='utf-8-sig') as handle:
            operator = {r['run']: r['intervention_class']
                        for r in csv.DictReader(handle)}
    labelled = {}
    if LABELLED.exists():
        with LABELLED.open(encoding='utf-8-sig') as handle:
            labelled = {r['run']: r['label'] for r in csv.DictReader(handle)}

    tasks = [(r['run'], r['vehicle'], r['scenario'], r['event_status'])
             for r in rows]
    print(f'scanning {len(tasks)} runs ...', flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_score_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if i % 100 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)

    for rec in out:
        rec['operator_label'] = operator.get(rec['run'],
                                             labelled.get(rec['run'], ''))
    out.sort(key=lambda r: (r['event_status'], r['vehicle'], r['run']))

    fields = list(out[0].keys())
    with (OUT_DIR / 'per_run_signature.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)

    # ---------- summary ----------
    by_status_class = defaultdict(Counter)
    for r in out:
        by_status_class[r['event_status']][r['screen_class']] += 1

    arch = defaultdict(lambda: {'dragged': 0, 'static': 0, 'tension': 0,
                                'foot_or_adj': 0, 'other': 0})
    for r in out:
        # AEB-signature set spans both screen statuses (noisy-flagged
        # robot_channel_active runs are brake-robot-quiet at the event)
        if r['screen_class'] not in ('aeb_pedal_dragged', 'aeb_pedal_static',
                                     'foot_compression',
                                     'drag_then_foot_adjudicate'):
            continue
        v = arch[r['vehicle']]
        if r['screen_class'] in ('aeb_pedal_dragged',):
            v['dragged'] += 1
            v['tension'] += int(bool(r['tension']))
        elif r['screen_class'] == 'aeb_pedal_static':
            v['static'] += 1
        else:
            v['foot_or_adj'] += 1

    agreement = defaultdict(Counter)
    for r in out:
        if r['operator_label']:
            agreement[r['operator_label']][r['screen_class']] += 1

    dup_groups = defaultdict(list)
    for r in out:
        if r['sha256']:
            dup_groups[r['sha256']].append(r['run'])
    duplicates = {h: runs for h, runs in dup_groups.items() if len(runs) > 1}

    d4f4 = [r for r in out if r['screen_class'] == 'd4f4_light_touch_then_aeb']
    d4f4_diag = {
        'n': len(d4f4),
        'cmd_abs_max_pre5': sorted(r['cmd_abs_max_pre5'] for r in d4f4),
        'pre_pulse_gap_s': sorted(r['pre_pulse_gap_s'] for r in d4f4),
        'signature_counts': Counter(r['signature'] for r in d4f4),
    }

    aeb_sig = [r for r in out if r['screen_class'] in ('aeb_pedal_dragged',
                                                       'aeb_pedal_static')]
    aeb_unique = len({r['sha256'] for r in aeb_sig})
    dedup_per_vehicle = Counter()
    seen_sha = set()
    for r in aeb_sig:
        if r['sha256'] and r['sha256'] not in seen_sha:
            seen_sha.add(r['sha256'])
            dedup_per_vehicle[r['vehicle']] += 1

    summary = {
        'scanned': len(out),
        'by_status_class': {k: dict(v) for k, v in by_status_class.items()},
        'per_vehicle_brzero': {k: dict(v) for k, v in sorted(arch.items())},
        'operator_label_agreement': {k: dict(v)
                                     for k, v in agreement.items()},
        'aeb_signature_runs': len(aeb_sig),
        'aeb_signature_unique_sha256': aeb_unique,
        'aeb_signature_unique_per_vehicle': dict(dedup_per_vehicle),
        'sha256_duplicate_groups': duplicates,
        'd4f4_diagnostics': d4f4_diag,
        'thresholds': {
            'static_travel_mm': 2.0, 'foot_force_n': 50.0,
            'tension_n': -5.0, 'pre_foot_travel_mm': 5.0,
            'cmd_eps': CMD_EPS, 'light_touch_max_eu': LIGHT_TOUCH_MAX,
            'pre_cmd_window_s': PRE_CMD_S,
        },
    }
    with (OUT_DIR / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    queue = [r for r in out if r['screen_class'] in (
        'foot_compression', 'drag_then_foot_adjudicate',
        'pre_pulse_large_adjudicate', 'cmd_nonzero_anomaly',
        'robot_joined_mid_window')]
    with (OUT_DIR / 'REVIEW_QUEUE.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)

    print(json.dumps(summary['by_status_class'], indent=2))
    print('review queue:', len(queue))
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
