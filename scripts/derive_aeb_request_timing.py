"""P3: back-calculate synthetic AEB request/active timestamps for aeb_dataset_v1.

No vehicle CAN was connected in any test, so AEB request/active signals were
never recorded. Operator decision (2026-09-13): the method pipeline does not
need them -- derive request time by back-calculating a fixed actuation delay
of 0.15-0.35 s from the observed braking onset:

    active  = onset of the detected deceleration event (what the log shows)
    request = onset - delay, delay in {0.15, 0.25, 0.35} (sensitivity bands)

This unblocks request-to-response delay calibration *as a method*; the delay
value itself is an injected prior, not a measurement, and every consumer must
keep the synthetic-request labeling.

Per run over aeb_dataset_v1 (680 unique pedal-signature rows):
- re-detect the event window (audit_abd_calibration._event_window)
- peak deceleration, stop detection (<=1 kph within 10 s of onset),
  distance-equivalent constant deceleration a_eff=(v0^2-v1^2)/(2*integral v dt)
  (same mapping as abd_supported_v1.json brake_deceleration)
- onset TTC (if the channel is present and sane) and margin proxy
  onset_ttc - v_onset/|peak|
- contact_outcome.csv join for stratified summaries (contact runs get their
  deceleration partly from the collision -> excluded from the config envelope)

Outputs runs/20260913_abd_request_timing/:
- timing.csv                 per-run timestamps and diagnostics
- abd_derived_v2_sensitivity.json  perturb_spec-consumable config
  (brake_deceleration = v1 measured envelope excluding contact runs;
   response_delay = U(0.15, 0.35) operator back-calculation prior)
- summary.json               distributions
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_abd_calibration import _event_window, _header, _stream_selected  # noqa: E402

V1 = ROOT / 'runs/20260912_abd_aeb_pedal_screen/aeb_dataset_v1.csv'
CONTACT = ROOT / 'runs/20260912_abd_aeb_pedal_screen/contact_outcome.csv'
OUT_DIR = ROOT / 'runs/20260913_abd_request_timing'

TTC_CH = 'Time to collision (longitudinal)'
NEED = {'Time', 'Speed', 'Forward acceleration (body)', TTC_CH}
STOP_KPH = 1.0        # stop detection threshold (same as abd_supported_v1)
STOP_HORIZON_S = 10.0
BACKCALC_S = (0.15, 0.25, 0.35)   # operator-prior actuation-delay bands
TTC_SANE = (0.0, 10.0)            # outside -> treated as missing


def _score(task):
    rel, vehicle, scenario, sha256, outcome = task
    path = ROOT / 'Data/ABD_Data' / rel
    rec = {'run': rel, 'vehicle': vehicle, 'scenario': scenario,
           'sha256': sha256, 'outcome': outcome,
           'onset_s': '', 'onset_speed_kph': '', 'peak_decel_ms2': '',
           'stopped': '', 'stop_rel_s': '', 'end_speed_kph': '',
           'equiv_decel_ms2': '', 'onset_ttc_s': '', 'margin_time_s': '',
           'active_s': '',
           **{f'request_s_d{int(d*1000):03d}': '' for d in BACKCALC_S}}
    try:
        _, delimiter, names, _, data_offset = _header(path)
        selected = [i for i, name in enumerate(names) if name in NEED]
        arrays, _, _, _ = _stream_selected(path, names, delimiter,
                                           data_offset, selected)
        time, speed = arrays['Time'], arrays['Speed']
        accel = arrays.get('Forward acceleration (body)')
        ev = _event_window(time, speed, accel)
        if ev is None:
            rec['stopped'] = 'no_event'
            return rec
        onset = ev['onset_index']
        dt = float(np.median(np.diff(time)))
        v0 = float(speed[onset])

        # deceleration phase: onset until stop (<=1 kph) or horizon
        horizon = min(len(time), onset + round(STOP_HORIZON_S / dt))
        stop_idx = None
        for i in range(onset, horizon):
            if speed[i] <= STOP_KPH:
                stop_idx = i
                break
        end = stop_idx if stop_idx is not None else horizon - 1
        acc = accel[onset:end + 1] if accel is not None else np.array([])
        peak = float(acc.min()) if len(acc) else float('nan')

        rec['onset_s'] = round(float(time[onset]), 3)
        rec['onset_speed_kph'] = round(v0, 2)
        rec['peak_decel_ms2'] = round(peak, 3)
        rec['stopped'] = int(stop_idx is not None)
        rec['stop_rel_s'] = (round(float(time[stop_idx] - time[onset]), 3)
                             if stop_idx is not None else '')
        rec['end_speed_kph'] = round(float(speed[end]), 2)
        if stop_idx is not None and v0 > STOP_KPH:
            seg_v = speed[onset:stop_idx + 1] / 3.6
            seg_t = time[onset:stop_idx + 1]
            dist = float(np.trapezoid(seg_v, seg_t))
            if dist > 0.5:
                v_end = float(seg_v[-1])
                rec['equiv_decel_ms2'] = round(
                    (float(seg_v[0]) ** 2 - v_end ** 2) / (2 * dist), 3)

        ttc = arrays.get(TTC_CH)
        ttc0 = float(ttc[onset]) if ttc is not None else float('nan')
        if TTC_SANE[0] < ttc0 < TTC_SANE[1] and np.isfinite(peak) and peak < 0:
            rec['onset_ttc_s'] = round(ttc0, 3)
            rec['margin_time_s'] = round(
                ttc0 - (v0 / 3.6) / abs(peak), 3)

        rec['active_s'] = rec['onset_s']
        for d in BACKCALC_S:
            rec[f'request_s_d{int(d*1000):03d}'] = round(
                float(time[onset]) - d, 3)
        return rec
    except Exception as exc:
        rec['stopped'] = f'parse_error:{type(exc).__name__}'
        return rec


def _stats(values):
    if not values:
        return {'n': 0}
    v = sorted(values)
    return {'n': len(v), 'min': round(v[0], 3),
            'p5': round(v[max(0, int(.05 * len(v)) - 1)], 3),
            'median': round(v[len(v) // 2], 3),
            'p95': round(v[min(len(v) - 1, int(.95 * len(v)))], 3),
            'max': round(v[-1], 3)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1 = list(csv.DictReader(V1.open(encoding='utf-8-sig')))
    outcome = {r['run']: r['outcome'] for r in
               csv.DictReader(CONTACT.open(encoding='utf-8-sig'))}
    tasks = [(r['run'], r['vehicle'], r['scenario'], r['sha256'],
              outcome.get(r['run'], '')) for r in v1]
    print(f'timing derivation over {len(tasks)} v1 runs ...', flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_score, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if i % 100 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    order = {t[0]: i for i, t in enumerate(tasks)}
    out.sort(key=lambda r: order[r['run']])
    with (OUT_DIR / 'timing.csv').open('w', encoding='utf-8-sig',
                                       newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    stopped = [r for r in out if str(r['stopped']) == '1']
    clean_stop = [r for r in stopped if r['outcome'] != 'contact_and_stopped'
                  and r['equiv_decel_ms2'] != '']
    equiv_clean = [float(r['equiv_decel_ms2']) for r in clean_stop]
    equiv_all = [float(r['equiv_decel_ms2']) for r in stopped
                 if r['equiv_decel_ms2'] != '']
    peak = [float(r['peak_decel_ms2']) for r in out
            if r['peak_decel_ms2'] != '']
    margin = [float(r['margin_time_s']) for r in out
              if r['margin_time_s'] != '']
    ttc_cov = sum(1 for r in out if r['onset_ttc_s'] != '')

    summary = {
        'n_runs': len(out),
        'parse_errors': sum(1 for r in out
                            if str(r['stopped']).startswith('parse_error')
                            or str(r['stopped']) == 'no_event'),
        'stopped_within_10s': len(stopped),
        'equiv_decel_clean_stop': _stats(equiv_clean),
        'equiv_decel_all_stopped': _stats(equiv_all),
        'peak_decel_signed': _stats(peak),
        'margin_time_proxy': _stats(margin),
        'ttc_coverage': ttc_cov,
        'backcalc_delay_bands_s': list(BACKCALC_S),
        'outcome_counts': dict(Counter(r['outcome'] for r in out)),
    }
    (OUT_DIR / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    env = summary['equiv_decel_clean_stop']
    if env['n'] == 0:
        raise SystemExit('no clean stopped runs with equiv deceleration; '
                         'refusing to write a brake_deceleration envelope')
    config = {
        'version': 'abd_derived_v2_sensitivity',
        'created': '2026-09-13',
        'purpose': 'scenario_lab perturb_spec calibrated parameter draws',
        'dataset': 'aeb_dataset_v1 (680 unique pedal-signature runs; '
                   'operator residual-risk removals applied)',
        'parameters': {
            'brake_deceleration': {
                'dist': 'uniform',
                'low': env['p5'],
                'high': env['p95'],
                'unit': 'm/s2 (positive distance-equivalent constant magnitude)',
                'status': 'abd_v1_measured_envelope_p5_p95_excluding_contact_runs',
                'summary_distance_equivalent_mps2': env,
                'n_source_runs': env['n'],
                'mapping': 'a_eff=(v_onset^2-v_end^2)/(2*integral(v dt)); '
                           'stop <=1 kph within 10 s of onset',
                'evidence': 'aeb_dataset_v1 stopped runs with outcome != '
                            'contact_and_stopped (contact deceleration is '
                            'partly collision dynamics). Sampling bounds are '
                            'the 5-95 percentile envelope: at n=612 the raw '
                            'minimum (1.405 m/s2, 10 runs <4.5) reflects '
                            'segmented intermittent braking (CPLA/CPTA '
                            'decel-release-restop) diluting the full-window '
                            'average, not weak actuators; full extrema are '
                            'recorded in summary_distance_equivalent_mps2. '
                            'Sensitivity envelope, not a fitted fleet '
                            'distribution.'
            },
            'response_delay': {
                'dist': 'uniform',
                'low': 0.15,
                'high': 0.35,
                'status': 'synthetic_backcalculation_operator_prior',
                'backcalculation': 'request_time = observed_onset - delay; '
                                   'active_time = observed_onset. Bands '
                                   '{0.15, 0.25, 0.35} s tabulated per run in '
                                   'timing.csv',
                'evidence': 'operator decision 2026-09-13: no vehicle CAN / '
                            'AEB request channel exists; the 0.15-0.35 s '
                            'actuation-delay prior is INJECTED, not measured. '
                            'This config unblocks the request-to-response '
                            'calibration METHOD; consumers must keep the '
                            'synthetic-request labeling and never report the '
                            'delay as an AEB timing measurement',
                'observed_margin_proxy_v1': summary['margin_time_proxy']
            },
            'action_delay_steps': {
                'dist': 'integers',
                'low': 0,
                'high': 2,
                'status': 'retained_assumed_not_identifiable',
                'evidence': '0-2 decision steps (<=40 ms) below 100 Hz log '
                            'resolution without the AEB request signal'
            },
            'target_accel_scale': {
                'dist': 'uniform',
                'low': 0.85,
                'high': 1.15,
                'status': 'retained_assumed_npc_side_not_in_vut_logs',
                'evidence': 'parameter scales NPC actions; VUT-side exports '
                            'have tracker reference/actual channels but no '
                            'NPC execution-error mapping was fitted'
            }
        },
        'limitations': [
            'response_delay is an injected prior; timing.csv request columns '
            'are synthetic, derived by subtracting fixed bands from onset',
            'brake_deceleration bounds are p5-p95 of 612 clean stopped runs; '
            'contact runs excluded (collision dynamics); 10 runs below '
            '4.5 m/s2 are segmented-braking dilution, kept in the recorded '
            'extrema but outside the sampling bounds',
            'equiv deceleration assumes constant-deceleration stopping '
            'distance; mid-event AEB release runs still included when they '
            'stopped within 10 s',
            'margin_time_proxy has negative tail (p5 -0.349 s): trigger '
            'occurred later than the ideal braking point at some speed '
            'regimes; descriptive only',
            'this is a method-pipeline deliverable, not an AEB timing '
            'measurement'
        ]
    }
    (OUT_DIR / 'abd_derived_v2_sensitivity.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

    # validate against the scenario_lab loader before declaring done
    sys.path.insert(0, str(ROOT))
    from scenario_lab.train import load_perturb_config  # noqa: E402
    load_perturb_config(OUT_DIR / 'abd_derived_v2_sensitivity.json')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('config passes load_perturb_config validation')
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
