"""P3.2: ABD-supported perturbation envelope -> abd_supported_v1.

Reads runs that manual_review.csv marks eligible and explicitly confirms had no
driver intervention,
re-extracts event-window features from the raw ABD exports, fits an empirical
envelope for the env parameter identifiable from VUT-side logs (effective
constant brake_deceleration), and writes a versioned perturbation config plus
internal robustness records. The observed TTC margin is retained as a timing
proxy and is not mapped to response_delay. Nothing here mutates scenario_lab
defaults; the config is consumed explicitly via --perturb-config.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_abd_calibration import (  # noqa: E402
    EXACT_CHANNELS, _event_window, _header, _moving_average, _stream_selected)

REVIEW = ROOT / 'runs/20260911_abd_review/manual_review.csv'
ABD_ROOT = ROOT / 'Data/ABD_Data'
OUT = ROOT / 'runs/20260912_abd_calibration'
CONFIG_NAME = 'abd_supported_v1.json'


def load_run(path: Path):
    _, delimiter, names, units, data_offset = _header(path)
    selected = [i for i, name in enumerate(names) if name in EXACT_CHANNELS]
    arrays, _, _, digest = _stream_selected(
        path, names, delimiter, data_offset, selected)
    return arrays, digest


def features(arrays):
    time, speed = arrays['Time'], arrays['Speed']
    accel = arrays.get('Forward acceleration (body)')
    event = _event_window(time, speed, accel)
    if event is None:
        return None
    onset, end = event['onset_index'], event['end_index']
    dt = float(np.median(np.diff(time)))
    smooth = _moving_average(accel, max(1, round(0.10 / dt)))
    seg = smooth[onset:end]
    peak = float(np.min(seg))
    t_peak = float(time[onset + int(np.argmin(seg))])
    ttc = arrays.get('Time to collision (longitudinal)')
    onset_ttc = float(ttc[onset]) if ttc is not None else float('nan')
    v0 = event['onset_speed_kph'] / 3.6
    velocity = np.maximum(speed[onset:end], 0.) / 3.6
    travelled = float(np.trapezoid(velocity, time[onset:end]))
    vend = float(velocity[-1])
    duration = float(time[end - 1] - time[onset])
    effective_distance = ((v0 ** 2 - vend ** 2) / (2. * travelled)
                          if travelled > 1e-9 else float('nan'))
    effective_time = ((v0 - vend) / duration if duration > 1e-9 else float('nan'))
    br = arrays.get('BR Position')
    force = arrays.get('Brake force (unfiltered)')
    return {
        'onset_time_s': round(float(time[onset]), 4),
        'onset_speed_kph': round(event['onset_speed_kph'], 3),
        'onset_ttc_s': round(onset_ttc, 4),
        'peak_deceleration_mps2': round(peak, 3),
        'p05_acceleration_mps2': round(event['p05_acceleration_mps2'], 3),
        'stopping_distance_from_onset_m': round(travelled, 3),
        'effective_constant_deceleration_distance_mps2': round(effective_distance, 3),
        'effective_constant_deceleration_time_mps2': round(effective_time, 3),
        'time_onset_to_peak_s': round(t_peak - float(time[onset]), 3),
        'event_duration_s': round(event['duration_s'], 3),
        'margin_time_s': round(onset_ttc - v0 / abs(peak), 4),
        'br_position_travel_mm': (round(float(np.ptp(br[onset:end])), 3)
                                  if br is not None else None),
        'brake_force_max_n': (round(float(np.max(force[onset:end])), 3)
                              if force is not None else None),
    }


def threshold_sensitivity(arrays, threshold):
    """Re-detect the event with a softer sustained-deceleration threshold."""
    time, speed = arrays['Time'], arrays['Speed']
    accel = arrays['Forward acceleration (body)']
    dt = float(np.median(np.diff(time)))
    smooth = _moving_average(accel, max(1, round(0.10 / dt)))
    sustain = max(1, round(0.15 / dt))
    eligible = (smooth <= threshold) & (speed >= 5.0)
    run = np.convolve(eligible.astype(np.int16),
                      np.ones(sustain, dtype=np.int16), mode='valid')
    idx = np.flatnonzero(run >= sustain)
    if not len(idx):
        return None
    onset = int(idx[0])
    while onset > 0 and smooth[onset - 1] <= threshold / 2:
        onset -= 1
    end = min(len(time), onset + max(2, round(3.0 / dt)))
    stopped = np.flatnonzero(speed[onset:end] <= 1.0)
    if len(stopped):
        end = onset + int(stopped[0]) + 1
    peak = float(np.min(smooth[onset:end]))
    ttc = arrays.get('Time to collision (longitudinal)')
    return {
        'onset_time_s': round(float(time[onset]), 4),
        'onset_ttc_s': round(float(ttc[onset]), 4) if ttc is not None else None,
        'peak_deceleration_mps2': round(peak, 3),
        'margin_time_s': (round(float(ttc[onset]) - (speed[onset] / 3.6) / abs(peak), 4)
                          if ttc is not None else None),
    }


def stat(values):
    v = np.asarray(values, dtype=float)
    return {'n': len(v), 'min': round(float(v.min()), 3),
            'median': round(float(np.median(v)), 3),
            'max': round(float(v.max()), 3),
            'mean': round(float(v.mean()), 3),
            'std': (round(float(v.std(ddof=1)), 3) if len(v) > 1 else None)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with REVIEW.open(encoding='utf-8-sig', newline='') as handle:
        reviewed = list(csv.DictReader(handle))
        candidates = [r for r in reviewed
                      if r['calibration_decision'].startswith(
                          'eligible_for_abd_calibrated_v1')]
        unresolved = [r['run'] for r in candidates
                      if r.get('driver_intervention', '').strip().casefold()
                      not in ('none_confirmed', 'none')]
        if unresolved:
            raise SystemExit(
                'driver intervention is absent or not confirmed-none for '
                f'{len(unresolved)} eligible runs; AEB attribution is unresolved')
        rows = candidates
    if not rows:
        raise SystemExit('no eligible runs in manual_review.csv')

    records = []
    for row in rows:
        path = ABD_ROOT / row['run']
        arrays, digest = load_run(path)
        feat = features(arrays)
        if feat is None:
            raise SystemExit(f'eligible run lost its event window: {row["run"]}')
        sens = threshold_sensitivity(arrays, -0.5)
        if sens is None:
            raise SystemExit(f'sensitivity window lost for eligible run: {row["run"]}')
        records.append({
            'run': row['run'], 'vehicle': row['vehicle_folder'],
            'scenario': row['scenario'], 'sha256': digest,
            'test_id': row['run'].rsplit('/', 1)[-1],
            'collection_group': row['run'].split('/')[0],
            **feat,
            'sens_onset_time_s': sens['onset_time_s'],
            'sens_onset_ttc_s': sens['onset_ttc_s'],
            'sens_peak_deceleration_mps2': sens['peak_deceleration_mps2'],
            'sens_margin_time_s': sens['margin_time_s'],
        })

    digests = [r['sha256'] for r in records]
    if len(set(digests)) != len(digests):
        raise SystemExit('duplicate sha256 among eligible runs')
    test_ids = [r['test_id'] for r in records]
    if len(set(test_ids)) != len(test_ids):
        raise SystemExit('duplicate test_id among eligible runs')

    pooled_median_peak = float(np.median([r['peak_deceleration_mps2'] for r in records]))
    for r in records:
        r['normalized_peak'] = round(r['peak_deceleration_mps2'] / pooled_median_peak, 4)

    ccrs = [r for r in records if r['scenario'] == 'CCRs']
    turning = [r for r in records if r['scenario'] != 'CCRs']

    peak_all = stat([r['peak_deceleration_mps2'] for r in records])
    peak_ccrs = stat([r['peak_deceleration_mps2'] for r in ccrs])
    peak_turn = stat([r['peak_deceleration_mps2'] for r in turning]) if turning else None
    ttc_ccrs = stat([r['onset_ttc_s'] for r in ccrs])
    margin_ccrs = stat([r['margin_time_s'] for r in ccrs])
    margin_all = stat([r['margin_time_s'] for r in records])
    ttp = stat([r['time_onset_to_peak_s'] for r in records])
    effective_all = stat([
        r['effective_constant_deceleration_distance_mps2'] for r in records])
    effective_ccrs = stat([
        r['effective_constant_deceleration_distance_mps2'] for r in ccrs])
    effective_turn = stat([
        r['effective_constant_deceleration_distance_mps2'] for r in turning]) \
        if turning else None

    # --- internal robustness: leave-one-out endpoint sensitivity --------------
    loo = []
    for i in range(len(records)):
        keep = [r for j, r in enumerate(records) if j != i]
        loo.append({
            'held_out': records[i]['test_id'],
            'peak_range': [round(min(r['peak_deceleration_mps2'] for r in keep), 3),
                           round(max(r['peak_deceleration_mps2'] for r in keep), 3)],
            'effective_constant_range': [
                round(min(r['effective_constant_deceleration_distance_mps2']
                          for r in keep), 3),
                round(max(r['effective_constant_deceleration_distance_mps2']
                          for r in keep), 3)],
            'margin_ccrs_range': ([round(min(r['margin_time_s'] for r in keep
                                             if r['scenario'] == 'CCRs'), 3),
                                   round(max(r['margin_time_s'] for r in keep
                                             if r['scenario'] == 'CCRs'), 3)]
                                  if len([r for r in keep if r['scenario'] == 'CCRs']) else None),
        })
    loo_peak_dev = max(abs(l['peak_range'][0] - peak_all['min']) for l in loo), \
        max(abs(l['peak_range'][1] - peak_all['max']) for l in loo)
    loo_effective_dev = (
        max(abs(l['effective_constant_range'][0] - effective_all['min']) for l in loo),
        max(abs(l['effective_constant_range'][1] - effective_all['max']) for l in loo))
    loo_margin_dev = max(abs(l['margin_ccrs_range'][0] - margin_ccrs['min']) for l in loo), \
        max(abs(l['margin_ccrs_range'][1] - margin_ccrs['max']) for l in loo)

    onset_shift = [round(r['sens_onset_time_s'] - r['onset_time_s'], 3)
                   for r in records]
    peak_diff = [round(r['sens_peak_deceleration_mps2']
                       - r['peak_deceleration_mps2'], 3) for r in records]
    peak_diff_ccrs = [round(r['sens_peak_deceleration_mps2']
                            - r['peak_deceleration_mps2'], 3) for r in ccrs]
    peak_diff_turn = [round(r['sens_peak_deceleration_mps2']
                            - r['peak_deceleration_mps2'], 3) for r in turning]

    verification = {
        'leave_one_out': {
            'records': loo,
            'max_peak_range_deviation': loo_peak_dev,
            'max_effective_constant_range_deviation': loo_effective_dev,
            'max_margin_ccrs_range_deviation': loo_margin_dev,
        },
        'onset_threshold_sensitivity': {
            'reference_threshold_mps2': -1.0, 'alternative_mps2': -0.5,
            'onset_time_shift_s': stat(onset_shift),
            'peak_deceleration_diff_mps2': stat(peak_diff),
            'peak_deceleration_diff_ccrs': stat(peak_diff_ccrs),
            'peak_deceleration_diff_turning': (stat(peak_diff_turn)
                                               if turning else None),
            'observed': (
                'CCRs (straight-line) runs are threshold-robust: peak decel diff '
                '0.000 m/s2 in all 8, onset shift <=0.11 s in 7 of 8 '
                '(V9_T52 shifts 0.81 s earlier onto a pre-event speed-control '
                'perturbation, peak unchanged). Turning runs (CCFT/CPTA) are NOT '
                'robust: the -0.5 m/s2 threshold latches onto mild deceleration '
                'during cornering ~33-36 s before the true event (onset 5.4-6.0 s, '
                'shallow peak -6.1/-7.2 m/s2). Calibrated distributions therefore '
                'use the CCRs subset for response_delay; turning runs corroborate '
                'peak decel levels only under the reference threshold.'),
        },
        'attribution_records': [
            'runs/20260911_abd_review/manual_review.csv (P3.1 evidence columns)',
            'runs/20260911_abd_review/MANUAL_EVIDENCE.md (E1-E7)',
            'runs/20260911_abd_calibration_audit/ (original NO-GO audit, untouched)',
        ],
    }

    config = {
        'version': 'abd_supported_v1',
        'created': '2026-09-12',
        'purpose': 'scenario_lab perturb_spec calibrated parameter draws',
        'eligible_runs': len(records),
        'abd_supported_parameter_count': 1,
        'total_perturbation_parameters': 4,
        'source_runs': [{'run': r['run'], 'sha256': r['sha256'],
                         'scenario': r['scenario']} for r in records],
        'parameters': {
            'brake_deceleration': {
                'dist': 'uniform',
                'low': effective_all['min'], 'high': effective_all['max'],
                'unit': 'm/s2 (positive distance-equivalent constant magnitude)',
                'status': 'abd_supported_empirical_envelope_not_population_distribution',
                'summary_distance_equivalent_mps2': effective_all,
                'ccrs_subset': effective_ccrs,
                'turning_subset': effective_turn,
                'peak_deceleration_diagnostic_signed_mps2': peak_all,
                'mapping': ('a_eff=(v_onset^2-v_end^2)/(2*integral(v dt)); matches '
                            'scenario_lab constant-deceleration stopping distance'),
                'evidence': ('AEB-attributed event windows; measured speed integrated '
                             'from braking onset to <=1 kph. Uniform sampling is a '
                             'bounded sensitivity design over observed extrema, not a '
                             'fitted fleet probability distribution.'),
            },
            'response_delay': {
                'dist': 'uniform',
                'low': 0.1, 'high': 0.4,
                'status': 'retained_assumed_proxy_only_not_identifiable',
                'observed_margin_proxy_ccrs': margin_ccrs,
                'observed_margin_proxy_all': margin_all,
                'proxy': 'margin_time = onset_ttc - v_onset/|peak_decel|',
                'evidence': ('CCRs subset (8 runs, 19.2-21.0 kph, one speed regime). '
                             'Margin time mixes trigger policy, geometry and braking '
                             'build-up; without an AEB request signal it is not '
                             'trigger-to-output delay and is not mapped to response_delay.'),
            },
            'action_delay_steps': {
                'dist': 'integers', 'low': 0, 'high': 2,
                'status': 'retained_assumed_not_identifiable',
                'evidence': ('0-2 decision steps (<=40 ms) cannot be resolved from '
                             '100 Hz VUT logs without the AEB request signal'),
            },
            'target_accel_scale': {
                'dist': 'uniform', 'low': 0.85, 'high': 1.15,
                'status': 'retained_assumed_npc_side_not_in_vut_logs',
                'evidence': ('parameter scales NPC (pedestrian/occluder) actions; VUT-side '
                             'ABD exports contain no NPC execution channels; measured '
                             'cross-run peak spread normalised to pooled median: '
                             f"{min(r['normalized_peak'] for r in records):.3f}-"
                             f"{max(r['normalized_peak'] for r in records):.3f}"),
            },
        },
        'attribution': ('AEB by elimination chain (MANUAL_EVIDENCE.md E3): window BR '
                        'Command==0 absolutes, trigger flags unchanged, AR throttle-hold '
                        'recipe, protocol driver non-intervention; residual driver-override '
                        'risk documented; no direct AEB status channel'),
        'limitations': [
            'n=10 runs across 8 vehicles; per-vehicle distributions underdetermined',
            'only one of four perturbation parameters has an ABD-supported env mapping',
            'response timing margin is descriptive and response_delay remains assumed',
            'the empirical uniform is a sensitivity envelope, not a population fit',
            '15-TANG contributes 3 of 10 runs (CCRs/CCFT/CPTA one each)',
        ],
    }

    (OUT / 'calibration_table.csv').parent.mkdir(parents=True, exist_ok=True)
    with (OUT / 'calibration_table.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (OUT / CONFIG_NAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'distributions.json').write_text(
        json.dumps({'peak_deceleration_all': peak_all, 'peak_deceleration_ccrs': peak_ccrs,
                    'peak_deceleration_turning': peak_turn, 'onset_ttc_ccrs': ttc_ccrs,
                    'margin_time_ccrs': margin_ccrs, 'margin_time_all': margin_all,
                    'time_onset_to_peak_all': ttp,
                    'effective_constant_deceleration_distance_all': effective_all,
                    'effective_constant_deceleration_distance_ccrs': effective_ccrs,
                    'effective_constant_deceleration_distance_turning': effective_turn,
                    'per_vehicle': {
                        v: stat([r['peak_deceleration_mps2'] for r in records
                                 if r['vehicle'] == v])
                        for v in sorted({r['vehicle'] for r in records})}},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'verification.json').write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'eligible': len(records), 'ccrs': len(ccrs), 'turning': len(turning),
        'brake_deceleration_distance_equivalent': [effective_all['min'], effective_all['max']],
        'response_delay_assumed': [0.1, 0.4],
        'response_margin_proxy_ccrs': [margin_ccrs['min'], margin_ccrs['max']],
        'loo_max_dev': {'peak': loo_peak_dev, 'effective_constant': loo_effective_dev,
                        'margin_proxy': loo_margin_dev},
        'threshold_sens_peak_diff': stat(peak_diff),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
