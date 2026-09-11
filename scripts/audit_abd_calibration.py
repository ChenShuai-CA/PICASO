"""Read-only audit of ABD braking-response evidence.

This script characterises measured kinematics and robot channels.  It never
turns a path name, an abort flag, or a deceleration trace into an AEB or
collision label.  Full run files are streamed and only a small, named channel
subset is retained in memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_lab.data import inspect_abd


EXACT_CHANNELS = {
    'Time', 'Speed', 'Desired Speed', 'Forward acceleration',
    'Forward acceleration (body)', 'Brake force (unfiltered)', 'BR Command',
    'BR Position', 'BR Velocity', 'BR start', 'BR end', 'BR test',
    'Motion Going BR', 'AR Command', 'AR start', 'AR end', 'Test phase',
    'Critical section start', 'Critical section end',
    'Time tolerance 1', 'Time to collision (longitudinal)',
    'Relative longitudinal distance', 'Relative longitudinal velocity',
}

DISCOVERY_TERMS = (
    'aeb', 'fcw', 'brake', 'decel', 'acceleration', 'pedal', 'pressure',
    'speed', 'time to collision', 'relative longitudinal', 'test phase',
    'critical section',
)


def _float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def _header(path: Path):
    with path.open('r', encoding='latin-1', errors='replace') as handle:
        first = [next(handle, '').rstrip('\r\n') for _ in range(4)]
    declared = None
    for line in first[:3]:
        if line.startswith('Points='):
            try:
                declared = int(float(line.partition('=')[2]))
            except ValueError:
                pass
    names_i = next((i for i, line in enumerate(first) if '\t' in line or ',' in line), None)
    if names_i is None:
        raise ValueError(f'{path}: no channel header')
    delimiter = '\t' if '\t' in first[names_i] else ','
    names = [x.strip() for x in first[names_i].split(delimiter)]
    units = [None] * len(names)
    data_offset = names_i + 1
    if data_offset < len(first):
        fields = first[data_offset].split(delimiter)
        if len(fields) == len(names) and not all(math.isfinite(_float(x)) for x in fields):
            units = [x.strip() or None for x in fields]
            data_offset += 1
    return declared, delimiter, names, units, data_offset


def _stream_selected(path: Path, names, delimiter, data_offset, selected):
    values = {i: [] for i in selected}
    total = bad = 0
    digest = hashlib.sha256()
    with path.open('rb') as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b''):
            digest.update(block)
    with path.open('r', encoding='latin-1', errors='replace') as handle:
        for _ in range(data_offset):
            next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip('\r\n').split(delimiter)
            if len(fields) != len(names):
                bad += 1
                continue
            parsed = [_float(fields[i]) for i in selected]
            if not all(math.isfinite(x) for x in parsed):
                bad += 1
                continue
            for i, value in zip(selected, parsed):
                values[i].append(value)
            total += 1
    arrays = {names[i]: np.asarray(values[i], dtype=float) for i in selected}
    return arrays, total, bad, digest.hexdigest()


def _moving_average(x, width):
    if width <= 1:
        return x.astype(float, copy=True)
    width = min(width, len(x))
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(x, (width // 2, width - 1 - width // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _first_sustained(mask, length):
    if length <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else None
    run = np.convolve(mask.astype(np.int16), np.ones(length, dtype=np.int16), mode='valid')
    idx = np.flatnonzero(run >= length)
    return int(idx[0]) if len(idx) else None


def _event_window(time, speed_kph, accel):
    if len(time) < 20:
        return None
    dt = float(np.median(np.diff(time)))
    if not math.isfinite(dt) or dt <= 0:
        return None
    smooth = _moving_average(accel, max(1, round(0.10 / dt)))
    sustain = max(1, round(0.15 / dt))
    horizon = max(2, round(2.0 / dt))
    eligible = (smooth <= -1.0) & (speed_kph >= 5.0)
    starts = []
    work = eligible.copy()
    while True:
        found = _first_sustained(work, sustain)
        if found is None:
            break
        end = min(len(speed_kph), found + horizon)
        # Small signed-speed overshoot around standstill is a sensor/path-frame
        # artifact, not additional braking speed.  Clamp only the drop metric;
        # the raw minimum remains available below.
        drop = float(speed_kph[found] - max(0.0, np.min(speed_kph[found:end])))
        starts.append((drop, found))
        work[found:found + sustain] = False
    if not starts:
        return None
    drop, onset = max(starts, key=lambda item: (item[0], -item[1]))
    while onset > 0 and smooth[onset - 1] <= -0.30:
        onset -= 1
    end = min(len(time), onset + max(2, round(3.0 / dt)))
    stopped = np.flatnonzero(speed_kph[onset:end] <= 1.0)
    if len(stopped):
        end = onset + int(stopped[0]) + 1
    segment = slice(onset, end)
    return {
        'onset_index': onset,
        'end_index': end,
        'onset_time_s': float(time[onset]),
        'end_time_s': float(time[end - 1]),
        'duration_s': float(time[end - 1] - time[onset]),
        'onset_speed_kph': float(speed_kph[onset]),
        'speed_drop_2s_kph': drop,
        'minimum_speed_kph': float(np.min(speed_kph[segment])),
        'peak_deceleration_mps2': float(np.min(smooth[segment])),
        'p05_acceleration_mps2': float(np.quantile(smooth[segment], 0.05)),
    }


def _signal_summary(values, baseline, event):
    result = {
        'min': float(np.min(values)), 'max': float(np.max(values)),
        'median': float(np.median(values)),
        'nonzero_fraction': float(np.mean(np.abs(values) > 1e-9)),
    }
    if event is not None:
        onset, end = event['onset_index'], event['end_index']
        before = values[max(0, onset - baseline):onset]
        during = values[onset:end]
        if len(before) and len(during):
            base = float(np.median(before))
            result.update(
                baseline_median=base,
                event_min=float(np.min(during)),
                event_max=float(np.max(during)),
                event_max_abs_delta=float(np.max(np.abs(during - base))),
            )
    return result


def analyze_run(path: Path):
    path = Path(path)
    declared, delimiter, names, units, data_offset = _header(path)
    discovered = [
        {'index': i, 'name': name, 'unit': units[i]}
        for i, name in enumerate(names)
        if any(term in name.casefold() for term in DISCOVERY_TERMS)
    ]
    selected = [i for i, name in enumerate(names) if name in EXACT_CHANNELS]
    arrays, rows, bad_rows, digest = _stream_selected(
        path, names, delimiter, data_offset, selected)
    sampled = inspect_abd(path, max_rows=1)
    config = sampled['config']
    result = {
        'path': str(path), 'sha256': digest, 'points_declared': declared,
        'rows_parsed': rows, 'bad_rows': bad_rows,
        'row_count_matches': declared == rows if declared is not None else None,
        'n_channels': len(names), 'discovered_channels': discovered,
        'config': config, 'event': None, 'signals': {},
        'aeb_event_source': 'unconfirmed', 'collision_label': None,
        'calibration_status': 'insufficient_kinematics',
    }
    if 'Time' not in arrays or 'Speed' not in arrays or len(arrays['Time']) < 2:
        return result
    time = arrays['Time']
    speed = arrays['Speed']
    accel_name = next((name for name in (
        'Forward acceleration (body)', 'Forward acceleration') if name in arrays), None)
    if accel_name is None:
        dt = np.gradient(time)
        accel = np.gradient(speed / 3.6) / dt
        accel_name = 'derived_from_speed'
    else:
        accel = arrays[accel_name]
    event = _event_window(time, speed, accel)
    result['event'] = event
    result['acceleration_source'] = accel_name
    result['time'] = {
        'start_s': float(time[0]), 'end_s': float(time[-1]),
        'dt_median_s': float(np.median(np.diff(time))),
        'monotonic': bool(np.all(np.diff(time) > 0)),
    }
    baseline = max(1, round(1.0 / result['time']['dt_median_s']))
    for name, values in arrays.items():
        if name != 'Time':
            result['signals'][name] = _signal_summary(values, baseline, event)
    if event is None:
        result['calibration_status'] = (
            'excluded_brake_robot_enabled_no_clear_response'
            if config.get('brake_robot_engaged') is True else 'no_clear_braking_response')
    elif config.get('brake_robot_engaged') is True:
        result['calibration_status'] = 'excluded_brake_robot_enabled_ambiguous_source'
    elif config.get('brake_robot_engaged') is False:
        result['calibration_status'] = 'response_candidate_source_unconfirmed'
    else:
        result['calibration_status'] = 'response_candidate_control_unknown'
    if event is not None and 'Time to collision (longitudinal)' in arrays:
        result['event']['onset_ttc_s'] = float(
            arrays['Time to collision (longitudinal)'][event['onset_index']])
    return result


def _flat_row(source, detail, root):
    event = detail.get('event') or {}
    config = detail['config']
    signals = detail['signals']
    def sig(name, field):
        return signals.get(name, {}).get(field, '')
    return {
        'run': source['run'], 'vehicle_folder': source['vehicle_folder'],
        'scenario': source['scenario'], 'sha256': detail['sha256'],
        'points_declared': detail['points_declared'], 'rows_parsed': detail['rows_parsed'],
        'bad_rows': detail['bad_rows'], 'row_count_matches': detail['row_count_matches'],
        'motion_control': config.get('motion_control'),
        'brake_control': config.get('brake_control'),
        'use_brake_robot': config.get('use_brake_robot'),
        'event_onset_s': event.get('onset_time_s', ''),
        'event_end_s': event.get('end_time_s', ''),
        'onset_speed_kph': event.get('onset_speed_kph', ''),
        'speed_drop_2s_kph': event.get('speed_drop_2s_kph', ''),
        'peak_deceleration_mps2': event.get('peak_deceleration_mps2', ''),
        'p05_acceleration_mps2': event.get('p05_acceleration_mps2', ''),
        'onset_ttc_s': event.get('onset_ttc_s', ''),
        'br_command_event_delta': sig('BR Command', 'event_max_abs_delta'),
        'br_position_event_delta': sig('BR Position', 'event_max_abs_delta'),
        'brake_force_event_delta_n': sig('Brake force (unfiltered)', 'event_max_abs_delta'),
        'br_start_event_delta': sig('BR start', 'event_max_abs_delta'),
        'br_test_event_delta': sig('BR test', 'event_max_abs_delta'),
        'br_motion_event_delta': sig('Motion Going BR', 'event_max_abs_delta'),
        'calibration_status': detail['calibration_status'],
        'aeb_event_source': 'unconfirmed', 'collision_label': '',
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path,
                        default=ROOT / 'runs/20260911_abd_review/manual_review.csv')
    parser.add_argument('--abd-root', type=Path, default=ROOT / 'Data/ABD_Data')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/20260911_abd_calibration_audit')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = list(csv.DictReader(args.review.open(encoding='utf-8-sig')))
    details = []
    rows = []
    channel_counts = Counter()
    for source in sources:
        detail = analyze_run(args.abd_root / source['run'])
        detail['run'] = source['run']
        detail['vehicle_folder'] = source['vehicle_folder']
        detail['scenario'] = source['scenario']
        details.append(detail)
        rows.append(_flat_row(source, detail, args.abd_root))
        channel_counts.update((c['name'], c['unit']) for c in detail['discovered_channels'])

    with (args.output / 'per_run.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'channel_inventory.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['channel', 'unit', 'run_count'])
        writer.writeheader()
        writer.writerows({'channel': name, 'unit': unit or '', 'run_count': count}
                         for (name, unit), count in sorted(channel_counts.items()))
    (args.output / 'evidence.json').write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')

    statuses = Counter(row['calibration_status'] for row in rows)
    direct_aeb = sorted({c['name'] for d in details for c in d['discovered_channels']
                         if 'aeb' in c['name'].casefold() or 'fcw' in c['name'].casefold()})
    candidate = [r for r in rows
                 if r['calibration_status'] == 'response_candidate_source_unconfirmed']
    def is_zero(value):
        return value != '' and abs(float(value)) <= 1e-6
    ccrs_subset = [r for r in candidate
                   if r['scenario'] == 'CCRs'
                   and is_zero(r['br_command_event_delta'])
                   and is_zero(r['br_start_event_delta'])
                   and is_zero(r['br_test_event_delta'])
                   and is_zero(r['br_motion_event_delta'])]
    envelope_fields = ('onset_ttc_s', 'onset_speed_kph', 'speed_drop_2s_kph',
                       'peak_deceleration_mps2', 'p05_acceleration_mps2',
                       'br_position_event_delta', 'brake_force_event_delta_n')
    envelopes = []
    for field in envelope_fields:
        values = np.asarray([float(r[field]) for r in ccrs_subset if r[field] != ''])
        if len(values):
            envelopes.append({'subset': 'ccrs_brake_robot_disabled_no_br_activity_flags',
                              'metric': field, 'n': len(values),
                              'minimum': float(values.min()),
                              'median': float(np.median(values)),
                              'maximum': float(values.max())})
    with (args.output / 'descriptive_envelope.csv').open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'subset', 'metric', 'n', 'minimum', 'median', 'maximum'])
        writer.writeheader()
        writer.writerows(envelopes)

    queue = []
    for row in candidate:
        anomaly = (row['br_command_event_delta'] != ''
                   and abs(float(row['br_command_event_delta'])) > 1e-6)
        question = (
            'Explain why BR Command changes during the detected window despite '
            'UseBrakeRobot=False, and identify the authoritative brake-event source.'
            if anomaly else
            'Confirm the authoritative AEB activation signal/time and whether BR Position '
            'is passive pedal motion while BR start/test/motion and BR Command remain zero.')
        queue.append({
            'run': row['run'], 'scenario': row['scenario'],
            'event_window_s': f"{float(row['event_onset_s']):.3f}..{float(row['event_end_s']):.3f}",
            'candidate_channels': ('Time; Speed; Forward acceleration (body); '
                                   'Time to collision (longitudinal); BR Command; '
                                   'BR Position; Brake force (unfiltered); BR start; '
                                   'BR test; Motion Going BR'),
            'blocking_question': question,
        })
    with (args.output / 'confirmation_queue.csv').open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]) if queue else ['run'])
        writer.writeheader()
        writer.writerows(queue)
    summary = {
        'runs': len(rows), 'scenarios': dict(Counter(r['scenario'] for r in rows)),
        'statuses': dict(statuses), 'direct_aeb_or_fcw_channels': direct_aeb,
        'candidate_response_runs': len(candidate),
        'descriptive_ccrs_subset_runs': len(ccrs_subset),
        'confirmed_aeb_source_runs': 0, 'collision_labels_created': 0,
        'decision': 'NO_GO_FOR_AEB_CALIBRATION',
        'reason': ('Measured braking responses can be characterised, but no direct AEB/FCW '
                   'state channel or independently confirmed brake-event source is present.'),
    }
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# P3 ABD calibration evidence audit', '',
        f"- Runs streamed: **{len(rows)}**; row-count mismatches: "
        f"**{sum(r['row_count_matches'] is not True for r in rows)}**.",
        f"- Direct AEB/FCW state channels: **{', '.join(direct_aeb) if direct_aeb else 'none'}**.",
        f"- Response candidates with `UseBrakeRobot=False`: **{len(candidate)}**.",
        f"- Consistent CCRs descriptive subset (no event-window BR command/start/test/motion): "
        f"**{len(ccrs_subset)}**.",
        '- Confirmed AEB event-source records: **0**.',
        '- Collision labels created: **0**.',
        '- Decision: **NO-GO for AEB-specific perturbation calibration**.', '',
        'The candidate traces support descriptive vehicle-response ranges only. They do not '
        'establish that the observed deceleration was caused by AEB. Robot steering, '
        'accelerator, and path-following blocks describe test execution and are reported '
        'separately from brake control.', '',
        'The seven-record CCRs envelope is descriptive only: TTC at detected onset, '
        'measured acceleration, passive/active robot-channel evidence, and file hashes are '
        'retained. It is not applied to `perturb_spec` and is not called AEB-calibrated.', '',
        '## Status counts', '',
    ]
    lines += [f'- `{key}`: {value}' for key, value in sorted(statuses.items())]
    lines += ['', '## Required external confirmation', '',
              'For each candidate run, identify the authoritative AEB/FCW activation signal '
              'or provide a synchronized test log/video annotation that fixes activation '
              'time and confirms that the brake robot did not command the event. Until then, '
              'all fitted ranges must retain `assumed_sensitivity_not_abd_calibrated`.', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
