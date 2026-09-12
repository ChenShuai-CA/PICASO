"""Screen historical ABD exports for manual no-takeover review.

The four operator-reviewed runs are prototypes, not training data for a
validated classifier.  This script ranks kinematically similar runs and keeps
the final intervention decision in a human-review column.  It also measures
target reference/actual tracking when both signals were captured.  A detected
deceleration onset is never relabelled as an ECU AEB request time.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_lab.data import inspect_abd  # noqa: E402
from scripts.audit_abd_calibration import (  # noqa: E402
    _event_window,
    _float,
    _header,
    _moving_average,
)


ANCHOR_FILENAMES = {
    'V14_T56_R1.txt', 'V14_T56_R2.txt', 'V14_T133_R2.txt', 'V14_T503_R4.txt'
}

CHANNELS = (
    'Time', 'Speed', 'Forward acceleration (body)', 'Forward acceleration',
    'Forward velocity', 'Lateral velocity', 'Lateral acceleration (body)',
    'Yaw angle', 'Yaw velocity', 'Yaw velocity (body)',
    'X position', 'Y position', 'BR Command', 'BR Position',
    'Brake force (unfiltered)', 'BR start', 'BR test', 'Motion Going BR',
    'Time to collision (longitudinal)',
    'Object 1 forward velocity (ref point)', 'Object 1 forward acceleration',
    'Object 1 actual X (front axle)', 'Object 1 actual Y (front axle)',
    'Object 1 reference X position', 'Object 1 reference Y position',
    'Head tracker forward velocity (ref point)',
    'Head tracker actual X (front axle)', 'Head tracker actual Y (front axle)',
    'Head tracker reference X position', 'Head tracker reference Y position',
)

SCENARIOS = (
    'CPNCO', 'CPTA', 'CPFA', 'CPNA', 'CPLA', 'CPLA', 'CCFT', 'CCRS', 'CCRM',
    'CCRB', 'CBLA', 'CBNA', 'CBFA', 'SCP', 'LSS', 'AES', 'AEB', 'FCW',
)

FEATURE_SCALES = {
    'lateral_velocity_delta_peak_mps': 0.25,
    'yaw_rate_delta_peak_dps': 3.0,
    'heading_change_deg': 3.0,
    'speed_rebound_kph': 0.75,
    'positive_speed_step_fraction': 0.03,
    'brake_pulse_count': 1.0,
    'jerk_p95_mps3': 12.0,
}


def infer_scenario(path: Path) -> str:
    upper = str(path).upper()
    for name in SCENARIOS:
        if re.search(rf'(?<![A-Z]){re.escape(name)}(?![A-Z])', upper):
            return name
    return 'OTHER'


def channel_capabilities(names: list[str]) -> dict:
    folded = [name.casefold() for name in names]
    def has(text):
        return any(text in name for name in folded)
    object_xy = all(name in names for name in (
        'Object 1 actual X (front axle)', 'Object 1 actual Y (front axle)',
        'Object 1 reference X position', 'Object 1 reference Y position'))
    tracker_xy = all(name in names for name in (
        'Head tracker actual X (front axle)', 'Head tracker actual Y (front axle)',
        'Head tracker reference X position', 'Head tracker reference Y position'))
    explicit = sorted({name for name in names
                       if 'aeb' in name.casefold() or 'fcw' in name.casefold()})
    return {
        'object_reference_actual_xy': object_xy,
        'head_tracker_reference_actual_xy': tracker_xy,
        'spt_reference_actual': has('spt desired') and has('spt '),
        'launchpad_low_level': has('launchpad drive') or has('launchpad sr'),
        'explicit_aeb_fcw_channels': explicit,
        'can_user_defined': has('can user defined'),
    }


def read_channels(path: Path, wanted=CHANNELS):
    declared, delimiter, names, units, data_offset = _header(path)
    index = {name: names.index(name) for name in wanted if name in names}
    values = {name: [] for name in index}
    bad = rows = 0
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
            parsed = {name: _float(fields[i]) for name, i in index.items()}
            if not all(math.isfinite(value) for value in parsed.values()):
                bad += 1
                continue
            for name, value in parsed.items():
                values[name].append(value)
            rows += 1
    return ({name: np.asarray(value, dtype=float) for name, value in values.items()},
            {'declared': declared, 'rows': rows, 'bad_rows': bad,
             'names': names, 'units': units})


def _event_slice(event, size, dt, pre_s=1.0, post_s=0.5):
    start = max(0, event['onset_index'] - round(pre_s / dt))
    end = min(size, event['end_index'] + round(post_s / dt))
    return slice(start, end)


def _count_pulses(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    return int(np.sum(np.diff(padded) == 1))


def extract_features(arrays: dict[str, np.ndarray], event: dict) -> dict:
    time = arrays['Time']
    speed = arrays['Speed']
    dt = float(np.median(np.diff(time)))
    window = _event_slice(event, len(time), dt)
    onset = event['onset_index']
    pre = slice(max(0, onset - max(1, round(1.0 / dt))), onset)
    acceleration = arrays.get('Forward acceleration (body)')
    if acceleration is None:
        acceleration = arrays.get('Forward acceleration')
    if acceleration is None:
        acceleration = np.gradient(speed / 3.6, time)
    acceleration = _moving_average(acceleration, max(1, round(0.10 / dt)))

    lateral = arrays.get('Lateral velocity', np.zeros_like(time))
    lateral_base = float(np.median(lateral[pre])) if onset else float(lateral[0])
    yaw_rate = arrays.get('Yaw velocity (body)')
    if yaw_rate is None:
        yaw_rate = arrays.get('Yaw velocity', np.zeros_like(time))
    yaw_base = float(np.median(yaw_rate[pre])) if onset else float(yaw_rate[0])
    yaw = arrays.get('Yaw angle', np.zeros_like(time))

    local_speed = speed[window]
    min_i = int(np.argmin(local_speed))
    rebound = float(max(0.0, np.max(local_speed[min_i:]) - local_speed[min_i]))
    steps = np.diff(local_speed)
    positive_fraction = float(np.mean(steps > 0.05)) if len(steps) else 0.0
    jerk = np.gradient(acceleration, time)
    pulses = _count_pulses(acceleration[window] <= -1.0)
    return {
        'lateral_velocity_delta_peak_mps': float(np.max(np.abs(lateral[window] - lateral_base))),
        'yaw_rate_delta_peak_dps': float(np.max(np.abs(yaw_rate[window] - yaw_base))),
        'heading_change_deg': float(np.max(yaw[window]) - np.min(yaw[window])),
        'speed_rebound_kph': rebound,
        'positive_speed_step_fraction': positive_fraction,
        'brake_pulse_count': pulses,
        'jerk_p95_mps3': float(np.quantile(np.abs(jerk[window]), 0.95)),
    }


def robot_activity(arrays: dict[str, np.ndarray], event: dict) -> dict:
    time = arrays['Time']
    dt = float(np.median(np.diff(time)))
    window = _event_slice(event, len(time), dt, pre_s=0.1, post_s=0.1)
    onset = event['onset_index']
    pre = slice(max(0, onset - max(1, round(0.5 / dt))), onset)
    evidence = {}
    for name in ('BR Command', 'BR start', 'BR test', 'Motion Going BR'):
        if name in arrays:
            baseline = float(np.median(arrays[name][pre])) if onset else float(arrays[name][0])
            evidence[name] = {
                'event_max_abs': float(np.max(np.abs(arrays[name][window]))),
                'event_max_abs_delta': float(np.max(np.abs(arrays[name][window] - baseline))),
            }
    # BR Command is the authoritative robot output in these exports.  The
    # start/test/motion flags are supporting evidence only: Motion Going BR can
    # remain high throughout a turning test even while BR Command is exactly 0.
    command_active = evidence.get('BR Command', {}).get('event_max_abs', 0.0) > 1e-6
    flag_transition = any(
        evidence.get(name, {}).get('event_max_abs_delta', 0.0) > 1e-6
        for name in ('BR start', 'BR test', 'Motion Going BR'))
    return {'active': command_active, 'command_active': command_active,
            'supporting_flag_transition': flag_transition, 'channels': evidence}


def target_tracking(arrays: dict[str, np.ndarray]) -> dict:
    for prefix in ('Object 1', 'Head tracker'):
        keys = [f'{prefix} actual X (front axle)', f'{prefix} actual Y (front axle)',
                f'{prefix} reference X position', f'{prefix} reference Y position']
        if not all(key in arrays for key in keys):
            continue
        ax, ay, rx, ry = (arrays[key] for key in keys)
        ref_step = np.hypot(np.diff(rx, prepend=rx[0]), np.diff(ry, prepend=ry[0]))
        actual_step = np.hypot(np.diff(ax, prepend=ax[0]), np.diff(ay, prepend=ay[0]))
        active = (ref_step > 1e-4) | (actual_step > 1e-4)
        if np.sum(active) < 10:
            return {'source': prefix, 'usable_dynamic': False}
        dx, dy = ax[active] - rx[active], ay[active] - ry[active]
        offset_x, offset_y = float(np.median(dx)), float(np.median(dy))
        raw_error = np.hypot(dx, dy)
        residual = np.hypot(dx - offset_x, dy - offset_y)
        return {
            'source': prefix,
            'usable_dynamic': True,
            'active_samples': int(np.sum(active)),
            'reference_path_range_m': float(np.hypot(np.ptp(rx), np.ptp(ry))),
            'reference_frame_offset_x_m': offset_x,
            'reference_frame_offset_y_m': offset_y,
            'reference_frame_offset_norm_m': float(math.hypot(offset_x, offset_y)),
            'raw_position_difference_p50_m': float(np.quantile(raw_error, 0.50)),
            'aligned_position_residual_rmse_m': float(np.sqrt(np.mean(residual ** 2))),
            'aligned_position_residual_p95_m': float(np.quantile(residual, 0.95)),
            'aligned_position_residual_max_m': float(np.max(residual)),
        }
    return {'source': None, 'usable_dynamic': False}


def prototype_model(anchor_rows: list[dict]) -> dict:
    model = {}
    for scenario in sorted({row['scenario'] for row in anchor_rows} | {'ALL'}):
        subset = anchor_rows if scenario == 'ALL' else [r for r in anchor_rows
                                                        if r['scenario'] == scenario]
        center, scale = {}, {}
        for feature, floor in FEATURE_SCALES.items():
            values = np.asarray([row[feature] for row in subset], dtype=float)
            center[feature] = float(np.median(values))
            mad = float(np.median(np.abs(values - np.median(values))))
            scale[feature] = max(floor, 1.4826 * mad)
        model[scenario] = {'n': len(subset), 'center': center, 'scale': scale}
    return model


def similarity(row: dict, model: dict) -> tuple[float, str]:
    key = row['scenario'] if row['scenario'] in model else 'ALL'
    prototype = model[key]
    z2 = []
    for feature in FEATURE_SCALES:
        z = (row[feature] - prototype['center'][feature]) / prototype['scale'][feature]
        z2.append(min(25.0, z * z))
    return float(math.exp(-0.5 * float(np.mean(z2)))), key


def analyze_candidate(path: Path, root: Path) -> dict:
    arrays, meta = read_channels(path)
    relative = path.relative_to(root)
    scenario = infer_scenario(relative)
    result = {
        'run': str(relative).replace('\\', '/'),
        'vehicle': relative.parts[0],
        'scenario': scenario,
        'filename': path.name,
        'rows': meta['rows'],
        'channels': len(meta['names']),
        'row_count_matches': meta['declared'] == meta['rows'] if meta['declared'] else None,
        'file_path_has_aeb': 'AEB' in str(relative).upper(),
        'channel_capabilities': channel_capabilities(meta['names']),
        'event_status': 'insufficient_kinematics',
    }
    if 'Time' not in arrays or 'Speed' not in arrays or len(arrays['Time']) < 20:
        return result
    accel = arrays.get('Forward acceleration (body)')
    if accel is None:
        accel = arrays.get('Forward acceleration')
    if accel is None:
        accel = np.gradient(arrays['Speed'] / 3.6, arrays['Time'])
    event = _event_window(arrays['Time'], arrays['Speed'], accel)
    if event is None:
        result['event_status'] = 'no_observed_braking_event'
        return result
    activity = robot_activity(arrays, event)
    result.update(
        event_status='robot_channel_active' if activity['active'] else 'br_zero_observed_braking',
        observed_braking_onset_s=event['onset_time_s'],
        observed_braking_onset_ttc_s=event.get('onset_ttc_s'),
        onset_speed_kph=event['onset_speed_kph'],
        peak_deceleration_mps2=event['peak_deceleration_mps2'],
        speed_drop_2s_kph=event['speed_drop_2s_kph'],
        robot_activity=activity,
        target_tracking=target_tracking(arrays),
        **extract_features(arrays, event),
    )
    if 'Time to collision (longitudinal)' in arrays:
        result['observed_braking_onset_ttc_s'] = float(
            arrays['Time to collision (longitudinal)'][event['onset_index']])
    try:
        result['control_config'] = inspect_abd(path, max_rows=1)['config']
    except Exception as exc:  # metadata is useful but must not block kinematic screening
        result['control_config_error'] = str(exc)
    return result


def _flatten(row: dict) -> dict:
    flat = {key: value for key, value in row.items()
            if not isinstance(value, (dict, list))}
    flat['explicit_aeb_fcw_channels'] = '; '.join(
        row['channel_capabilities']['explicit_aeb_fcw_channels'])
    for key, value in row['channel_capabilities'].items():
        if key != 'explicit_aeb_fcw_channels':
            flat[f'has_{key}'] = value
    for key, value in row.get('target_tracking', {}).items():
        flat[f'target_{key}'] = value
    flat['robot_max_abs_json'] = json.dumps(
        row.get('robot_activity', {}).get('channels', {}), ensure_ascii=False)
    return flat


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def build_review_queue(rows: list[dict], limit: int) -> list[dict]:
    eligible = [row for row in rows
                if row['event_status'] == 'br_zero_observed_braking'
                and row['filename'] not in ANCHOR_FILENAMES]
    eligible.sort(key=lambda row: (-row['prototype_similarity_0_1'], row['run']))
    chosen, used = [], set()
    # First guarantee vehicle breadth, then vehicle/scenario breadth, then score.
    used_vehicles = set()
    for row in eligible:
        if row['vehicle'] in used_vehicles:
            continue
        chosen.append(row)
        used_vehicles.add(row['vehicle'])
        if len(chosen) >= limit:
            break
    for row in eligible:
        group = (row['vehicle'], row['scenario'])
        if group in used:
            continue
        used.add(group)
        if row not in chosen:
            chosen.append(row)
        if len(chosen) >= limit:
            break
    for row in eligible:
        if row in chosen:
            continue
        if len(chosen) >= limit:
            break
        chosen.append(row)
    queue = []
    for row in chosen:
        queue.append({
            'run': row['run'], 'vehicle': row['vehicle'], 'scenario': row['scenario'],
            'prototype_similarity_0_1': round(row['prototype_similarity_0_1'], 6),
            'prototype_scope': row['prototype_scope'],
            'observed_braking_onset_s': round(row['observed_braking_onset_s'], 4),
            'peak_deceleration_mps2': round(row['peak_deceleration_mps2'], 4),
            'lateral_velocity_delta_peak_mps': round(
                row['lateral_velocity_delta_peak_mps'], 4),
            'heading_change_deg': round(row['heading_change_deg'], 4),
            'target_reference_actual_usable': row['target_tracking']['usable_dynamic'],
            'algorithm_status': 'manual_review_required',
            'driver_intervention': 'unknown',
        })
    return queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abd-root', type=Path, default=ROOT / 'Data/ABD_Data')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/20260912_abd_no_takeover_screen')
    parser.add_argument('--review-limit', type=int, default=40)
    parser.add_argument('--all-paths', action='store_true',
                        help='screen all TXT files, not only paths containing AEB')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.abd_root.rglob('*.txt'))
    header_rows, candidates = [], []
    for number, path in enumerate(paths, 1):
        rel = path.relative_to(args.abd_root)
        if path.name.casefold() in {'currenttestspec.txt', 'expinfo.txt'}:
            header_rows.append({
                'run': str(rel).replace('\\', '/'), 'vehicle': rel.parts[0],
                'scenario': infer_scenario(rel),
                'header_status': 'metadata_not_data_export',
            })
            continue
        try:
            declared, delimiter, names, units, offset = _header(path)
            caps = channel_capabilities(names)
            header_rows.append({
                'run': str(rel).replace('\\', '/'), 'vehicle': rel.parts[0],
                'scenario': infer_scenario(rel), 'channels': len(names),
                'points_declared': declared, 'header_status': 'data_export',
                **{f'has_{k}': v for k, v in caps.items()
                                                if k != 'explicit_aeb_fcw_channels'},
                'explicit_aeb_fcw_channels': '; '.join(caps['explicit_aeb_fcw_channels']),
            })
            if args.all_paths or 'AEB' in str(rel).upper():
                candidates.append(analyze_candidate(path, args.abd_root))
        except Exception as exc:
            rel = path.relative_to(args.abd_root)
            header_rows.append({'run': str(rel).replace('\\', '/'),
                                'vehicle': rel.parts[0], 'header_status': 'parse_error',
                                'error': str(exc)})
        if number % 250 == 0:
            print(json.dumps({'headers': number, 'screened': len(candidates)}), flush=True)

    anchors = [row for row in candidates
               if row['filename'] in ANCHOR_FILENAMES
               and row['event_status'] == 'br_zero_observed_braking']
    if len(anchors) != len(ANCHOR_FILENAMES):
        found = {row['filename'] for row in anchors}
        raise RuntimeError(f'missing usable reviewed anchors: {sorted(ANCHOR_FILENAMES - found)}')
    model = prototype_model(anchors)
    for row in candidates:
        if row['event_status'] == 'br_zero_observed_braking':
            row['prototype_similarity_0_1'], row['prototype_scope'] = similarity(row, model)

    flat_candidates = [_flatten(row) for row in candidates]
    write_csv(args.output / 'channel_inventory.csv', header_rows)
    write_csv(args.output / 'screening.csv', flat_candidates)
    queue = build_review_queue(candidates, args.review_limit)
    write_csv(args.output / 'manual_intervention_review.csv', queue)
    (args.output / 'prototype_model.json').write_text(
        json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')

    capability_counts = Counter()
    explicit = Counter()
    for row in header_rows:
        for name in ('object_reference_actual_xy', 'head_tracker_reference_actual_xy',
                     'spt_reference_actual', 'launchpad_low_level', 'can_user_defined'):
            capability_counts[name] += bool(row.get(f'has_{name}'))
        for channel in str(row.get('explicit_aeb_fcw_channels', '')).split('; '):
            if channel:
                explicit[channel] += 1
    statuses = Counter(row['event_status'] for row in candidates)
    target_residuals = np.asarray([
        row['target_tracking']['aligned_position_residual_p95_m']
        for row in candidates
        if row['event_status'] == 'br_zero_observed_braking'
        and row.get('target_tracking', {}).get('usable_dynamic')
    ], dtype=float)
    target_quantiles = ({
        str(q): float(np.quantile(target_residuals, q))
        for q in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)
    } if len(target_residuals) else {})
    summary = {
        'txt_files_encountered': len(paths),
        'data_export_files': sum(row.get('header_status') == 'data_export'
                                 for row in header_rows),
        'metadata_text_files': sum(row.get('header_status') == 'metadata_not_data_export'
                                   for row in header_rows),
        'aeb_path_files_screened': len(candidates),
        'header_parse_errors': sum(row.get('header_status') == 'parse_error'
                                   for row in header_rows),
        'channel_capability_file_counts': dict(capability_counts),
        'explicit_aeb_fcw_channel_file_counts': dict(explicit),
        'screen_status_counts': dict(statuses),
        'br_zero_dynamic_target_runs': int(len(target_residuals)),
        'aligned_target_position_residual_p95_m_quantiles': target_quantiles,
        'reviewed_anchor_runs': len(anchors), 'manual_review_queue_runs': len(queue),
        'manual_review_vehicle_count': len({row['vehicle'] for row in queue}),
        'manual_review_scenario_count': len({row['scenario'] for row in queue}),
        'interpretation': {
            'prototype_similarity': (
                'Unvalidated ranking against four operator-reviewed kinematic prototypes; '
                'it is not a no-takeover probability or classifier accuracy.'),
            'observed_braking_onset': (
                'First sustained -1 m/s2 response backtracked to -0.3 m/s2; it is not '
                'the unavailable ECU AEB request time.'),
            'target_reference_actual': (
                'Synchronized object/head-tracker reference and actual position channels '
                'from the same exported Time rows; low-level actuator commands are separate.'),
        },
    }
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report = [
        '# ABD no-takeover candidate screen', '',
        f"- TXT files encountered: **{summary['txt_files_encountered']}**; channel data "
        f"exports: **{summary['data_export_files']}**; configuration metadata TXT: "
        f"**{summary['metadata_text_files']}**.",
        f"- AEB-path exports screened: **{summary['aeb_path_files_screened']}**.",
        f"- BR-zero exports with an observed braking event: "
        f"**{statuses.get('br_zero_observed_braking', 0)}**.",
        f"- Manual-review queue: **{len(queue)}** runs across "
        f"**{summary['manual_review_vehicle_count']}** vehicle folders and "
        f"**{summary['manual_review_scenario_count']}** scenario labels.", '',
        f"Dynamic target reference/actual channels are usable in **{len(target_residuals)}** "
        f"BR-zero braking runs. The median per-run P95 aligned position residual is "
        f"**{target_quantiles.get('0.5', math.nan):.6f} m** and its across-run 95th "
        f"percentile is **{target_quantiles.get('0.95', math.nan):.6f} m**. The maximum "
        f"is **{target_quantiles.get('1.0', math.nan):.6f} m**, so outliers require a "
        f"quality gate before fitting a distribution.", '',
        'The queue is ranked from four operator-reviewed 14-BZ3X examples. With only four '
        'positive examples and no independently instrumented manual-braking negatives, the '
        'score is a review-priority measure. Only the operator may change '
        '`driver_intervention` from `unknown`.', '',
        'The detected time is `observed_braking_onset`: a sustained measured deceleration '
        'threshold backtracked to -0.3 m/s2. It cannot establish ECU AEB request time without '
        'vehicle CAN or another authoritative trigger channel.', '',
        'Target `reference` and `actual` X/Y channels share the exported `Time` rows. Their '
        'tracking errors are reported where the target trajectory is dynamic. This supports '
        'target execution-error analysis, while LaunchPad low-level actuator channels remain '
        'available only when captured by the corresponding target system.', '',
    ]
    (args.output / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
