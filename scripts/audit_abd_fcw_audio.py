"""Recover AVAD3-observed FCW audio times from RC time-tolerance mappings.

Historical TXT exports need not contain a raw ``CAN User Defined`` column.
The companion SPEC can map that input into a Time Tolerance Trigger (TTT), and
the TXT then records the TTT's within-tolerance, true-time-qualified, and final
delayed states.  This audit follows that mapping explicitly and never treats a
generic CAN Omni trigger or an unmapped TTT as FCW audio evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_abd_calibration import (  # noqa: E402
    _event_window, _float, _header, _stream_selected)
from scripts.screen_abd_no_takeover import robot_activity  # noqa: E402

MAPPING_KEY = re.compile(r'^TimeTolerance(\d+)Channel([12])$', re.I)
CAN_USER_VALUE = re.compile(
    r'^CAN User Defined\s+([12])(?:\s*\(([^)]*)\))?\s*$', re.I)
SOUND_TERMS = ('sound', 'alarm', 'avad', 'audio', 'buzzer')


def read_spec(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding='latin-1', errors='replace').splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip()
    return values


def find_can_user_mappings(spec: dict[str, str], context='') -> list[dict]:
    mappings = []
    description = spec.get('Description', '')
    fcw_context = 'fcw' in f'{context} {description}'.casefold()
    for key, value in spec.items():
        key_match = MAPPING_KEY.match(key)
        value_match = CAN_USER_VALUE.match(value)
        if not key_match or not value_match:
            continue
        trigger, slot = int(key_match.group(1)), int(key_match.group(2))
        can_user = int(value_match.group(1))
        label = (value_match.group(2) or '').strip()
        explicit_sound_label = any(term in label.casefold() for term in SOUND_TERMS)
        if explicit_sound_label:
            semantic_status = 'avad3_fcw_audio_explicit_sound_label'
        elif fcw_context:
            semantic_status = 'avad3_fcw_audio_by_operator_convention_in_fcw_context'
        else:
            semantic_status = 'can_user_defined_mapping_semantics_unconfirmed'
        prefix = f'TimeTolerance{trigger}'
        mappings.append({
            'ttt_index': trigger,
            'ttt_channel_slot': slot,
            'can_user_defined_index': can_user,
            'configured_label': label,
            'description': description,
            'fcw_context': fcw_context,
            'explicit_sound_label': explicit_sound_label,
            'semantic_status': semantic_status,
            'minimum': _float(spec.get(f'{prefix}MinTrigger{slot}')),
            'maximum': _float(spec.get(f'{prefix}MaxTrigger{slot}')),
            'true_time_s': _float(spec.get(f'{prefix}TriggerTime', 0)),
            'delay_time_s': _float(spec.get(f'{prefix}TriggerDelayTime', 0)),
            'or_true': spec.get(f'{prefix}OrTrue', '').casefold() == 'true',
        })
    return mappings


def first_rising_time(time: np.ndarray, values: np.ndarray, threshold=0.5):
    high = np.asarray(values) >= threshold
    rising = high & np.r_[True, ~high[:-1]]
    indices = np.flatnonzero(rising)
    return float(time[indices[0]]) if len(indices) else None


def analyze_ttt(path: Path, mapping: dict) -> dict:
    trigger = mapping['ttt_index']
    required = [
        'Time',
        f'Time tolerance {trigger} (within tolerances)',
        f'Time tolerance {trigger} (excl. delay)',
        f'Time tolerance {trigger}',
    ]
    declared, delimiter, names, units, data_offset = _header(path)
    missing = [name for name in required if name not in names]
    if missing:
        return {'trace_status': 'missing_ttt_export_channels',
                'missing_channels': '; '.join(missing)}
    wanted = required + [name for name in (
        'Speed', 'Forward acceleration (body)', 'Forward acceleration',
        'Time to collision (longitudinal)', 'BR Command', 'BR start', 'BR test',
        'Motion Going BR') if name in names]
    selected = [names.index(name) for name in wanted]
    arrays, rows, bad_rows, digest = _stream_selected(
        path, names, delimiter, data_offset, selected)
    time = arrays['Time']
    observed = first_rising_time(time, arrays[required[1]])
    qualified = first_rising_time(time, arrays[required[2]])
    final = first_rising_time(time, arrays[required[3]])
    result = {
        'trace_status': ('fcw_audio_rising_edge_found' if observed is not None
                         else 'no_within_tolerance_rising_edge'),
        'rows': rows, 'bad_rows': bad_rows, 'sha256': digest,
        't_fcw_audio_observed_s': observed,
        't_ttt_true_time_complete_s': qualified,
        't_ttt_final_after_delay_s': final,
        'time_source': 'Time tolerance X (within tolerances) first 0-to-1 edge',
    }
    if observed is not None and qualified is not None:
        result['observed_to_qualified_s'] = qualified - observed
    if qualified is not None and final is not None:
        result['qualified_to_final_s'] = final - qualified
    acceleration = arrays.get('Forward acceleration (body)')
    if acceleration is None:
        acceleration = arrays.get('Forward acceleration')
    if acceleration is not None and 'Speed' in arrays:
        event = _event_window(time, arrays['Speed'], acceleration)
        if event is not None:
            braking_onset = event['onset_time_s']
            result['observed_braking_onset_s'] = braking_onset
            activity = robot_activity(arrays, event)
            result['braking_source_status'] = (
                'robot_channel_active' if activity['active']
                else 'br_zero_braking_source_unconfirmed')
            result['br_command_active'] = activity['command_active']
            if observed is not None:
                result['fcw_audio_to_observed_braking_s'] = braking_onset - observed
    return result


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ['run'],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_review_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f'cannot decode review CSV: {path}')


def fcw_intervention_class(value: str) -> str:
    note = value.strip()
    folded = note.casefold()
    if '人工接管' in note or 'manual' in folded:
        return 'manual_after_fcw_audio_expected_by_test_procedure'
    if folded in ('none', 'none_confirmed'):
        return 'none_confirmed'
    return 'unresolved'


def write_manual_review(path: Path, rows: list[dict]):
    prior = {}
    if path.exists():
        prior = {row['run']: row.get('driver_intervention', 'unknown')
                 for row in read_review_csv(path)}
    review = []
    for row in rows:
        interval = float(row['fcw_audio_to_observed_braking_s'])
        review.append({
            'run': row['run'], 'vehicle': row['vehicle'],
            'description': row['description'], 'ttt_index': row['ttt_index'],
            'can_user_defined_index': row['can_user_defined_index'],
            'configured_label': row['configured_label'],
            't_fcw_audio_observed_s': round(row['t_fcw_audio_observed_s'], 4),
            'observed_braking_onset_s': round(row['observed_braking_onset_s'], 4),
            'fcw_audio_to_observed_braking_s': round(interval, 4),
            'timing_quality_flag': ('braking_precedes_fcw_audio_review_event_selection'
                                    if interval < 0 else 'nonnegative_interval'),
            'driver_intervention': prior.get(row['run'], 'unknown'),
        })
    write_csv(path, review)
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abd-root', type=Path, default=ROOT / 'Data/ABD_Data')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/20260912_abd_fcw_audio_audit')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec_path in sorted(args.abd_root.rglob('*.spec')):
        relative = spec_path.relative_to(args.abd_root)
        spec = read_spec(spec_path)
        mappings = find_can_user_mappings(spec, str(relative))
        for mapping in mappings:
            txt_path = spec_path.with_suffix('.txt')
            row = {
                'run': str(txt_path.relative_to(args.abd_root)).replace('\\', '/'),
                'spec': str(relative).replace('\\', '/'),
                'vehicle': relative.parts[0],
                **mapping,
            }
            if not txt_path.exists():
                row['trace_status'] = 'missing_companion_txt'
            else:
                try:
                    row.update(analyze_ttt(txt_path, mapping))
                except Exception as exc:
                    row.update(trace_status='trace_parse_error', error=str(exc))
            rows.append(row)

    write_csv(args.output / 'fcw_audio_mappings.csv', rows)
    statuses = Counter(row['trace_status'] for row in rows)
    semantics = Counter(row['semantic_status'] for row in rows)
    labels = Counter(row['configured_label'] or '<blank>' for row in rows)
    audio_rows = [row for row in rows
                  if row['semantic_status'] != 'can_user_defined_mapping_semantics_unconfirmed']
    observed = [row for row in audio_rows
                if row['trace_status'] == 'fcw_audio_rising_edge_found']
    paired = [row for row in observed
              if row.get('fcw_audio_to_observed_braking_s') is not None]
    br_zero_paired = [row for row in paired
                      if row.get('braking_source_status') ==
                      'br_zero_braking_source_unconfirmed']
    paired_intervals = np.asarray([
        row['fcw_audio_to_observed_braking_s'] for row in paired], dtype=float)
    br_zero_intervals = np.asarray([
        row['fcw_audio_to_observed_braking_s'] for row in br_zero_paired], dtype=float)
    review = write_manual_review(
        args.output / 'manual_intervention_review.csv', br_zero_paired)
    intervention_classes = Counter(
        fcw_intervention_class(row.get('driver_intervention', '')) for row in review)
    delays = np.asarray([row['qualified_to_final_s'] for row in observed
                         if row.get('qualified_to_final_s') is not None], dtype=float)
    summary = {
        'spec_mappings_to_can_user_defined_1_or_2': len(rows),
        'fcw_audio_semantic_mappings': len(audio_rows),
        'fcw_audio_observed_rising_edges': len(observed),
        'vehicles_with_observed_fcw_audio': len({row['vehicle'] for row in observed}),
        'fcw_audio_with_observed_braking_pair': len(paired),
        'fcw_audio_to_observed_braking_s': ({
            'n': int(len(paired_intervals)),
            'min': float(paired_intervals.min()),
            'median': float(np.median(paired_intervals)),
            'max': float(paired_intervals.max()),
        } if len(paired_intervals) else None),
        'braking_source_status_counts': dict(Counter(
            row.get('braking_source_status', 'no_observed_braking')
            for row in observed)),
        'br_zero_fcw_audio_to_observed_braking_s': ({
            'n': int(len(br_zero_intervals)),
            'min': float(br_zero_intervals.min()),
            'median': float(np.median(br_zero_intervals)),
            'max': float(br_zero_intervals.max()),
        } if len(br_zero_intervals) else None),
        'manual_intervention_review_queue_runs': len(review),
        'manual_intervention_review_counts': dict(intervention_classes),
        'fcw_only_test_runs_with_observed_audio': len(observed),
        'fcw_runs_eligible_for_aeb_response_or_proxy': 0,
        'trace_status_counts': dict(statuses),
        'semantic_status_counts': dict(semantics),
        'configured_label_counts': dict(labels),
        'qualified_to_final_delay_s': ({
            'n': int(len(delays)), 'min': float(delays.min()),
            'median': float(np.median(delays)), 'max': float(delays.max())
        } if len(delays) else None),
        'interpretation': {
            't_fcw_audio_observed_s': (
                'First 0-to-1 edge of the mapped TTT within-tolerances channel. '
                'This is AVAD3-observed audible FCW onset under the operator-provided '
                'channel convention, not an ECU internal FCW request time.'),
            'time_tolerance_final': (
                'Includes configured TTT true time and delay; do not use as audible-onset '
                'time when the within-tolerances trace is available.'),
            'historical_test_intent': (
                'These are FCW-only tests. Braking after the warning is robot braking or '
                'the expected driver takeover and is not an AEB response.'),
        },
    }
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report = [
        '# ABD AVAD3 / FCW audio timing audit', '',
        f"- SPEC mappings to CAN User Defined 1/2: **{len(rows)}**.",
        f"- Mappings supported as FCW audio by an explicit sound label or FCW-context "
        f"operator convention: **{len(audio_rows)}**.",
        f"- Runs with an observed 0-to-1 audio edge: **{len(observed)}**, across "
        f"**{summary['vehicles_with_observed_fcw_audio']}** vehicle folders.", '',
        f"- Runs that also contain an observed braking onset: **{len(paired)}**. These are "
        "FCW-only tests, so the subsequent braking is not an AEB response.", '',
        f"- Of the paired runs, **{sum(row.get('braking_source_status') == 'robot_channel_active' for row in paired)}** "
        "have robot braking and **"
        f"{intervention_classes.get('manual_after_fcw_audio_expected_by_test_procedure', 0)}** "
        "BR-zero runs were confirmed as the expected manual takeover after the warning.", '',
        '- All recovered audio edges remain usable FCW timing observations. No historical '
        'FCW-only run is eligible for AEB response, AEB proxy timing, or FCW-to-AEB delay.', '',
        '`T_FCW_audio_observed` is the first rising edge of `Time tolerance X (within '
        'tolerances)` after the companion SPEC maps that trigger input to CAN User Defined '
        '1 or 2. This recovers the AVAD3-observed audible warning time even when the raw '
        'CAN User Defined column was not included in the TXT export.', '',
        '`Time tolerance X (excl. delay)` completes the configured TTT true-time condition; '
        '`Time tolerance X` additionally includes the configured trigger delay. Neither '
        'should replace the within-tolerance first edge when estimating audible onset.', '',
        'The recovered time is an external acoustic observation. It includes the vehicle '
        'warning generation, speaker/acoustic propagation, AVAD3 detection, and RC sampling '
        'chain; it is not the ECU internal FCW request transition.', '',
    ]
    (args.output / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
