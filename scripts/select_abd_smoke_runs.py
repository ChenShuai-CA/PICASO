"""Create the reviewed 14-BZ3X historical smoke-run manifest."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_abd_calibration import _header, analyze_run  # noqa: E402


SELECTION = {
    'V14_T56_R1.txt': ('CCRs', 'zero_br_unknown_repeat_1'),
    'V14_T56_R2.txt': ('CCRs', 'zero_br_unknown_repeat_2'),
    'V14_T133_R2.txt': ('CCFT', 'zero_br_unknown_turning'),
    'V14_T503_R4.txt': ('CPTA', 'zero_br_unknown_turning'),
    'V14_T133_R1.txt': ('CCFT', 'positive_br_robot_negative_control'),
}

AVAILABLE_CORE = (
    'Time', 'Speed', 'Forward acceleration (body)',
    'Time to collision (longitudinal)', 'Relative longitudinal distance',
    'Relative longitudinal velocity', 'BR Command', 'BR Position',
    'BR Velocity', 'Brake force (unfiltered)', 'AR Command',
)

REVIEW_FIELDS = ('run', 'vehicle', 'scenario', 'driver_intervention')
REVIEW_VALUES = ('none_confirmed', 'manual', 'unknown')

REVIEW_METHOD = {
    'type': 'indirect_kinematic_human_review',
    'software': 'Robot Controller',
    'views': [
        'Results > Check Paths',
        'Motion Pack > Forward velocity [m/s]',
        'Motion Pack > Lateral velocity [m/s]',
    ],
    'decision_basis': (
        'The project test operator reviewed the path together with longitudinal and '
        'lateral velocity curves. A takeover is marked when the combined trajectory '
        'and velocity response differs materially from a normal AEB stop, including '
        'avoidance-related lateral motion or a different longitudinal convergence to zero.'
    ),
    'confirmation_date': '2026-09-12',
    'evidence_level': 'operator-reviewed indirect kinematic evidence',
    'limitation': (
        'No independent brake-pedal/pressure marker is recorded. Manual straight-line '
        'braking that closely resembles an AEB stop may therefore remain undetected.'
    ),
}


def load_or_create_review(path, selected_paths):
    """Create a stable human-review surface once and never overwrite answers."""
    if not path.exists():
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS,
                                    lineterminator='\n')
            writer.writeheader()
            for filename, (scenario, role) in SELECTION.items():
                if not role.startswith('zero_br_'):
                    continue
                writer.writerow({
                    'run': str(selected_paths[filename].relative_to(ROOT)),
                    'vehicle': '14-BZ3X', 'scenario': scenario,
                    'driver_intervention': 'unknown',
                })
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing_fields = set(REVIEW_FIELDS) - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f'missing manual review fields: {sorted(missing_fields)}')
        rows = list(reader)
    expected = {str(path.relative_to(ROOT)) for filename, path in selected_paths.items()
                if SELECTION[filename][1].startswith('zero_br_')}
    actual = {row['run'] for row in rows}
    if actual != expected:
        raise ValueError('manual intervention review rows do not match four BR-zero runs')
    invalid = [row['driver_intervention'] for row in rows
               if row['driver_intervention'] not in REVIEW_VALUES]
    if invalid:
        raise ValueError(f'invalid driver_intervention values: {invalid}')
    return {row['run']: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='Data/ABD_Data/14-BZ3X')
    parser.add_argument('--output', default='runs/20260912_abd_smoke_selection')
    args = parser.parse_args()
    data = (ROOT / args.data).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in data.rglob('*.txt'):
        if path.name in SELECTION:
            if path.name in found:
                raise ValueError(f'duplicate selected filename: {path.name}')
            found[path.name] = path
    missing = set(SELECTION) - set(found)
    if missing:
        raise FileNotFoundError(f'missing selected runs: {sorted(missing)}')

    review_path = output / 'manual_intervention_review.csv'
    reviews = load_or_create_review(review_path, found)

    rows = []
    for filename, (scenario, role) in SELECTION.items():
        path = found[filename]
        detail = analyze_run(path)
        names = set(_header(path)[2])
        event = detail.get('event') or {}
        signals = detail['signals']
        br = signals.get('BR Command', {})
        companions = {suffix: path.with_suffix(suffix).exists()
                      for suffix in ('.spec', '.log', '.CRUN')}
        br_active = max(abs(br.get('event_min', 0.)),
                        abs(br.get('event_max', 0.))) > 1e-6
        relative = str(path.relative_to(ROOT))
        review = reviews.get(relative)
        driver_status = review['driver_intervention'] if review else 'not_applicable'
        if br_active:
            classification = 'robot_brake_negative_control'
        elif driver_status == 'manual':
            classification = 'driver_brake_contaminated'
        elif driver_status == 'none_confirmed':
            classification = 'observed_braking_no_takeover_signature_aeb_unconfirmed'
        else:
            classification = 'unknown_aeb_or_driver_brake'
        rows.append({
            'run': relative, 'sha256': detail['sha256'],
            'vehicle': '14-BZ3X', 'scenario': scenario,
            'selection_role': role, 'classification': classification,
            'calibration_eligible': False,
            'observed_response_fit_eligible': (
                not br_active and driver_status == 'none_confirmed'),
            'driver_intervention_review': review,
            'driver_intervention_evidence_level': (
                REVIEW_METHOD['evidence_level'] if review else 'not_applicable'),
            'calibration_blocker': (
                'robot braking is active' if br_active else
                'direct AEB request/status is unavailable'),
            'rows': detail['rows_parsed'], 'channels': detail['n_channels'],
            'dt_median_s': detail['time']['dt_median_s'],
            'event': {
                'onset_s': event.get('onset_time_s'),
                'onset_ttc_s': event.get('onset_ttc_s'),
                'peak_deceleration_mps2': event.get('peak_deceleration_mps2'),
                'speed_drop_2s_kph': event.get('speed_drop_2s_kph'),
                'br_command_min': br.get('event_min'),
                'br_command_max': br.get('event_max'),
            },
            'available_core_channels': [name for name in AVAILABLE_CORE if name in names],
            'companions': companions,
            'missing_for_source_separation': [
                'direct AEB request/active/state',
                'independent driver brake pedal/pressure marker',
                'target platform command and actual motion on common timebase',
            ],
        })
    manifest = {
        'purpose': 'historical non-CAN pipeline and source-classification smoke test',
        'vehicle': '14-BZ3X', 'selected_runs': len(rows),
        'selection_reason': (
            'three scenario types, repeated CCRs response, four BR-zero unknown-source '
            'events, and one BR-active negative control; all selected runs have the '
            'same 415-channel export and complete companion files'),
        'manual_intervention_review_method': REVIEW_METHOD,
        'aeb_calibration_status': (
            'NO-GO for AEB request timing: direct vehicle AEB request/status is unavailable; '
            'manual takeover was reviewed indirectly from path and velocity curves'),
        'rows': rows,
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# ABD historical smoke selection: 14-BZ3X', '',
        'Selected for parser, event-window and brake-source classification smoke testing. '
        'The project test operator reviewed manual takeover using Robot Controller Check Paths '
        'and Motion Pack forward/lateral velocity curves. The four BR-zero runs have no observed '
        'takeover signature and may enter observed braking response analysis. They are not AEB '
        'request-timing samples because the exports contain no direct vehicle AEB request/status.', '',
        '| run | scenario | role | driver review | onset s | onset TTC s | peak m/s2 | BR cmd range | class |',
        '|---|---|---|---|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        event = row['event']
        lines.append(
            f"| {Path(row['run']).name} | {row['scenario']} | {row['selection_role']} | "
            f"{(row['driver_intervention_review'] or {}).get('driver_intervention', 'n/a')} | "
            f"{event['onset_s']:.3f} | {event['onset_ttc_s']:.3f} | "
            f"{event['peak_deceleration_mps2']:.3f} | "
            f"{event['br_command_min']:.3f}..{event['br_command_max']:.3f} | "
            f"{row['classification']} |")
    lines += [
        '', 'All five exports contain 415 channels at approximately 100 Hz and have '
        'matching `.spec`, `.log` and `.CRUN` files. The manifest records SHA256 values '
        'and exact relative paths; raw files were not copied or modified.', '',
        'Before using new runs for response calibration, add an independent driver-brake '
        'marker and synchronised target command/actual logs. Without vehicle CAN, retain '
        'the Post Processor threshold time as `observed_braking_onset`, not AEB request time.',
        '',
        'Human review is entered only in the `driver_intervention` column of '
        '`manual_intervention_review.csv`. Allowed values are `none_confirmed`, `manual`, and '
        '`unknown`. The shared review method and its limitation are stored in `manifest.json`. '
        'Re-run this script after editing; it validates the four paths and preserves the review file.',
        '',
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'vehicle': '14-BZ3X',
                      'selected': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
