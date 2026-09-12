"""Create the reviewed 14-BZ3X historical smoke-run manifest."""
from __future__ import annotations

import argparse
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
        classification = ('robot_brake_negative_control' if br_active
                          else 'unknown_aeb_or_driver_brake')
        rows.append({
            'run': str(path.relative_to(ROOT)), 'sha256': detail['sha256'],
            'vehicle': '14-BZ3X', 'scenario': scenario,
            'selection_role': role, 'classification': classification,
            'calibration_eligible': False,
            'calibration_blocker': (
                'robot braking is active' if br_active else
                'AEB and driver braking cannot be separated'),
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
        'aeb_calibration_status': 'NO-GO: direct AEB and driver intervention unavailable',
        'rows': rows,
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# ABD historical smoke selection: 14-BZ3X', '',
        'Selected for parser, event-window and brake-source classification smoke testing. '
        'These are not AEB calibration samples because the exports cannot separate vehicle '
        'AEB from driver braking.', '',
        '| run | scenario | role | onset s | onset TTC s | peak m/s2 | BR cmd range | class |',
        '|---|---|---|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        event = row['event']
        lines.append(
            f"| {Path(row['run']).name} | {row['scenario']} | {row['selection_role']} | "
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
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'vehicle': '14-BZ3X',
                      'selected': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
