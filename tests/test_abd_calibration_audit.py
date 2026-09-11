"""Synthetic checks for the read-only P3 braking-response audit."""
import numpy as np

from scripts.audit_abd_calibration import analyze_run


def write_run(path, use_brake_robot):
    channels = [
        'Point', 'Time', 'Speed', 'Forward acceleration (body)',
        'Time to collision (longitudinal)', 'BR Command', 'BR Position',
        'Brake force (unfiltered)', 'BR start', 'BR test', 'Motion Going BR',
    ]
    units = ['', 's', 'km/h', 'm/s2', 's', 'mixed units', 'mm', 'N', '', '', '']
    rows = []
    speed = 20.0
    for i in range(400):
        accel = -5.0 if 100 <= i < 210 else 0.0
        if accel < 0:
            speed = max(0.0, speed + accel * 0.01 * 3.6)
        rows.append([i, i * 0.01, speed, accel, 1.5 - i * 0.005,
                     0.0, -20.0 + (10.0 if accel < 0 else 0.0),
                     2.0 + (15.0 if accel < 0 else 0.0), 0, 0, 0])
    lines = ['Anthony Best Dynamics Ltd', f'Points={len(rows)}',
             '\t'.join(channels), '\t'.join(units)]
    lines.extend('\t'.join(map(str, row)) for row in rows)
    path.write_text('\n'.join(lines) + '\n', encoding='latin-1')
    path.with_suffix('.spec').write_text(
        'Type=SR/AR Combination\nType=AR Speed Throttle Event\n'
        f'UseBrakeRobot={use_brake_robot}\n', encoding='latin-1')


def test_analyze_run_keeps_response_candidate_unconfirmed(tmp_path):
    path = tmp_path / 'V1_T1_R1.txt'
    write_run(path, 'False')
    result = analyze_run(path)
    assert result['row_count_matches'] is True
    assert result['bad_rows'] == 0
    assert result['config']['motion_control'] == 'robot_assisted'
    assert result['config']['brake_control'] == 'robot_disabled'
    assert result['calibration_status'] == 'response_candidate_source_unconfirmed'
    assert result['aeb_event_source'] == 'unconfirmed'
    assert result['collision_label'] is None
    assert 0.9 <= result['event']['onset_time_s'] <= 1.1
    assert result['event']['peak_deceleration_mps2'] < -4.0
    assert result['signals']['BR Command']['event_max_abs_delta'] == 0.0
    assert len(result['sha256']) == 64


def test_analyze_run_excludes_enabled_brake_robot(tmp_path):
    path = tmp_path / 'V1_T2_R1.txt'
    write_run(path, 'True')
    result = analyze_run(path)
    assert result['config']['brake_control'] == 'robot_enabled'
    assert result['calibration_status'] == 'excluded_brake_robot_enabled_ambiguous_source'
