import numpy as np

from scripts.screen_abd_no_takeover import (
    aeb_request_proxy,
    build_review_queue,
    channel_capabilities,
    extract_features,
    infer_scenario,
    prototype_model,
    robot_activity,
    similarity,
    target_tracking,
)


def test_channel_capabilities_finds_synchronised_object_reference_actual():
    names = [
        'Object 1 actual X (front axle)', 'Object 1 actual Y (front axle)',
        'Object 1 reference X position', 'Object 1 reference Y position',
        'Calculated AEB state',
    ]
    result = channel_capabilities(names)
    assert result['object_reference_actual_xy'] is True
    assert result['explicit_aeb_fcw_channels'] == ['Calculated AEB state']


def test_aeb_request_proxy_back_calculates_interval_without_false_precision():
    result = aeb_request_proxy(10.0)
    assert result['aeb_request_proxy_earliest_s'] == 9.65
    assert result['aeb_request_proxy_nominal_s'] == 9.75
    assert result['aeb_request_proxy_latest_s'] == 9.85
    assert result['aeb_request_proxy_status'].endswith('not_observed_ecu_signal')


def test_target_tracking_reports_dynamic_position_error():
    x = np.linspace(0.0, 10.0, 101)
    arrays = {
        'Object 1 actual X (front axle)': x + 0.1,
        'Object 1 actual Y (front axle)': np.zeros_like(x),
        'Object 1 reference X position': x,
        'Object 1 reference Y position': np.zeros_like(x),
    }
    result = target_tracking(arrays)
    assert result['usable_dynamic'] is True
    assert abs(result['reference_frame_offset_norm_m'] - 0.1) < 1e-9
    assert result['aligned_position_residual_rmse_m'] < 1e-9


def test_similarity_is_one_at_prototype_center():
    base = {'scenario': 'CCRS', **{name: float(i + 1)
                                   for i, name in enumerate((
                                       'lateral_velocity_delta_peak_mps',
                                       'yaw_rate_delta_peak_dps', 'heading_change_deg',
                                       'speed_rebound_kph', 'positive_speed_step_fraction',
                                       'brake_pulse_count', 'jerk_p95_mps3'))}}
    model = prototype_model([base])
    score, scope = similarity(base, model)
    assert score == 1.0
    assert scope == 'CCRS'


def test_extract_features_distinguishes_lateral_excursion():
    time = np.arange(0.0, 4.0, 0.01)
    speed = np.maximum(0.0, 20.0 - np.maximum(0.0, time - 1.0) * 10.0)
    accel = np.where((time >= 1.0) & (time <= 3.0), -3.0, 0.0)
    event = {'onset_index': 100, 'end_index': 300}
    calm = {'Time': time, 'Speed': speed, 'Forward acceleration (body)': accel,
            'Lateral velocity': np.zeros_like(time)}
    avoid = {**calm, 'Lateral velocity': np.where((time > 1.2) & (time < 2.0), 1.5, 0.0)}
    assert extract_features(avoid, event)['lateral_velocity_delta_peak_mps'] > 1.0
    assert extract_features(calm, event)['lateral_velocity_delta_peak_mps'] == 0.0


def test_motion_going_br_level_does_not_override_zero_br_command():
    time = np.arange(0.0, 4.0, 0.01)
    arrays = {
        'Time': time,
        'BR Command': np.zeros_like(time),
        'Motion Going BR': np.ones_like(time),
    }
    result = robot_activity(arrays, {'onset_index': 100, 'end_index': 250})
    assert result['active'] is False
    assert result['supporting_flag_transition'] is False


def test_nonzero_br_command_marks_robot_activity():
    time = np.arange(0.0, 4.0, 0.01)
    command = np.zeros_like(time)
    command[100:250] = 20.0
    result = robot_activity({'Time': time, 'BR Command': command},
                            {'onset_index': 100, 'end_index': 250})
    assert result['active'] is True


def test_infer_scenario_uses_path_tokens():
    from pathlib import Path
    assert infer_scenario(Path('car/11-CCRs/56-AEB/run.txt')) == 'CCRS'


def test_review_queue_respects_limit_after_diversity_pass():
    rows = []
    for i in range(8):
        rows.append({
            'event_status': 'br_zero_observed_braking', 'filename': f'run{i}.txt',
            'vehicle': f'car{i}', 'scenario': 'CCRS', 'run': f'car{i}/run{i}.txt',
            'prototype_similarity_0_1': 1.0 - i / 100,
            'prototype_scope': 'CCRS', 'observed_braking_onset_s': 1.0,
            'peak_deceleration_mps2': -5.0,
            'lateral_velocity_delta_peak_mps': 0.0, 'heading_change_deg': 0.0,
            'target_tracking': {'usable_dynamic': True},
        })
    assert len(build_review_queue(rows, 5)) == 5
