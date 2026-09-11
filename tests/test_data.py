"""Tests for scenario_lab.data adapters (synthetic fixtures only, bounded reads)."""
import csv
import json
import math

import numpy as np
import pytest

from scenario_lab.data import audit_dataset, inspect_abd, iter_interaction_tracks, split_group


# ---------------------------------------------------------------- helpers

def write_tracks_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_abd_txt(path, channels, units, rows):
    lines = ['Anthony Best Dynamics Ltd', f'Points={len(rows)}',
             '\t'.join(channels), '\t'.join(units)]
    for row in rows:
        lines.append('\t'.join(str(v) for v in row))
    path.write_text('\n'.join(lines) + '\n', encoding='latin-1')


def write_spec(path, lines):
    path.write_text('\n'.join(lines) + '\n', encoding='latin-1')


VEHICLE_HEADER = ['track_id', 'frame_id', 'timestamp_ms', 'agent_type',
                  'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']


def vehicle_rows(n, t0_ms=100, dt_ms=100):
    return [[1, i + 1, t0_ms + i * dt_ms, 'car',
             10.0 + 0.5 * i, 20.0 + 0.1 * i, 5.0, 1.0, 0.2, 4.2, 1.8]
            for i in range(n)]


# ---------------------------------------------------------------- split_group

def test_split_group_is_deterministic_and_roughly_balanced():
    ids = [f'LOC_{i}' for i in range(120)]
    a = [split_group('interaction', g) for g in ids]
    b = [split_group('interaction', g) for g in ids]
    assert a == b
    assert set(a) == {'train', 'val', 'test'}
    counts = {s: a.count(s) for s in ('train', 'val', 'test')}
    assert 80 <= counts['train'] <= 110
    assert 4 <= counts['val'] <= 20
    assert 4 <= counts['test'] <= 20


def test_split_group_normalizes_source_casing_only():
    assert split_group('Interaction', 'g1') == split_group('interaction', 'g1')
    assert split_group('interaction', 'g1') == split_group(' interaction ', 'g1')
    # different logical groups may land in different buckets
    buckets = {split_group('interaction', f'c{i}') for i in range(60)}
    assert buckets <= {'train', 'val', 'test'}


# ---------------------------------------------------------------- iter_interaction_tracks

def test_iter_tracks_basic_schema_and_units(tmp_path):
    p = tmp_path / 'LOC_DEMO' / 'vehicle_tracks_000.csv'
    p.parent.mkdir()
    write_tracks_csv(p, VEHICLE_HEADER, vehicle_rows(5))
    recs = list(iter_interaction_tracks(p))
    assert len(recs) == 1
    r = recs[0]
    for key in ('source', 'group_id', 'track_id', 'kind', 'split', 't', 'xy',
                'velocity', 'heading', 'provenance'):
        assert key in r
    assert r['source'] == 'interaction'
    assert r['group_id'] == 'LOC_DEMO'
    assert r['track_id'] == '1'
    assert r['kind'] == 'vehicle'
    assert r['split'] == split_group('interaction', 'LOC_DEMO')
    assert r['t'].shape == (5,)
    np.testing.assert_allclose(r['t'], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert r['xy'].shape == (5, 2)
    np.testing.assert_allclose(r['xy'][:, 0], [10.0 + 0.5 * i for i in range(5)])
    np.testing.assert_allclose(r['velocity'][:, 0], 5.0)
    np.testing.assert_allclose(r['velocity'][:, 1], 1.0)
    np.testing.assert_allclose(r['heading'], 0.2)
    assert r['provenance']['heading_source'] == 'psi_rad'
    assert r['provenance']['n_segments'] == 1
    assert r['provenance']['segment_index'] == 0


def test_iter_tracks_splits_at_time_gap_only(tmp_path):
    p = tmp_path / 'LOC_GAP' / 'vehicle_tracks_000.csv'
    p.parent.mkdir()
    rows = vehicle_rows(5)                                   # t = 0.1 .. 0.5 s
    rows += vehicle_rows(5, t0_ms=1500)                      # t = 1.5 .. 1.9 s (gap)
    write_tracks_csv(p, VEHICLE_HEADER, rows)
    recs = list(iter_interaction_tracks(p))
    assert len(recs) == 2
    assert {r['provenance']['segment_index'] for r in recs} == {0, 1}
    assert all(r['provenance']['n_segments'] == 2 for r in recs)
    for r in recs:
        dt = np.diff(r['t'])
        assert dt.max() < 0.15                               # never bridged across gap
    assert recs[0]['t'][-1] < recs[1]['t'][0]


def test_iter_tracks_never_bridges_across_cases(tmp_path):
    p = tmp_path / 'LOC_CASE' / 'vehicle_tracks_000.csv'
    p.parent.mkdir()
    header = ['case_id'] + VEHICLE_HEADER
    rows = ([1] + r for r in vehicle_rows(4))
    rows = list(rows) + list([2] + r for r in vehicle_rows(3))
    write_tracks_csv(p, header, rows)
    recs = list(iter_interaction_tracks(p))
    assert len(recs) == 2
    by_group = {r['group_id']: r for r in recs}
    assert set(by_group) == {'LOC_CASE::1', 'LOC_CASE::2'}
    assert by_group['LOC_CASE::1']['t'].shape == (4,)
    assert by_group['LOC_CASE::2']['t'].shape == (3,)


def test_iter_tracks_splits_on_track_id_time_reset(tmp_path):
    p = tmp_path / 'LOC_RESET' / 'vehicle_tracks_000.csv'
    p.parent.mkdir()
    rows = vehicle_rows(5)                                   # t = 0.1 .. 0.5
    rows += vehicle_rows(3, t0_ms=100)                       # clock restarts
    write_tracks_csv(p, VEHICLE_HEADER, rows)
    recs = list(iter_interaction_tracks(p))
    assert len(recs) == 2
    assert all(r['group_id'] == 'LOC_RESET' for r in recs)


def test_iter_tracks_derives_missing_heading_from_velocity(tmp_path):
    p = tmp_path / 'LOC_PED' / 'pedestrian_tracks_000.csv'
    p.parent.mkdir()
    header = ['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy']
    rows = [['P2', 671 + i, 67100 + 100 * i, 'pedestrian',
             1016.9 - 0.35 * i, 988.6 - 0.22 * i, -3.4, -2.2] for i in range(6)]
    write_tracks_csv(p, header, rows)
    recs = list(iter_interaction_tracks(p))
    assert len(recs) == 1
    r = recs[0]
    assert r['provenance']['heading_source'] == 'derived_from_velocity'
    expected = math.atan2(-2.2, -3.4)
    np.testing.assert_allclose(r['heading'], expected, rtol=1e-6)


def test_iter_tracks_unknown_agent_type_maps_to_other_with_reason(tmp_path):
    p = tmp_path / 'LOC_UNK' / 'pedestrian_tracks_000.csv'
    p.parent.mkdir()
    header = ['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy']
    rows = [['P1', 1 + i, 100 + 100 * i, 'mystery_bot', 1.0, 2.0, 0.0, 0.0]
            for i in range(3)]
    write_tracks_csv(p, header, rows)
    recs = list(iter_interaction_tracks(p))
    assert len(recs) == 1
    r = recs[0]
    assert r['kind'] == 'other'
    assert r['provenance']['agent_type_raw'] == 'mystery_bot'
    assert any('mystery_bot' in reason for reason in r['provenance']['reasons'])


def test_iter_tracks_official_split_dir_respected(tmp_path):
    p = tmp_path / 'LOC_A' / 'train' / 'LOC_A_train.csv'
    p.parent.mkdir(parents=True)
    write_tracks_csv(p, VEHICLE_HEADER, vehicle_rows(3))
    recs = list(iter_interaction_tracks(p))
    assert recs[0]['split'] == 'train'
    assert recs[0]['provenance']['split_source'] == 'official'
    assert recs[0]['group_id'] == 'LOC_A'                    # split dir not in group id


def test_iter_tracks_duplicate_exports_share_group_and_split(tmp_path):
    rows = vehicle_rows(4)
    p1 = tmp_path / 'exports_v1' / 'LOC_B' / 'vehicle_tracks_000.csv'
    p2 = tmp_path / 'exports_v2' / 'LOC_B' / 'tracks_77.csv'
    for p in (p1, p2):
        p.parent.mkdir(parents=True)
        write_tracks_csv(p, VEHICLE_HEADER, rows)
    r1 = list(iter_interaction_tracks(p1))[0]
    r2 = list(iter_interaction_tracks(p2))[0]
    assert r1['group_id'] == r2['group_id'] == 'LOC_B'
    assert r1['split'] == r2['split']
    assert r1['provenance']['split_source'] == 'hash'


def test_iter_tracks_observed_file_is_test_not_training_label(tmp_path):
    p = tmp_path / 'LOC_C' / 'val' / 'LOC_C_val_observed.csv'
    p.parent.mkdir(parents=True)
    write_tracks_csv(p, VEHICLE_HEADER, vehicle_rows(3))
    recs = list(iter_interaction_tracks(p))
    r = recs[0]
    assert r['split'] == 'test'
    assert r['provenance']['observed_only'] is True


def test_iter_tracks_rejects_missing_required_columns(tmp_path):
    p = tmp_path / 'LOC_BAD' / 'vehicle_tracks_000.csv'
    p.parent.mkdir()
    # no timestamp_ms/time_s/frame_id -> no usable time source
    write_tracks_csv(p, ['track_id', 'x', 'y'], [[1, 0.0, 0.0]])
    with pytest.raises(ValueError):
        list(iter_interaction_tracks(p))


# ---------------------------------------------------------------- inspect_abd

def basic_abd_rows(n, abort_at=None):
    rows = []
    for i in range(n):
        abort = 1 if (abort_at is not None and i == abort_at) else 0
        rows.append([i, round(0.01 * i, 4), 50.0 + 0.1 * i, abort])
    return rows


def test_inspect_abd_parses_channels_units_and_time(tmp_path):
    p = tmp_path / 'V1_T1_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed', 'SR Angle'],
                  ['', 's', 'km/h', 'deg'], basic_abd_rows(6))
    info = inspect_abd(p)
    assert info['parse_status'] == 'ok'
    assert info['role'] == 'unconfirmed'
    channels = {c['name']: c['unit'] for c in info['channels']}
    assert channels['Time'] == 's'
    assert channels['Speed'] == 'km/h'
    assert channels['Point'] is None
    assert info['n_channels'] == 4
    assert info['n_points_declared'] == 6
    assert info['rows_read'] == 6
    t = info['time']
    assert t['channel'] == 'Time'
    assert t['monotonic'] is True
    assert t['uniform'] is True
    assert abs(t['dt_median_s'] - 0.01) < 1e-9
    assert json.dumps(info)                                  # JSON serializable


def test_inspect_abd_reads_bounded_rows(tmp_path):
    p = tmp_path / 'V1_T2_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed'], ['', 's', 'km/h'],
                  [row[:3] for row in basic_abd_rows(40)])
    info = inspect_abd(p, max_rows=10)
    assert info['rows_read'] == 10
    assert info['truncated'] is True
    assert any('max_rows' in reason for reason in info['reasons'])


def test_inspect_abd_unknown_control_not_suitable(tmp_path):
    p = tmp_path / 'V1_T3_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed'], ['', 's', 'km/h'],
                  [row[:3] for row in basic_abd_rows(5)])
    info = inspect_abd(p)
    assert info['config']['spec_found'] is False
    assert info['config']['control'] == 'unknown'
    assert info['config']['brake_robot_engaged'] is None
    assert info['suitable_for_aeb_calibration'] is False
    assert any('unknown' in reason for reason in info['reasons'])


def test_inspect_abd_brake_robot_true_not_suitable(tmp_path):
    p = tmp_path / 'V1_T4_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed'], ['', 's', 'km/h'],
                  [row[:3] for row in basic_abd_rows(5)])
    write_spec(p.with_suffix('.spec'),
               ['Type=SR/AR Combination', 'Description=demo', 'UseBrakeRobot=True'])
    info = inspect_abd(p)
    assert info['config']['use_brake_robot'] is True
    assert info['config']['brake_robot_engaged'] is True
    assert info['config']['control'] == 'robot'
    assert info['suitable_for_aeb_calibration'] is False


def test_inspect_abd_brake_robot_false_is_candidate_but_unconfirmed(tmp_path):
    p = tmp_path / 'V1_T5_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed'], ['', 's', 'km/h'],
                  [row[:3] for row in basic_abd_rows(5)])
    write_spec(p.with_suffix('.spec'),
               ['Type=SR/AR Combination', 'Description=demo', 'UseBrakeRobot=False'])
    info = inspect_abd(p)
    assert info['config']['use_brake_robot'] is False
    assert info['config']['brake_robot_engaged'] is False
    assert info['calibration_candidate'] is True
    assert info['suitable_for_aeb_calibration'] is False
    assert info['role'] == 'unconfirmed'
    assert any('human' in reason for reason in info['reasons'])


def test_inspect_abd_br_type_spec_counts_as_brake_robot(tmp_path):
    p = tmp_path / 'V1_T6_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed'], ['', 's', 'km/h'],
                  [row[:3] for row in basic_abd_rows(5)])
    write_spec(p.with_suffix('.spec'),
               ['Type=BR Ramp Triggered Hold', 'Control=Motor Encoder'])
    info = inspect_abd(p)
    assert info['config']['use_brake_robot'] is None
    assert info['config']['brake_robot_engaged'] is True
    assert info['suitable_for_aeb_calibration'] is False


def test_inspect_abd_never_labels_collision_from_abort(tmp_path):
    p = tmp_path / 'V1_T7_R1.txt'
    write_abd_txt(p, ['Point', 'Time', 'Speed', 'SR path abort'],
                  ['', 's', 'km/h', ''], basic_abd_rows(6, abort_at=3))
    info = inspect_abd(p)
    assert info['collision_label'] is None
    assert 'SR path abort' in info['abort_channels']
    assert any('abort' in r.lower() and 'collision' in r.lower() for r in info['reasons'])
    assert not any(k.startswith('collision') and info[k] for k in info
                   if k != 'collision_label')


def test_inspect_abd_unsupported_format(tmp_path):
    p = tmp_path / 'run.tem'
    p.write_text('Version=30\n0.25\n0\n', encoding='latin-1')
    info = inspect_abd(p)
    assert info['parse_status'] == 'unsupported_format'
    assert info['suitable_for_aeb_calibration'] is False
    assert info['reasons']


# ---------------------------------------------------------------- audit_dataset

def build_audit_tree(root):
    abd_dir = root / 'Data' / 'ABD_Data' / 'BrandA' / 'g1'
    abd_dir.mkdir(parents=True)
    write_abd_txt(abd_dir / 'V1_T1_R1.txt', ['Point', 'Time', 'Speed'],
                  ['', 's', 'km/h'], [row[:3] for row in basic_abd_rows(5)])
    write_spec(abd_dir / 'V1_T1_R1.spec',
               ['Type=SR/AR Combination', 'UseBrakeRobot=False'])
    inter_dir = root / 'Data' / 'INTERACTION' / 'recorded_trackfiles' / 'LOC_Q'
    inter_dir.mkdir(parents=True)
    write_tracks_csv(inter_dir / 'vehicle_tracks_000.csv', VEHICLE_HEADER,
                     vehicle_rows(4))
    inv_dir = root / 'research_audit_20260910' / 'expanded_inventory'
    inv_dir.mkdir(parents=True)
    with open(inv_dir / 'all_runs_inventory.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['brand', 'data_role', 'validity_label'])
        for i in range(3):
            w.writerow(['BrandA', 'engineering_envelope', 'pass_parser_v1'])


def test_audit_dataset_writes_json_and_manifest(tmp_path):
    root = tmp_path / 'repo'
    build_audit_tree(root)
    out = tmp_path / 'audit_out'
    result = audit_dataset(root, out, limit=4)
    assert result['abd']['status'] == 'ok'
    assert result['interaction']['status'] == 'ok'
    assert result['abd']['n_files_total'] == 1
    assert result['interaction']['n_files_total'] == 1
    inv = result['inventory']
    assert inv['status'] == 'ok'
    assert inv['n_rows'] == 3
    assert inv['by_brand'] == {'BrandA': 3}
    assert 'not valid episodes' in inv['note']
    json_path = out / 'dataset_audit.json'
    manifest_path = out / 'manifest.csv'
    assert json_path.exists() and manifest_path.exists()
    loaded = json.loads(json_path.read_text(encoding='utf-8'))
    assert {'abd', 'interaction', 'inventory'} <= set(loaded)
    with open(manifest_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2                                     # one ABD + one INTERACTION
    assert {r['source'] for r in rows} == {'abd', 'interaction'}
    abd_row = next(r for r in rows if r['source'] == 'abd')
    assert abd_row['suitable_for_aeb_calibration'] == 'False'


def test_audit_dataset_tolerates_missing_sources(tmp_path):
    root = tmp_path / 'empty_repo'
    root.mkdir()
    out = tmp_path / 'audit_out2'
    result = audit_dataset(root, out, limit=2)
    assert result['abd']['status'] == 'missing'
    assert result['interaction']['status'] == 'missing'
    assert result['inventory']['status'] == 'missing'
    assert (out / 'dataset_audit.json').exists()
    assert (out / 'manifest.csv').exists()
