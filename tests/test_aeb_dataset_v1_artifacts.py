import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scenario_lab.train import load_perturb_config


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'runs/20260912_abd_aeb_pedal_screen'
TIMING = ROOT / 'runs/20260913_abd_request_timing'


def read_rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def key(row):
    return row['run'], row['sha256']


def test_v1_is_an_exact_partition_of_v0_after_residual_risk_removal():
    v0 = read_rows(DATASET / 'aeb_dataset_v0.csv')
    v1 = read_rows(DATASET / 'aeb_dataset_v1.csv')
    removed = read_rows(DATASET / 'aeb_dataset_v1_removed.csv')
    assert (len(v0), len(v1), len(removed)) == (707, 680, 27)
    assert {key(r) for r in v1}.isdisjoint({key(r) for r in removed})
    assert {key(r) for r in v1} | {key(r) for r in removed} == {
        key(r) for r in v0}
    assert sum(r['removal_reason'] == 'foot_pre_event' for r in removed) == 12
    assert sum(r['removal_reason'] == 'no_tension_force_ge_30n'
               for r in removed) == 15


def test_timing_proxies_align_with_v1_and_keep_observation_separate():
    v1 = read_rows(DATASET / 'aeb_dataset_v1.csv')
    timing = read_rows(TIMING / 'timing.csv')
    assert len(timing) == 680
    assert {key(r) for r in timing} == {key(r) for r in v1}
    for row in timing:
        onset = float(row['onset_s'])
        assert float(row['observed_response_onset_s']) == onset
        for milliseconds in (150, 250, 350):
            request = float(row[f'aeb_request_proxy_s_d{milliseconds}'])
            active = float(row[f'aeb_active_proxy_s_d{milliseconds}'])
            assert request == active
            assert onset - request == pytest.approx(milliseconds / 1000)


def test_shipped_v2_config_uses_split_delay_and_standard_quantiles():
    config = load_perturb_config(TIMING / 'abd_derived_v2_sensitivity.json')
    rows = read_rows(TIMING / 'timing.csv')
    values = [float(r['equiv_decel_ms2']) for r in rows
              if r['stopped'] == '1'
              and r['outcome'] != 'contact_and_stopped'
              and r['equiv_decel_ms2']]
    brake = config['parameters']['brake_deceleration']
    assert len(values) == 612
    assert brake['low'] == round(float(np.quantile(values, 0.05)), 3)
    assert brake['high'] == round(float(np.quantile(values, 0.95)), 3)
    assert 'response_delay' not in config['parameters']
    assert config['parameters']['aeb_actuation_delay']['low'] == 0.15
    assert config['parameters']['aeb_actuation_delay']['high'] == 0.35
    assert config['parameters']['controller_preview_delay']['low'] == 0.25
    assert config['parameters']['controller_preview_delay']['high'] == 0.25
