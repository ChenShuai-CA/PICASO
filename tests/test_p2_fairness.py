"""P2 fairness infrastructure: budget-exact multi-condition CEM, interaction-budget
and naturalness aggregates, and the sampler-v2 training path (preregistered in
GLM_CHANGELOG P2 R2-R4)."""
import json
from pathlib import Path

import numpy as np
import pytest

from scenario_lab.evaluate import ScriptPolicy, run_episode, search, search_conditions, summarize
from scenario_lab.sampling import sample_spec_v2
from scenario_lab.schema import ScenarioSpec
from scenario_lab.train import TrainConfig, train
from scripts.summarize_p2 import paired_cluster_stats
from scripts.summarize_p21_rescue import two_way_paired_delta


def test_search_conditions_budget_exact(tmp_path):
    specs = [sample_spec_v2(np.random.default_rng(50 + i), 'single', i) for i in range(2)]
    result = search_conditions(specs, tmp_path, kind='parameters', budget=4, seed=9,
                               population=2, branches=('single',))
    assert result['total_evaluations'] == 8  # budget per condition, nothing dropped
    assert [c['evaluations'] for c in result['per_condition']] == [4, 4]
    assert result['valid_attempts'] == sum(c['valid_attempts'] for c in result['per_condition'])
    assert result['total_interaction_steps'] == sum(c['interaction_steps'] for c in result['per_condition'])
    assert result['mean_steps_per_condition'] == result['total_interaction_steps'] / 2
    assert 'best-of-search' in result['warning']
    lines = (tmp_path / 'attempts.jsonl').read_text().strip().splitlines()
    assert len(lines) == 8  # invalid attempts stay in the log and the denominators
    for c in result['per_condition']:
        assert c['best_parameters'] is not None
    saved = json.loads((tmp_path / 'conditions_search.json').read_text())
    assert saved['budget_per_condition'] == 4 and saved['n_conditions'] == 2


def test_search_conditions_interaction_budget_exact_and_lane_locked(tmp_path):
    specs = [ScenarioSpec(branch='single', scenario_id=f's{i}', horizon=.3)
             for i in range(2)]
    result = search_conditions(
        specs, tmp_path, kind='trajectory', interaction_budget=7, seed=9,
        population=2, branches=('single',), role_action_mode='lane_locked',
        condition_set_version='test-dev')
    assert result['total_interaction_steps'] == 14
    assert result['mean_steps_per_condition'] == 7
    assert result['interaction_budget_per_condition'] == 7
    assert result['budget_per_condition'] is None
    assert result['role_action_mode'] == 'lane_locked'
    assert result['condition_set_version'] == 'test-dev'
    assert result['search_dimensions'] == [8]
    assert [row['interaction_steps'] for row in result['per_condition']] == [7, 7]
    attempts = [json.loads(line) for line in (tmp_path / 'attempts.jsonl').read_text().splitlines()]
    assert sum(row['budget_truncated'] for row in attempts) == 2
    assert all(row['spec']['role_action_mode'] == 'lane_locked' for row in attempts)
    assert all(row['role_projection_events'] == 0 for row in attempts)
    assert all(len(row['search_parameters']) == 8 for row in attempts)


def test_lane_locked_cem_uses_only_active_longitudinal_dimensions(tmp_path):
    single = search(ScenarioSpec(branch='single', horizon=.3), tmp_path / 'single',
                    kind='parameters', interaction_budget=7, seed=3,
                    population=2, role_action_mode='lane_locked')
    dual = search(ScenarioSpec(branch='dual', horizon=.3), tmp_path / 'dual',
                  kind='trajectory', interaction_budget=7, seed=3,
                  population=2, role_action_mode='lane_locked')
    assert single['search_dimension'] == 3
    assert dual['search_dimension'] == 16
    assert single['total_interaction_steps'] == dual['total_interaction_steps'] == 7


def test_truncated_episode_is_explicit():
    row = run_episode(ScriptPolicy(), ScenarioSpec(branch='single'), 42,
                      max_decision_steps=2)
    assert row['decision_steps'] == 2
    assert row['budget_truncated'] is True and row['terminated'] is False


def test_budget_truncated_cem_candidate_cannot_be_selected(tmp_path):
    result = search(ScenarioSpec(branch='single', horizon=1.0), tmp_path,
                    kind='parameters', interaction_budget=2, seed=3,
                    population=2, role_action_mode='lane_locked')
    assert result['total_interaction_steps'] == 2
    assert result['best'] is None and result['parameters'] is None


def test_search_single_spec_mode_unchanged(tmp_path):
    result = search(ScenarioSpec(branch='single'), tmp_path, kind='parameters',
                    budget=4, seed=3, population=2)
    assert result['budget'] == 4 and len(result['parameters']) == 6
    assert (tmp_path / 'search.json').exists() and (tmp_path / 'attempts.jsonl').exists()


def test_total_effort_accumulates_and_zero_for_script_policy():
    info = run_episode(ScriptPolicy(), ScenarioSpec(branch='single'), 42)
    assert info['decision_steps'] > 0
    assert info['total_effort'] == 0.0  # zero action: never a control discontinuity


def _row(i, scenario, dangerous):
    return dict(branch='single', valid=not dangerous, dangerous=dangerous, collision=False,
                collision_speed=0., min_clearance=1.2, scenario_id=scenario,
                spec=dict(ego_speed=8., crossing_x=5., pedestrian_speed=1.4, pedestrian_delay=.2),
                decision_steps=30 + i, wall_s=.01, clipped_actions=i, first_brake_time=None,
                total_effort=float(i))


def test_summarize_budget_and_naturalness_fields():
    rows = [_row(0, 's0', True), _row(1, 's0', False), _row(0, 's1', False), _row(1, 's1', False)]
    out = summarize(rows, seed=2026, b_rounds=50)
    m = out['single']
    assert m['bootstrap_rounds'] == 50 and m['attempts'] == 4
    assert m['independent_scenarios'] == 2
    assert m['total_steps'] == 30 + 31 + 30 + 31
    assert m['steps_mean'] == pytest.approx(30.5) and m['steps_median'] == pytest.approx(30.5)
    assert m['effort_mean'] == pytest.approx(0.5)
    assert m['clipped_rate'] == pytest.approx((0 / 30 + 1 / 31 + 0 / 30 + 1 / 31) / 4)
    assert m['brake_coverage'] == 0.
    assert len(m['dangerous_rate_cluster_ci95']) == 2


def test_paired_cluster_difference_keeps_common_scenarios():
    left = [_row(0, 's0', True), _row(0, 's1', False)]
    right = [_row(0, 's0', False), _row(0, 's1', False)]
    result = paired_cluster_stats(left, right, 'single', b_rounds=100, seed=3)
    assert result['scenarios'] == 2
    assert result['dangerous_rate_delta'] == pytest.approx(.5)
    assert result['valid_rate_delta'] == pytest.approx(-.5)
    assert result['dangerous_rate_delta_ci95'][0] >= 0.


def test_paired_cluster_difference_rejects_mismatched_scenarios():
    with pytest.raises(ValueError, match='same single scenario ids'):
        paired_cluster_stats([_row(0, 's0', False)], [_row(0, 's1', False)], 'single')


def test_two_way_bootstrap_uses_paired_seed_scenario_grid():
    left = {7: {'s0': 1., 's1': 0.}, 17: {'s0': 1., 's1': 1.}}
    right = {7: {'s0': 0., 's1': 0.}, 17: {'s0': 0., 's1': 1.}}
    result = two_way_paired_delta(left, right, b_rounds=100, seed=4)
    assert result['delta'] == pytest.approx(.5)
    assert result['seeds'] == 2 and result['scenarios'] == 2
    assert result['ci95'][0] >= 0.


def test_train_sampler_v2_path_records_version(tmp_path):
    history = train(tmp_path, TrainConfig(updates=1, episodes_per_update=2, epochs=1,
                                          hidden=32, mode='dual', algorithm='ippo',
                                          sampler_version=2, device='cpu'))
    assert len(history) == 1
    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['sampler_version'] == 2
    assert (tmp_path / 'policy.pt').exists()
