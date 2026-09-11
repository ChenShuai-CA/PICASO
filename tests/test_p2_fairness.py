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


def test_train_sampler_v2_path_records_version(tmp_path):
    history = train(tmp_path, TrainConfig(updates=1, episodes_per_update=2, epochs=1,
                                          hidden=32, mode='dual', algorithm='ippo',
                                          sampler_version=2, device='cpu'))
    assert len(history) == 1
    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['sampler_version'] == 2
    assert (tmp_path / 'policy.pt').exists()
