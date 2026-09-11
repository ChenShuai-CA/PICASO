import numpy as np
import torch
from scenario_lab.env import ScenarioEnv
from scenario_lab.schema import ScenarioSpec
from scenario_lab.policy import Actor, Critics, PolicyRunner, action_log_prob, load_bundle
from scenario_lab.train import advantages, TrainConfig, train

torch.set_num_threads(1)


def test_actor_absence_and_sequence_replay():
    torch.manual_seed(1)
    actor = Actor(32)
    env = ScenarioEnv()
    obs = env.reset()
    tokens = torch.tensor(obs['tokens'])[None, None].repeat(1, 4, 1, 1, 1)
    masks = torch.tensor(obs['token_mask'])[None, None].repeat(1, 4, 1, 1)
    active = torch.tensor(obs['actor_mask'])[None, None].repeat(1, 4, 1)
    full, _ = actor(tokens, masks, active)
    hidden, outputs = None, []
    for t in range(4):
        dist, hidden = actor(tokens[:, t:t+1], masks[:, t:t+1], active[:, t:t+1], hidden)
        outputs.append(dist.mean)
    torch.testing.assert_close(full.mean, torch.cat(outputs, 1))
    assert torch.isfinite(full.mean).all()
    assert not full.mean[:, :, 1].any()


def test_latent_logp_is_finite_for_large_actions():
    dist = torch.distributions.Normal(torch.zeros(3, 2), torch.ones(3, 2))
    assert torch.isfinite(action_log_prob(dist, torch.full((3, 2), 40.))).all()


def test_gae_terminal_and_padding_contract():
    adv, ret = advantages(np.array([1., 2.]), np.zeros((2, 2)), gamma=1, lam=1)
    np.testing.assert_allclose(ret, [[3., 3.], [2., 2.]])
    _, truncated_ret = advantages(np.array([1.]), np.zeros((1, 2)), gamma=1, lam=1,
                                  bootstrap=np.array([2., 3.]))
    np.testing.assert_allclose(truncated_ret, [[3., 4.]])


def test_end_to_end_update_and_bundle(tmp_path):
    result = train(tmp_path, TrainConfig(updates=1, episodes_per_update=2, epochs=1, hidden=32))
    assert result[0]['steps'] > 0 and np.isfinite(result[0]['loss'])
    runner, bundle = load_bundle(tmp_path / 'policy.pt')
    for branch in ('single', 'dual'):
        runner.reset()
        action = runner.act(ScenarioEnv().reset(ScenarioSpec(branch=branch)))
        assert action.shape == (2, 2) and np.abs(action).max() <= 1
    assert bundle['extra']['results_kind'] == 'pilot_not_paper_evidence'


def test_exact_interaction_budget_and_diagnostics(tmp_path):
    result = train(tmp_path, TrainConfig(updates=20, episodes_per_update=2, epochs=1,
                                         hidden=32, interaction_budget=37,
                                         role_action_mode='lane_locked',
                                         mixed_warmup_fraction=0.0))
    assert result[-1]['steps'] == 37
    assert sum(row['truncated_episodes'] for row in result) >= 1
    assert all('invalid_reasons' in row and 'projection_event_rate' in row
               and 'grad_norm' in row and 'branch_steps' in row
               and 'branch_episodes' in row for row in result)
    assert sum(sum(row['branch_steps'].values()) for row in result) == 37
    _, bundle = load_bundle(tmp_path / 'policy.pt')
    assert bundle['config']['interaction_budget'] == 37
    assert bundle['config']['role_action_mode'] == 'lane_locked'
    assert bundle['config']['mixed_warmup_fraction'] == 0.0


def test_gpu_update_and_both_branches(tmp_path):
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('CUDA runtime unavailable')
    result = train(tmp_path, TrainConfig(updates=1, episodes_per_update=2, epochs=1,
                                        hidden=32, mode='dual', device='cuda'))
    assert np.isfinite(result[0]['loss'])
    runner, bundle = load_bundle(tmp_path/'policy.pt', device='cuda')
    assert next(runner.actor.parameters()).is_cuda
    for branch in ('single', 'dual'):
        runner.reset()
        action = runner.act(ScenarioEnv().reset(ScenarioSpec(branch=branch)))
        assert np.isfinite(action).all()
        if branch == 'single':
            assert not action[1].any()


def test_exact_per_branch_interaction_budget(tmp_path):
    result = train(tmp_path, TrainConfig(
        updates=10, episodes_per_update=2, epochs=1, hidden=32, mode='mixed',
        robust=False, interaction_budget=36, branch_interaction_budget=18,
        mixed_warmup_fraction=0.0, role_action_mode='lane_locked'))
    totals = {'single': 0, 'dual': 0}
    for row in result:
        for branch, steps in row['branch_steps'].items():
            totals[branch] += steps
    assert totals == {'single': 18, 'dual': 18}
    assert result[-1]['cumulative_branch_steps'] == totals
