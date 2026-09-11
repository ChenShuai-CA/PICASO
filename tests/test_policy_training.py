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


def test_end_to_end_update_and_bundle(tmp_path):
    result = train(tmp_path, TrainConfig(updates=1, episodes_per_update=2, epochs=1, hidden=32))
    assert result[0]['steps'] > 0 and np.isfinite(result[0]['loss'])
    runner, bundle = load_bundle(tmp_path / 'policy.pt')
    for branch in ('single', 'dual'):
        runner.reset()
        action = runner.act(ScenarioEnv().reset(ScenarioSpec(branch=branch)))
        assert action.shape == (2, 2) and np.abs(action).max() <= 1
    assert bundle['extra']['results_kind'] == 'pilot_not_paper_evidence'


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
