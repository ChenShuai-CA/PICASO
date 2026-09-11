import numpy as np
import torch

from scenario_lab.env import OBS_DIM
from scenario_lab.evaluate import ParamPolicy
from scenario_lab.pulse import PulsePolicy, PulsePredictor, train_pulse_predictor


def observation():
    return {
        'tokens': np.zeros((2, 3, OBS_DIM), dtype=np.float32),
        'token_mask': np.ones((2, 3), dtype=bool),
        'actor_mask': np.ones(2, dtype=np.float32),
    }


def test_pulse_predictor_keeps_actor_observations_separate():
    torch.manual_seed(3)
    predictor = PulsePredictor(hidden=16).eval()
    tokens = torch.zeros((1, 5, 2, 3, OBS_DIM))
    masks = torch.ones((1, 5, 2, 3), dtype=torch.bool)
    active = torch.ones((1, 5, 2))
    original = predictor(tokens, masks, active)
    tokens[:, :, 1] = 100
    changed = predictor(tokens, masks, active)
    assert torch.allclose(original[:, 0], changed[:, 0])
    assert not torch.allclose(original[:, 1], changed[:, 1])


def test_pulse_policy_reproduces_parameter_policy_after_zero_history():
    parameters = torch.tensor([[.8, -.75, 0.], [-.4, -.5, -.5]])

    class FixedPredictor(torch.nn.Module):
        def forward(self, tokens, token_mask, actor_mask):
            return parameters.to(tokens.device)[None].expand(tokens.shape[0], -1, -1)

    learned = PulsePolicy(FixedPredictor(), history_steps=5)
    teacher = ParamPolicy(parameters.numpy().ravel(), active_actors=2)
    obs = observation()
    learned.reset(); teacher.reset()
    for _ in range(30):
        assert np.allclose(learned.act(obs), teacher.act(obs))


def test_pulse_training_selects_a_validation_checkpoint(tmp_path):
    count = 8
    rng = np.random.default_rng(4)
    arrays = {
        'tokens': rng.normal(size=(count, 5, 2, 3, OBS_DIM)).astype(np.float32),
        'token_mask': np.ones((count, 5, 2, 3), dtype=bool),
        'actor_mask': np.ones((count, 5, 2), dtype=np.float32),
        'target': rng.uniform(-.5, .5, size=(count, 2, 3)).astype(np.float32),
        'branch': np.asarray(['single', 'dual', 'single', 'dual'] * 2),
        'split': np.asarray(['train'] * 4 + ['val'] * 4),
        'scenario_id': np.asarray([f's{i}' for i in range(count)]),
        'schema_version': np.asarray('incremental-pulse-v1'),
    }
    arrays['actor_mask'][arrays['branch'] == 'single', :, 1] = 0
    corpus = tmp_path / 'pulse.npz'
    np.savez_compressed(corpus, **arrays)
    result = train_pulse_predictor(corpus, tmp_path / 'model', epochs=2,
                                   hidden=16, seed=5, device='cpu', batch_size=2)
    assert 1 <= result['selected_epoch'] <= 2
    bundle = torch.load(tmp_path / 'model' / 'pulse.pt', map_location='cpu',
                        weights_only=True)
    assert bundle['config']['selection_metric'].startswith('branch-balanced')
    assert bundle['extra']['heldout_read'] is False
