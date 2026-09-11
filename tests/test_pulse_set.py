import numpy as np
import torch

from scenario_lab.env import OBS_DIM
from scenario_lab.pulse_set import PulseSetPredictor, SCHEMA, set_loss, train_pulse_set
from scenario_lab.pulse_set_data import select_diverse_candidates


def test_set_loss_rewards_covering_distinct_teachers():
    target = torch.tensor([[[[1., 0., 0.]], [[-1., 0., 0.]]]])
    candidate_mask = torch.tensor([[True, True]])
    actor_mask = torch.tensor([[1.]])
    covered = target.clone()
    collapsed = torch.zeros_like(covered)
    assert float(set_loss(covered, target, candidate_mask, actor_mask)) == 0.
    assert float(set_loss(collapsed, target, candidate_mask, actor_mask)) > 0.


def test_predictor_keeps_actor_inputs_separate():
    torch.manual_seed(3)
    model = PulseSetPredictor(hidden=16, heads=4).eval()
    tokens = torch.randn(2, 5, 2, 3, OBS_DIM)
    masks = torch.ones(2, 5, 2, 3, dtype=torch.bool)
    actors = torch.ones(2, 5, 2)
    changed = tokens.clone()
    changed[:, :, 1] += 50.
    with torch.no_grad():
        original = model(tokens, masks, actors)
        perturbed = model(changed, masks, actors)
    torch.testing.assert_close(original[:, :, 0], perturbed[:, :, 0])
    assert torch.all(original[..., 1] >= -.75)


def test_candidate_selection_applies_score_half_before_action_diversity():
    rows = []
    for index, (score, amplitude) in enumerate(
            [(6., .1), (5., .15), (4., -.9), (3., 1.), (2., -1.), (1., .8)]):
        rows.append({'score': score, '_search_seed': 10,
                     'cem_iteration': 0, 'cem_candidate': index,
                     'search_parameters': [amplitude, 0., 0.]})
    selected = select_diverse_candidates(rows, 'single', maximum=2)
    assert selected[0]['score'] == 6.
    assert selected[1]['score'] == 4.
    assert all(row['score'] >= 4. for row in selected)


def test_tiny_pulse_set_training_writes_selected_checkpoint(tmp_path):
    rng = np.random.default_rng(4)
    count = 4
    actor_mask = np.ones((count, 5, 2), dtype=np.float32)
    actor_mask[:2, :, 1] = 0.
    target = np.zeros((count, 8, 2, 3), dtype=np.float32)
    target[:, :2] = rng.uniform(-.7, .7, (count, 2, 2, 3))
    target[:2, :, 1] = 0.
    candidate_mask = np.zeros((count, 8), dtype=bool)
    candidate_mask[:, :2] = True
    corpus = tmp_path / 'tiny.npz'
    np.savez_compressed(
        corpus,
        tokens=rng.normal(size=(count, 5, 2, 3, OBS_DIM)).astype(np.float32),
        token_mask=np.ones((count, 5, 2, 3), dtype=bool),
        actor_mask=actor_mask, target=target, candidate_mask=candidate_mask,
        branch=np.asarray(['single', 'single', 'dual', 'dual']),
        split=np.asarray(['train', 'val', 'train', 'val']),
        uid=np.asarray(['a', 'b', 'c', 'd']), schema_version=np.asarray(SCHEMA))
    result = train_pulse_set(corpus, tmp_path / 'model', seed=5, epochs=2,
                             hidden=16, heads=4, batch_size=2, device='cpu')
    assert 1 <= result['selected_epoch'] <= 2
    assert (tmp_path / 'model' / 'model.pt').exists()
    assert (tmp_path / 'model' / 'training.json').exists()
