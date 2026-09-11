"""Small actor-local multi-head predictor for sets of successful CEM pulses."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .env import OBS_DIM


SCHEMA = 'pulse-set-v1'


class PulseSetPredictor(nn.Module):
    def __init__(self, hidden=16, heads=4):
        super().__init__()
        self.hidden, self.n_heads = hidden, heads
        self.token_encoder = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh())
        self.role_embedding = nn.Embedding(2, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.ModuleList([nn.Linear(hidden, heads * 3) for _ in range(2)])
        for layer in self.output:
            nn.init.orthogonal_(layer.weight, .01)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens, token_mask, actor_mask):
        batch, length, actors, entities, _ = tokens.shape
        encoded = self.token_encoder(tokens).reshape(-1, entities, self.hidden)
        mask = token_mask.reshape(-1, entities).clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True
        roles = torch.arange(actors, device=tokens.device).view(1, 1, actors)
        query = self.role_embedding(roles.expand(batch, length, actors).reshape(-1)).unsqueeze(1)
        pooled, _ = self.attention(query, encoded, encoded,
                                   key_padding_mask=~mask, need_weights=False)
        sequence = pooled.reshape(batch, length, actors, self.hidden)
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * actors, length, self.hidden)
        sequence, _ = self.memory(sequence)
        final = sequence[:, -1].reshape(batch, actors, self.hidden)
        values = torch.stack([
            self.output[actor](final[:, actor]).reshape(batch, self.n_heads, 3)
            for actor in range(actors)], dim=2)
        values = values.tanh()
        # The observation prefix ends at 0.5 s, hence normalized start must be
        # at least -0.75 because onset=(start+1)*2.
        values = torch.stack((values[..., 0], .875 * values[..., 1] + .125,
                              values[..., 2]), dim=-1)
        return values * actor_mask[:, -1, None, :, None]


def set_loss(prediction, target, candidate_mask, actor_mask):
    """Per-example coverage + precision Chamfer loss."""
    diff = prediction[:, :, None] - target[:, None]
    active = actor_mask[:, None, None, :, None]
    distance = (diff.square() * active).sum((-1, -2)) \
        / (actor_mask.sum(-1)[:, None, None].clamp_min(1.) * 3.)
    coverage = distance.min(1).values
    coverage = (coverage * candidate_mask).sum(-1) / candidate_mask.sum(-1).clamp_min(1.)
    masked = distance.masked_fill(~candidate_mask[:, None], float('inf'))
    precision = masked.min(2).values.mean(1)
    return coverage + .25 * precision


def train_pulse_set(corpus, output, seed=7, epochs=100, hidden=16, heads=4,
                    batch_size=16, device='cpu'):
    from .runtime import resolve_device, record_runtime
    device, output = resolve_device(device), Path(output)
    if (output / 'model.pt').exists():
        raise FileExistsError('pulse-set model exists; use a fresh output')
    record_runtime(output, device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(corpus, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data['schema_version']) != SCHEMA:
        raise ValueError('unsupported pulse-set corpus')
    train = np.flatnonzero(data['split'] == 'train')
    val = np.flatnonzero(data['split'] == 'val')
    if not len(train) or not len(val):
        raise ValueError('nonempty train and val required')
    model = PulseSetPredictor(hidden, heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    counts = {branch: max(1, int(np.sum(data['branch'][train] == branch)))
              for branch in ('single', 'dual')}

    def loss_for(indices, branch_weighted):
        tensors = {key: torch.as_tensor(data[key][indices], device=device)
                   for key in ('tokens', 'token_mask', 'actor_mask',
                               'target', 'candidate_mask')}
        prediction = model(tensors['tokens'], tensors['token_mask'], tensors['actor_mask'])
        losses = set_loss(prediction, tensors['target'], tensors['candidate_mask'],
                          tensors['actor_mask'][:, -1])
        if branch_weighted:
            weights = torch.as_tensor(
                [1. / counts[str(data['branch'][index])] for index in indices],
                device=device)
            return (losses * weights).sum() / weights.sum()
        return losses.mean()

    history, best = [], None
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        order = rng.permutation(train)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            loss = loss_for(indices, True)
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('non-finite pulse-set update')
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); branch_values = {}
        with torch.no_grad():
            for branch in ('single', 'dual'):
                indices = val[data['branch'][val] == branch]
                branch_values[branch] = float(loss_for(indices, False))
        balanced = float(np.mean(list(branch_values.values())))
        row = {'epoch': epoch, 'train_set_loss': float(np.mean(losses)),
               'val_single_set_loss': branch_values['single'],
               'val_dual_set_loss': branch_values['dual'],
               'val_branch_balanced_set_loss': balanced}
        history.append(row)
        if best is None or balanced < best['metric']:
            best = {'metric': balanced, 'epoch': epoch,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    config = {'seed': seed, 'epochs': epochs, 'selected_epoch': best['epoch'],
              'hidden': hidden, 'heads': heads, 'batch_size': batch_size,
              'learning_rate': 3e-4, 'history_steps': 5,
              'loss': 'coverage_chamfer_plus_0.25_precision_branch_weighted'}
    torch.save({'schema_version': SCHEMA, 'config': config,
                'model': model.state_dict(),
                'corpus_sha256': hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
                'heldout_read': False}, output / 'model.pt')
    (output / 'training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return {'selected_epoch': best['epoch'], 'selected_val_set_loss': best['metric']}


def load_pulse_set(path, device='cpu'):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get('schema_version') != SCHEMA:
        raise ValueError('unsupported pulse-set bundle')
    config = bundle['config']
    model = PulseSetPredictor(config['hidden'], config['heads']).to(device).eval()
    model.load_state_dict(bundle['model'])
    return model, bundle
