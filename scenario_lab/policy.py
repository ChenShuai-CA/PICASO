"""Shared role-conditioned recurrent actor; privileged critic has a separate API."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .env import OBS_DIM, CRITIC_DIM


class Actor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.token_encoder = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh())
        self.role_embedding = nn.Embedding(2, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, 2), nn.Linear(hidden, 2)])
        self.log_std = nn.Parameter(torch.full((2, 2), -.9))
        for head in self.heads:
            nn.init.orthogonal_(head.weight, .01)
            nn.init.zeros_(head.bias)

    def forward(self, tokens, token_mask, actor_mask, hidden=None):
        # [batch,time,actor,entity,feature]; no global state argument exists here.
        batch, length, actors, entities, _ = tokens.shape
        x = self.token_encoder(tokens).reshape(-1, entities, self.hidden)
        mask = token_mask.reshape(-1, entities).clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True  # Prevent all-masked attention NaNs for absent actors.
        roles = torch.arange(actors, device=tokens.device).view(1, 1, actors).expand(batch, length, actors)
        query = self.role_embedding(roles.reshape(-1)).unsqueeze(1)
        pooled, _ = self.attention(query, x, x, key_padding_mask=~mask, need_weights=False)
        pooled = pooled.reshape(batch, length, actors, self.hidden)
        sequence = pooled.permute(0, 2, 1, 3).reshape(batch * actors, length, self.hidden)
        sequence, next_hidden = self.memory(sequence, hidden)
        sequence = sequence.reshape(batch, actors, length, self.hidden).permute(0, 2, 1, 3)
        mean = torch.stack([self.heads[a](sequence[:, :, a]) for a in range(actors)], dim=2)
        mean = mean * actor_mask[..., None]
        std = self.log_std.clamp(-4, 1).exp().view(1, 1, 2, 2).expand_as(mean)
        return Normal(mean, std), next_hidden


class Critics(nn.Module):
    def __init__(self, hidden=64, decentralized=False):
        super().__init__()
        self.decentralized = decentralized
        dim = OBS_DIM * 3 + 3 if decentralized else CRITIC_DIM
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state, tokens, token_mask):
        if self.decentralized:
            x = torch.cat((tokens.flatten(-2), token_mask.float()), dim=-1)
            return self.net(x).squeeze(-1)
        return self.net(state).expand(*state.shape[:-1], 2)


def action_log_prob(distribution, latent):
    # Stable tanh Jacobian: log(1-tanh(z)^2) = 2(log2-z-softplus(-2z)).
    correction = 2 * (np.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
    return (distribution.log_prob(latent) - correction).sum(-1)


class PolicyRunner:
    def __init__(self, actor, device='cpu', deterministic=True):
        self.actor = actor.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.hidden = None

    def reset(self):
        self.hidden = None

    @torch.no_grad()
    def act(self, obs):
        d = self.device
        tokens = torch.as_tensor(obs['tokens'], device=d)[None, None]
        mask = torch.as_tensor(obs['token_mask'], device=d)[None, None]
        active = torch.as_tensor(obs['actor_mask'], device=d)[None, None]
        dist, self.hidden = self.actor(tokens, mask, active, self.hidden)
        latent = dist.mean if self.deterministic else dist.sample()
        return (latent.tanh() * active[..., None])[0, 0].cpu().numpy()


def save_bundle(path, actor, config, critic=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version='0.1', hidden=actor.hidden, actor=actor.state_dict(),
                    critic=critic.state_dict() if critic else None,
                    config=config, extra=extra or {}), path)


def load_bundle(path, device='cpu'):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get('schema_version') != '0.1':
        raise ValueError('unsupported policy bundle schema')
    actor = Actor(bundle['hidden'])
    actor.load_state_dict(bundle['actor'])
    return PolicyRunner(actor, device), bundle
