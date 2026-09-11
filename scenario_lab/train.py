"""Episode-sequence PPO/MAPPO with presence masks and full recurrent replay.

No claim of reproducing an external reference implementation. Reference-policy KL
and grouped mean-minus-std REINFORCE post-training are explicit optional mechanisms.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from .env import ScenarioEnv
from .sampling import sample_spec
from .policy import Actor, Critics, action_log_prob, save_bundle


@dataclass
class TrainConfig:
    seed: int = 7
    updates: int = 12
    episodes_per_update: int = 4
    epochs: int = 3
    hidden: int = 64
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .005
    reference_kl: float = .02
    robust_weight: float = .2
    mode: str = 'mixed'
    algorithm: str = 'mappo'
    role_constraints: bool = True
    robust: bool = True
    # Scenario sampler used for on-policy spec draws; 1 keeps the legacy training
    # distribution, 2 pairs physics v2 with constructive reference feasibility.
    sampler_version: int = 1
    device: str = 'cpu'
    threads: int = 1


def perturb_spec(spec, rng):
    s = deepcopy(spec)
    s.response_delay = float(rng.uniform(.1, .4))
    s.brake_deceleration = float(rng.uniform(5.5, 8.0))
    s.action_delay_steps = int(rng.integers(0, 3))
    s.target_accel_scale = float(rng.uniform(.85, 1.15))
    s.perturbation_source = 'assumed_sensitivity_not_abd_calibrated'
    return s


def advantages(rewards, values, gamma=.99, lam=.95):
    # Full finite-horizon episodes: terminal bootstrap zero by task definition.
    adv = np.zeros_like(values)
    last = np.zeros(2, dtype=np.float32)
    next_v = np.zeros(2, dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_v = values[t]
    return adv, adv + values


@torch.no_grad()
def collect_episode(actor, critic, spec, seed, device):
    env = ScenarioEnv()
    obs = env.reset(spec, seed)
    hidden = None
    items = {k: [] for k in ('tokens', 'token_mask', 'actor_mask', 'states', 'latent', 'logp', 'values', 'rewards')}
    while not env.done:
        tokens = torch.as_tensor(obs['tokens'], device=device)[None, None]
        masks = torch.as_tensor(obs['token_mask'], device=device)[None, None]
        active = torch.as_tensor(obs['actor_mask'], device=device)[None, None]
        state = env.critic_state()
        dist, hidden = actor(tokens, masks, active, hidden)
        latent = dist.sample()
        logp = action_log_prob(dist, latent)[0, 0].cpu().numpy()
        values = critic(torch.as_tensor(state, device=device)[None, None], tokens, masks)[0, 0].cpu().numpy()
        for key in ('tokens', 'token_mask', 'actor_mask'):
            items[key].append(obs[key])
        items['states'].append(state)
        items['latent'].append(latent[0, 0].cpu().numpy())
        items['logp'].append(logp)
        items['values'].append(values)
        action = (latent.tanh() * active[..., None])[0, 0].cpu().numpy()
        obs, reward, _, info = env.step(action)
        items['rewards'].append(reward)
    items = {k: np.asarray(v) for k, v in items.items()}
    items['summary'] = info
    return items


def pad_episodes(episodes, config):
    max_t = max(len(e['rewards']) for e in episodes)
    for e in episodes:
        e['advantages'], e['returns'] = advantages(e['rewards'], e['values'], config.gamma, config.gae_lambda)
    result = {}
    for key in episodes[0]:
        if key == 'summary':
            continue
        template = episodes[0][key]
        arr = np.zeros((len(episodes), max_t, *template.shape[1:]), dtype=template.dtype)
        for b, e in enumerate(episodes):
            arr[b, :len(e[key])] = e[key]
        result[key] = torch.as_tensor(arr, device=config.device)
    active = result['actor_mask']
    adv = result['advantages']
    selected = adv[active > 0]
    result['advantages'] = (adv - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6)
    return result


def train(output, config=None, pretrained=None):
    cfg = config or TrainConfig()
    from .runtime import resolve_device, record_runtime
    cfg.device = resolve_device(cfg.device)
    if cfg.updates < 1 or cfg.episodes_per_update < 2 or cfg.mode not in ('single', 'dual', 'mixed'):
        raise ValueError('invalid training configuration')
    if cfg.algorithm not in ('mappo', 'ippo', 'ppo') or (cfg.algorithm == 'ppo' and cfg.mode != 'single'):
        raise ValueError('PPO is restricted to single mode; choose MAPPO/IPPO for mixed')
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    actor = Actor(cfg.hidden).to(cfg.device)
    if pretrained:
        bundle = torch.load(pretrained, map_location=cfg.device, weights_only=True)
        actor.load_state_dict(bundle['actor'])
    critic = Critics(cfg.hidden, cfg.algorithm == 'ippo').to(cfg.device)
    reference = deepcopy(actor).eval() if pretrained else None
    if reference:
        for p in reference.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.learning_rate)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training.jsonl').exists():
        raise FileExistsError('Training output already contains history; use a fresh directory')
    record_runtime(output, cfg.device)
    (output / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    history = []
    started = time.perf_counter()
    total_steps = 0
    for update in range(cfg.updates):
        fraction = update / max(cfg.updates, 1)
        robust_phase = cfg.robust and fraction >= .8
        phase = 'robust_grouped' if robust_phase else ('single_warmup' if fraction < .2 and cfg.mode == 'mixed' else 'mixed')
        episodes = []
        # Robust groups share base initial state, differ only execution perturbations.
        base = None
        group_ids = []
        for e in range(cfg.episodes_per_update):
            branch = cfg.mode if cfg.mode != 'mixed' else ('single' if phase == 'single_warmup' else ('single' if e % 2 == 0 else 'dual'))
            if not robust_phase or e % 2 == 0:
                if robust_phase and cfg.mode == 'mixed':
                    branch = 'single' if (e // 2 + update) % 2 == 0 else 'dual'
                base = sample_spec(rng, branch, update * cfg.episodes_per_update + e,
                                   version=cfg.sampler_version)
                base.role_constraints = cfg.role_constraints
            spec = perturb_spec(base, rng) if robust_phase else deepcopy(base)
            ep = collect_episode(actor, critic, spec, int(rng.integers(2**31)), cfg.device)
            episodes.append(ep)
            group_ids.append(e // 2)
        total_steps += sum(len(e['rewards']) for e in episodes)
        batch = pad_episodes(episodes, cfg)
        active = batch['actor_mask']
        denominator = active.sum().clamp_min(1.)
        robust_baseline = float(np.mean([h['robust_utility'] for h in history[-5:]])) if history else 0.
        utilities = np.zeros(len(episodes), dtype=np.float32)
        if robust_phase:
            for gid in set(group_ids):
                indices = [i for i, g in enumerate(group_ids) if g == gid]
                scores = np.array([episodes[i]['summary']['risk'] if episodes[i]['summary']['valid'] else -1. for i in indices])
                utility = float(scores.mean() - cfg.robust_weight * scores.std())
                utilities[indices] = utility
        for epoch in range(1 if robust_phase else cfg.epochs):
            dist, _ = actor(batch['tokens'], batch['token_mask'], active)
            logp = action_log_prob(dist, batch['latent'])
            ratio = (logp - batch['logp']).clamp(-20, 20).exp()
            unclipped = ratio * batch['advantages']
            clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * batch['advantages']
            policy_loss = -(torch.minimum(unclipped, clipped) * active).sum() / denominator
            if robust_phase:
                # Joint group utility score-function estimator, single on-policy update.
                utility_tensor = torch.as_tensor(utilities - robust_baseline, device=cfg.device)
                episode_logp = (logp * active).sum((1, 2))
                policy_loss = -(utility_tensor * episode_logp).mean() / batch['tokens'].shape[1]
            predicted = critic(batch['states'], batch['tokens'], batch['token_mask'])
            value_loss = (((predicted - batch['returns']) ** 2) * active).sum() / denominator
            entropy = (dist.entropy().sum(-1) * active).sum() / denominator
            kl = torch.tensor(0., device=cfg.device)
            if reference is not None:
                with torch.no_grad():
                    reference_dist, _ = reference(batch['tokens'], batch['token_mask'], active)
                kl = (torch.distributions.kl_divergence(dist, reference_dist).sum(-1) * active).sum() / denominator
            loss = policy_loss + .5 * value_loss - cfg.entropy * entropy + cfg.reference_kl * kl
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), .5)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise FloatingPointError('non-finite training update')
            optimizer.step()
        row = dict(update=update + 1, phase=phase, steps=total_steps,
                   loss=float(loss.detach()), reference_kl=float(kl.detach()),
                   mean_return=float(np.mean([e['rewards'].sum() for e in episodes])),
                   valid_rate=float(np.mean([e['summary']['valid'] for e in episodes])),
                   collision_rate=float(np.mean([e['summary']['collision'] for e in episodes])),
                   robust_utility=float(utilities.mean()), elapsed_s=time.perf_counter() - started)
        history.append(row)
        with (output / 'training.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        save_bundle(output / 'policy.pt', actor, asdict(cfg), critic,
                    dict(steps=total_steps, pretrained=bool(pretrained), results_kind='pilot_not_paper_evidence'))
    return history
