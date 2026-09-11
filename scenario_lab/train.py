"""Episode-sequence PPO/MAPPO with presence masks and full recurrent replay.

No claim of reproducing an external reference implementation. Reference-policy KL
and grouped mean-minus-std REINFORCE post-training are explicit optional mechanisms.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
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
    # Fraction of the training budget reserved for legacy single-branch warmup.
    # Fair shared-vs-specialized studies set this to zero.
    mixed_warmup_fraction: float = .2
    algorithm: str = 'mappo'
    role_constraints: bool = True
    role_action_mode: str = 'none'
    robust: bool = True
    # Scenario sampler used for on-policy spec draws; 1 keeps the legacy training
    # distribution, 2 pairs physics v2 with constructive reference feasibility.
    sampler_version: int = 1
    # When set, data collection stops at this exact number of decision steps.
    # The final episode may be truncated and bootstraps its critic value.
    interaction_budget: int | None = None
    # Optional exact budget for each branch in mixed mode.  The global budget
    # must equal twice this value; branch episodes may be truncated separately.
    branch_interaction_budget: int | None = None
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


def advantages(rewards, values, gamma=.99, lam=.95, bootstrap=None):
    # Full episodes use terminal bootstrap zero. Budget-truncated episodes pass
    # the critic value at the first uncollected state.
    adv = np.zeros_like(values)
    last = np.zeros(2, dtype=np.float32)
    next_v = (np.zeros(2, dtype=np.float32) if bootstrap is None
              else np.asarray(bootstrap, dtype=np.float32))
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_v = values[t]
    return adv, adv + values


@torch.no_grad()
def collect_episode(actor, critic, spec, seed, device, max_steps=None):
    env = ScenarioEnv()
    obs = env.reset(spec, seed)
    hidden = None
    items = {k: [] for k in ('tokens', 'token_mask', 'actor_mask', 'states', 'latent',
                             'actions', 'logp', 'values', 'rewards')}
    steps = 0
    while not env.done and (max_steps is None or steps < max_steps):
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
        items['actions'].append(action := (latent.tanh() * active[..., None])[0, 0].cpu().numpy())
        items['logp'].append(logp)
        items['values'].append(values)
        obs, reward, _, info = env.step(action)
        items['rewards'].append(reward)
        steps += 1
    truncated = not env.done
    bootstrap = None
    if truncated:
        tokens = torch.as_tensor(obs['tokens'], device=device)[None, None]
        masks = torch.as_tensor(obs['token_mask'], device=device)[None, None]
        state = env.critic_state()
        bootstrap = critic(torch.as_tensor(state, device=device)[None, None],
                           tokens, masks)[0, 0].cpu().numpy()
        info = dict(info, truncated_for_budget=True)
    else:
        info = dict(info, truncated_for_budget=False)
    items = {k: np.asarray(v) for k, v in items.items()}
    items['summary'] = info
    items['bootstrap'] = bootstrap
    return items


def pad_episodes(episodes, config):
    max_t = max(len(e['rewards']) for e in episodes)
    for e in episodes:
        e['advantages'], e['returns'] = advantages(
            e['rewards'], e['values'], config.gamma, config.gae_lambda, e['bootstrap'])
    result = {}
    for key in episodes[0]:
        if key in ('summary', 'bootstrap'):
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
    if cfg.role_action_mode not in ('none', 'lane_locked'):
        raise ValueError('invalid role_action_mode')
    if cfg.reference_kl < 0:
        raise ValueError('reference_kl must be nonnegative')
    if not 0 <= cfg.mixed_warmup_fraction < 1:
        raise ValueError('mixed_warmup_fraction must be in [0, 1)')
    if cfg.interaction_budget is not None and cfg.interaction_budget < cfg.episodes_per_update:
        raise ValueError('interaction_budget must allow at least one normal batch')
    if cfg.branch_interaction_budget is not None:
        if (cfg.mode != 'mixed' or cfg.robust
                or cfg.interaction_budget != 2 * cfg.branch_interaction_budget):
            raise ValueError('branch_interaction_budget requires non-robust mixed mode and '
                             'interaction_budget == 2 * branch_interaction_budget')
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
    reference = deepcopy(actor).eval() if pretrained and cfg.reference_kl > 0 else None
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
    total_branch_steps = Counter()
    for update in range(cfg.updates):
        if cfg.interaction_budget is not None and total_steps >= cfg.interaction_budget:
            break
        fraction = (total_steps / cfg.interaction_budget if cfg.interaction_budget
                    else update / max(cfg.updates, 1))
        robust_phase = cfg.robust and fraction >= .8
        phase = ('robust_grouped' if robust_phase else
                 ('single_warmup' if (fraction < cfg.mixed_warmup_fraction
                                      and cfg.mode == 'mixed') else 'mixed'))
        episodes = []
        # Robust groups share base initial state, differ only execution perturbations.
        base = None
        group_ids = []
        update_branch_steps = Counter()
        for e in range(cfg.episodes_per_update):
            branch = cfg.mode if cfg.mode != 'mixed' else ('single' if phase == 'single_warmup' else ('single' if e % 2 == 0 else 'dual'))
            if cfg.branch_interaction_budget is not None:
                remaining_by_branch = {
                    name: (cfg.branch_interaction_budget - total_branch_steps[name]
                           - update_branch_steps[name])
                    for name in ('single', 'dual')}
                if remaining_by_branch[branch] <= 0:
                    branch = 'dual' if branch == 'single' else 'single'
                if remaining_by_branch[branch] <= 0:
                    break
            if not robust_phase or e % 2 == 0:
                if robust_phase and cfg.mode == 'mixed':
                    branch = 'single' if (e // 2 + update) % 2 == 0 else 'dual'
                base = sample_spec(rng, branch, update * cfg.episodes_per_update + e,
                                   version=cfg.sampler_version)
                base.role_constraints = cfg.role_constraints
                base.role_action_mode = cfg.role_action_mode
            spec = perturb_spec(base, rng) if robust_phase else deepcopy(base)
            remaining = (None if cfg.interaction_budget is None
                         else cfg.interaction_budget - total_steps
                         - sum(len(item['rewards']) for item in episodes))
            if cfg.branch_interaction_budget is not None:
                branch_remaining = (cfg.branch_interaction_budget - total_branch_steps[branch]
                                    - update_branch_steps[branch])
                remaining = min(remaining, branch_remaining)
            if remaining is not None and remaining <= 0:
                break
            ep = collect_episode(actor, critic, spec, int(rng.integers(2**31)), cfg.device,
                                 max_steps=remaining)
            episodes.append(ep)
            update_branch_steps[branch] += len(ep['rewards'])
            group_ids.append(e // 2)
        if not episodes:
            break
        total_steps += sum(len(e['rewards']) for e in episodes)
        total_branch_steps.update(update_branch_steps)
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
        invalid_counts = Counter(reason for e in episodes
                                 for reason in e['summary']['invalid_reasons'])
        branch_episodes = Counter(e['summary']['branch'] for e in episodes)
        branch_steps = Counter()
        for e in episodes:
            branch_steps[e['summary']['branch']] += len(e['rewards'])
        action_values = np.concatenate([e['actions'].reshape(-1, 2, 2) for e in episodes])
        row = dict(update=update + 1, phase=phase, steps=total_steps,
                   loss=float(loss.detach()), reference_kl=float(kl.detach()),
                   grad_norm=float(grad_norm.detach()),
                   mean_return=float(np.mean([e['rewards'].sum() for e in episodes])),
                   valid_rate=float(np.mean([e['summary']['valid'] for e in episodes])),
                   collision_rate=float(np.mean([e['summary']['collision'] for e in episodes])),
                   invalid_reasons=dict(sorted(invalid_counts.items())),
                   branch_episodes=dict(sorted(branch_episodes.items())),
                   branch_steps=dict(sorted(branch_steps.items())),
                   cumulative_branch_steps=dict(sorted(total_branch_steps.items())),
                   action_abs_mean=float(np.abs(action_values).mean()),
                   action_std=float(action_values.std()),
                   projection_event_rate=float(sum(e['summary']['role_projection_events']
                                                   for e in episodes)
                                               / sum(len(e['rewards']) for e in episodes)),
                   truncated_episodes=sum(e['summary']['truncated_for_budget'] for e in episodes),
                   robust_utility=float(utilities.mean()), elapsed_s=time.perf_counter() - started)
        history.append(row)
        with (output / 'training.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        save_bundle(output / 'policy.pt', actor, asdict(cfg), critic,
                    dict(steps=total_steps, pretrained=bool(pretrained), results_kind='pilot_not_paper_evidence'))
    if cfg.interaction_budget is not None and total_steps != cfg.interaction_budget:
        raise RuntimeError(f'interaction budget not reached: {total_steps}/{cfg.interaction_budget}; '
                           'increase --updates')
    if cfg.branch_interaction_budget is not None and any(
            total_branch_steps[name] != cfg.branch_interaction_budget
            for name in ('single', 'dual')):
        raise RuntimeError(f'branch interaction budget not reached: {dict(total_branch_steps)}')
    return history
