"""Paired evaluation, trace export, CEM search, replay and latency measurement."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import time
import numpy as np
from .env import ScenarioEnv, sample_spec
from .schema import ScenarioSpec
from .train import perturb_spec


class ScriptPolicy:
    def reset(self):
        pass

    def act(self, obs):
        return np.zeros((2, 2), dtype=np.float32)


class ParamPolicy(ScriptPolicy):
    """Low-dimensional bounded acceleration pulses; steering stays role aligned."""
    def __init__(self, parameters):
        self.parameters = np.asarray(parameters).reshape(2, 3)
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, obs):
        out = np.zeros((2, 2))
        for actor, (amplitude, start, duration) in enumerate(self.parameters):
            if (start + 1) * 2 <= self.t < (start + 1) * 2 + (duration + 1) * 1.5:
                out[actor, 0] = amplitude
        self.t += .1
        return out * obs['actor_mask'][:, None]


class TrajectoryPolicy(ScriptPolicy):
    """Fixed action knots, independent of new observations (except presence mask)."""
    def __init__(self, parameters, knots=8):
        self.sequence = np.asarray(parameters).reshape(knots, 2, 2)
        self.knots = knots
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, obs):
        action = self.sequence[min(self.t // 10, self.knots - 1)]
        self.t += 1
        return action * obs['actor_mask'][:, None]


class OneLearningPolicy:
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.reset()

    def act(self, obs):
        action = self.runner.act(obs)
        action[1] = 0
        return action


def run_episode(policy, spec, seed=0, record_path=None):
    env = ScenarioEnv(record=record_path is not None)
    obs = env.reset(spec, seed)
    policy.reset()
    steps, total_reward = 0, 0.
    start = time.perf_counter()
    while not env.done:
        action = policy.act(obs)
        obs, reward, _, info = env.step(action)
        total_reward += reward
        steps += 1
    info.update(seed=seed, decision_steps=steps, reward=total_reward,
                wall_s=time.perf_counter() - start, spec=spec.to_dict())
    if record_path:
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(asdict(env.record), ensure_ascii=False), encoding='utf-8')
    return info


def signature(row):
    # A preregistered coarse parameter+outcome bin signature, not learned diversity.
    spec = row['spec']
    bins = (row['branch'], round(spec['ego_speed']), round(spec['crossing_x'] / 2),
            round(spec['pedestrian_speed'] / .25), round(spec['pedestrian_delay'] / .25),
            round(row['min_clearance'] / .25), round(row['collision_speed'] / 2))
    return hashlib.sha256(repr(bins).encode()).hexdigest()[:16]


def summarize(rows, seed=0):
    result = {}
    rng = np.random.default_rng(seed)
    for branch in ('single', 'dual'):
        subset = [r for r in rows if r['branch'] == branch]
        if not subset:
            continue
        valid = np.array([r['valid'] for r in subset])
        danger = np.array([r['dangerous'] for r in subset])
        # Cluster bootstrap: all perturbations of a base scenario remain together.
        groups = sorted({r['scenario_id'] for r in subset})
        group_rates = np.array([np.mean([r['dangerous'] for r in subset if r['scenario_id'] == g]) for g in groups])
        estimates = np.array([np.mean(rng.choice(group_rates, len(groups), replace=True)) for _ in range(1000)])
        result[branch] = dict(attempts=len(subset), independent_scenarios=len(groups),
                              valid_rate=float(valid.mean()), dangerous_rate=float(danger.mean()),
                              dangerous_rate_cluster_ci95=np.quantile(estimates, [.025, .975]).tolist(),
                              collision_rate=float(np.mean([r['collision'] for r in subset])),
                              unique_valid_dangerous=len({signature(r) for r in subset if r['dangerous']}),
                              mean_clearance=float(np.mean([r['min_clearance'] for r in subset])),
                              total_steps=sum(r['decision_steps'] for r in subset),
                              wall_s=sum(r['wall_s'] for r in subset))
    return result


def evaluate(policy, output, count=20, seed=1000, branches=('single', 'dual'), perturbations=1, export=2,
             controller='stopping', conditions=None, condition_set_version=None):
    if count < 1 or perturbations < 1:
        raise ValueError('evaluation count and perturbations must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if conditions is None:
        rng = np.random.default_rng(seed)
        plan = []
        for branch in branches:
            for i in range(count):
                plan.append((branch, i, sample_spec(rng, branch, i, controller)))
        header = dict(seed=seed, count_per_branch=count, branches=list(branches))
    else:
        # Explicit condition manifest: identical specs for every method regardless of
        # how many branches an evaluation run covers (removes rng draw-order coupling).
        plan = []
        seen = {}
        for spec in conditions:
            seen[spec.branch] = seen.get(spec.branch, 0)
            plan.append((spec.branch, seen[spec.branch], spec))
            seen[spec.branch] += 1
        versions = {(spec.physics_version, spec.sampler_version) for spec in conditions}
        if len(versions) != 1:
            raise ValueError('condition set mixes physics/sampler versions; refusing to evaluate')
        header = dict(seed=seed, count_per_branch={b: n for b, n in seen.items()},
                      branches=sorted(seen), condition_set_version=condition_set_version,
                      physics_version=conditions[0].physics_version,
                      sampler_version=conditions[0].sampler_version)
    for branch, i, spec in plan:
        for j in range(perturbations):
            s = perturb_spec(spec, np.random.default_rng(seed + i * 100 + j)) if perturbations > 1 else spec
            path = output / f'{branch}_{i:04d}_{j:02d}.json' if i < export else None
            row = run_episode(policy, s, seed + i * 100 + j, path)
            rows.append(row)
    report = dict(results_kind='pilot_until_preregistered_multiseed_study', controller=controller,
                  perturbations=perturbations, metrics=summarize(rows, seed), **header)
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def search(spec, output, kind='parameters', budget=40, seed=0, population=8):
    if budget < 2 or population < 2:
        raise ValueError('CEM requires budget and population >=2')
    rng = np.random.default_rng(seed)
    dim = 6 if kind == 'parameters' else 32
    cls = ParamPolicy if kind == 'parameters' else TrajectoryPolicy
    mean, std = np.zeros(dim), np.ones(dim) * .6
    attempts = []
    best = None
    best_params = None
    started = time.perf_counter()
    while len(attempts) < budget:
        n = min(population, budget - len(attempts))
        candidates = np.clip(rng.normal(mean, std, (n, dim)), -1, 1)
        scored = []
        for parameters in candidates:
            row = run_episode(cls(parameters), spec, seed)
            score = row['risk'] + row['collision_speed'] / spec.ego_speed if row['valid'] else -2.
            row['score'] = score
            attempts.append(row)
            scored.append(score)
            if best is None or score > best['score']:
                best, best_params = row, parameters.copy()
        elite = candidates[np.argsort(scored)[-max(1, n // 4):]]
        mean = .3 * mean + .7 * elite.mean(0)
        std = np.maximum(.1, .3 * std + .7 * elite.std(0))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_episode(cls(best_params), spec, seed, output / 'best_trace.json')
    result = dict(kind=kind, parameters=best_params.tolist(), best=best, budget=budget,
                  total_interaction_steps=sum(r['decision_steps'] for r in attempts),
                  elapsed_s=time.perf_counter() - started,
                  valid_attempts=sum(r['valid'] for r in attempts),
                  warning='best-of-search result; compare only with equivalent search budget')
    (output / 'search.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'attempts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in attempts), encoding='utf-8')
    return result


def replay(path, atol=1e-9):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    scenario = dict(record['scenario'])
    version = scenario.setdefault('physics_version', 1)
    if version == 1:
        print('warning: legacy physics v1 trace; replaying under v1 semantics for compatibility')
    env = ScenarioEnv()
    env.reset(ScenarioSpec(**scenario), record['seed'])
    for frame in record['frames'][1:]:
        env.step(frame['requested_actions'])
        for expected, actual in zip(frame['bodies'], env.bodies):
            for key in ('x', 'y', 'speed', 'heading'):
                if abs(expected[key] - getattr(actual, key)) > atol:
                    raise AssertionError(f'replay mismatch {key} at t={env.time}')
    return dict(matched=True, frames=len(record['frames']), physics_version=version,
                schema_version=record.get('schema_version', 'unknown'),
                summary=env.summary(),
                scope='action trace determinism; closed-loop reproduction also requires policy bundle')


def benchmark(policy, output, steps=10000, seed=91):
    if steps < 1:
        raise ValueError('steps must be positive')
    import platform
    import torch
    torch.set_num_threads(1)
    result = dict(hardware=dict(platform=platform.platform(), torch=torch.__version__,
                                inference_device=str(getattr(policy, 'device', 'cpu'))), branches={})
    for branch in ('single', 'dual'):
        env = ScenarioEnv()
        rng = np.random.default_rng(seed)
        obs = env.reset(sample_spec(rng, branch), seed)
        policy.reset()
        # Warmup excludes initialization and framework first-use overhead.
        for _ in range(20):
            policy.act(obs)
        policy.reset()
        timings = []
        simulation_s = 0.
        started = time.perf_counter()
        for i in range(steps):
            if env.done:
                obs = env.reset(sample_spec(rng, branch, i), seed + i)
                policy.reset()
            if str(getattr(policy, 'device', 'cpu')).startswith('cuda'):
                torch.cuda.synchronize()
            t = time.perf_counter()
            obs = env.observe()
            action = np.clip(policy.act(obs), -1, 1) * obs['actor_mask'][:, None]
            if str(getattr(policy, 'device', 'cpu')).startswith('cuda'):
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t)
            before = env.time
            obs, _, _, _ = env.step(action)
            simulation_s += env.time - before
        elapsed = time.perf_counter() - started
        result['branches'][branch] = dict(decision_steps=steps, p50_ms=float(np.quantile(timings, .5) * 1000),
                                          p99_ms=float(np.quantile(timings, .99) * 1000),
                                          deadline_miss_rate=float(np.mean(np.array(timings) >= .1)),
                                          realtime_factor=simulation_s / elapsed, wall_s=elapsed,
                                          rendering=False, scope='observe+actor+action_bounds; full RTF includes physics/tracking')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
