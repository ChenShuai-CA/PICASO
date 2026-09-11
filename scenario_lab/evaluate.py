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
    def __init__(self, parameters, active_actors=2):
        values = np.asarray(parameters)
        if values.size != active_actors * 3:
            raise ValueError('parameter policy size does not match active actors')
        self.parameters = np.zeros((2, 3))
        self.parameters[:active_actors] = values.reshape(active_actors, 3)
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
    def __init__(self, parameters, knots=8, active_actors=2, longitudinal_only=False):
        values = np.asarray(parameters)
        if longitudinal_only:
            if values.size != knots * active_actors:
                raise ValueError('longitudinal trajectory size does not match active actors')
            self.sequence = np.zeros((knots, 2, 2))
            self.sequence[:, :active_actors, 0] = values.reshape(knots, active_actors)
        else:
            self.sequence = values.reshape(knots, 2, 2)
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


def run_episode(policy, spec, seed=0, record_path=None, max_decision_steps=None):
    if max_decision_steps is not None and max_decision_steps < 1:
        raise ValueError('max_decision_steps must be positive')
    env = ScenarioEnv(record=record_path is not None)
    obs = env.reset(spec, seed)
    policy.reset()
    steps, total_reward = 0, 0.
    start = time.perf_counter()
    while not env.done and (max_decision_steps is None or steps < max_decision_steps):
        action = policy.act(obs)
        obs, reward, _, info = env.step(action)
        total_reward += reward
        steps += 1
    info.update(seed=seed, decision_steps=steps, reward=total_reward,
                terminated=env.done, budget_truncated=not env.done,
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


def summarize(rows, seed=0, b_rounds=2000):
    """Branch metrics with group-clustered bootstrap CIs (preregistered B and seed).

    Episode-length, control-effort and clipping aggregates sit next to the outcome
    rates so the interaction budget and the naturalness raw material are explicit;
    invalid attempts stay in every denominator. b_rounds default upgraded from the
    pilot's hard-coded 1000 (P2 preregistration); pilot artifacts are not recomputed.
    """
    result = {}
    rng = np.random.default_rng(seed)
    for branch in ('single', 'dual'):
        subset = [r for r in rows if r['branch'] == branch]
        if not subset:
            continue
        valid = np.array([r['valid'] for r in subset])
        danger = np.array([r['dangerous'] for r in subset])
        steps = np.array([r['decision_steps'] for r in subset], dtype=float)
        # Cluster bootstrap: all perturbations of a base scenario remain together.
        groups = sorted({r['scenario_id'] for r in subset})
        group_rates = np.array([np.mean([r['dangerous'] for r in subset if r['scenario_id'] == g]) for g in groups])
        estimates = np.array([np.mean(rng.choice(group_rates, len(groups), replace=True)) for _ in range(b_rounds)])
        efforts = [r['total_effort'] for r in subset if 'total_effort' in r]
        result[branch] = dict(attempts=len(subset), independent_scenarios=len(groups),
                              bootstrap_rounds=b_rounds,
                              valid_rate=float(valid.mean()), dangerous_rate=float(danger.mean()),
                              dangerous_rate_cluster_ci95=np.quantile(estimates, [.025, .975]).tolist(),
                              collision_rate=float(np.mean([r['collision'] for r in subset])),
                              unique_valid_dangerous=len({signature(r) for r in subset if r['dangerous']}),
                              mean_clearance=float(np.mean([r['min_clearance'] for r in subset])),
                              total_steps=int(steps.sum()),
                              steps_mean=float(steps.mean()), steps_median=float(np.median(steps)),
                              effort_mean=float(np.mean(efforts)) if efforts else None,
                              clipped_rate=float(np.mean([r['clipped_actions'] / max(r['decision_steps'], 1)
                                                           for r in subset])),
                              brake_coverage=float(np.mean([r['first_brake_time'] is not None
                                                             for r in subset])),
                              wall_s=sum(r['wall_s'] for r in subset))
    return result


def evaluate(policy, output, count=20, seed=1000, branches=('single', 'dual'), perturbations=1, export=2,
             controller='stopping', conditions=None, condition_set_version=None,
             role_action_mode='none'):
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
        spec = deepcopy(spec)
        spec.role_action_mode = role_action_mode
        for j in range(perturbations):
            s = perturb_spec(spec, np.random.default_rng(seed + i * 100 + j)) if perturbations > 1 else spec
            path = output / f'{branch}_{i:04d}_{j:02d}.json' if i < export else None
            row = run_episode(policy, s, seed + i * 100 + j, path)
            rows.append(row)
    report = dict(results_kind='pilot_until_preregistered_multiseed_study', controller=controller,
                  role_action_mode=role_action_mode,
                  perturbations=perturbations, metrics=summarize(rows, seed), **header)
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def _cem_policy(spec, kind, parameters):
    active_actors = 1 if spec.branch == 'single' else 2
    if kind == 'parameters':
        searched_actors = active_actors if spec.role_action_mode == 'lane_locked' else 2
        return ParamPolicy(parameters, active_actors=searched_actors)
    return TrajectoryPolicy(parameters, active_actors=active_actors,
                            longitudinal_only=spec.role_action_mode == 'lane_locked')


def _cem_dimension(spec, kind):
    active_actors = 1 if spec.branch == 'single' else 2
    if kind == 'parameters':
        return 3 * active_actors if spec.role_action_mode == 'lane_locked' else 6
    if spec.role_action_mode == 'lane_locked':
        return 8 * active_actors
    return 32


def _cem(spec, budget, seed, population, kind, interaction_budget=None):
    """One CEM run on one condition; every sampled episode is kept and counted."""
    if interaction_budget is not None and interaction_budget < 2:
        raise ValueError('CEM interaction budget must be >=2')
    rng = np.random.default_rng(seed)
    dim = _cem_dimension(spec, kind)
    mean, std = np.zeros(dim), np.ones(dim) * .6
    attempts = []
    interaction_steps = 0
    best = None
    best_params = None
    while ((interaction_budget is not None and interaction_steps < interaction_budget)
           or (interaction_budget is None and len(attempts) < budget)):
        n = population if interaction_budget is not None else min(population, budget - len(attempts))
        candidates = np.clip(rng.normal(mean, std, (n, dim)), -1, 1)
        scored = []
        executed = []
        for candidate_index, parameters in enumerate(candidates):
            remaining = (None if interaction_budget is None
                         else interaction_budget - interaction_steps)
            if remaining is not None and remaining <= 0:
                break
            row = run_episode(_cem_policy(spec, kind, parameters), spec, seed,
                              max_decision_steps=remaining)
            interaction_steps += row['decision_steps']
            eligible = row['valid'] and not row['budget_truncated']
            score = (row['risk'] + row['collision_speed'] / spec.ego_speed
                     if eligible else (-3. if row['budget_truncated'] else -2.))
            row.update(score=score, search_parameters=parameters.tolist(),
                       cem_iteration=len(attempts) // population,
                       cem_candidate=candidate_index)
            attempts.append(row)
            scored.append(score)
            executed.append(parameters)
            if eligible and (best is None or score > best['score']):
                best, best_params = row, parameters.copy()
        executed = np.asarray(executed)
        elite = executed[np.argsort(scored)[-max(1, len(executed) // 4):]]
        mean = .3 * mean + .7 * elite.mean(0)
        std = np.maximum(.1, .3 * std + .7 * elite.std(0))
    return attempts, best, best_params


def search(spec, output, kind='parameters', budget=40, seed=0, population=8,
           interaction_budget=None, role_action_mode='none'):
    if (interaction_budget is None and budget < 2) or population < 2:
        raise ValueError('CEM requires budget and population >=2')
    started = time.perf_counter()
    spec = deepcopy(spec)
    spec.role_action_mode = role_action_mode
    attempts, best, best_params = _cem(spec, budget, seed, population, kind,
                                       interaction_budget=interaction_budget)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if best_params is not None:
        run_episode(_cem_policy(spec, kind, best_params), spec, seed, output / 'best_trace.json')
    result = dict(kind=kind, seed=seed,
                  parameters=best_params.tolist() if best_params is not None else None,
                  best=best, budget=budget if interaction_budget is None else None,
                  interaction_budget=interaction_budget,
                  role_action_mode=role_action_mode,
                  search_dimension=_cem_dimension(spec, kind),
                  total_interaction_steps=sum(r['decision_steps'] for r in attempts),
                  elapsed_s=time.perf_counter() - started,
                  valid_attempts=sum(r['valid'] for r in attempts),
                  warning='best-of-search result; compare only with equivalent search budget')
    (output / 'search.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'attempts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in attempts), encoding='utf-8')
    return result


def search_conditions(conditions, output, kind='parameters', budget=40, seed=0,
                      population=8, branches=('dual',), interaction_budget=None,
                      role_action_mode='none', condition_set_version=None):
    """Per-condition CEM across a frozen condition manifest (P2 preregistration).

    Budget is per condition; invalid attempts count toward evaluations and the
    denominators. Best-of-search rows are never comparable to single-sample
    policies without matching search budgets.
    """
    if (interaction_budget is None and budget < 2) or population < 2:
        raise ValueError('CEM requires budget and population >=2')
    specs = [c for c in conditions if c.branch in branches]
    if not specs:
        raise ValueError('no conditions match the requested branches')
    per_condition, all_attempts = [], []
    started = time.perf_counter()
    for i, original_spec in enumerate(specs):
        spec = deepcopy(original_spec)
        spec.role_action_mode = role_action_mode
        attempts, best, best_params = _cem(spec, budget, seed + i, population, kind,
                                           interaction_budget=interaction_budget)
        for row in attempts:
            row['condition_index'] = i
        all_attempts.extend(attempts)
        per_condition.append(dict(condition_index=i, scenario_id=spec.scenario_id,
                                  branch=spec.branch, evaluations=len(attempts),
                                  search_dimension=_cem_dimension(spec, kind),
                                  interaction_steps=sum(r['decision_steps'] for r in attempts),
                                  valid_attempts=sum(r['valid'] for r in attempts),
                                  completed_attempts=sum(not r['budget_truncated'] for r in attempts),
                                  completed_valid_attempts=sum(r['valid'] and not r['budget_truncated']
                                                               for r in attempts),
                                  best_score=best['score'] if best else None,
                                  best_valid=best['valid'] if best else None,
                                  best_dangerous=best['dangerous'] if best else None,
                                  best_parameters=best_params.tolist() if best_params is not None else None))
    steps = sum(r['decision_steps'] for r in all_attempts)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(kind=kind, seed=seed,
                  budget_per_condition=budget if interaction_budget is None else None,
                  interaction_budget_per_condition=interaction_budget,
                  population=population, role_action_mode=role_action_mode,
                  condition_set_version=condition_set_version,
                  search_dimensions=sorted({row['search_dimension'] for row in per_condition}),
                  branches=list(branches),
                  n_conditions=len(specs), total_evaluations=len(all_attempts),
                  total_interaction_steps=steps,
                  mean_steps_per_condition=steps / len(specs),
                  valid_attempts=sum(r['valid'] for r in all_attempts),
                  per_condition=per_condition,
                  elapsed_s=time.perf_counter() - started,
                  warning='best-of-search result; compare only with equivalent search budget')
    (output / 'conditions_search.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'attempts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in all_attempts),
                                           encoding='utf-8')
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
