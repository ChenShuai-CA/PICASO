"""Versioned scenario samplers and explicit, shareable condition manifests.

sampler v1 reproduces the sequential draws behind every 20260911 pilot evaluation
and must never be edited. v2 adds a constructive reference-feasibility constraint
(analytic pedestrian/occluder occupancy windows) so reference sets are executable
by the zero-action script policy by construction; stress sets stay unconstrained.
The constraint reads only spec parameters, never any evaluated policy, so it is a
sampling-time design rule rather than outcome-based filtering.
"""
import json
from pathlib import Path
import numpy as np
from .schema import ScenarioSpec

SAMPLER_VERSION = 2
PHYSICS_VERSION = 2
PURPOSES = ('diagnostic', 'development', 'heldout')
# Resampling keeps the occluder near the ego-pedestrian sight line band; values far
# outside it would break the occlusion mechanism the dual branch exists to study.
_OCCLUDER_FRACTION_RESAMPLE = (.5, .75)


def _draw_params(rng, branch, index, controller):
    # Draw order is load-bearing: it must match the legacy env.sample_spec exactly.
    branch = branch or ('single' if rng.random() < .5 else 'dual')
    x = float(rng.uniform(24, 42))
    return dict(branch=branch, scenario_id=f'{branch}-{index:05d}',
                ego_speed=float(rng.uniform(8, 14)), crossing_x=x,
                pedestrian_y=float(rng.uniform(-6, -4)),
                pedestrian_speed=float(rng.uniform(1.0, 1.9)),
                pedestrian_delay=float(rng.uniform(0, 1.2)),
                occluder_x=x * .6, occluder_y=-2.8,
                occluder_speed=float(rng.uniform(2, 5)), controller=controller)


def sample_spec_v1(rng, branch=None, index=0, controller='stopping'):
    return ScenarioSpec(**_draw_params(rng, branch, index, controller),
                        physics_version=1, sampler_version=1, condition_role='unspecified')


def occupancy_windows(spec):
    """1-D analytic times when the pedestrian occupies the occluder lane and the
    occluder occupies the crossing line (constant-speed kinematic approximation)."""
    lateral = (1.8 + .6) / 2       # occluder width + pedestrian width, half sum
    longitudinal = (4.5 + .6) / 2  # occluder length + pedestrian width, half sum
    ped_lo = spec.pedestrian_delay + (spec.occluder_y - lateral - spec.pedestrian_y) / spec.pedestrian_speed
    ped_hi = spec.pedestrian_delay + (spec.occluder_y + lateral - spec.pedestrian_y) / spec.pedestrian_speed
    occ_lo = (spec.crossing_x - longitudinal - spec.occluder_x) / spec.occluder_speed
    occ_hi = (spec.crossing_x + longitudinal - spec.occluder_x) / spec.occluder_speed
    return (ped_lo, ped_hi), (occ_lo, occ_hi)


def reference_feasible(spec, time_margin=.5):
    if spec.pedestrian_speed <= 0 or spec.occluder_speed <= 0:
        return False
    (ped_lo, ped_hi), (occ_lo, occ_hi) = occupancy_windows(spec)
    return ped_lo - occ_hi >= time_margin or occ_lo - ped_hi >= time_margin


def sample_spec_v2(rng, branch=None, index=0, controller='stopping', role='reference',
                   max_resamples=40, return_tries=False):
    if role not in ('reference', 'stress'):
        raise ValueError("role must be 'reference' or 'stress'")
    params = _draw_params(rng, branch, index, controller)
    kwargs = dict(physics_version=PHYSICS_VERSION, sampler_version=2, condition_role=role)
    spec = ScenarioSpec(**params, **kwargs)
    tries = 0
    if role == 'reference' and spec.branch == 'dual':
        # Redraw the five parameters that shape the pedestrian/occluder time windows.
        # crossing_x and ego_speed keep their first draw; v1 never paired parameters
        # across branches, so no cross-branch comparability is lost.
        while not reference_feasible(spec) and tries < max_resamples:
            tries += 1
            params['occluder_speed'] = float(rng.uniform(2, 5))
            params['pedestrian_delay'] = float(rng.uniform(0, 1.2))
            params['pedestrian_y'] = float(rng.uniform(-6, -4))
            params['pedestrian_speed'] = float(rng.uniform(1.0, 1.9))
            params['occluder_x'] = params['crossing_x'] * float(rng.uniform(*_OCCLUDER_FRACTION_RESAMPLE))
            spec = ScenarioSpec(**params, **kwargs)
        if not reference_feasible(spec):
            spec.condition_role = 'reference_infeasible'  # kept and counted, never dropped
    return (spec, tries) if return_tries else spec


def sample_spec(rng, branch=None, index=0, controller='stopping', version=SAMPLER_VERSION,
                role='reference', max_resamples=40):
    if version == 1:
        return sample_spec_v1(rng, branch, index, controller)
    if version == 2:
        return sample_spec_v2(rng, branch, index, controller, role, max_resamples)
    raise ValueError('unknown sampler version')


def export_conditions(path, seed=1000, count=20, branches=('single', 'dual'),
                      sampler_version=SAMPLER_VERSION, role='reference',
                      purpose='diagnostic', controller='stopping'):
    if purpose not in PURPOSES:
        raise ValueError(f'purpose must be one of {PURPOSES}')
    if count < 1:
        raise ValueError('count must be positive')
    rng = np.random.default_rng(seed)
    conditions, resample_counts = [], {}
    for branch in branches:
        counts = []
        for i in range(count):
            if sampler_version == 1:
                spec = sample_spec_v1(rng, branch, i, controller)
                counts.append(0)
            else:
                spec, tries = sample_spec_v2(rng, branch, i, controller, role, return_tries=True)
                counts.append(tries)
            conditions.append(dict(branch=branch, index=i, spec=spec.to_dict()))
        resample_counts[branch] = counts
    manifest = dict(condition_set_version=f'{purpose}_seed{seed}_samplerv{sampler_version}',
                    purpose=purpose, sampler_version=sampler_version,
                    physics_version=1 if sampler_version == 1 else PHYSICS_VERSION,
                    role='unspecified' if sampler_version == 1 else role,
                    seed=seed, count_per_branch=count, branches=list(branches),
                    controller=controller,
                    rng_note=('sequential draw over branches, matching evaluate() order; '
                              'v1 dual specs coincide with the joint single+dual pilot evaluations only'),
                    resample_counts=resample_counts, conditions=conditions)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load_conditions(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = {'condition_set_version', 'purpose', 'sampler_version',
               'physics_version', 'conditions'} - set(manifest)
    if missing:
        raise ValueError(f'condition manifest missing fields: {sorted(missing)}')
    if manifest['purpose'] not in PURPOSES:
        raise ValueError(f"unknown purpose {manifest['purpose']!r}")
    specs = []
    for entry in manifest['conditions']:
        spec = ScenarioSpec(**entry['spec'])
        if spec.sampler_version != manifest['sampler_version'] or spec.physics_version != manifest['physics_version']:
            raise ValueError('condition spec version does not match manifest header; refusing silent mismatch')
        specs.append(spec)
    return specs, manifest
