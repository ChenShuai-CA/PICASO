"""Observable-state simulation. Only critic_state/rewards may contain hidden truth."""
from collections import deque
from copy import deepcopy
import math
import numpy as np
from .schema import Body, ScenarioSpec, EpisodeRecord
from .geometry import clearance, segment_blocked
from .sampling import sample_spec, sample_spec_v1, sample_spec_v2  # versioned re-export; legacy name kept

OBS_DIM = 15
CRITIC_DIM = 18
MAX_ACTORS = 2
DT = 0.02
DECISION_DT = 0.1


class ScenarioEnv:
    def __init__(self, record=False):
        self.record_enabled = record

    def reset(self, spec=None, seed=0):
        self.spec = deepcopy(spec or ScenarioSpec())
        self.rng = np.random.default_rng(seed)
        s = self.spec
        # physics v2: a waiting pedestrian stands at rest until pedestrian_delay.
        self.pedestrian_started = not (s.physics_version >= 2 and s.pedestrian_delay > 0)
        ped_initial = s.pedestrian_speed if self.pedestrian_started else 0.
        self.bodies = [Body(0., 0., s.ego_speed, 0., 'vehicle'),
                       Body(s.crossing_x, s.pedestrian_y, ped_initial,
                            math.pi / 2, 'pedestrian', 0.6, 0.6)]
        if s.branch == 'dual':
            self.bodies.append(Body(s.occluder_x, s.occluder_y, s.occluder_speed, 0., 'vehicle'))
        if any(clearance(a, b) == 0 for i, a in enumerate(self.bodies) for b in self.bodies[i + 1:]):
            raise ValueError('initial footprint overlap')
        self.time = 0.
        self.done = False
        self.tracks = [{} for _ in self.bodies]
        self.mask = np.array([1., float(s.branch == 'dual')], dtype=np.float32)
        self.action_queue = deque([np.zeros((2, 2)) for _ in range(s.action_delay_steps)])
        self.brake_queue = deque([0.] * int(round(s.response_delay / DT)))
        self.last_actions = np.zeros((2, 2))
        self.min_clearance = float('inf')
        self.peak_risk = 0.
        self.collision = False
        self.collision_speed = 0.
        self.invalid_reasons = set()
        self.clipped_actions = 0
        self.occluded_steps = 0
        self.total_steps = 0
        self.effort_total = 0.
        self.first_brake_time = None
        self.record = EpisodeRecord(scenario=s.to_dict(), seed=int(seed))
        self._refresh_tracks()
        obs = self.observe()
        if self.record_enabled:
            self._record(obs, np.zeros((2, 2)))
        return obs

    def _visible(self, observer_id, target_id):
        if observer_id == target_id:
            return True
        a, b = self.bodies[observer_id], self.bodies[target_id]
        if math.hypot(a.x - b.x, a.y - b.y) > 80:
            return False
        return not any(k not in (observer_id, target_id) and body.kind == 'vehicle'
                       and segment_blocked((a.x, a.y), (b.x, b.y), body)
                       for k, body in enumerate(self.bodies))

    def _refresh_tracks(self):
        for i, observer in enumerate(self.bodies):
            for j, body in enumerate(self.bodies):
                if self._visible(i, j):
                    noise = self.rng.normal(0, self.spec.observation_noise, 2) if i != j else np.zeros(2)
                    self.tracks[i][j] = dict(x=body.x + noise[0], y=body.y + noise[1],
                                            vx=body.speed * math.cos(body.heading),
                                            vy=body.speed * math.sin(body.heading),
                                            heading=body.heading, last_seen=self.time,
                                            kind=body.kind, visible=True)
                elif j in self.tracks[i]:
                    self.tracks[i][j]['visible'] = False

    def _estimate(self, observer, target):
        track = self.tracks[observer].get(target)
        if track is None or self.time - track['last_seen'] > 2.0:
            return None
        est = dict(track)
        est['age'] = max(0., self.time - track['last_seen'])
        est['x'] += est['vx'] * est['age']
        est['y'] += est['vy'] * est['age']
        return est

    def observe(self):
        tokens = np.zeros((2, 3, OBS_DIM), dtype=np.float32)
        token_mask = np.zeros((2, 3), dtype=bool)
        for actor in range(len(self.bodies) - 1):
            observer = actor + 1
            own = self.bodies[observer]
            for target in range(len(self.bodies)):
                est = self._estimate(observer, target)
                if est is None:
                    continue
                token_mask[actor, target] = True
                # All target dynamic values below come from observer's track, never truth.
                tokens[actor, target] = [(est['x'] - own.x) / 40, (est['y'] - own.y) / 10,
                                         est['vx'] / 15, est['vy'] / 15,
                                         math.cos(est['heading']), math.sin(est['heading']),
                                         est['visible'], est['age'] / 2, 1.,
                                         est['kind'] == 'vehicle', est['kind'] == 'pedestrian',
                                         observer == target, target == 0,
                                         (self.spec.crossing_x - own.x) / 40 if actor == 0 else 0.,
                                         (self.spec.occluder_y - own.y) / 10 if actor == 1 else 0.]
        return dict(tokens=tokens, token_mask=token_mask, actor_mask=self.mask.copy(),
                    roles=np.array([0, 1], dtype=np.int64))

    def critic_state(self):
        state = np.zeros(CRITIC_DIM, dtype=np.float32)
        for i, b in enumerate(self.bodies):
            state[i * 5:(i + 1) * 5] = [b.x / 40, b.y / 10, b.speed / 15,
                                       math.cos(b.heading), math.sin(b.heading)]
        state[15:17] = self.mask
        state[17] = self.time / self.spec.horizon
        return state

    def _ego_request(self):
        ego = self.bodies[0]
        for j in range(1, len(self.bodies)):
            est = self._estimate(0, j)
            if est is None:
                continue
            dx, dy = est['x'] - ego.x, est['y'] - ego.y
            relative_speed = ego.speed - est['vx']
            if dx < -3 or relative_speed <= 0.05:
                continue
            ttc = max(0., (dx - 2.6) / relative_speed)
            predicted_y = dy + est['vy'] * ttc
            lateral_margin = 1.2 if est['kind'] == 'pedestrian' else 1.8
            crosses = abs(predicted_y) <= lateral_margin or abs(dy) <= lateral_margin
            if not crosses:
                continue
            if self.spec.controller == 'ttc':
                trigger = ttc <= self.spec.controller_threshold
            else:
                stopping = ego.speed ** 2 / (2 * self.spec.brake_deceleration)
                trigger = dx - 2.6 <= stopping + ego.speed * self.spec.response_delay + self.spec.controller_threshold
            if trigger:
                return -self.spec.brake_deceleration
        return 0.

    def _integrate(self, actions, brake):
        s = self.spec
        ego = self.bodies[0]
        ego.speed = max(0., ego.speed + brake * DT)
        ego.x += ego.speed * DT
        ego.acceleration = brake
        for i, b in enumerate(self.bodies[1:]):
            if i == 0 and s.physics_version >= 2 and self.time < s.pedestrian_delay:
                # Waiting at the curb: no dynamics at all, so reported speed matches
                # the frozen position and constant-velocity occlusion extrapolation
                # can no longer invent motion that never happens.
                b.speed = b.acceleration = 0.
                continue
            desired = float(actions[i, 0]) * (1.5 if i == 0 else 3.) * s.target_accel_scale
            jerk = 5. if i == 0 else 6.
            b.acceleration += float(np.clip(desired - b.acceleration, -jerk * DT, jerk * DT))
            b.speed = float(np.clip(b.speed + b.acceleration * DT, 0., 3. if i == 0 else 12.))
            if i == 0:
                if s.physics_version >= 2 and not self.pedestrian_started:
                    self.pedestrian_started = True
                    b.speed = s.pedestrian_speed  # known approximation: instantaneous start
                b.heading += float(actions[i, 1]) * .45 * DT
                if s.physics_version < 2 and self.time < s.pedestrian_delay:
                    # legacy v1 semantics: position frozen while speed/heading keep
                    # evolving (replay of 20260911 traces only; never use for new runs)
                    continue
            else:
                b.steering = float(np.clip(b.steering + actions[i, 1] * .2 * DT, -.3, .3))
                b.heading += b.speed / 2.7 * math.tan(b.steering) * DT
            b.x += b.speed * math.cos(b.heading) * DT
            b.y += b.speed * math.sin(b.heading) * DT

    def step(self, actions):
        if self.done:
            raise RuntimeError('reset required after termination')
        requested = np.asarray(actions, dtype=float)
        if requested.shape != (2, 2) or not np.isfinite(requested).all():
            raise ValueError('actions must be finite shape (2,2)')
        self.clipped_actions += int(np.any(np.abs(requested[self.mask > 0]) > 1.))
        bounded = np.clip(requested, -1, 1) * self.mask[:, None]
        self.action_queue.append(bounded.copy())
        applied = self.action_queue.popleft()
        previous_risk = self.peak_risk
        for _ in range(5):
            self.brake_queue.append(self._ego_request())
            brake = self.brake_queue.popleft()
            if brake < 0 and self.first_brake_time is None:
                self.first_brake_time = self.time
            self._integrate(applied, brake)
            self.time += DT
            self._refresh_tracks()
            distances = [clearance(self.bodies[0], b) for b in self.bodies[1:]]
            self.min_clearance = min(self.min_clearance, *distances)
            self.collision = min(distances) <= 1e-9
            if self.collision:
                self.collision_speed = self.bodies[0].speed
            p = self.bodies[1]
            if self.spec.role_constraints:
                if abs(p.x - self.spec.crossing_x) > 1.5 or not .8 <= p.heading <= 2.35:
                    self.invalid_reasons.add('pedestrian_role')
                if len(self.bodies) == 3:
                    car = self.bodies[2]
                    if abs(car.y - self.spec.occluder_y) > .7 or abs(car.heading) > .35:
                        self.invalid_reasons.add('occluder_role')
            if len(self.bodies) == 3 and clearance(self.bodies[1], self.bodies[2]) <= 1e-9:
                self.invalid_reasons.add('target_target_collision')
            if self.collision or self.invalid_reasons:
                break
        self.total_steps += 1
        self.occluded_steps += int(not self._visible(0, 1))
        self.peak_risk = max(self.peak_risk, math.exp(-self.min_clearance / 2.))
        self.done = bool(self.collision or self.invalid_reasons or self.time + 1e-9 >= self.spec.horizon)
        effort = float(np.square(applied - self.last_actions).sum() / max(self.mask.sum(), 1))
        self.effort_total += effort  # naturalness raw material, reported outside the reward
        reward = self.peak_risk - previous_risk - .002 * effort
        if self.collision:
            reward += self.collision_speed / max(self.spec.ego_speed, 1)
        if self.invalid_reasons:
            reward = -2.0
        self.last_actions = applied.copy()
        obs = self.observe()
        info = self.summary()
        if self.record_enabled:
            self._record(obs, bounded)
            self.record.summary = info
        return obs, float(reward), self.done, info

    def summary(self):
        return dict(scenario_id=self.spec.scenario_id, branch=self.spec.branch,
                    valid=not bool(self.invalid_reasons), invalid_reasons=sorted(self.invalid_reasons),
                    collision=self.collision, collision_speed=self.collision_speed,
                    min_clearance=self.min_clearance, risk=self.peak_risk if not self.invalid_reasons else 0.,
                    dangerous=not self.invalid_reasons and (self.collision or self.min_clearance < .5),
                    first_brake_time=self.first_brake_time, elapsed=self.time,
                    occlusion_fraction=self.occluded_steps / max(self.total_steps, 1),
                    clipped_actions=self.clipped_actions,
                    total_effort=self.effort_total,
                    perturbation_source=self.spec.perturbation_source)

    def _record(self, obs, requested):
        self.record.frames.append(dict(time=self.time, bodies=[b.to_dict() for b in self.bodies],
                                       observation={k: v.tolist() for k, v in obs.items()},
                                       requested_actions=requested.tolist(),
                                       applied_actions=self.last_actions.tolist()))
