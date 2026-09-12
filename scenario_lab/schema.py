"""Serializable, versioned experiment inputs. Distances m, time s, angles rad."""
from dataclasses import asdict, dataclass, field
from typing import Literal
import math


@dataclass
class ScenarioSpec:
    branch: Literal['single', 'dual'] = 'single'
    scenario_id: str = 'crossing-0000'
    ego_speed: float = 10.0
    crossing_x: float = 32.0
    pedestrian_y: float = -5.0
    pedestrian_speed: float = 1.4
    pedestrian_delay: float = 0.0
    occluder_x: float = 19.0
    occluder_y: float = -2.8
    occluder_speed: float = 3.0
    horizon: float = 8.0
    controller: Literal['ttc', 'stopping'] = 'stopping'
    controller_threshold: float = 2.0
    brake_deceleration: float = 7.0
    # Legacy combined parameter retained for replay of experiments created
    # before the trigger-preview and brake-actuation delays were separated.
    response_delay: float = 0.2
    # New experiments may set these independently.  ``None`` preserves the
    # legacy behavior by falling back to ``response_delay`` for each role.
    controller_preview_delay: float | None = None
    aeb_actuation_delay: float | None = None
    action_delay_steps: int = 0
    target_accel_scale: float = 1.0
    observation_noise: float = 0.0
    role_constraints: bool = True
    # Optional execution projection for learned target actions. ``lane_locked``
    # keeps the crossing pedestrian and moving occluder on their assigned axes;
    # legacy experiments use ``none`` and remain replay-compatible.
    role_action_mode: Literal['none', 'lane_locked'] = 'none'
    protocol_level: str = 'research_extension'
    perturbation_source: str = 'assumed_sensitivity_not_abd_calibrated'
    # Version pinning so traces/conditions stay auditable across physics fixes.
    # physics v1: legacy pre-delay pedestrian semantics (replay compatibility only).
    # physics v2: pedestrian fully at rest (speed 0, no actions) before pedestrian_delay.
    physics_version: int = 2
    sampler_version: int = 2
    condition_role: str = 'unspecified'

    def __post_init__(self):
        if self.branch not in ('single', 'dual'):
            raise ValueError('branch must be single or dual')
        if self.controller not in ('ttc', 'stopping'):
            raise ValueError('unknown frozen controller')
        if self.physics_version not in (1, 2):
            raise ValueError('physics_version must be 1 or 2')
        if self.condition_role not in ('unspecified', 'reference', 'reference_infeasible', 'stress'):
            raise ValueError('unknown condition_role')
        if self.role_action_mode not in ('none', 'lane_locked'):
            raise ValueError('unknown role_action_mode')
        if self.sampler_version not in (1, 2):
            raise ValueError('sampler_version must be 1 or 2')
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('scenario contains non-finite values')
        if not (0 < self.ego_speed <= 25 and self.crossing_x > 8 and self.horizon > 0):
            raise ValueError('invalid initial geometry, speed or horizon')
        if not (0 <= self.pedestrian_speed <= 3 and 0 <= self.occluder_speed <= 15):
            raise ValueError('target speed out of bounds')
        optional_delays = [v for v in (
            self.controller_preview_delay, self.aeb_actuation_delay) if v is not None]
        if min(self.response_delay, self.observation_noise, self.pedestrian_delay,
               *optional_delays) < 0:
            raise ValueError('negative delay/noise')
        if not (0 <= self.action_delay_steps <= 10 and self.brake_deceleration > 0
                and 0 < self.target_accel_scale <= 2):
            raise ValueError('invalid execution parameters')

    def to_dict(self):
        return asdict(self)


@dataclass
class Body:
    x: float
    y: float
    speed: float
    heading: float
    kind: str
    length: float = 4.5
    width: float = 1.8
    steering: float = 0.0
    acceleration: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class EpisodeRecord:
    schema_version: str = '0.2'
    scenario: dict = field(default_factory=dict)
    seed: int = 0
    frames: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
