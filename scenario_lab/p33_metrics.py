"""P3.3.2 evaluation metrics: CV baseline, displacement errors, honest aggregation.

All errors are computed in the anchor-local metric frame of ``scene-shard-v1``.
Aggregation is two-level by construction: per-agent errors are first averaged
inside an independent group (Waymo ``scenario_id`` / INTERACTION
``location::case``), then across groups.  Pooled agent-level numbers are
reported separately and are never substituted for group-level results.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .p33_spec import CONFIG_PATH, load_p33_config

FUTURE_TIMES = np.arange(1, 51) * 0.1  # t = 0.1 s ... 5.0 s at 10 Hz
ENDPOINT_INDEX = 49  # t = 5.0 s
MAIN_EVAL_ROLES = (1, 2)  # anchor, prediction_target

# Engineering diagnostic thresholds (not G1 gates); recorded in every report.
KINEMATIC_LIMITS = {
    "speed_limit_mps": 35.0,
    "accel_limit_mps2": 10.0,
    "jerk_limit_mps3": 20.0,
}


@dataclass
class AgentErrorRecord:
    source: str
    group_id: str
    sample_id: str
    agent_slot: int
    agent_type: int
    agent_role: int
    agents_truncated: bool
    map_truncated: bool
    ade: float
    fde: float
    min_ade: float
    min_fde: float
    valid_steps: int
    endpoint_valid: bool


def constant_velocity_prediction(agent_history: np.ndarray,
                                 state_valid_mask: np.ndarray) -> np.ndarray:
    """p_hat(t) = p_last_valid + v_last_valid * (t - t_last_valid) as [A,50,2].

    The last valid history state may sit before t = 0 (e.g. t = -0.1 s); the
    extrapolation runs forward from that state's own timestamp, never from an
    unobserved t = 0 position.
    """
    agents, steps, _ = agent_history.shape
    prediction = np.zeros((agents, len(FUTURE_TIMES), 2), dtype=np.float64)
    for agent in range(agents):
        valid = np.flatnonzero(state_valid_mask[agent])
        if not len(valid):
            continue  # no observable state; masked out of evaluation anyway
        last = int(valid[-1])
        t_last = (last - (steps - 1)) * 0.1
        prediction[agent] = (agent_history[agent, last, :2]
                             + agent_history[agent, last, 2:4]
                             * (FUTURE_TIMES - t_last)[:, None])
    return prediction


def evaluate_agent_mask(agent_present_mask: np.ndarray, agent_role: np.ndarray,
                        future_valid_mask: np.ndarray) -> np.ndarray:
    """Main evaluation agents: present AND role in {anchor, prediction_target} AND future."""
    return (agent_present_mask
            & np.isin(agent_role, MAIN_EVAL_ROLES)
            & future_valid_mask.any(axis=-1))


def displacement_errors(prediction: np.ndarray, truth: np.ndarray,
                        valid: np.ndarray) -> dict[str, np.ndarray]:
    """Per-agent ADE/FDE for one trajectory sample each.

    FDE is defined only at the t = 5 s endpoint; agents whose endpoint is
    invalid get NaN and ``endpoint_valid=False``.
    """
    agents = prediction.shape[0]
    per_step = np.linalg.norm(prediction - truth, axis=-1)  # [A, 50]
    steps = valid.sum(axis=1)
    ade = np.full(agents, np.nan)
    has_steps = steps > 0
    ade[has_steps] = (per_step * valid).sum(axis=1)[has_steps] / steps[has_steps]
    endpoint_valid = valid[:, ENDPOINT_INDEX].copy()
    fde = np.where(endpoint_valid, per_step[:, ENDPOINT_INDEX], np.nan)
    return {"ade": ade, "fde": fde, "valid_steps": steps, "endpoint_valid": endpoint_valid}


def min_k_displacement_errors(predictions: np.ndarray, truth: np.ndarray,
                              valid: np.ndarray) -> dict[str, np.ndarray]:
    """Per-agent minADE@K / minFDE@K over K sampled trajectories."""
    samples, agents = predictions.shape[0], predictions.shape[1]
    per_step = np.linalg.norm(predictions - truth[None], axis=-1)  # [K, A, 50]
    steps = valid.sum(axis=1)
    ade = np.full((samples, agents), np.nan)
    has_steps = steps > 0
    ade[:, has_steps] = (per_step * valid[None]).sum(axis=2)[:, has_steps] / steps[None, has_steps]
    endpoint_valid = valid[:, ENDPOINT_INDEX]
    fde = np.where(endpoint_valid[None], per_step[:, :, ENDPOINT_INDEX], np.nan)
    min_ade = ade.min(axis=0)
    # min over valid samples per agent; agents with no valid endpoint stay NaN (no warning)
    finite = np.where(np.isnan(fde), np.inf, fde)
    min_fde = np.where(np.isfinite(finite).any(axis=0), finite.min(axis=0), np.nan)
    return {"min_ade": min_ade, "min_fde": min_fde}


def _records_for_batch(trajectories: np.ndarray, batch: dict,
                       eval_mask: np.ndarray | None = None) -> list[AgentErrorRecord]:
    """Build per-agent records from [B, K, A, 50, 2] rollouts (K=1 allowed)."""
    config = load_p33_config(CONFIG_PATH)
    records: list[AgentErrorRecord] = []
    size = len(batch["sample_id"])
    for index in range(size):
        truth = batch["future_xy"][index]
        valid = batch["future_valid_mask"][index]
        if eval_mask is None:
            mask = evaluate_agent_mask(batch["agent_present_mask"][index],
                                       batch["agent_role"][index], valid)
        else:
            mask = eval_mask[index]
        single = displacement_errors(trajectories[index, 0], truth, valid)
        if trajectories.shape[1] > 1:
            best = min_k_displacement_errors(trajectories[index], truth, valid)
            min_ade, min_fde = best["min_ade"], best["min_fde"]
        else:
            min_ade, min_fde = single["ade"], single["fde"]
        for slot in np.flatnonzero(mask):
            records.append(AgentErrorRecord(
                source=batch["sample_source"][index],
                group_id=batch["group_id"][index],
                sample_id=batch["sample_id"][index],
                agent_slot=int(slot),
                agent_type=int(batch["agent_type"][index, slot]),
                agent_role=int(batch["agent_role"][index, slot]),
                agents_truncated=bool(batch["agents_truncated"][index]),
                map_truncated=bool(batch["map_truncated"][index]),
                ade=float(single["ade"][slot]),
                fde=float(single["fde"][slot]),
                min_ade=float(min_ade[slot]),
                min_fde=float(min_fde[slot]),
                valid_steps=int(single["valid_steps"][slot]),
                endpoint_valid=bool(single["endpoint_valid"][slot]),
            ))
    return records


def cv_records_for_batch(batch: dict) -> list[AgentErrorRecord]:
    """Constant-velocity records replicated to K=6 identical samples."""
    config = load_p33_config(CONFIG_PATH)
    size = len(batch["sample_id"])
    cv = np.stack([constant_velocity_prediction(batch["agent_history"][i],
                                                batch["state_valid_mask"][i])
                   for i in range(size)])
    six = np.repeat(cv[:, None], 6, axis=1)  # [B, K=6, A, 50, 2]
    return _records_for_batch(six, batch)


def rollout_records_for_batch(trajectories: np.ndarray, batch: dict,
                              eval_mask: np.ndarray | None = None) -> list[AgentErrorRecord]:
    """Model rollout records for [B, K, A, 50, 2] sampled trajectories."""
    return _records_for_batch(trajectories, batch, eval_mask)


def group_level_table(records: list[AgentErrorRecord]) -> dict:
    """Per-source two-level aggregation: mean of per-group agent means."""
    groups: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        groups[record.source][record.group_id]["ade"].append(record.ade)
        groups[record.source][record.group_id]["min_ade"].append(record.min_ade)
        if record.endpoint_valid:
            groups[record.source][record.group_id]["fde"].append(record.fde)
            groups[record.source][record.group_id]["min_fde"].append(record.min_fde)
    table: dict[str, dict] = {}
    for source, source_groups in sorted(groups.items()):
        def across(metric: str, pooled: bool = False) -> float:
            if pooled:
                values = [value for group in source_groups.values() for value in group[metric]]
            else:
                values = [float(np.mean(group[metric])) for group in source_groups.values()
                          if group[metric]]
            return float(np.mean(values)) if values else float("nan")
        table[source] = {
            "groups": len(source_groups),
            "agent_records": sum(len(group["ade"]) for group in source_groups.values()),
            "ade": across("ade"),
            "min_ade": across("min_ade"),
            "fde": across("fde"),
            "min_fde": across("min_fde"),
            "pooled_ade": across("ade", pooled=True),
            "pooled_min_ade": across("min_ade", pooled=True),
            "pooled_fde": across("fde", pooled=True),
            "pooled_min_fde": across("min_fde", pooled=True),
        }
    return table


def _stratum(values: list[AgentErrorRecord]) -> dict:
    def mean(metric: str) -> float:
        selected = [getattr(record, metric) for record in values
                    if not (metric in ("fde", "min_fde") and not record.endpoint_valid)]
        return float(np.mean(selected)) if selected else float("nan")
    return {
        "count": len(values),
        "ade": mean("ade"),
        "min_ade": mean("min_ade"),
        "fde": mean("fde"),
        "min_fde": mean("min_fde"),
        "endpoint_missing": sum(1 for record in values if not record.endpoint_valid),
    }


def summarize_records(records: list[AgentErrorRecord]) -> dict:
    """Full summary: group-level tables, strata, and honest counts."""
    strata_dims = {
        "source": lambda record: record.source,
        "agent_type": lambda record: str(record.agent_type),
        "agent_role": lambda record: str(record.agent_role),
        "agents_truncated": lambda record: str(record.agents_truncated),
        "map_truncated": lambda record: str(record.map_truncated),
    }
    strata: dict[str, dict[str, dict]] = {}
    for name, key in strata_dims.items():
        buckets: dict[str, list[AgentErrorRecord]] = defaultdict(list)
        for record in records:
            buckets[key(record)].append(record)
        strata[name] = {value: _stratum(bucket) for value, bucket in sorted(buckets.items())}
    by_source: dict[str, int] = defaultdict(int)
    for record in records:
        by_source[record.source] += 1
    return {
        "group_level": group_level_table(records),
        "strata": strata,
        "counts": {
            "agents": len(records),
            "endpoint_missing": sum(1 for record in records if not record.endpoint_valid),
            "by_source": dict(sorted(by_source.items())),
            "total_valid_future_steps": int(sum(record.valid_steps for record in records)),
        },
    }


def kinematic_diagnostics(trajectories: np.ndarray) -> dict:
    """Speed/acceleration/jerk diagnostics from predicted xy; geometric proxy only.

    ``offroad_rate`` cannot be identified from ``scene-shard-v1`` because the
    schema stores no drivable-area polygons; it is reported as such instead of
    an implied 0.
    """
    flat = trajectories.reshape(-1, trajectories.shape[-2], 2)
    velocity = np.diff(flat, axis=1) * 10.0
    speed = np.linalg.norm(velocity, axis=-1)
    accel = np.diff(velocity, axis=1) * 10.0
    accel_norm = np.linalg.norm(accel, axis=-1)
    jerk = np.diff(accel, axis=1) * 10.0
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    return {
        **KINEMATIC_LIMITS,
        "agent_count": int(flat.shape[0]),
        "speed_violations": int((speed > KINEMATIC_LIMITS["speed_limit_mps"]).any(axis=1).sum()),
        "accel_violations": int((accel_norm > KINEMATIC_LIMITS["accel_limit_mps2"]).any(axis=1).sum()),
        "jerk_violations": int((jerk_norm > KINEMATIC_LIMITS["jerk_limit_mps3"]).any(axis=1).sum()),
        "offroad_rate": "not_identifiable_from_current_scene_shard",
        "collision_rate": "geometric_proxy_only_if_reported_separately",
    }


def token_frequency_report(true_tokens: np.ndarray, valid_mask: np.ndarray,
                           predicted_tokens: np.ndarray) -> dict:
    """True vs predicted motion-token frequencies and macro recall."""
    config = load_p33_config(CONFIG_PATH)
    vocabulary = config["motion_tokens"]["vocabulary_size"]
    truth = true_tokens[valid_mask]
    predicted = predicted_tokens[valid_mask]
    true_counts = np.bincount(truth, minlength=vocabulary)
    predicted_counts = np.bincount(predicted, minlength=vocabulary)
    recalls = []
    for token in range(vocabulary):
        total = true_counts[token]
        if total:
            recalls.append(float((predicted == token)[truth == token].mean()))
    occupied = int((true_counts > 0).sum())
    return {
        "vocabulary": vocabulary,
        "valid_tokens": int(len(truth)),
        "occupied_true_tokens": occupied,
        "true_top_share": float(true_counts.max() / max(1, len(truth))),
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_recall_occupied": float(np.mean(recalls)) if recalls else float("nan"),
        "true_counts": true_counts.tolist(),
        "predicted_counts": predicted_counts.tolist(),
    }
