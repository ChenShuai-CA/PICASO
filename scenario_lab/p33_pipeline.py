"""Streaming P3.3.1 public-data conversion into ``scene-shard-v1``.

The converter keeps one Waymo Scenario or one INTERACTION recording case in
memory at a time.  It has no TensorFlow, protobuf, lanelet2, or pyproj runtime
dependency: only the documented protobuf wire fields and the local Lanelet2
OSM files are decoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from .p33_spec import data_schema_version, validate_scene_arrays, waymo_training_split
from .waymo import fields, iter_tfrecord


MPH_TO_MPS = 0.44704
AGENT_TYPES = {"vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
MAP_TYPES = {
    "lane_center": 1,
    "lane_boundary": 2,
    "road_boundary": 3,
    "crosswalk": 4,
    "stop_line": 5,
    "speed_bump": 6,
    "other": 7,
}


@dataclass
class AgentTrack:
    track_id: str
    kind: str
    # Columns: x, y, vx, vy, heading, length, width. Invalid rows are zero.
    states: np.ndarray
    valid: np.ndarray


@dataclass
class MapPolyline:
    feature_id: str
    kind: str
    points: np.ndarray
    speed_limit_mps: float = 0.0


@dataclass
class Scene:
    source: str
    group_id: str
    timestamps: np.ndarray
    current_index: int
    tracks: list[AgentTrack]
    anchor_indices: list[int]
    objects_of_interest: list[str]
    prediction_indices: list[int]
    map_features: list[MapPolyline]
    source_file: str
    source_content_sha256: str


def _scalar(message: bytes, key: int, default=None):
    for field, _wire, value in fields(message):
        if field == key:
            return value
    return default


def _repeated_int(wire: int, value) -> list[int]:
    if wire == 0:
        return [int(value)]
    if wire != 2:
        return []
    result = []
    data = value
    index = 0
    while index < len(data):
        number = 0
        for shift in range(0, 70, 7):
            byte = data[index]
            index += 1
            number |= (byte & 0x7f) << shift
            if not byte & 0x80:
                break
        result.append(number)
    return result


def _parse_map_point(message: bytes) -> tuple[float, float] | None:
    values = {key: value for key, _wire, value in fields(message)}
    if 1 not in values or 2 not in values:
        return None
    point = float(values[1]), float(values[2])
    return point if np.isfinite(point).all() else None


def _nested_points(message: bytes, key: int) -> np.ndarray:
    points = []
    for field, wire, value in fields(message):
        if field == key and wire == 2:
            point = _parse_map_point(value)
            if point is not None:
                points.append(point)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _parse_waymo_map_feature(message: bytes) -> MapPolyline | None:
    feature_id = str(_scalar(message, 1, "unknown"))
    for key, wire, value in fields(message):
        if wire != 2:
            continue
        if key == 3:  # LaneCenter: speed_limit_mph=1, polyline=8.
            speed = float(_scalar(value, 1, 0.0)) * MPH_TO_MPS
            points = _nested_points(value, 8)
            kind = "lane_center"
        elif key == 4:  # RoadLine.
            speed, points, kind = 0.0, _nested_points(value, 2), "lane_boundary"
        elif key == 5:  # RoadEdge.
            speed, points, kind = 0.0, _nested_points(value, 2), "road_boundary"
        elif key == 7:  # StopSign position.
            point_message = _scalar(value, 2)
            point = _parse_map_point(point_message) if point_message else None
            speed = 0.0
            points = np.asarray([point], dtype=np.float64) if point else np.zeros((0, 2))
            kind = "stop_line"
        elif key == 8:
            speed, points, kind = 0.0, _nested_points(value, 1), "crosswalk"
        elif key == 9:
            speed, points, kind = 0.0, _nested_points(value, 1), "speed_bump"
        elif key == 10:
            speed, points, kind = 0.0, _nested_points(value, 1), "other"
        else:
            continue
        if len(points):
            return MapPolyline(feature_id, kind, points, speed)
    return None


def parse_waymo_scene(record: bytes, source_file: str,
                      source_content_sha256: str) -> Scene:
    """Decode the Motion Scenario fields required by ``scene-shard-v1``."""
    scenario_id = None
    timestamps = []
    tracks: list[AgentTrack] = []
    map_features = []
    sdc_track_index = None
    current_index = None
    objects_of_interest: list[str] = []
    prediction_indices: list[int] = []

    for key, wire, value in fields(record):
        if key == 1:
            timestamps.extend(np.frombuffer(value, dtype="<f8").tolist()
                              if wire == 2 else [float(value)])
        elif key == 2 and wire == 2:
            track_id = str(_scalar(value, 1, ""))
            kind_value = int(_scalar(value, 2, 4))
            kind = {1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}.get(
                kind_value, "other")
            rows, valid = [], []
            for state_key, state_wire, state_value in fields(value):
                if state_key != 3 or state_wire != 2:
                    continue
                state = {field: item for field, _wire, item in fields(state_value)}
                is_valid = bool(state.get(11, 0)) and all(
                    field in state for field in (2, 3, 5, 6, 8, 9, 10))
                values = [float(state.get(field, 0.0)) for field in (2, 3, 9, 10, 8, 5, 6)]
                is_valid = is_valid and bool(np.isfinite(values).all())
                rows.append(values if is_valid else [0.0] * 7)
                valid.append(is_valid)
            tracks.append(AgentTrack(track_id, kind, np.asarray(rows, dtype=np.float64),
                                     np.asarray(valid, dtype=bool)))
        elif key == 4:
            objects_of_interest.extend(str(item) for item in _repeated_int(wire, value))
        elif key == 5 and wire == 2:
            scenario_id = value.decode("utf-8")
        elif key == 6:
            sdc_track_index = int(value)
        elif key == 8 and wire == 2:
            parsed = _parse_waymo_map_feature(value)
            if parsed is not None:
                map_features.append(parsed)
        elif key == 10:
            current_index = int(value)
        elif key == 11 and wire == 2:
            prediction_indices.append(int(_scalar(value, 1, -1)))

    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    if not scenario_id or current_index is None or sdc_track_index is None:
        raise ValueError("Waymo Scenario is missing id/current/sdc fields")
    if not len(timestamps_array) or np.any(np.diff(timestamps_array) <= 0):
        raise ValueError(f"{scenario_id}: invalid timestamps")
    if not (0 <= sdc_track_index < len(tracks)):
        raise ValueError(f"{scenario_id}: invalid sdc_track_index")
    if current_index < 10 or current_index + 50 >= len(timestamps_array):
        raise ValueError(f"{scenario_id}: insufficient 1 s history or 5 s future")
    if any(len(track.states) != len(timestamps_array) for track in tracks):
        raise ValueError(f"{scenario_id}: track/timestamp length mismatch")
    prediction_indices = [index for index in prediction_indices if 0 <= index < len(tracks)]
    return Scene(
        source="waymo",
        group_id=scenario_id,
        timestamps=timestamps_array,
        current_index=current_index,
        tracks=tracks,
        anchor_indices=[sdc_track_index],
        objects_of_interest=objects_of_interest,
        prediction_indices=prediction_indices,
        map_features=map_features,
        source_file=source_file,
        source_content_sha256=source_content_sha256,
    )


def iter_waymo_scenes(path: Path, content_sha256: str,
                      record_limit: int | None = None) -> Iterator[Scene]:
    for record in iter_tfrecord(path, record_limit):
        yield parse_waymo_scene(record, str(path), content_sha256)


def interaction_case_id(path: Path) -> str:
    match = re.search(r"_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"cannot infer INTERACTION recording case from {path.name}")
    return match.group(1).zfill(3)


def interaction_location(path: Path) -> str:
    return path.parent.name


def group_interaction_case_files(paths: Iterable[Path]) -> list[list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        key = interaction_location(path), interaction_case_id(path)
        groups.setdefault(key, []).append(path)
    return [sorted(group) for _key, group in sorted(groups.items())]


def _utm_zone31_xy(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Project WGS84 to UTM zone 31 and subtract the projection of (0, 0).

    INTERACTION's official visualizer uses a UTM projector with the default
    origin (0, 0).  All supplied map coordinates lie close to that origin.
    """
    a = 6378137.0
    e2 = 0.0066943799901413165
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lat = math.radians(lat_deg)
    lon_delta = math.radians(lon_deg - 3.0)  # central meridian of UTM zone 31
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    tan_lat = math.tan(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    aa = cos_lat * lon_delta
    m = a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lat)
    )
    x = k0 * n * (aa + (1 - t + c) * aa ** 3 / 6
                  + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * aa ** 5 / 120)
    y = k0 * (m + n * tan_lat * (aa ** 2 / 2
                                  + (5 - t + 9 * c + 4 * c ** 2) * aa ** 4 / 24
                                  + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2)
                                  * aa ** 6 / 720))
    # Subtract the same truncated-series projection of lat=0/lon=0.  Using the
    # same approximation on both sides preserves the dataset's local origin.
    origin_a = math.radians(-3.0)
    origin_c = ep2
    origin_x = k0 * a * (
        origin_a + (1 + origin_c) * origin_a ** 3 / 6
        + (5 + 72 * origin_c - 58 * ep2) * origin_a ** 5 / 120
    )
    return x - origin_x, y


def _tag_map(element: ET.Element) -> dict[str, str]:
    return {tag.get("k", ""): tag.get("v", "") for tag in element.findall("tag")}


def _resample_xy(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return points.copy()
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distances[-1] <= 1e-9:
        return points[:1]
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack([np.interp(targets, distances, points[:, axis]) for axis in (0, 1)])


def load_interaction_map(path: Path) -> list[MapPolyline]:
    root = ET.parse(path).getroot()
    nodes = {}
    for node in root.findall("node"):
        nodes[node.get("id")] = _utm_zone31_xy(float(node.get("lat")),
                                                float(node.get("lon")))
    ways: dict[str, tuple[np.ndarray, dict[str, str]]] = {}
    result: list[MapPolyline] = []
    for way in root.findall("way"):
        refs = [item.get("ref") for item in way.findall("nd")]
        points = np.asarray([nodes[ref] for ref in refs if ref in nodes], dtype=np.float64)
        tags = _tag_map(way)
        ways[way.get("id")] = points, tags
        if len(points) == 0:
            continue
        raw_type = tags.get("type", "")
        if raw_type in {"line_thin", "line_thick", "virtual"}:
            kind = "lane_boundary"
        elif raw_type in {"curbstone", "guard_rail", "road_border"}:
            kind = "road_boundary"
        elif raw_type in {"pedestrian_marking"}:
            kind = "crosswalk"
        elif raw_type in {"stop_line"}:
            kind = "stop_line"
        else:
            kind = "other"
        result.append(MapPolyline(f"way:{way.get('id')}", kind, points))

    regulatory_speed: dict[str, float] = {}
    for relation in root.findall("relation"):
        tags = _tag_map(relation)
        if tags.get("type") == "regulatory_element" and tags.get("subtype") == "speed_limit":
            match = re.search(r"(\d+(?:\.\d+)?)", tags.get("sign_type", ""))
            if match:
                regulatory_speed[relation.get("id")] = float(match.group(1)) / 3.6

    for relation in root.findall("relation"):
        tags = _tag_map(relation)
        if tags.get("type") != "lanelet":
            continue
        members = {(item.get("type"), item.get("role")): item.get("ref")
                   for item in relation.findall("member")}
        left = ways.get(members.get(("way", "left"), ""), (np.zeros((0, 2)), {}))[0]
        right = ways.get(members.get(("way", "right"), ""), (np.zeros((0, 2)), {}))[0]
        if len(left) == 0 or len(right) == 0:
            continue
        count = min(40, max(2, min(len(left), len(right))))
        left_r, right_r = _resample_xy(left, count), _resample_xy(right, count)
        if np.linalg.norm(left_r[0] - right_r[-1]) < np.linalg.norm(left_r[0] - right_r[0]):
            right_r = right_r[::-1]
        speed = 0.0
        for item in relation.findall("member"):
            if item.get("type") == "relation" and item.get("role") == "regulatory_element":
                speed = max(speed, regulatory_speed.get(item.get("ref"), 0.0))
        result.append(MapPolyline(f"lanelet:{relation.get('id')}", "lane_center",
                                  (left_r + right_r) / 2.0, speed))
    return result


def _parse_interaction_csv(path: Path) -> dict[str, dict[int, list[float]]]:
    tracks: dict[str, dict[int, list[float]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig", errors="strict") as handle:
        reader = csv.DictReader(handle)
        required = {"track_id", "frame_id", "x", "y", "vx", "vy"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path}: missing INTERACTION columns {sorted(required)}")
        for row in reader:
            try:
                frame = int(row["frame_id"])
                track_id = row["track_id"].strip()
                values = [float(row[key]) for key in ("x", "y", "vx", "vy")]
                heading = float(row["psi_rad"]) if row.get("psi_rad", "").strip() else math.nan
                length = float(row["length"]) if row.get("length", "").strip() else 1.8
                width = float(row["width"]) if row.get("width", "").strip() else 0.8
                values.extend((heading, length, width))
            except (TypeError, ValueError):
                continue
            if not track_id or not np.isfinite(values[:4] + values[5:]).all():
                continue
            if not math.isfinite(heading) and math.hypot(values[2], values[3]) > 0.5:
                values[4] = math.atan2(values[3], values[2])
            tracks.setdefault(track_id, {})[frame] = values
    return tracks


def resolve_interaction_map(data_root: Path, location: str) -> Path:
    candidates = [
        data_root / "INTERACTION" / "INTERACTION" / "maps" / f"{location}.osm",
        data_root / "INTERACTION" / "INTERACTION-Dataset-DR-multi-v1_2" / "maps" / f"{location}.osm",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no INTERACTION map for {location}")


def iter_interaction_scenes(case_paths: Sequence[Path], data_root: Path,
                            content_hashes: dict[str, str], stride_frames: int = 10,
                            anchors_per_window: int = 8) -> Iterator[Scene]:
    """Yield fixed-window scenes for one paired INTERACTION recording case."""
    if not case_paths:
        return
    location = interaction_location(case_paths[0])
    case_id = interaction_case_id(case_paths[0])
    if any(interaction_location(path) != location or interaction_case_id(path) != case_id
           for path in case_paths):
        raise ValueError("INTERACTION case files do not share location/case id")
    raw: dict[str, tuple[str, dict[int, list[float]]]] = {}
    for path in case_paths:
        # The raw pedestrian export uses the unresolved label
        # ``pedestrian/bicycle`` and omits dimensions, so it cannot supervise a
        # pedestrian-only or cyclist-only type head.
        family = "other" if path.name.startswith("pedestrian_tracks_") else "vehicle"
        for track_id, states in _parse_interaction_csv(path).items():
            raw[f"{family}:{track_id}"] = family, states
    if not raw:
        return
    all_frames = sorted({frame for _kind, states in raw.values() for frame in states})
    if not all_frames:
        return
    start, stop = all_frames[0] + 10, all_frames[-1] - 50
    if stop < start:
        return
    frames = np.arange(all_frames[0], all_frames[-1] + 1, dtype=int)
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    tracks = []
    for track_id in sorted(raw):
        kind, state_rows = raw[track_id]
        states = np.zeros((len(frames), 7), dtype=np.float64)
        valid = np.zeros(len(frames), dtype=bool)
        last_heading = math.nan
        for frame in sorted(state_rows):
            row = list(state_rows[frame])
            if math.isfinite(row[4]):
                last_heading = row[4]
            elif math.isfinite(last_heading):
                row[4] = last_heading
            if math.isfinite(row[4]):
                index = frame_to_index[frame]
                states[index] = row
                valid[index] = True
        tracks.append(AgentTrack(track_id, kind, states, valid))
    map_path = resolve_interaction_map(data_root, location)
    map_features = load_interaction_map(map_path)
    source_file = "+".join(str(path) for path in case_paths)
    combined_hash = sha256("\n".join(content_hashes[str(path)] for path in case_paths).encode()).hexdigest()
    timestamps = frames.astype(np.float64) / 10.0
    group_id = f"{location}::{case_id}"

    for current_frame in range(start, stop + 1, stride_frames):
        current = frame_to_index[current_frame]
        eligible = [index for index, track in enumerate(tracks)
                    if track.kind == "vehicle" and track.valid[current - 10:current + 51].all()]
        if not eligible:
            continue

        def anchor_key(index: int):
            anchor = tracks[index]
            position = anchor.states[current, :2]
            nonvehicle = [np.linalg.norm(other.states[current, :2] - position)
                          for other in tracks if other.kind != "vehicle" and other.valid[current]]
            vehicle = [np.linalg.norm(other.states[current, :2] - position)
                       for j, other in enumerate(tracks) if j != index and other.kind == "vehicle"
                       and other.valid[current]]
            return (min(nonvehicle, default=math.inf), min(vehicle, default=math.inf),
                    anchor.track_id)

        anchor_indices = sorted(eligible, key=anchor_key)[:anchors_per_window]
        yield Scene(
            source="interaction",
            group_id=group_id,
            timestamps=timestamps,
            current_index=current,
            tracks=tracks,
            anchor_indices=anchor_indices,
            objects_of_interest=[],
            prediction_indices=[index for index, track in enumerate(tracks)
                                if track.valid[current:current + 51].all()],
            map_features=map_features,
            source_file=source_file,
            source_content_sha256=combined_hash,
        )


def _local_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, s], [-s, c]], dtype=np.float64)


def _wrap_angle(angle: np.ndarray | float):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _closing_speed(anchor: AgentTrack, other: AgentTrack, current: int) -> float:
    relative_position = other.states[current, :2] - anchor.states[current, :2]
    distance = float(np.linalg.norm(relative_position))
    if distance < 1e-6:
        return 0.0
    relative_velocity = other.states[current, 2:4] - anchor.states[current, 2:4]
    return max(0.0, -float(relative_position @ relative_velocity) / distance)


def _select_agents(scene: Scene, anchor_index: int, maximum: int) -> list[int]:
    current = scene.current_index
    anchor = scene.tracks[anchor_index]
    if not anchor.valid[current]:
        raise ValueError(f"{scene.group_id}: anchor invalid at current time")
    selected = [anchor_index]
    by_id = {track.track_id: index for index, track in enumerate(scene.tracks)}
    for track_id in scene.objects_of_interest:
        index = by_id.get(track_id)
        if index is not None and index not in selected and scene.tracks[index].valid[current]:
            selected.append(index)
    for index in scene.prediction_indices:
        if index not in selected and scene.tracks[index].valid[current]:
            selected.append(index)
    remaining = [index for index, track in enumerate(scene.tracks)
                 if index not in selected and track.valid[current]]
    remaining.sort(key=lambda index: (
        float(np.linalg.norm(scene.tracks[index].states[current, :2]
                             - anchor.states[current, :2])),
        -_closing_speed(anchor, scene.tracks[index], current),
        scene.tracks[index].track_id,
    ))
    return (selected + remaining)[:maximum]


def _segment_box_intersection(start: np.ndarray, end: np.ndarray,
                              center: np.ndarray, heading: float,
                              half_length: float, half_width: float) -> bool:
    rotation = _local_rotation(heading)
    p0, p1 = rotation @ (start - center), rotation @ (end - center)
    direction = p1 - p0
    low, high = 0.0, 1.0
    for origin, delta, extent in zip(p0, direction, (half_length, half_width)):
        if abs(delta) < 1e-12:
            if origin < -extent or origin > extent:
                return False
            continue
        first, second = (-extent - origin) / delta, (extent - origin) / delta
        if first > second:
            first, second = second, first
        low, high = max(low, first), min(high, second)
        if low > high:
            return False
    return True


def geometric_visibility(history_world: np.ndarray, state_valid: np.ndarray,
                         max_range_m: float, inflation_m: float) -> np.ndarray:
    """Return [query,time,source] visibility with dynamic-box occlusion."""
    agents, timesteps = state_valid.shape
    result = np.zeros((agents, timesteps, agents), dtype=bool)
    for time in range(timesteps):
        valid_indices = np.flatnonzero(state_valid[:, time])
        if not len(valid_indices):
            continue
        states = history_world[valid_indices, time]
        positions = states[:, :2]
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        visible = distance <= max_range_m
        count = len(valid_indices)
        for third in range(count):
            heading = states[third, 4]
            relative = positions - positions[third]
            local = (_local_rotation(heading) @ relative.T).T
            p0 = local[:, None, :]
            direction = local[None, :, :] - p0
            t_low = np.zeros((count, count), dtype=np.float64)
            t_high = np.ones((count, count), dtype=np.float64)
            intersects = np.ones((count, count), dtype=bool)
            extents = (
                max(0.05, states[third, 5] / 2 + inflation_m),
                max(0.05, states[third, 6] / 2 + inflation_m),
            )
            for axis, extent in enumerate(extents):
                origin = p0[:, :, axis]
                delta = direction[:, :, axis]
                parallel = np.abs(delta) < 1e-12
                intersects &= ~(parallel & ((origin < -extent) | (origin > extent)))
                safe_delta = np.where(parallel, 1.0, delta)
                first = (-extent - origin) / safe_delta
                second = (extent - origin) / safe_delta
                lower = np.minimum(first, second)
                upper = np.maximum(first, second)
                t_low = np.maximum(t_low, np.where(parallel, 0.0, lower))
                t_high = np.minimum(t_high, np.where(parallel, 1.0, upper))
            intersects &= t_low <= t_high
            intersects[third, :] = False
            intersects[:, third] = False
            np.fill_diagonal(intersects, False)
            visible &= ~intersects
        np.fill_diagonal(visible, True)
        result[np.ix_(valid_indices, [time], valid_indices)] = visible[:, None, :]
    return result


def _polyline_tensor(features: Sequence[MapPolyline], origin: np.ndarray,
                     rotation: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = config["data"]
    maximum, max_points = data["max_map_polylines"], data["max_points_per_polyline"]
    radius = data["map_selection"]["radius_m"]
    candidates = []
    for feature in features:
        if len(feature.points) == 0 or not np.isfinite(feature.points).all():
            continue
        distance = float(np.linalg.norm(feature.points - origin, axis=1).min())
        if distance <= radius:
            candidates.append((distance, MAP_TYPES.get(feature.kind, MAP_TYPES["other"]),
                               feature.feature_id, feature))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    kept = candidates[:maximum]
    polylines = np.zeros((maximum, max_points, 6), dtype=np.float32)
    mask = np.zeros((maximum, max_points), dtype=bool)
    types = np.zeros(maximum, dtype=np.uint8)
    for slot, (_distance, type_id, _feature_id, feature) in enumerate(kept):
        points = _resample_xy(feature.points, max_points)
        local = (rotation @ (points - origin).T).T
        count = len(local)
        direction = np.zeros_like(local)
        if count > 1:
            derivative = np.gradient(local, axis=0)
            norm = np.linalg.norm(derivative, axis=1, keepdims=True)
            direction = derivative / np.maximum(norm, 1e-8)
        curvature = np.zeros(count, dtype=np.float64)
        if count > 2:
            heading = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
            if np.all(np.diff(arc) > 1e-8):
                curvature = np.gradient(heading, arc)
        polylines[slot, :count, :2] = local
        polylines[slot, :count, 2:4] = direction
        polylines[slot, :count, 4] = curvature
        polylines[slot, :count, 5] = feature.speed_limit_mps
        mask[slot, :count] = True
        types[slot] = type_id
    return polylines, mask, types, len(candidates)


def batched_motion_vectors(arrays: dict[str, np.ndarray], config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return [sample,agent,chunk,10] displacements and their validity masks."""
    chunk_steps = config["motion_tokens"]["chunk_steps"]
    future = arrays["future_xy"]
    valid = arrays["future_valid_mask"]
    current_xy = arrays["agent_history"][:, :, -1, :2]
    current_valid = arrays["state_valid_mask"][:, :, -1]
    samples, agents, future_steps, _ = future.shape
    chunks = future_steps // chunk_steps
    positions = np.concatenate((current_xy[:, :, None, :], future), axis=2)
    displacements = np.diff(positions, axis=2)
    transition_valid = np.empty((samples, agents, future_steps), dtype=bool)
    transition_valid[:, :, 0] = current_valid & valid[:, :, 0]
    transition_valid[:, :, 1:] = valid[:, :, :-1] & valid[:, :, 1:]
    vectors = displacements.reshape(
        samples, agents, chunks, chunk_steps * 2).astype(np.float32)
    vector_valid = transition_valid.reshape(
        samples, agents, chunks, chunk_steps).all(axis=3)
    vectors[~vector_valid] = 0
    return vectors, vector_valid


def motion_vectors(arrays: dict[str, np.ndarray], config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return [agent,chunk,10] consecutive local displacements and validity."""
    batched = {key: value[None, ...] for key, value in arrays.items()
               if key in {"future_xy", "future_valid_mask", "agent_history",
                          "state_valid_mask"}}
    vectors, valid = batched_motion_vectors(batched, config)
    return vectors[0], valid[0]


def build_scene_arrays(scene: Scene, anchor_index: int, config: dict,
                       codebook: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
    data = config["data"]
    maximum = data["max_agents"]
    current = scene.current_index
    history_indices = np.arange(current - data["history_steps"] + 1, current + 1)
    future_indices = np.arange(current + 1, current + 1 + data["future_steps"])
    selected = _select_agents(scene, anchor_index, maximum)
    anchor = scene.tracks[anchor_index]
    origin = anchor.states[current, :2].copy()
    yaw = float(anchor.states[current, 4])
    rotation = _local_rotation(yaw)

    agent_history = np.zeros((maximum, data["history_steps"], 8), dtype=np.float32)
    state_valid = np.zeros((maximum, data["history_steps"]), dtype=bool)
    present = np.zeros(maximum, dtype=bool)
    agent_type = np.zeros(maximum, dtype=np.uint8)
    agent_role = np.zeros(maximum, dtype=np.uint8)
    future_xy = np.zeros((maximum, data["future_steps"], 2), dtype=np.float32)
    future_valid = np.zeros((maximum, data["future_steps"]), dtype=bool)
    history_world = np.zeros((maximum, data["history_steps"], 7), dtype=np.float64)
    prediction_set = set(scene.prediction_indices)

    for slot, track_index in enumerate(selected):
        track = scene.tracks[track_index]
        present[slot] = True
        agent_type[slot] = AGENT_TYPES.get(track.kind, AGENT_TYPES["other"])
        agent_role[slot] = 1 if slot == 0 else (2 if track_index in prediction_set else 3)
        hist_valid = track.valid[history_indices]
        state_valid[slot] = hist_valid
        history_world[slot, hist_valid] = track.states[history_indices[hist_valid]]
        states = track.states[history_indices]
        local_xy = (rotation @ (states[:, :2] - origin).T).T
        local_velocity = (rotation @ states[:, 2:4].T).T
        local_heading = _wrap_angle(states[:, 4] - yaw)
        agent_history[slot, hist_valid, :2] = local_xy[hist_valid]
        agent_history[slot, hist_valid, 2:4] = local_velocity[hist_valid]
        agent_history[slot, hist_valid, 4] = np.cos(local_heading[hist_valid])
        agent_history[slot, hist_valid, 5] = np.sin(local_heading[hist_valid])
        agent_history[slot, hist_valid, 6:8] = states[hist_valid, 5:7]
        fut_valid = track.valid[future_indices]
        future_valid[slot] = fut_valid
        fut_local = (rotation @ (track.states[future_indices, :2] - origin).T).T
        future_xy[slot, fut_valid] = fut_local[fut_valid]

    visibility_cfg = data["visibility_proxy"]
    visibility = geometric_visibility(
        history_world, state_valid,
        visibility_cfg["maximum_range_m"],
        visibility_cfg["dynamic_actor_box_inflation_m"],
    )
    map_polylines, map_mask, map_type, map_available = _polyline_tensor(
        scene.map_features, origin, rotation, config)
    chunks = data["future_steps"] // config["motion_tokens"]["chunk_steps"]
    token_target = np.full((maximum, chunks), config["motion_tokens"]["ignore_index"],
                           dtype=np.uint8)
    arrays = {
        "agent_history": agent_history,
        "state_valid_mask": state_valid,
        "pairwise_visibility_mask": visibility,
        "agent_present_mask": present,
        "agent_type": agent_type,
        "agent_role": agent_role,
        "map_polylines": map_polylines,
        "map_point_mask": map_mask,
        "map_type": map_type,
        "future_xy": future_xy,
        "future_valid_mask": future_valid,
        "motion_token_target": token_target,
        "motion_token_valid_mask": np.zeros((maximum, chunks), dtype=bool),
    }
    vectors, vector_valid = motion_vectors(arrays, config)
    if codebook is not None:
        labels = assign_motion_tokens(vectors, vector_valid, codebook,
                                      config["motion_tokens"]["ignore_index"])
        arrays["motion_token_target"] = labels
        arrays["motion_token_valid_mask"] = vector_valid
        validate_scene_arrays(arrays, config)
    metadata = {
        "schema_version": data_schema_version(config),
        "sample_id": "",  # Assigned by the caller from source/group/time/anchor.
        "source": scene.source,
        "split": (waymo_training_split(scene.group_id) if scene.source == "waymo" else ""),
        "group_id": scene.group_id,
        "anchor_track_id": anchor.track_id,
        "source_file": scene.source_file,
        "source_content_sha256": scene.source_content_sha256,
        "coordinate_transform": {
            "origin_x_m": float(origin[0]), "origin_y_m": float(origin[1]), "yaw_rad": yaw,
        },
        "visibility_source": data["masks"]["public_visibility_source"],
        "truncation": {
            "agents_available": int(sum(track.valid[current] for track in scene.tracks)),
            "agents_kept": len(selected),
            "map_polylines_available": map_available,
            "map_polylines_kept": int(map_mask.any(axis=1).sum()),
        },
        "arrays": {},
    }
    return arrays, metadata


def assign_motion_tokens(vectors: np.ndarray, valid: np.ndarray, codebook: np.ndarray,
                         ignore_index: int = 255) -> np.ndarray:
    labels = np.full(valid.shape, ignore_index, dtype=np.uint8)
    if valid.any():
        rows = vectors[valid].astype(np.float64)
        distance = ((rows[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        labels[valid] = distance.argmin(axis=1).astype(np.uint8)
    return labels


class BalancedMotionReservoir:
    """Deterministic bounded reservoir, stratified by source and agent type."""

    def __init__(self, capacity_per_stratum: int, seed: int):
        self.capacity = int(capacity_per_stratum)
        self.rng = np.random.default_rng(seed)
        self.rows: dict[str, list[np.ndarray]] = {}
        self.seen: dict[str, int] = {}

    def add(self, key: str, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float32)
        if not len(rows):
            return
        bucket = self.rows.setdefault(key, [])
        seen = self.seen.get(key, 0)
        fill = min(self.capacity - len(bucket), len(rows))
        if fill > 0:
            bucket.extend(row.copy() for row in rows[:fill])
            seen += fill
            rows = rows[fill:]
        if len(rows):
            totals = np.arange(seen + 1, seen + len(rows) + 1, dtype=np.float64)
            replacements = np.floor(self.rng.random(len(rows)) * totals).astype(np.int64)
            for row_index in np.flatnonzero(replacements < self.capacity):
                bucket[int(replacements[row_index])] = rows[row_index].copy()
            seen += len(rows)
        self.seen[key] = int(seen)

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.stack(rows) for key, rows in sorted(self.rows.items()) if rows}


def fit_balanced_minibatch_kmeans(strata: dict[str, np.ndarray], clusters: int,
                                  batch_size: int, max_steps: int, tolerance: float,
                                  seed: int) -> tuple[np.ndarray, dict]:
    if not strata:
        raise ValueError("no valid training motion vectors for codebook")
    rng = np.random.default_rng(seed)
    keys = sorted(strata)
    if sum(len(value) for value in strata.values()) < clusters:
        raise ValueError("fewer motion vectors than codebook clusters")

    def balanced_batch(size: int) -> np.ndarray:
        chosen = []
        for _ in range(size):
            key = keys[int(rng.integers(0, len(keys)))]
            values = strata[key]
            chosen.append(values[int(rng.integers(0, len(values)))])
        return np.asarray(chosen, dtype=np.float64)

    initialization_pool = balanced_batch(max(batch_size, clusters * 16))
    centroids = np.empty((clusters, initialization_pool.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, len(initialization_pool)))
    centroids[0] = initialization_pool[first]
    nearest = ((initialization_pool - centroids[0]) ** 2).sum(axis=1)
    for cluster in range(1, clusters):
        total = float(nearest.sum())
        if total <= 1e-20:
            choice = int(rng.integers(0, len(initialization_pool)))
        else:
            choice = int(rng.choice(len(initialization_pool), p=nearest / total))
        centroids[cluster] = initialization_pool[choice]
        nearest = np.minimum(nearest,
                             ((initialization_pool - centroids[cluster]) ** 2).sum(axis=1))
    counts = np.zeros(clusters, dtype=np.int64)
    history = []
    for step in range(max_steps):
        batch = balanced_batch(batch_size)
        distances = ((batch[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        previous = centroids.copy()
        batch_counts = np.bincount(labels, minlength=clusters).astype(np.int64)
        batch_sums = np.zeros_like(centroids)
        np.add.at(batch_sums, labels, batch)
        updated = batch_counts > 0
        new_counts = counts + batch_counts
        centroids[updated] = (
            centroids[updated] * counts[updated, None] + batch_sums[updated]
        ) / new_counts[updated, None]
        counts = new_counts
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            residual = np.take_along_axis(distances, labels[:, None], axis=1)[:, 0]
            farthest = np.argsort(residual)[::-1]
            for cluster, row_index in zip(empty, farthest):
                centroids[cluster] = batch[row_index]
                counts[cluster] = 1
        shift = float(np.linalg.norm(centroids - previous, axis=1).max())
        inertia = float(np.take_along_axis(distances, labels[:, None], axis=1).mean())
        history.append({"step": step + 1, "max_centroid_shift": shift,
                        "batch_inertia": inertia})
        if step >= 9 and shift <= tolerance:
            break
    return centroids.astype(np.float32), {
        "steps": len(history),
        "final_max_centroid_shift": history[-1]["max_centroid_shift"],
        "final_batch_inertia": history[-1]["batch_inertia"],
        "cluster_update_count_min": int(counts.min()),
        "cluster_update_count_max": int(counts.max()),
    }


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
