"""Bounded streaming data adapters: INTERACTION tracks + ABD run inspection.

Design constraints (research plan, 2026-09):
- stdlib + numpy only; every read is row-bounded (never load a full dataset file
  unless explicitly iterating tracks).
- No data leakage across INTERACTION export variants: group ids derive from the
  canonical location + case_id only, never from filesystem paths or split dirs.
- ABD run files are inspected, never interpreted: channel semantics stay
  role='unconfirmed' until human validation, and abort/path-exit events are
  never treated as collision evidence.
"""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = ['split_group', 'iter_interaction_tracks', 'inspect_abd', 'audit_dataset']

# Split rule: sha256(f'{source.casefold()}\x1f{group_id}'), first 8 digest bytes
# as big-endian uint64 v; bucket = v*10 // 2**64 in {0..9}; <8 train, ==8 val,
# ==9 test. Pure integer arithmetic: stable across platforms and Python versions.
_SPLIT_SEPARATOR = '\x1f'

# A time gap > GAP_DT_FACTOR * median dt cuts a track into segments. A single
# dropped frame therefore counts as a gap: we never bridge across missing data.
GAP_DT_FACTOR = 1.5
# Below this speed (m/s) a velocity-derived heading is too noisy to trust.
HEADING_MIN_SPEED = 0.5
# Fallback frame rate when only frame_id exists (INTERACTION is 10 Hz).
DEFAULT_FPS = 10.0
# Segments shorter than this carry no trajectory and are dropped (counted).
_MIN_SEGMENT_ROWS = 2

AGENT_TYPE_KIND = {
    'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'trailer': 'vehicle',
    'van': 'vehicle', 'motorcycle': 'other', 'bicycle': 'other',
    'pedestrian': 'pedestrian', 'ped': 'pedestrian',
}

_SPLIT_TOKEN_MAP = {'train': 'train', 'val': 'val', 'validation': 'val',
                    'test': 'test', 'observed': 'test'}

# Tokens that never identify a recording location (split dirs, export variants,
# generic chunk-file family names, dataset scaffolding).
_NON_LOCATION_TOKENS = {
    'train', 'val', 'test', 'validation', 'observed', 'full', 'multiagent',
    'singleagent', 'dense', 'reactive', 'nonreactive', 'vehicle', 'pedestrian',
    'tracks', 'recorded_trackfiles', 'records', 'interaction', 'data', 'exports',
}

# ABD .spec Type= prefixes that indicate robot control of the vehicle.
_ROBOT_TYPE_PREFIXES = ('SR', 'AR', 'BR', 'PF', 'GR', 'CR', 'CBAR')

# Known ABD robot path/track template suffixes (not run measurement exports).
_ABD_UNSUPPORTED_SUFFIXES = {'.tem', '.pmc', '.spf', '.path'}


def split_group(source: str, group_id: str) -> str:
    """Deterministic 80/10/10 train/val/test assignment for one leak-proof group.

    Rule (stable, do not change without migrating downstream caches): take
    sha256 of f'{source.strip().casefold()}\\x1f{group_id}' (group_id verbatim),
    read the first 8 digest bytes as a big-endian unsigned integer v, and map
    bucket = v * 10 // 2**64 to train (<8), val (==8), test (==9). Integer math
    only, so the partition is reproducible on any platform/Python version.
    """
    key = f'{source.strip().casefold()}{_SPLIT_SEPARATOR}{group_id}'
    digest = hashlib.sha256(key.encode('utf-8')).digest()
    bucket = (int.from_bytes(digest[:8], 'big') * 10) // (1 << 64)
    if bucket < 8:
        return 'train'
    if bucket == 8:
        return 'val'
    return 'test'


# --------------------------------------------------------------------------- helpers

def _to_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_bool(raw):
    value = raw.strip().casefold()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    return None


def _pick_column(cols, names):
    for name in names:
        if name in cols:
            return cols[name]
    return None


def _location_and_split(path: Path):
    """Canonical location, official split token and observed-only flag.

    Only the file stem and the immediate parent directory are scanned for
    split tokens ('train'/'val'/'validation'/'test'/'observed'); higher
    ancestors are ignored so unrelated directory names cannot flip the split.
    'observed' maps to test: observed-only files are never training labels.
    """
    stem = re.sub(r'[_\-]\d{1,4}$', '', path.stem)  # drop chunk index (_000)
    tokens = {t.casefold() for t in stem.split('_') + path.parent.name.split('_') if t}
    official = None
    for token in ('test', 'observed', 'val', 'validation', 'train'):
        if token in tokens:
            official = _SPLIT_TOKEN_MAP[token]
            break
    observed = 'observed' in tokens

    parts = [t for t in stem.split('_') if t]
    while parts and parts[-1].casefold() in _NON_LOCATION_TOKENS:
        parts.pop()
    candidate = '_'.join(parts)
    if candidate and candidate.casefold() not in _NON_LOCATION_TOKENS:
        location = candidate
    else:
        location = path.stem
        for part in reversed(path.parts[:-1]):
            part = part.strip()
            if part and part.casefold() not in _NON_LOCATION_TOKENS:
                location = part
                break
    return location, official, observed


# --------------------------------------------------------------------------- INTERACTION

def iter_interaction_tracks(path: Path) -> Iterator[dict]:
    """Stream INTERACTION track CSVs as per-segment records.

    Yields one dict per contiguous segment of one (case_id, track_id):
    source='interaction', group_id, track_id (str), kind
    ('vehicle'/'pedestrian'/'other'), split, t (s), xy (N,2 m),
    velocity (N,2 m/s), heading (rad), provenance (JSON-serializable dict).

    Leakage control: group_id = canonical location [+ '::' + case_id]; split
    dirs, chunk indices and export variants are canonicalized away, so the
    same scene exported twice always lands in the same split. An official
    split token (train/val/validation/test/observed in stem or parent dir)
    overrides the hash partition; 'observed' files are test-only, never
    training labels. Segments are cut at time gaps (> 1.5x median dt) and at
    clock resets, never bridged across cases. Missing psi_rad headings are
    derived from velocity where speed > 0.5 m/s (then forward-filled);
    unresolvable headings stay NaN rather than being fabricated.
    """
    path = Path(path)
    location, official_split, observed = _location_and_split(path)
    file_reasons = []
    with open(path, 'r', newline='', encoding='utf-8-sig', errors='replace') as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f'interaction file has no header row: {path}')
        cols = {name.strip().casefold(): i for i, name in enumerate(header) if name.strip()}

        track_i = _pick_column(cols, ('track_id', 'trackid', 'track'))
        x_i, y_i = _pick_column(cols, ('x',)), _pick_column(cols, ('y',))
        case_i = _pick_column(cols, ('case_id', 'caseid', 'case'))
        agent_i = _pick_column(cols, ('agent_type', 'agenttype', 'object_type'))
        tms_i = _pick_column(cols, ('timestamp_ms', 'time_ms'))
        tss_i = _pick_column(cols, ('time_s', 'time', 'timestamp_s'))
        frame_i = _pick_column(cols, ('frame_id', 'frameid', 'frame'))
        vx_i = _pick_column(cols, ('vx', 'velocity_x'))
        vy_i = _pick_column(cols, ('vy', 'velocity_y'))
        psi_i = _pick_column(cols, ('psi_rad', 'psi', 'heading_rad', 'heading'))

        missing = [name for name, idx in (('track_id', track_i), ('x', x_i), ('y', y_i))
                   if idx is None]
        if tms_i is None and tss_i is None and frame_i is None:
            missing.append('time (timestamp_ms|time_s|frame_id)')
        if missing:
            raise ValueError(f'{path}: missing required columns: {", ".join(missing)}')

        if tms_i is not None:
            t_i, t_scale, t_source = tms_i, 0.001, 'timestamp_ms'
        elif tss_i is not None:
            t_i, t_scale, t_source = tss_i, 1.0, 'time_s'
        else:
            t_i, t_scale, t_source = frame_i, 1.0 / DEFAULT_FPS, 'frame_id'
            file_reasons.append(
                f'no timestamp column; t derived from frame_id at {DEFAULT_FPS:.0f} Hz')

        tracks = {}
        order = []
        skipped = 0
        for row in reader:
            if not row:
                continue

            def cell(idx):
                if idx is None or idx >= len(row):
                    return None
                return _to_float(row[idx])

            track = row[track_i].strip() if track_i < len(row) else ''
            t = cell(t_i)
            x, y = cell(x_i), cell(y_i)
            if not track or t is None or x is None or y is None:
                skipped += 1
                continue
            case = row[case_i].strip() if (case_i is not None and case_i < len(row)) else ''
            key = (case, track)
            entry = tracks.get(key)
            if entry is None:
                entry = tracks[key] = {'rows': [], 'agent_raw': None}
                order.append(key)
            if entry['agent_raw'] is None and agent_i is not None and agent_i < len(row):
                entry['agent_raw'] = row[agent_i].strip() or None
            entry['rows'].append((t * t_scale, x, y, cell(vx_i), cell(vy_i), cell(psi_i)))

    for key in order:
        case, track = key
        entry = tracks[key]
        group_id = f'{location}::{case}' if case else location
        split = official_split or split_group('interaction', group_id)

        rows = entry['rows']
        positive_dts = [b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[0] - a[0] > 0]
        median_dt = statistics.median(positive_dts) if positive_dts else None
        gap_threshold = median_dt * GAP_DT_FACTOR if median_dt is not None else float('inf')

        segments, prev_t = [[]], None
        for row in rows:
            if prev_t is not None:
                dt = row[0] - prev_t
                if dt <= 1e-12 or dt > gap_threshold:  # clock reset or time gap
                    segments.append([])
            segments[-1].append(row)
            prev_t = row[0]
        kept = [s for s in segments if len(s) >= _MIN_SEGMENT_ROWS]
        dropped_short = len(segments) - len(kept)

        raw_agent = entry['agent_raw']
        kind = AGENT_TYPE_KIND.get((raw_agent or '').casefold(), 'other')
        track_reasons = list(file_reasons)
        if raw_agent is None:
            track_reasons.append("agent_type absent/empty -> kind='other'")
        elif kind == 'other':
            track_reasons.append(
                f"agent_type '{raw_agent}' not in supported mapping -> kind='other'")

        for segment_index, segment in enumerate(kept):
            t = np.array([r[0] for r in segment], dtype=float)
            xy = np.array([[r[1], r[2]] for r in segment], dtype=float)

            if vx_i is not None and vy_i is not None:
                velocity = np.array([[r[3] if r[3] is not None else np.nan,
                                      r[4] if r[4] is not None else np.nan]
                                     for r in segment], dtype=float)
                velocity_source = 'vx_vy'
            else:
                velocity = np.gradient(xy, t, axis=0)
                velocity_source = 'derived_from_position'

            if psi_i is not None:
                heading = np.array([r[5] if r[5] is not None else np.nan
                                    for r in segment], dtype=float)
                heading_source = 'psi_rad'
            else:
                heading = np.full(len(segment), np.nan)
                heading_source = 'derived_from_velocity'
            speed = np.hypot(velocity[:, 0], velocity[:, 1])
            resolvable = (~np.isfinite(heading)
                          & np.isfinite(velocity).all(axis=1)
                          & (speed > HEADING_MIN_SPEED))
            heading[resolvable] = np.arctan2(velocity[resolvable, 1],
                                             velocity[resolvable, 0])
            last = np.nan
            for i in range(len(heading)):  # forward-fill, never fabricate
                if np.isnan(heading[i]):
                    heading[i] = last
                else:
                    last = heading[i]

            provenance = {
                'source_file': str(path),
                'location': location,
                'case_id': case or None,
                'time_source': t_source,
                'agent_type_raw': raw_agent,
                'split_source': 'official' if official_split else 'hash',
                'observed_only': bool(observed),
                'segment_index': segment_index,
                'n_segments': len(kept),
                'n_samples': len(segment),
                'dt_median_s': float(median_dt) if median_dt is not None else None,
                'velocity_source': velocity_source,
                'heading_source': heading_source,
                'rows_skipped_unparsable': skipped,
                'short_segments_dropped': dropped_short,
                'reasons': track_reasons,
            }
            yield {
                'source': 'interaction',
                'group_id': group_id,
                'track_id': track,
                'kind': kind,
                'split': split,
                't': t,
                'xy': xy,
                'velocity': velocity,
                'heading': heading,
                'provenance': provenance,
            }


# --------------------------------------------------------------------------- ABD

def _read_spec(path: Path):
    """Parse a key=value .spec file; returns (type_values, key_values) or None."""
    try:
        handle = open(path, 'r', encoding='latin-1', errors='replace')
    except OSError:
        return None
    types, kv = [], {}
    with handle:
        for line in handle:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if key.casefold() == 'type':
                types.append(value)
            else:
                kv.setdefault(key.casefold(), value)
    return types, kv


def _config_from_spec(path: Path):
    """Control markers from the run's .spec (stem sibling, CurrentTestSpec.txt)."""
    config = {'spec_found': False, 'spec_path': None, 'spec_types': [],
              'use_brake_robot': None, 'brake_robot_engaged': None,
              'control': 'unknown'}
    parsed = None
    for candidate in (path.with_suffix('.spec'), path.parent / 'CurrentTestSpec.txt'):
        if candidate.exists():
            parsed = _read_spec(candidate)
            if parsed is not None:
                config['spec_found'] = True
                config['spec_path'] = str(candidate)
                break
    if not config['spec_found'] or parsed is None:
        return config, ['control unknown: no readable .spec sidecar for run file']
    types, kv = parsed
    config['spec_types'] = types
    use_brake = _parse_bool(kv.get('usebrakerobot', ''))
    config['use_brake_robot'] = use_brake
    # 'BR' brake-robot blocks and 'CBAR' combined brake+steering robots both
    # command the brakes: their runs cannot serve as naturalistic AEB response.
    brake_type = any(t.upper().startswith(('BR', 'CBAR')) for t in types)
    config['brake_robot_engaged'] = (True if use_brake is True or brake_type
                                     else (False if use_brake is False else None))
    robot_types = [t for t in types
                   if t.upper().startswith(_ROBOT_TYPE_PREFIXES)]
    config['control'] = 'robot' if robot_types else 'unknown'
    reasons = []
    if config['brake_robot_engaged'] is True:
        reasons.append('brake robot engaged (UseBrakeRobot or BR-type spec block): '
                       'brake events are robot inputs, not vehicle/AEB response')
    if config['control'] == 'unknown':
        reasons.append('control unknown: spec contains no recognizable robot '
                       'control type block')
    return config, reasons


def inspect_abd(path: Path, max_rows: int = 2000) -> dict:
    """Bounded inspection of one ABD run export (.txt, tab-delimited channels).

    Reads at most max_rows data rows plus a few header lines; never loads the
    full file. Returns a JSON-serializable dict with channels+units, sample
    time validation, .spec configuration markers (UseBrakeRobot / Type blocks),
    parse status and reasons. role is always 'unconfirmed' (channel semantics
    require human validation) and collision_label is always None: abort,
    path-exit and sync channels are run-control events, never collision
    evidence. suitable_for_aeb_calibration is False whenever control is
    unknown or the brake robot is engaged; vehicle mass is never assumed.
    """
    path = Path(path)
    info = {
        'source': 'abd',
        'path': str(path),
        'file': path.name,
        'parse_status': 'error',
        'reasons': [],
        'role': 'unconfirmed',
        'suitable_for_aeb_calibration': False,
        'collision_label': None,
        'channels': [],
        'n_channels': 0,
        'n_points_declared': None,
        'rows_read': 0,
        'truncated': False,
        'time': {'channel': None, 'unit': None, 'monotonic': None,
                 'uniform': None, 'dt_median_s': None, 'dt_max_dev_s': None,
                 'n_samples': 0},
        'abort_channels': [],
        'config': {},
        'provenance': {'max_rows': int(max_rows), 'encoding': 'latin-1'},
    }
    if max_rows is not None and max_rows <= 0:
        info['reasons'].append(f'invalid max_rows={max_rows}')
        return info

    if path.suffix.casefold() in _ABD_UNSUPPORTED_SUFFIXES:
        info['parse_status'] = 'unsupported_format'
        info['reasons'].append(
            f"unsupported suffix '{path.suffix}': expected ABD .txt measurement "
            'export (.tem/.pmc/.spf/.path are robot path/track templates, not '
            'run channel data)')
        info['config'], _ = _config_from_spec(path)
        return info

    read_limit = max_rows + 12
    try:
        handle = open(path, 'r', encoding='latin-1', errors='replace')
    except OSError as exc:
        info['reasons'].append(f'cannot open file: {exc}')
        info['config'], _ = _config_from_spec(path)
        return info
    with handle:
        lines = []
        for i, line in enumerate(handle):
            lines.append(line.rstrip('\r\n'))
            if i + 1 >= read_limit:
                break
    hit_read_cap = len(lines) >= read_limit

    points_idx = next((i for i, l in enumerate(lines[:6])
                       if l.startswith('Points=')), None)
    if points_idx is not None:
        declared = _to_float(lines[points_idx].partition('=')[2])
        info['n_points_declared'] = int(declared) if declared is not None else None
    search_start = (points_idx + 1) if points_idx is not None else 1
    names_idx = next((i for i in range(search_start, len(lines))
                      if '\t' in lines[i] or ',' in lines[i]), None)
    if names_idx is None:
        info['reasons'].append('no delimited channel header line found')
        info['config'], _ = _config_from_spec(path)
        return info

    names_line = lines[names_idx]
    delimiter = '\t' if '\t' in names_line else ','
    names = [n.strip() for n in names_line.split(delimiter)]
    info['channels'] = [{'name': n, 'unit': None} for n in names]
    info['n_channels'] = len(names)

    data_start = names_idx + 1
    units_line = lines[data_start] if data_start < len(lines) else None
    unit_fields = (units_line.split(delimiter)
                   if units_line is not None and delimiter in units_line else None)
    if (unit_fields is not None and len(unit_fields) == len(names)
            and not all(_to_float(u) is not None for u in unit_fields)):
        for channel, unit in zip(info['channels'], unit_fields):
            channel['unit'] = unit.strip() or None
        data_start += 1

    rows, bad_rows = [], 0
    for line in lines[data_start:]:
        if not line.strip():
            continue
        if len(rows) >= max_rows:
            break
        fields = line.split(delimiter)
        values = [_to_float(f) for f in fields]
        if len(values) != len(names) or any(v is None for v in values):
            bad_rows += 1
            continue
        rows.append(values)
    info['rows_read'] = len(rows)
    info['truncated'] = bool(rows) and len(rows) >= max_rows and (
        hit_read_cap or data_start + len(rows) < len(lines))
    info['provenance']['delimiter'] = repr(delimiter)

    time_idx = next((i for i, n in enumerate(names)
                     if n.strip().casefold() == 'time'), None)
    if time_idx is None:
        time_idx = next((i for i, n in enumerate(names)
                         if 'time' in n.casefold()
                         and 'system' not in n.casefold()
                         and 'mp ' not in n.casefold()), None)
    if time_idx is not None and rows:
        stamps = [r[time_idx] for r in rows]
        dts = [b - a for a, b in zip(stamps, stamps[1:])]
        tinfo = info['time']
        tinfo['channel'] = names[time_idx]
        tinfo['unit'] = info['channels'][time_idx]['unit']
        tinfo['n_samples'] = len(stamps)
        if dts:
            tinfo['monotonic'] = all(d > 0 for d in dts)
            median_dt = statistics.median(dts)
            tinfo['dt_median_s'] = float(median_dt)
            tinfo['dt_max_dev_s'] = float(max(abs(d - median_dt) for d in dts))
            tinfo['uniform'] = tinfo['dt_max_dev_s'] <= max(0.05 * median_dt, 1e-9)

    info['abort_channels'] = [n for n in names if 'abort' in n.casefold()]
    abort_active = []
    for name in info['abort_channels']:
        idx = names.index(name)
        if any(abs(r[idx]) > 1e-12 for r in rows):
            abort_active.append(name)

    config, config_reasons = _config_from_spec(path)
    info['config'] = config
    info['reasons'].extend(config_reasons)

    status = 'ok'
    if not rows:
        status, reason = 'partial', 'no parsable data rows in sampled window'
        info['reasons'].append(reason)
    if time_idx is None:
        status = 'partial'
        info['reasons'].append('no usable time channel among sampled channels')
    elif rows and info['time']['monotonic'] is False:
        status = 'partial'
        info['reasons'].append('time channel is non-monotonic in sampled rows')
    if bad_rows:
        info['reasons'].append(f'{bad_rows} sampled rows unparsable and skipped')
    if (info['n_points_declared'] is not None and not info['truncated']
            and info['n_points_declared'] != info['rows_read']):
        info['reasons'].append(
            f"declared Points={info['n_points_declared']} != rows_read="
            f"{info['rows_read']} within sampled window")
    if info['truncated']:
        info['reasons'].append(
            f'row sampling capped at max_rows={int(max_rows)}; file not fully read')
    info['parse_status'] = status

    # Policy: never infer collision from run-control events.
    info['reasons'].append(
        'collision_label is None by policy: abort/path-exit/sync/status channels '
        'are run-control events and are never interpreted as collision evidence')
    for name in abort_active:
        info['reasons'].append(
            f"abort activity on '{name}' in sampled rows (run-control event, "
            'NOT a collision label)')

    info['calibration_candidate'] = bool(
        status == 'ok'
        and config['control'] == 'robot'
        and config['brake_robot_engaged'] is False)
    # A configuration flag cannot establish the causal source of a brake event.
    info['suitable_for_aeb_calibration'] = False
    if info['calibration_candidate']:
        info['reasons'].append(
            'calibration candidate only: role=unconfirmed, human validation of '
            'channel semantics required before AEB calibration use')
    return info


# --------------------------------------------------------------------------- audit

def _evenly_sample(items, limit):
    if limit is None or len(items) <= limit:
        return list(items)
    if limit < 1:
        raise ValueError('limit must be positive')
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / (limit - 1)
    picked, last_idx = [], -1
    for i in range(limit):
        idx = min(round(i * step), len(items) - 1)
        if idx != last_idx:
            picked.append(items[idx])
            last_idx = idx
    return picked


def _peek_interaction_csv(path: Path, max_rows: int = 60) -> dict:
    rec = {'source': 'interaction', 'path': str(path), 'file': path.name,
           'parse_status': 'error', 'reasons': [], 'columns': [], 'n_columns': 0,
           'location': None, 'official_split': None, 'observed_only': False,
           'split': None, 'agent_types': [], 'dt_median_s': None,
           'rows_read': 0}
    location, official, observed = _location_and_split(path)
    rec['location'] = location
    rec['official_split'] = official
    rec['observed_only'] = bool(observed)
    rec['split'] = official or split_group('interaction', location)
    try:
        handle = open(path, 'r', newline='', encoding='utf-8-sig', errors='replace')
    except OSError as exc:
        rec['reasons'].append(str(exc))
        return rec
    with handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            rec['reasons'].append('empty file')
            return rec
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    rec['columns'] = [h.strip() for h in header]
    rec['n_columns'] = len(header)
    rec['rows_read'] = len(rows)
    cols = {h.strip().casefold(): i for i, h in enumerate(header) if h.strip()}
    agent_i = _pick_column(cols, ('agent_type', 'agenttype', 'object_type'))
    if agent_i is not None:
        rec['agent_types'] = sorted({r[agent_i].strip() for r in rows
                                     if agent_i < len(r) and r[agent_i].strip()})
    tms_i = _pick_column(cols, ('timestamp_ms', 'time_ms'))
    if tms_i is not None:
        stamps = [_to_float(r[tms_i]) for r in rows
                  if tms_i < len(r) and _to_float(r[tms_i]) is not None]
        dts = [b - a for a, b in zip(stamps, stamps[1:]) if b - a > 0]
        if dts:
            rec['dt_median_s'] = float(statistics.median(dts)) / 1000.0
    rec['parse_status'] = 'ok'
    return rec


def _manifest_row(source, rec):
    config = rec.get('config') or {}
    return {
        'source': source,
        'file': rec.get('file', ''),
        'path': rec.get('path', ''),
        'status': rec.get('parse_status', ''),
        'location': rec.get('location') or '',
        'split': rec.get('split') if rec.get('split') is not None else '',
        'n_channels': rec.get('n_channels', ''),
        'rows_sampled': rec.get('rows_read', ''),
        'control': config.get('control', ''),
        'use_brake_robot': ('' if config.get('use_brake_robot') is None
                            else str(config['use_brake_robot'])),
        'suitable_for_aeb_calibration': (
            '' if rec.get('suitable_for_aeb_calibration') is None
            else str(rec['suitable_for_aeb_calibration'])),
        'reasons': ' ; '.join(rec.get('reasons', [])),
    }


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def _inventory_summary(root: Path) -> dict:
    inv = {'status': 'missing', 'path': None, 'n_rows': 0, 'by_brand': {},
           'by_validity_label': {},
           'note': 'inventory rows are run-level metadata labels, not valid '
                   'episodes; never report them as scene/episode counts'}
    path = _first_existing([
        root / 'research_audit_20260910' / 'expanded_inventory' / 'all_runs_inventory.csv',
        root / 'research_audit_20260910' / 'inventory' / 'all_runs_inventory.csv',
    ])
    if path is None:
        return inv
    inv['status'], inv['path'] = 'ok', str(path)
    brands, labels = {}, {}
    with open(path, 'r', newline='', encoding='utf-8-sig', errors='replace') as handle:
        for row in csv.DictReader(handle):
            if not row:
                continue
            inv['n_rows'] += 1
            brand = (row.get('brand') or '').strip()
            if brand:
                brands[brand] = brands.get(brand, 0) + 1
            label = (row.get('validity_label') or '').strip()
            if label:
                labels[label] = labels.get(label, 0) + 1
    inv['by_brand'] = brands
    inv['by_validity_label'] = labels
    return inv


def audit_dataset(root: Path, output: Path, limit: int = 8) -> dict:
    """Bounded representative audit of the ABD + INTERACTION data under root.

    Inspects at most `limit` files per source with row-capped reads (ABD:
    inspect_abd at 300 rows; INTERACTION: header + 60-row peek), enumerates
    file totals without reading them, and summarizes the existing run
    inventory (metadata rows, explicitly not valid episodes). Writes
    <output>/dataset_audit.json and <output>/manifest.csv and returns the same
    summary dict. Never loads a full dataset file.
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    abd_base = _first_existing([root / 'Data' / 'ABD_Data', root / 'ABD_Data'])
    if abd_base is None:
        abd = {'status': 'missing', 'base_dir': None, 'n_files_total': 0,
               'reason': 'no ABD_Data directory under root'}
    else:
        txts = sorted(p for p in abd_base.rglob('*.txt')
                      if re.fullmatch(r'V\d+_T\d+_R\d+\.txt', p.name, re.IGNORECASE))
        samples = [_inspect_bounded(p) for p in _evenly_sample(txts, limit)]
        manifest_rows.extend(_manifest_row('abd', s) for s in samples)
        abd = {'status': 'ok' if txts else 'empty', 'base_dir': str(abd_base),
               'n_files_total': len(txts),
               'n_files_inspected': len(samples), 'samples': samples}

    inter_base = _first_existing([root / 'Data' / 'INTERACTION', root / 'INTERACTION'])
    if inter_base is None:
        interaction = {'status': 'missing', 'base_dir': None, 'n_files_total': 0,
                       'reason': 'no INTERACTION directory under root'}
    else:
        csvs = sorted(inter_base.rglob('*.csv'))
        samples = [_peek_interaction_csv(p) for p in _evenly_sample(csvs, limit)]
        manifest_rows.extend(_manifest_row('interaction', s) for s in samples)
        interaction = {'status': 'ok' if csvs else 'empty',
                       'base_dir': str(inter_base), 'n_files_total': len(csvs),
                       'n_files_inspected': len(samples), 'samples': samples}

    inventory = _inventory_summary(root)

    result = {
        'root': str(root),
        'limit': limit,
        'abd': abd,
        'interaction': interaction,
        'inventory': inventory,
        'notes': [
            'bounded inspection only: ABD runs capped at 300 sampled rows, '
            'INTERACTION peeks at 60 rows; no full dataset file was loaded',
            'interaction splits shown here follow iter_interaction_tracks rules '
            '(official token or deterministic sha256 group hash)',
            inventory['note'],
        ],
    }
    json_path = output / 'dataset_audit.json'
    manifest_path = output / 'manifest.csv'
    with open(manifest_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'source', 'file', 'path', 'status', 'location', 'split', 'n_channels',
            'rows_sampled', 'control', 'use_brake_robot',
            'suitable_for_aeb_calibration', 'reasons'])
        writer.writeheader()
        writer.writerows(manifest_rows)
    result['outputs'] = {'json': str(json_path), 'manifest': str(manifest_path)}
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    return result


def _inspect_bounded(path: Path, max_rows: int = 300) -> dict:
    return inspect_abd(path, max_rows=max_rows)
