"""Bounded Motion TFRecord reader without TensorFlow.

Reads the documented Scenario/Track/ObjectState wire fields only (not sensor frames).
Schema: github.com/waymo-research/waymo-open-dataset, protos/scenario.proto.
Unknown fields are skipped; frame CRC32C and required field shapes are checked.
"""
from pathlib import Path
import struct
import numpy as np


def _crc_table():
    table = []
    for i in range(256):
        for _ in range(8):
            i = (i >> 1) ^ (0x82F63B78 if i & 1 else 0)
        table.append(i)
    return table


_CRC = _crc_table()


def masked_crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord(path, limit=None):
    with Path(path).open('rb') as f:
        index = 0
        while limit is None or index < limit:
            length = f.read(8)
            if not length:
                return
            crc = f.read(4)
            if len(length) != 8 or len(crc) != 4 or struct.unpack('<I', crc)[0] != masked_crc32c(length):
                raise ValueError(f'TFRecord length CRC/truncation at record {index}')
            size = struct.unpack('<Q', length)[0]
            if size > 128 * 1024 * 1024:
                raise ValueError('record exceeds bounded Motion reader limit')
            record, crc = f.read(size), f.read(4)
            if len(record) != size or len(crc) != 4 or struct.unpack('<I', crc)[0] != masked_crc32c(record):
                raise ValueError(f'TFRecord data CRC/truncation at record {index}')
            yield record
            index += 1


def _varint(data, index):
    value = 0
    for shift in range(0, 70, 7):
        if index >= len(data):
            raise ValueError('truncated protobuf varint')
        byte = data[index]
        index += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, index
    raise ValueError('oversized protobuf varint')


def fields(data):
    index = 0
    while index < len(data):
        tag, index = _varint(data, index)
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError('invalid protobuf field number')
        if wire == 0:
            value, index = _varint(data, index)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            if index + size > len(data):
                raise ValueError('truncated protobuf fixed field')
            value = struct.unpack_from('<d' if wire == 1 else '<f', data, index)[0]
            index += size
        elif wire == 2:
            size, index = _varint(data, index)
            if index + size > len(data):
                raise ValueError('truncated protobuf bytes field')
            value = data[index:index + size]
            index += size
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        yield field, wire, value


def parse_scenario(record):
    scenario_id, timestamps, tracks = None, [], []
    for key, wire, value in fields(record):
        if key == 5 and wire == 2:
            scenario_id = value.decode('utf-8')
        elif key == 1:
            timestamps.extend(np.frombuffer(value, dtype='<f8').tolist() if wire == 2 else [value])
        elif key == 2 and wire == 2:
            track = dict(id=None, kind='other', states=[])
            for k, w, v in fields(value):
                if k == 1:
                    track['id'] = str(v)
                elif k == 2:
                    track['kind'] = {1: 'vehicle', 2: 'pedestrian'}.get(v, 'other')
                elif k == 3 and w == 2:
                    state = {f: x for f, _, x in fields(v)}
                    valid = state.get(11, 0) == 1 and all(f in state for f in (2, 3, 8, 9, 10))
                    track['states'].append([state.get(2, np.nan), state.get(3, np.nan),
                                            state.get(9, np.nan), state.get(10, np.nan),
                                            state.get(8, np.nan), valid])
            tracks.append(track)
    if not scenario_id or len(timestamps) < 2 or not tracks:
        raise ValueError('not a supported Waymo Motion Scenario')
    t = np.asarray(timestamps, dtype=float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('invalid Scenario timestamps')
    return scenario_id, t, tracks


def iter_waymo_tracks(path, limit=8):
    from .data import split_group
    path = Path(path)
    official = 'test' if any('test' in p.lower() for p in path.parts[-3:]) else ('val' if any('val' in p.lower() for p in path.parts[-3:]) else None)
    for record in iter_tfrecord(path, limit):
        sid, t, tracks = parse_scenario(record)
        for track in tracks:
            if len(track['states']) != len(t):
                raise ValueError('track and scenario timestamp lengths differ')
            array = np.asarray(track['states'], dtype=float)
            valid = (array[:, 5] > 0) & np.isfinite(array[:, :5]).all(axis=1)
            indices = np.flatnonzero(valid)
            for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if len(run) < 3:
                    continue
                yield dict(source='waymo', group_id=sid, track_id=track['id'], kind=track['kind'],
                           split=official or split_group('waymo', sid), t=t[run], xy=array[run, :2],
                           velocity=array[run, 2:4], heading=array[run, 4],
                           provenance=dict(path=str(path), crc_verified=True,
                                           action_source='kinematic_estimate', segment_start=int(run[0]),
                                           limitations='no map or sensor fields decoded'))
