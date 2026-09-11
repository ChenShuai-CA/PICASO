import struct
import pytest
from scenario_lab.waymo import masked_crc32c, iter_tfrecord, fields, parse_scenario


def test_crc_and_corruption(tmp_path):
    payload = b'hello test'
    length = struct.pack('<Q', len(payload))
    framed = length + struct.pack('<I', masked_crc32c(length)) + payload + struct.pack('<I', masked_crc32c(payload))
    path = tmp_path / 'sample.tfrecord'
    path.write_bytes(framed)
    assert list(iter_tfrecord(path)) == [payload]
    path.write_bytes(framed[:-1])
    with pytest.raises(ValueError, match='CRC/truncation'):
        list(iter_tfrecord(path))


def test_not_scenario_rejected():
    with pytest.raises(ValueError):
        parse_scenario(b'\x08\x01')
    with pytest.raises(ValueError):
        list(fields(b'\x0a\x10a'))
