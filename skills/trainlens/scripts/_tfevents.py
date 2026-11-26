"""Zero-dependency pure-stdlib TensorBoard tfevents scalar reader.

Parses TFRecord framing and the minimal protobuf wire subset needed to extract
(step, wall_time, tag, value) scalar events from ``events.out.tfevents.*``
without requiring ``tensorflow`` or ``tensorboard``.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    length = len(buf)
    while pos < length:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
        if shift > 64:
            raise ValueError("Malformed protobuf varint")
    raise ValueError("Truncated protobuf varint")


def _skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:  # varint
        _, pos = _read_varint(buf, pos)
        return pos
    if wire_type == 1:  # 64-bit
        return pos + 8
    if wire_type == 2:  # length-delimited
        size, pos = _read_varint(buf, pos)
        return pos + size
    if wire_type == 5:  # 32-bit
        return pos + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def _is_scalar_shape(buf: bytes) -> bool:
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        if key == (2 << 3 | 2):  # TensorShapeProto.dim
            return False
        if key == (3 << 3):  # unknown_rank
            unknown, pos = _read_varint(buf, pos)
            if unknown:
                return False
        else:
            pos = _skip_field(buf, pos, key & 7)
    return True


def _parse_tensor_scalar(buf: bytes) -> Optional[float]:
    """Extract a scalar float from TensorProto (dtype=DT_FLOAT=1 or DT_DOUBLE=2)."""
    pos = 0
    end = len(buf)
    dtype = 0
    float_val: Optional[float] = None
    double_val: Optional[float] = None
    tensor_content: Optional[bytes] = None

    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_num = key >> 3
        wire_type = key & 0x07
        if field_num == 1 and wire_type == 0:  # dtype
            dtype, pos = _read_varint(buf, pos)
        elif field_num == 2 and wire_type == 2:  # tensor_shape
            size, pos = _read_varint(buf, pos)
            if not _is_scalar_shape(buf[pos:pos + size]):
                return None
            pos += size
        elif field_num == 4 and wire_type == 2:  # tensor_content
            size, pos = _read_varint(buf, pos)
            tensor_content = buf[pos : pos + size]
            pos += size
        elif field_num == 5 and wire_type == 5:  # float_val
            float_val = struct.unpack_from("<f", buf, pos)[0]
            pos += 4
        elif field_num == 5 and wire_type == 2:  # packed float_val
            size, pos = _read_varint(buf, pos)
            if size == 4:
                float_val = struct.unpack_from("<f", buf, pos)[0]
            pos += size
        elif field_num == 6 and wire_type == 1:  # double_val
            double_val = struct.unpack_from("<d", buf, pos)[0]
            pos += 8
        elif field_num == 6 and wire_type == 2:  # packed double_val
            size, pos = _read_varint(buf, pos)
            if size == 8:
                double_val = struct.unpack_from("<d", buf, pos)[0]
            pos += size
        else:
            pos = _skip_field(buf, pos, wire_type)

    if dtype == 1 and float_val is not None:
        return float(float_val)
    if dtype == 2 and double_val is not None:
        return float(double_val)
    if tensor_content:
        if dtype == 2 and len(tensor_content) == 8:
            return float(struct.unpack("<d", tensor_content)[0])
        if dtype == 1 and len(tensor_content) == 4:
            return float(struct.unpack("<f", tensor_content)[0])
    return None


def _parse_summary_value(buf: bytes) -> Optional[Tuple[str, float]]:
    pos = 0
    end = len(buf)
    tag: Optional[str] = None
    val: Optional[float] = None

    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_num = key >> 3
        wire_type = key & 0x07
        if field_num == 1 and wire_type == 2:  # tag
            size, pos = _read_varint(buf, pos)
            tag = buf[pos : pos + size].decode("utf-8", errors="replace")
            pos += size
        elif field_num == 2 and wire_type == 5:  # simple_value (float)
            val = float(struct.unpack_from("<f", buf, pos)[0])
            pos += 4
        elif field_num == 8 and wire_type == 2:  # tensor (TF 2.x scalar)
            size, pos = _read_varint(buf, pos)
            tensor_val = _parse_tensor_scalar(buf[pos : pos + size])
            if tensor_val is not None:
                val = tensor_val
            pos += size
        else:
            pos = _skip_field(buf, pos, wire_type)

    if tag is not None and val is not None:
        return tag, val
    return None


def _parse_summary(buf: bytes) -> List[Tuple[str, float]]:
    pos = 0
    end = len(buf)
    items: List[Tuple[str, float]] = []
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_num = key >> 3
        wire_type = key & 0x07
        if field_num == 1 and wire_type == 2:  # repeated Summary.Value
            size, pos = _read_varint(buf, pos)
            parsed = _parse_summary_value(buf[pos : pos + size])
            if parsed is not None:
                items.append(parsed)
            pos += size
        else:
            pos = _skip_field(buf, pos, wire_type)
    return items


def _parse_event(buf: bytes) -> Tuple[int, float, List[Tuple[str, float]]]:
    pos = 0
    end = len(buf)
    wall_time = 0.0
    step = 0
    scalars: List[Tuple[str, float]] = []

    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_num = key >> 3
        wire_type = key & 0x07
        if field_num == 1 and wire_type == 1:  # wall_time
            wall_time = struct.unpack_from("<d", buf, pos)[0]
            pos += 8
        elif field_num == 2 and wire_type == 0:  # step
            step, pos = _read_varint(buf, pos)
        elif field_num == 5 and wire_type == 2:  # summary
            size, pos = _read_varint(buf, pos)
            scalars.extend(_parse_summary(buf[pos : pos + size]))
            pos += size
        else:
            pos = _skip_field(buf, pos, wire_type)

    return step, wall_time, scalars


def iter_tfevents_records(path: Path) -> Iterator[bytes]:
    """Yield raw protobuf Event payloads from a TFRecord file, tolerating truncated tails."""
    with path.open("rb") as fh:
        while True:
            header = fh.read(12)
            if len(header) < 12:
                break
            length = struct.unpack("<Q", header[:8])[0]
            if length > 64 * 1024 * 1024:  # guard against corrupted length
                break
            payload = fh.read(length)
            if len(payload) < length:
                break
            crc_tail = fh.read(4)
            if len(crc_tail) < 4:
                break
            yield payload


def read_tfevents_scalars(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Return ``{tag: [(step, value), ...]}`` parsed from a single tfevents file."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    for payload in iter_tfevents_records(path):
        try:
            step, _wall_time, scalars = _parse_event(payload)
        except (ValueError, struct.error):
            continue
        for tag, value in scalars:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                series.setdefault(tag, []).append((step, value))
            else:
                series.setdefault(tag, []).append((step, float(value)))
    return series
