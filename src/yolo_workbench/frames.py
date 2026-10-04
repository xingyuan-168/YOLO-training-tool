"""Bounded latest-frame transport; safe to import from the GUI without numpy/Qt."""

from __future__ import annotations

import struct
import uuid
from multiprocessing import shared_memory

_HEADER = struct.Struct("<8sQQIIII")
_HEADER_SIZE = 64
_MAGIC = b"YWBFRM01"
_MAX_BYTES = 64 * 1024 * 1024


class SharedFrameWriter:
    """One producer, two slots, at most one retired allocation while resizing.

    References may expire as the producer advances. Readers copy and validate the
    sequence twice, and retain their own OS handle. On Windows a mapping remains
    valid until the last reader closes it, even when its producer has exited.
    """

    def __init__(self, *, max_bytes=_MAX_BYTES):
        self.max_bytes = min(int(max_bytes), _MAX_BYTES)
        if self.max_bytes < 1:
            raise ValueError("共享帧容量必须为正数")
        self.sequence = 0
        self.capacity = 0
        self._memory = None
        self._retired = None
        self._closed = False

    @staticmethod
    def _dispose(memory):
        if memory is not None:
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                pass

    def write(self, data, width=None, height=None, channels=3, *, stride=None):
        if self._closed:
            raise RuntimeError("共享帧写入器已关闭")
        if width is None:
            height, width, channels = data.shape
            data = data.tobytes(order="C")
        width, height, channels = int(width), int(height), int(channels)
        stride = int(stride or width * channels)
        if width < 1 or height < 1 or channels not in (3, 4) or stride < width * channels:
            raise ValueError("非法帧尺寸")
        data = bytes(data)
        count = stride * height
        if len(data) != count or count > self.max_bytes:
            raise ValueError("帧缓冲长度无效或超出容量上限")
        if count > self.capacity:
            self._dispose(self._retired)
            self._retired = self._memory
            self.capacity = min(1 << max(12, (count - 1).bit_length()), self.max_bytes)
            self._memory = shared_memory.SharedMemory(
                name=f"ywb_{uuid.uuid4().hex}", create=True, size=2 * (_HEADER_SIZE + self.capacity)
            )
        self.sequence += 1
        slot = self.sequence % 2
        offset = slot * (_HEADER_SIZE + self.capacity)
        generation = self.sequence * 2
        _HEADER.pack_into(
            self._memory.buf, offset, _MAGIC, generation - 1, count, width, height, channels, stride
        )
        self._memory.buf[offset + _HEADER_SIZE : offset + _HEADER_SIZE + count] = data
        # Publish only after all pixels and geometry are complete.
        struct.pack_into("<Q", self._memory.buf, offset + 8, generation)
        return {
            "schema_version": 1,
            "shm_name": self._memory.name,
            "slot": slot,
            "capacity": self.capacity,
            "sequence": self.sequence,
            "width": width,
            "height": height,
            "channels": channels,
            "stride": stride,
            "pixel_format": "BGR" if channels == 3 else "BGRA",
        }

    def close(self):
        self._dispose(self._memory)
        self._dispose(self._retired)
        self._memory = self._retired = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SharedFrameReader:
    """Read a stable copied frame or None when a reference has already expired."""

    def __init__(self):
        self._memory = None

    def read(self, reference: dict) -> dict | None:
        if reference.get("schema_version") != 1:
            raise ValueError("不支持的共享帧版本")
        name = reference.get("shm_name", "")
        capacity, slot, sequence = (reference.get(key) for key in ("capacity", "slot", "sequence"))
        if (
            not isinstance(name, str)
            or not name.startswith("ywb_")
            or any(type(value) is not int for value in (capacity, slot, sequence))
            or not 1 <= capacity <= _MAX_BYTES
            or slot not in (0, 1)
            or sequence < 1
        ):
            raise ValueError("非法共享帧引用")
        if self._memory is None or self._memory.name != name:
            try:
                memory = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                return None
            self.close()
            self._memory = memory
        # Windows reports the page-rounded allocation size when attaching.
        if not 2 * (_HEADER_SIZE + capacity) <= self._memory.size < 2 * (_HEADER_SIZE + capacity) + 65536:
            raise ValueError("共享帧容量不匹配")
        offset = slot * (_HEADER_SIZE + capacity)
        before = bytes(self._memory.buf[offset : offset + _HEADER.size])
        magic, generation, count, width, height, channels, stride = _HEADER.unpack(before)
        if magic != _MAGIC or generation != sequence * 2:
            return None
        if (
            channels not in (3, 4)
            or width < 1
            or height < 1
            or stride < width * channels
            or count != stride * height
            or count > capacity
        ):
            raise ValueError("共享帧头损坏")
        pixels = bytes(self._memory.buf[offset + _HEADER_SIZE : offset + _HEADER_SIZE + count])
        after = bytes(self._memory.buf[offset : offset + _HEADER.size])
        if before != after:
            return None
        return {
            "data": pixels,
            "width": width,
            "height": height,
            "channels": channels,
            "stride": stride,
            "pixel_format": "BGR" if channels == 3 else "BGRA",
            "sequence": sequence,
        }

    def close(self):
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
