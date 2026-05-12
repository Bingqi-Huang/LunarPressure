from __future__ import annotations

import functools
from typing import Any

import numpy as np


def _load_openpi_msgpack_numpy() -> Any | None:
    try:
        from openpi_client import msgpack_numpy

        return msgpack_numpy
    except ImportError:
        return None


def pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack transport: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class MsgpackNumpyCodec:
    def __init__(self):
        openpi_msgpack_numpy = _load_openpi_msgpack_numpy()
        if openpi_msgpack_numpy is not None:
            self._packer = openpi_msgpack_numpy.Packer()
            self._unpackb = openpi_msgpack_numpy.unpackb
        else:
            try:
                import msgpack
            except ImportError as exc:
                raise RuntimeError("msgpack is required for OpenPI websocket transport") from exc
            self._packer = msgpack.Packer(default=pack_array)
            self._unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array, raw=False)

    def pack(self, value: Any) -> bytes:
        return self._packer.pack(value)

    def unpack(self, value: bytes) -> Any:
        return self._unpackb(value)

