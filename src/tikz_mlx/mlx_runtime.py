from __future__ import annotations

import ctypes
import sys
from types import ModuleType


class MlxRuntimeUnavailableError(RuntimeError):
    """Raised before importing MLX when the current process cannot see Metal."""


def metal_device_count() -> int | None:
    """Return visible Metal device count on macOS without importing MLX.

    MLX's native extension aborts the interpreter on some macOS 26 sandboxed
    processes when Metal returns an empty device list.  This probe uses the
    system Metal/CoreFoundation APIs directly so callers can fail cleanly before
    importing ``mlx.core``.
    """
    if sys.platform != "darwin":
        return None
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        corefoundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        metal.MTLCopyAllDevices.restype = ctypes.c_void_p
        corefoundation.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        corefoundation.CFArrayGetCount.restype = ctypes.c_long
        corefoundation.CFRelease.argtypes = [ctypes.c_void_p]

        devices = metal.MTLCopyAllDevices()
        if not devices:
            return 0
        try:
            return int(corefoundation.CFArrayGetCount(devices))
        finally:
            corefoundation.CFRelease(devices)
    except Exception:
        return None


def ensure_mlx_runtime_available() -> None:
    count = metal_device_count()
    if count == 0:
        raise MlxRuntimeUnavailableError(
            "No Metal devices are visible to this process. Importing mlx.core in this state aborts "
            "the Python interpreter on macOS 26. Run MLX training/tests outside the sandbox or grant "
            "GPU/Metal access to the process."
        )


def import_mlx_core() -> ModuleType:
    ensure_mlx_runtime_available()
    import mlx.core as mx

    return mx


def import_mlx_nn() -> ModuleType:
    ensure_mlx_runtime_available()
    import mlx.nn as nn

    return nn
