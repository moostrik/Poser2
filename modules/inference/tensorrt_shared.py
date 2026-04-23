"""Shared TensorRT resources to prevent Myelin graph conflicts between multiple models."""
from threading import Lock
import tensorrt as trt

# Suppress the mismatched logger warning - it's harmless
# TensorRT will use whichever logger was created first
_trt_logger: trt.Logger = trt.Logger(trt.Logger.ERROR)  # type: ignore

# Global TensorRT runtime singleton
_trt_runtime: trt.Runtime | None = None  # type: ignore
_trt_runtime_lock = Lock()

# Global initialization lock prevents concurrent engine/context creation
# This prevents TensorRT Myelin graph conflicts when multiple models load simultaneously
_trt_init_lock = Lock()

# Global execution lock - all TensorRT inference calls are serialized
# This prevents Myelin graph conflicts during concurrent inference
_trt_exec_lock = Lock()


def get_tensorrt_runtime() -> trt.Runtime:  # type: ignore
    """Get or create shared TensorRT runtime singleton.

    All TensorRT models should use this shared runtime to avoid resource conflicts.
    Uses a shared logger to prevent TensorRT warning about mismatched loggers.
    """
    global _trt_runtime
    with _trt_runtime_lock:
        if _trt_runtime is None:
            _trt_runtime = trt.Runtime(_trt_logger)  # type: ignore
        return _trt_runtime


def get_init_lock() -> Lock:
    """Get the global TensorRT initialization lock."""
    return _trt_init_lock


def get_exec_lock() -> Lock:
    """Get the global TensorRT execution lock."""
    return _trt_exec_lock
