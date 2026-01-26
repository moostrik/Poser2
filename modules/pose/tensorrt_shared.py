"""Shared TensorRT resources to prevent Myelin graph conflicts between multiple models."""
from threading import Lock
import tensorrt as trt

# Global TensorRT runtime singleton
_trt_runtime: trt.Runtime | None = None  # type: ignore
_trt_runtime_lock = Lock()

# Global initialization lock prevents concurrent engine/context creation
# This prevents TensorRT Myelin graph conflicts when multiple models load simultaneously
_trt_init_lock = Lock()

# Global execution lock prevents concurrent inference
# TensorRT Myelin graphs conflict when multiple contexts execute simultaneously
_trt_exec_lock = Lock()


def get_tensorrt_runtime() -> trt.Runtime:  # type: ignore
    """Get or create shared TensorRT runtime singleton.

    All TensorRT models should use this shared runtime to avoid resource conflicts.
    """
    global _trt_runtime
    with _trt_runtime_lock:
        if _trt_runtime is None:
            _trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))  # type: ignore
        return _trt_runtime


def get_init_lock() -> Lock:
    """Get the global TensorRT initialization lock.

    Models should acquire this lock during engine deserialization and context creation
    to prevent concurrent Myelin graph loading which causes TensorRT errors.
    """
    return _trt_init_lock


def get_exec_lock() -> Lock:
    """Get the global TensorRT execution lock.

    Models should acquire this lock during execute_async_v3() calls to prevent
    concurrent inference which causes Myelin graph conflicts.
    """
    return _trt_exec_lock
