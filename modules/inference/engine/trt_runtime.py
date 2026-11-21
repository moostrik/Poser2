"""The TensorRT runtime and logger the models share. Each model loads its engine and runs its own execution context on
its own thread and stream, without a lock between them."""
from threading import Lock
import tensorrt as trt

# One logger for every runtime: TensorRT warns about a second, and uses the first anyway
_trt_logger: trt.Logger = trt.Logger(trt.Logger.ERROR)

_trt_runtime: trt.Runtime | None = None
_trt_runtime_lock = Lock()


def get_tensorrt_runtime() -> trt.Runtime:
    """The shared TensorRT runtime, made at the first call."""
    global _trt_runtime
    with _trt_runtime_lock:
        if _trt_runtime is None:
            _trt_runtime = trt.Runtime(_trt_logger)
        return _trt_runtime
