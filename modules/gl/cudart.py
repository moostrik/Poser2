"""The CUDA runtime's OpenGL interop, called through ctypes in the CUDA runtime library that torch loaded.

torch's wheel brings its own CUDA runtime (`cudart64_<major>.dll` on Windows, `libcudart.so.<major>` on
Linux) and has it loaded by the time this module is imported, so no CUDA toolkit and no cuda-python
package is needed. Only the calls the `Tensor` texture and the `ScreenCapture` use are bound.
"""

import ctypes
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SUCCESS = 0
REGISTER_FLAGS_READ_ONLY = 0x01         # cudaGraphicsRegisterFlagsReadOnly
REGISTER_FLAGS_WRITE_DISCARD = 0x02     # cudaGraphicsRegisterFlagsWriteDiscard
MEMCPY_DEVICE_TO_HOST = 2               # cudaMemcpyDeviceToHost
MEMCPY_DEVICE_TO_DEVICE = 3             # cudaMemcpyDeviceToDevice
STREAM_NON_BLOCKING = 0x01              # cudaStreamNonBlocking: no implicit sync with the default stream
EVENT_DISABLE_TIMING = 0x02             # cudaEventDisableTiming: an event only to wait on


class CudaError(RuntimeError):
    """A CUDA runtime call that did not return cudaSuccess."""


def _load() -> ctypes.CDLL:
    """The CUDA runtime library torch uses, by the name torch's CUDA version gives it."""
    if torch.version.cuda is None:
        raise OSError("torch has no CUDA")
    major = torch.version.cuda.split('.')[0]
    names = [f"cudart64_{major}.dll", f"libcudart.so.{major}"]
    # The library is already loaded by torch, so its bare name resolves; the paths are the fallback
    torch_lib = Path(torch.__file__).parent / 'lib'
    site = torch_lib.parents[1]
    candidates = names + [str(p) for name in names
                                             for p in (list(torch_lib.glob(name)) + list(site.glob(f"nvidia/**/{name}")))]
    errors: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as e:
            errors.append(f"{candidate}: {e}")
    raise OSError("; ".join(errors))


_lib = _load()

_lib.cudaGetErrorString.restype = ctypes.c_char_p
_lib.cudaGetErrorString.argtypes = [ctypes.c_int]
_lib.cudaGraphicsGLRegisterBuffer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint]
_lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
_lib.cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_lib.cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_lib.cudaGraphicsResourceGetMappedPointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
                                                      ctypes.c_void_p]
_lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_lib.cudaFree.argtypes = [ctypes.c_void_p]
_lib.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
_lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
_lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
_lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
_lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
_lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
for _name in ('cudaGraphicsGLRegisterBuffer', 'cudaGraphicsUnregisterResource', 'cudaGraphicsMapResources',
              'cudaGraphicsUnmapResources', 'cudaGraphicsResourceGetMappedPointer', 'cudaMemcpyAsync',
              'cudaMalloc', 'cudaFree', 'cudaMallocHost', 'cudaFreeHost', 'cudaMemGetInfo',
              'cudaStreamCreateWithFlags', 'cudaStreamDestroy', 'cudaStreamSynchronize',
              'cudaEventCreateWithFlags', 'cudaEventDestroy', 'cudaEventRecord', 'cudaEventSynchronize'):
    getattr(_lib, _name).restype = ctypes.c_int


def _check(err: int, call: str) -> None:
    if err != SUCCESS:
        message = _lib.cudaGetErrorString(err)
        raise CudaError(f"{call} failed: {message.decode() if message else err}")


def register_gl_buffer(buffer: int, flags: int = REGISTER_FLAGS_WRITE_DISCARD) -> int:
    """Register an OpenGL buffer object with CUDA; returns the graphics resource handle."""
    resource = ctypes.c_void_p()
    _check(_lib.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), buffer, flags), 'cudaGraphicsGLRegisterBuffer')
    return resource.value or 0


def unregister_resource(resource: int) -> None:
    _check(_lib.cudaGraphicsUnregisterResource(resource), 'cudaGraphicsUnregisterResource')


def map_resource(resource: int, stream: int) -> None:
    handle = ctypes.c_void_p(resource)
    _check(_lib.cudaGraphicsMapResources(1, ctypes.byref(handle), stream), 'cudaGraphicsMapResources')


def unmap_resource(resource: int, stream: int) -> None:
    handle = ctypes.c_void_p(resource)
    _check(_lib.cudaGraphicsUnmapResources(1, ctypes.byref(handle), stream), 'cudaGraphicsUnmapResources')


def mapped_pointer(resource: int) -> tuple[int, int]:
    """The device pointer and size in bytes of a mapped resource."""
    pointer, size = ctypes.c_void_p(), ctypes.c_size_t()
    _check(_lib.cudaGraphicsResourceGetMappedPointer(ctypes.byref(pointer), ctypes.byref(size), resource),
           'cudaGraphicsResourceGetMappedPointer')
    return pointer.value or 0, size.value


def memcpy_device_to_device(dst: int, src: int, size: int, stream: int) -> None:
    """A device-to-device copy queued on `stream`; the caller does not wait for it. Work queued on the same stream
    after it runs after it, and the unmap of a mapped resource on that stream finishes it before OpenGL reads."""
    _check(_lib.cudaMemcpyAsync(dst, src, size, MEMCPY_DEVICE_TO_DEVICE, stream), 'cudaMemcpyAsync')


def memcpy_device_to_host(dst: int, src: int, size: int, stream: int) -> None:
    """A device-to-host copy queued on `stream`; asynchronous only into pinned host memory (`malloc_host`)."""
    _check(_lib.cudaMemcpyAsync(dst, src, size, MEMCPY_DEVICE_TO_HOST, stream), 'cudaMemcpyAsync')


def malloc(size: int) -> int:
    """`size` bytes of device memory; the device pointer."""
    pointer = ctypes.c_void_p()
    _check(_lib.cudaMalloc(ctypes.byref(pointer), size), 'cudaMalloc')
    return pointer.value or 0


def free(pointer: int) -> None:
    _check(_lib.cudaFree(pointer), 'cudaFree')


def malloc_host(size: int) -> int:
    """`size` bytes of pinned (page-locked) host memory; the host pointer."""
    pointer = ctypes.c_void_p()
    _check(_lib.cudaMallocHost(ctypes.byref(pointer), size), 'cudaMallocHost')
    return pointer.value or 0


def free_host(pointer: int) -> None:
    _check(_lib.cudaFreeHost(pointer), 'cudaFreeHost')


def mem_get_info() -> tuple[int, int]:
    """The device's free and total memory in bytes, counting what every process holds."""
    free_bytes, total_bytes = ctypes.c_size_t(), ctypes.c_size_t()
    _check(_lib.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)), 'cudaMemGetInfo')
    return free_bytes.value, total_bytes.value


def stream_create() -> int:
    """A stream of its own that does not sync with the default stream."""
    stream = ctypes.c_void_p()
    _check(_lib.cudaStreamCreateWithFlags(ctypes.byref(stream), STREAM_NON_BLOCKING), 'cudaStreamCreateWithFlags')
    return stream.value or 0


def stream_destroy(stream: int) -> None:
    _check(_lib.cudaStreamDestroy(stream), 'cudaStreamDestroy')


def stream_synchronize(stream: int) -> None:
    """Wait until all work queued on `stream` is done."""
    _check(_lib.cudaStreamSynchronize(stream), 'cudaStreamSynchronize')


def event_create() -> int:
    """An event to wait on, without timing."""
    event = ctypes.c_void_p()
    _check(_lib.cudaEventCreateWithFlags(ctypes.byref(event), EVENT_DISABLE_TIMING), 'cudaEventCreateWithFlags')
    return event.value or 0


def event_destroy(event: int) -> None:
    _check(_lib.cudaEventDestroy(event), 'cudaEventDestroy')


def event_record(event: int, stream: int) -> None:
    """Mark the point in `stream` after the work queued so far."""
    _check(_lib.cudaEventRecord(event, stream), 'cudaEventRecord')


def event_synchronize(event: int) -> None:
    """Wait until the work before the event's last record is done."""
    _check(_lib.cudaEventSynchronize(event), 'cudaEventSynchronize')
