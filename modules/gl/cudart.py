"""The CUDA runtime's OpenGL interop, called through ctypes in the CUDA runtime library that torch loaded.

torch's wheel brings its own CUDA runtime (`cudart64_<major>.dll` on Windows, `libcudart.so.<major>` on
Linux) and has it loaded by the time this module is imported, so no CUDA toolkit and no cuda-python
package is needed. Only the six calls the `Tensor` texture uses are bound.
"""

import ctypes
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SUCCESS = 0
REGISTER_FLAGS_WRITE_DISCARD = 0x02     # cudaGraphicsRegisterFlagsWriteDiscard
MEMCPY_DEVICE_TO_DEVICE = 3             # cudaMemcpyDeviceToDevice


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
_lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
for _name in ('cudaGraphicsGLRegisterBuffer', 'cudaGraphicsUnregisterResource', 'cudaGraphicsMapResources',
              'cudaGraphicsUnmapResources', 'cudaGraphicsResourceGetMappedPointer', 'cudaMemcpy'):
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


def memcpy_device_to_device(dst: int, src: int, size: int) -> None:
    _check(_lib.cudaMemcpy(dst, src, size, MEMCPY_DEVICE_TO_DEVICE), 'cudaMemcpy')
