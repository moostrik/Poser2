"""CUDA C kernels compiled at runtime (NVRTC) and launched on a torch stream, through ctypes.

torch's wheel brings NVRTC (`nvrtc64_<major>0_0.dll` on Windows, `libnvrtc.so.<major>` on Linux) and the driver
brings its API (`nvcuda.dll`, `libcuda.so.1`), so no CUDA toolkit, compiler or package is needed: the source is
compiled for the GPU's own architecture once, on first use, and the kernels run on the context torch made current.
The libraries are loaded on first use too: a machine without them imports this, and `CudaKernels.launch` raises.
"""

import ctypes
import logging
from pathlib import Path
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

SUCCESS = 0


class CudaKernelError(RuntimeError):
    """An NVRTC or CUDA driver call that did not succeed."""


def _load(names: list[str]) -> ctypes.CDLL:
    torch_lib: Path = Path(torch.__file__).parent / 'lib'
    candidates: list[str] = names + [str(path) for name in names for path in torch_lib.glob(name)]
    errors: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as e:
            errors.append(f"{candidate}: {e}")
    raise OSError("; ".join(errors))


def _libraries() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    if torch.version.cuda is None:
        raise OSError("torch has no CUDA")
    major: str = torch.version.cuda.split('.')[0]
    nvrtc: ctypes.CDLL = _load([f"nvrtc64_{major}0_0.dll", f"libnvrtc.so.{major}"])
    driver: ctypes.CDLL = _load(["nvcuda.dll", "libcuda.so.1"])
    p, c, i = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    nvrtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(p), c, c, i, ctypes.POINTER(c), ctypes.POINTER(c)]
    nvrtc.nvrtcCompileProgram.argtypes = [p, i, ctypes.POINTER(c)]
    nvrtc.nvrtcGetProgramLogSize.argtypes = [p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes = [p, c]
    nvrtc.nvrtcGetCUBINSize.argtypes = [p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetCUBIN.argtypes = [p, c]
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(p)]
    nvrtc.nvrtcGetErrorString.argtypes = [i]
    nvrtc.nvrtcGetErrorString.restype = c
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(p)]
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(p), p]
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(p), p, c]
    driver.cuLaunchKernel.argtypes = [p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_uint, p, ctypes.POINTER(p), ctypes.POINTER(p)]
    driver.cuGetErrorString.argtypes = [i, ctypes.POINTER(c)]
    for name in ('nvrtcCreateProgram', 'nvrtcCompileProgram', 'nvrtcGetProgramLogSize', 'nvrtcGetProgramLog', 'nvrtcGetCUBINSize',
                 'nvrtcGetCUBIN', 'nvrtcDestroyProgram'):
        getattr(nvrtc, name).restype = i
    for name in ('cuCtxGetCurrent', 'cuModuleLoadData', 'cuModuleGetFunction', 'cuLaunchKernel', 'cuGetErrorString'):
        getattr(driver, name).restype = i
    return nvrtc, driver


_loaded: tuple[ctypes.CDLL, ctypes.CDLL] | None = None


def _nvrtc() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = _libraries()
    return _loaded[0]


def _driver() -> ctypes.CDLL:
    _nvrtc()
    assert _loaded is not None
    return _loaded[1]


def _check_nvrtc(result: int, call: str) -> None:
    if result != SUCCESS:
        raise CudaKernelError(f"{call} failed: {_nvrtc().nvrtcGetErrorString(result).decode()}")


def _check_driver(result: int, call: str) -> None:
    if result != SUCCESS:
        text = ctypes.c_char_p()
        _driver().cuGetErrorString(result, ctypes.byref(text))
        raise CudaKernelError(f"{call} failed: {(text.value or b'error %d' % result).decode()}")


def compile_cubin(source: str, name: str) -> bytes:
    """`source` compiled by NVRTC for the current GPU's architecture."""
    major, minor = torch.cuda.get_device_capability()
    program = ctypes.c_void_p()
    nvrtc: ctypes.CDLL = _nvrtc()
    _check_nvrtc(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source.encode(), name.encode(), 0, None, None), 'nvrtcCreateProgram')
    try:
        options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        result: int = nvrtc.nvrtcCompileProgram(program, len(options), (ctypes.c_char_p * len(options))(*options))
        if result != SUCCESS:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise CudaKernelError(f"{name} did not compile:\n{log.value.decode(errors='replace')}")
        size = ctypes.c_size_t()
        _check_nvrtc(nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)), 'nvrtcGetCUBINSize')
        cubin = ctypes.create_string_buffer(size.value)
        _check_nvrtc(nvrtc.nvrtcGetCUBIN(program, cubin), 'nvrtcGetCUBIN')
        return cubin.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


class CudaKernels:
    """The `__global__` functions `names` of the CUDA C `source`, compiled and loaded on first launch, on the
    context torch made current on the calling thread. `launch` queues one on a torch stream; a kernel's arguments
    are ctypes values, in the order its signature takes them."""

    def __init__(self, source: str, names: Sequence[str], label: str) -> None:
        self._source: str = source
        self._names: tuple[str, ...] = tuple(names)
        self._label: str = label
        self._cubin: bytes | None = None        # kept alive while the module uses it
        self._functions: dict[str, ctypes.c_void_p] = {}

    def _load(self) -> None:
        driver: ctypes.CDLL = _driver()
        context = ctypes.c_void_p()
        _check_driver(driver.cuCtxGetCurrent(ctypes.byref(context)), 'cuCtxGetCurrent')
        if not context.value:
            raise CudaKernelError(f"{self._label}: no CUDA context on this thread; run a torch CUDA call on it first")
        self._cubin = compile_cubin(self._source, self._label)
        module = ctypes.c_void_p()
        _check_driver(driver.cuModuleLoadData(ctypes.byref(module), self._cubin), 'cuModuleLoadData')
        for name in self._names:
            function = ctypes.c_void_p()
            _check_driver(driver.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), f'cuModuleGetFunction({name})')
            self._functions[name] = function
        logger.info("%s: %d CUDA kernels compiled", self._label, len(self._functions))

    def launch(self, name: str, grid: tuple[int, int], block: tuple[int, int], args: Sequence[ctypes._SimpleCData | ctypes.Structure],
               stream: torch.cuda.Stream) -> None:
        """Queue kernel `name` on `stream`; raises `CudaKernelError` or `OSError` where they cannot be compiled or loaded."""
        if not self._functions:
            self._load()
        pointers = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in args])
        _check_driver(_driver().cuLaunchKernel(self._functions[name], grid[0], grid[1], 1, block[0], block[1], 1, 0,
                                             ctypes.c_void_p(stream.cuda_stream), pointers, None), f'cuLaunchKernel({name})')
