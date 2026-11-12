"""The camera's depth moved onto the color frame's view on the GPU: what `DepthAlign` does in GL. `align_staged`, the
uploader's, runs it as two CUDA kernels compiled at runtime (`SOURCE`, `cuda_kernels`); `align` is the same in torch
ops, which run on the CPU too, the reference the tests and the harness hold the kernels to.

Every depth pixel goes to its point in space (`to_ray`, at its distance), into the color frame (`to_uv`), through the
straightening homography and the mirror, and lands on the 2x2 grid cells around it; where two land on one cell the
nearest is kept, and a cell nothing lands on is 0. The value is the depth along the level camera's axis, the point's
depth times the homography's third row (`video.z * corrected.z`), as a level camera would have measured it. Rows run
down, as in the frame. `to_ray` and `to_uv` are `modules.camera.depth_projection`'s, the size `aligned_depth_size`.
`scripts/depth_align_check.py` compares it with the GL shader on recorded frames.
"""

import ctypes
import logging

import numpy as np
import torch

from .cuda_kernels import CudaKernelError, CudaKernels

logger = logging.getLogger(__name__)

IDENTITY: np.ndarray = np.eye(3)
BLOCK: tuple[int, int] = (16, 16)

# The GL `DepthAlign` shader's two stages as CUDA kernels. SPLAT runs over the depth pixels: each goes to its point
# in space, into the color frame, through the homography and the mirror, and lands on the 2x2 cells around it, the
# nearest kept by an atomic minimum on the float's bits (positive floats order as their bits do). RESOLVE runs over
# the cells: what landed becomes metres, the rest 0, and the minimum is emptied for the next frame. Rows run down.
SOURCE: str = r"""
struct Geometry {
    float to_ray[9];        // row-major 3x3: a depth pixel to its ray at unit depth
    float to_uv[12];        // row-major 3x4: a point to the color frame's homogeneous uv
    float homography[9];    // row-major 3x3: uv to straightened uv
    int depth_width, depth_height, out_width, out_height, flip_h;
};

#define EMPTY 0xFFFFFFFFu

extern "C" __global__ void splat(const unsigned short* depth, unsigned int* nearest, Geometry g) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.depth_width || y >= g.depth_height) return;
    float metres = depth[y * g.depth_width + x] / 1000.0f;
    if (metres <= 0.0f) return;
    const float* r = g.to_ray;
    float px = metres * (r[0] * x + r[1] * y + r[2]);
    float py = metres * (r[3] * x + r[4] * y + r[5]);
    float pz = metres * (r[6] * x + r[7] * y + r[8]);
    const float* m = g.to_uv;
    float vx = m[0] * px + m[1] * py + m[2] * pz + m[3];
    float vy = m[4] * px + m[5] * py + m[6] * pz + m[7];
    float vz = m[8] * px + m[9] * py + m[10] * pz + m[11];
    if (vz <= 0.0f) return;
    float fx = vx / vz, fy = vy / vz;
    const float* h = g.homography;
    float cz = h[6] * fx + h[7] * fy + h[8];
    if (cz <= 0.0f) return;
    float u = (h[0] * fx + h[1] * fy + h[2]) / cz;
    float v = (h[3] * fx + h[4] * fy + h[5]) / cz;
    if (g.flip_h) u = 1.0f - u;
    unsigned int level = __float_as_uint(vz * cz);
    int cx = (int)floorf(u * g.out_width - 0.5f);
    int cy = (int)floorf(v * g.out_height - 0.5f);
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int tx = cx + dx, ty = cy + dy;
            if (tx >= 0 && ty >= 0 && tx < g.out_width && ty < g.out_height) {
                atomicMin(&nearest[ty * g.out_width + tx], level);
            }
        }
    }
}

extern "C" __global__ void resolve(unsigned int* nearest, float* aligned, Geometry g) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.out_width || y >= g.out_height) return;
    int i = y * g.out_width + x;
    unsigned int level = nearest[i];
    aligned[i] = level == EMPTY ? 0.0f : __uint_as_float(level);
    nearest[i] = EMPTY;
}
"""


class _Geometry(ctypes.Structure):
    """`SOURCE`'s `Geometry`, passed to the kernels by value."""
    _fields_ = [('to_ray', ctypes.c_float * 9), ('to_uv', ctypes.c_float * 12), ('homography', ctypes.c_float * 9),
                ('depth_width', ctypes.c_int), ('depth_height', ctypes.c_int), ('out_width', ctypes.c_int),
                ('out_height', ctypes.c_int), ('flip_h', ctypes.c_int)]


def _geometry(to_ray: np.ndarray, to_uv: np.ndarray, homography: np.ndarray | None, flip_h: bool, depth_size: tuple[int, int],
              size: tuple[int, int]) -> _Geometry:
    def floats(matrix: np.ndarray, count: int):
        return (ctypes.c_float * count)(*np.asarray(matrix, dtype=np.float32).ravel().tolist())
    return _Geometry(floats(to_ray, 9), floats(to_uv, 12), floats(IDENTITY if homography is None else homography, 9),
                     depth_size[0], depth_size[1], size[0], size[1], int(flip_h))


def _grid(width: int, height: int) -> tuple[int, int]:
    return ((width + BLOCK[0] - 1) // BLOCK[0], (height + BLOCK[1] - 1) // BLOCK[1])


CORNER_OFFSETS: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])     # the 2x2 cells around a point: x, y


def depth_metres(depth: torch.Tensor) -> torch.Tensor:
    """uint16 millimetres (H, W) as float32 metres; 0 stays 0, no depth."""
    return depth.to(torch.float32) / 1000.0


class DepthAligner:
    """`align` keeps what does not change from frame to frame: each depth pixel's ray, made once per lens and
    size, and the matrices on the GPU, made again only when a different array is passed."""

    def __init__(self) -> None:
        self._rays: tuple[int, tuple[int, int], torch.device, torch.Tensor] | None = None     # to_ray's id, size, device: rays (3, N)
        self._matrices: dict[str, tuple[int, torch.Tensor]] = {}
        # `align_staged`'s kernels, the depth frame on the GPU and the cells' minimum, kept empty between frames
        self._kernels: CudaKernels = CudaKernels(SOURCE, ('splat', 'resolve'), 'DepthAligner')
        self._kernels_failed: bool = False
        self._source: torch.Tensor | None = None
        self._nearest: torch.Tensor | None = None
        # without the kernels, `align` as a CUDA graph: what it was captured for, the graph, its input and its output
        self._captured: tuple[tuple, torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor] | None = None

    def _matrix(self, name: str, matrix: np.ndarray, device: torch.device) -> torch.Tensor:
        cached: tuple[int, torch.Tensor] | None = self._matrices.get(name)
        if cached is None or cached[0] != id(matrix) or cached[1].device != device:
            cached = (id(matrix), torch.as_tensor(np.asarray(matrix, dtype=np.float32), device=device))
            self._matrices[name] = cached
        return cached[1]

    def _ray_grid(self, to_ray: np.ndarray, size: tuple[int, int], device: torch.device) -> torch.Tensor:
        if self._rays is None or self._rays[0] != id(to_ray) or self._rays[1] != size or self._rays[2] != device:
            width, height = size
            ys, xs = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                    torch.arange(width, device=device, dtype=torch.float32), indexing='ij')
            pixels: torch.Tensor = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.ones(height * width, device=device)))
            self._rays = (id(to_ray), size, device, self._matrix('to_ray', to_ray, device) @ pixels)
        return self._rays[3]

    def align(self, depth: torch.Tensor, to_ray: np.ndarray, to_uv: np.ndarray, homography: np.ndarray | None, flip_h: bool,
              size: tuple[int, int]) -> torch.Tensor:
        """`depth` (H, W) metres, 0 for none, on the depth camera's view, as (height, width) = `size[::-1]` metres on
        the color frame's view, queued on the current stream."""
        device: torch.device = depth.device
        height, width = depth.shape
        out_w, out_h = size
        metres: torch.Tensor = depth.reshape(-1)
        rays: torch.Tensor = self._ray_grid(to_ray, (width, height), device)
        uv_m: torch.Tensor = self._matrix('to_uv', to_uv, device)
        # The homography's nine numbers go into the kernels as numbers: a tensor of it would be copied to the GPU
        # with each new tilt, a copy that waits for the GPU's queue
        h: list[list[float]] = np.asarray(IDENTITY if homography is None else homography, dtype=np.float64).tolist()

        video: torch.Tensor = torch.addmm(uv_m[:, 3:], uv_m[:, :3], rays * metres)                     # (3, N)
        vz: torch.Tensor = video[2]
        fx: torch.Tensor = video[0] / vz
        fy: torch.Tensor = video[1] / vz
        cz: torch.Tensor = fx * h[2][0] + fy * h[2][1] + h[2][2]
        u: torch.Tensor = (fx * h[0][0] + fy * h[0][1] + h[0][2]) / cz
        v: torch.Tensor = (fx * h[1][0] + fy * h[1][1] + h[1][2]) / cz
        if flip_h:
            u = 1.0 - u
        valid: torch.Tensor = (metres > 0.0) & (vz > 0.0) & (cz > 0.0)
        level: torch.Tensor = torch.where(valid, vz * cz, float('inf'))
        u = torch.where(valid, u, 0.0)     # a point that lands nowhere: a cell index that is a number
        v = torch.where(valid, v, 0.0)

        # The four cells around each point at once, (4, N): a kernel per step, not per step and corner
        corners: torch.Tensor = self._matrix('corners', CORNER_OFFSETS, device)      # (4, 2)
        tx: torch.Tensor = torch.floor(u * out_w - 0.5) + corners[:, :1]
        ty: torch.Tensor = torch.floor(v * out_h - 0.5) + corners[:, 1:]
        inside: torch.Tensor = (tx >= 0) & (ty >= 0) & (tx < out_w) & (ty < out_h)
        indices: torch.Tensor = (ty.clamp(0, out_h - 1) * out_w + tx.clamp(0, out_w - 1)).to(torch.int64)
        values: torch.Tensor = torch.where(inside, level, float('inf'))
        nearest: torch.Tensor = torch.full((out_h * out_w,), float('inf'), device=device, dtype=torch.float32)
        nearest.scatter_reduce_(0, indices.reshape(-1), values.reshape(-1), reduce='amin')
        return torch.where(torch.isinf(nearest), 0.0, nearest).reshape(out_h, out_w)

    def align_staged(self, staged: torch.Tensor, to_ray: np.ndarray, to_uv: np.ndarray, homography: np.ndarray | None, flip_h: bool,
                     size: tuple[int, int], kernels: bool = True) -> torch.Tensor:
        """`staged`, the depth frame's uint16 millimetres (H, W) in a pinned buffer, aligned as `align` does, queued on
        the current stream; the result is a tensor of its own. With `kernels`, as a copy and two CUDA kernels compiled
        at runtime (`SOURCE`), the GL shader's two stages; without, or where they cannot be compiled (logged once), as
        `align` captured in a CUDA graph, its hundred small kernels launched as one."""
        if kernels and not self._kernels_failed:
            try:
                return self._align_kernels(staged, to_ray, to_uv, homography, flip_h, size)
            except (CudaKernelError, OSError):
                logger.exception("DepthAligner: the CUDA kernels failed; the depth is aligned by torch ops from now on")
                self._kernels_failed = True
        return self._align_graph(staged, to_ray, to_uv, homography, flip_h, size)

    def _align_kernels(self, staged: torch.Tensor, to_ray: np.ndarray, to_uv: np.ndarray, homography: np.ndarray | None, flip_h: bool,
                       size: tuple[int, int]) -> torch.Tensor:
        """SPLAT: each depth pixel lands on its 2x2 cells with an atomic minimum; RESOLVE: the cells become metres and
        are emptied for the next frame."""
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        height, width = staged.shape
        out_w, out_h = size
        if self._source is None or self._source.shape != staged.shape:
            self._source = torch.empty(staged.shape, dtype=torch.int16, device='cuda')     # the millimetres' bits, read as unsigned
        if self._nearest is None or self._nearest.numel() != out_w * out_h:
            self._nearest = torch.full((out_h * out_w,), -1, dtype=torch.int32, device='cuda')     # empty: all bits set
        self._source.copy_(staged.view(torch.int16), non_blocking=True)
        aligned: torch.Tensor = torch.empty((out_h, out_w), dtype=torch.float32, device='cuda')
        geometry: _Geometry = _geometry(to_ray, to_uv, homography, flip_h, (width, height), size)
        self._kernels.launch('splat', _grid(width, height), BLOCK, [ctypes.c_void_p(self._source.data_ptr()),
                             ctypes.c_void_p(self._nearest.data_ptr()), geometry], stream)
        self._kernels.launch('resolve', _grid(out_w, out_h), BLOCK, [ctypes.c_void_p(self._nearest.data_ptr()),
                             ctypes.c_void_p(aligned.data_ptr()), geometry], stream)
        return aligned

    def _align_graph(self, staged: torch.Tensor, to_ray: np.ndarray, to_uv: np.ndarray, homography: np.ndarray | None, flip_h: bool,
                     size: tuple[int, int]) -> torch.Tensor:
        """`align` as one CUDA graph, captured once per lens, homography and size, which waits for the GPU once; its
        output is copied out, as the next frame writes it again."""
        key: tuple = (id(to_ray), id(to_uv), id(homography), flip_h, size, tuple(staged.shape), staged.dtype)
        if self._captured is None or self._captured[0] != key:
            self._capture(key, staged, to_ray, to_uv, homography, flip_h, size)
        assert self._captured is not None
        _, graph, source, aligned = self._captured
        source.copy_(staged, non_blocking=True)
        graph.replay()
        return aligned.clone()

    def _capture(self, key: tuple, staged: torch.Tensor, to_ray: np.ndarray, to_uv: np.ndarray, homography: np.ndarray | None,
                 flip_h: bool, size: tuple[int, int]) -> None:
        self._captured = None
        source: torch.Tensor = torch.zeros(staged.shape, dtype=staged.dtype, device='cuda')
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        warm_up: torch.cuda.Stream = torch.cuda.Stream()
        warm_up.wait_stream(stream)
        with torch.cuda.stream(warm_up):            # the rays, the matrices and the kernels made before the capture
            for _ in range(2):
                self.align(depth_metres(source), to_ray, to_uv, homography, flip_h, size)
        stream.wait_stream(warm_up)
        graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode='thread_local'):      # the other threads' GPU work goes on
            aligned: torch.Tensor = self.align(depth_metres(source), to_ray, to_uv, homography, flip_h, size)
        self._captured = (key, graph, source, aligned)
