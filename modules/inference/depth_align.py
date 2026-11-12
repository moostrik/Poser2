"""The camera's depth moved onto the color frame's view, in torch on the GPU: what `DepthAlign` does in GL.

Every depth pixel goes to its point in space (`to_ray`, at its distance), into the color frame (`to_uv`), through the
straightening homography and the mirror, and lands on the 2x2 grid cells around it; where two land on one cell the
nearest is kept, and a cell nothing lands on is 0. The value is the depth along the level camera's axis, the point's
depth times the homography's third row (`video.z * corrected.z`), as a level camera would have measured it. Rows run
down, as in the frame. `to_ray` and `to_uv` are `modules.camera.depth_projection`'s, the size `aligned_depth_size`.
`scripts/depth_align_check.py` compares it with the GL shader on recorded frames.
"""

import numpy as np
import torch

IDENTITY: np.ndarray = np.eye(3)
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
        # `align_staged`'s graph: what it was captured for, the graph, its input and its output
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
                     size: tuple[int, int]) -> torch.Tensor:
        """`staged`, the depth frame's uint16 millimetres (H, W) in a pinned buffer, aligned as `align` does, queued on
        the current stream as a copy, one CUDA graph and a copy out: about a hundred small kernels launched as one, as
        each launch waits while the GPU's queue is full. The graph is captured once per lens, homography and size,
        which waits for the GPU once; the result is a tensor of its own, the graph's output being written again with
        the next frame."""
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
