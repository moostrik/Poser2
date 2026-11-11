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
CORNERS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def depth_metres(depth: torch.Tensor) -> torch.Tensor:
    """uint16 millimetres (H, W) as float32 metres; 0 stays 0, no depth."""
    return depth.to(torch.float32) / 1000.0


class DepthAligner:
    """`align` keeps what does not change from frame to frame: each depth pixel's ray, made once per lens and
    size, and the matrices on the GPU, made again only when a different array is passed."""

    def __init__(self) -> None:
        self._rays: tuple[int, tuple[int, int], torch.device, torch.Tensor] | None = None     # to_ray's id, size, device: rays (3, N)
        self._matrices: dict[str, tuple[int, torch.Tensor]] = {}

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
        h_m: torch.Tensor = self._matrix('homography', IDENTITY if homography is None else homography, device)

        video: torch.Tensor = torch.addmm(uv_m[:, 3:], uv_m[:, :3], rays * metres)                     # (3, N)
        vz: torch.Tensor = video[2]
        flat: torch.Tensor = torch.stack((video[0] / vz, video[1] / vz, torch.ones_like(vz)))
        corrected: torch.Tensor = h_m @ flat
        cz: torch.Tensor = corrected[2]
        u: torch.Tensor = corrected[0] / cz
        v: torch.Tensor = corrected[1] / cz
        if flip_h:
            u = 1.0 - u
        valid: torch.Tensor = (metres > 0.0) & (vz > 0.0) & (cz > 0.0)
        level: torch.Tensor = torch.where(valid, vz * cz, torch.full_like(vz, float('inf')))
        u = torch.where(valid, u, torch.zeros_like(u))     # a point that lands nowhere: a cell index that is a number
        v = torch.where(valid, v, torch.zeros_like(v))

        corner_x: torch.Tensor = torch.floor(u * out_w - 0.5)
        corner_y: torch.Tensor = torch.floor(v * out_h - 0.5)
        indices: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for dx, dy in CORNERS:
            tx: torch.Tensor = corner_x + dx
            ty: torch.Tensor = corner_y + dy
            inside: torch.Tensor = (tx >= 0) & (ty >= 0) & (tx < out_w) & (ty < out_h)
            indices.append((ty.clamp(0, out_h - 1) * out_w + tx.clamp(0, out_w - 1)).to(torch.int64))
            values.append(torch.where(inside, level, torch.full_like(level, float('inf'))))
        nearest: torch.Tensor = torch.full((out_h * out_w,), float('inf'), device=device, dtype=torch.float32)
        nearest.scatter_reduce_(0, torch.cat(indices), torch.cat(values), reduce='amin')
        return torch.where(torch.isinf(nearest), torch.zeros_like(nearest), nearest).reshape(out_h, out_w)
