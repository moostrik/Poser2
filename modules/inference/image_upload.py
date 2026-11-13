"""ImageUpload: camera color frames onto the GPU, the work both uploaders share.

One frame, or a stack of frames of one size and layout, goes through a pinned staging buffer to the GPU, is
converted there (NV12, BGR or mono) to float16 RGB CHW in [0, 1], straightened by its camera's homography and
mirrored when its frame says so, and comes out with one event recorded after it on the caller's stream. A stack is
one copy and one round of kernels: the conversion and the straightening run over the whole stack (the cameras'
grids stacked), and only the mirror runs per mirrored camera. Neither the caller's thread nor this waits for the GPU
while it has room: a staging buffer is reused once the copy queued from it has run, and only a GPU `STAGING_COUNT`
uploads behind makes the caller wait. Not thread-safe: one uploader's thread uses it."""

import numpy as np
import torch
import torch.nn.functional as F

from modules.camera import CameraFrame

STAGING_COUNT: int = 3      # pinned buffers per stack of cameras: uploads the GPU may be behind before the caller waits


def nv12_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """RGB float32 in [0, 1], channels before the rows, from NV12's planes, uint8: Y (…, H, W) and interleaved UV
    (…, H / 2, W), any leading dimensions (a stack) kept: (…, 3, H, W).

    BT.601 limited range with the chroma repeated, which is how `getCvFrame` reads an OAK4's frame,
    so the colors match the BGR it replaces.
    """
    # The chroma repeated by a broadcast view: `repeat_interleave` waits for the GPU to size its output
    lead: tuple[int, ...] = tuple(uv.shape[:-2])
    half: torch.Tensor = uv.reshape(*lead, uv.shape[-2], -1, 2)
    rows, columns = half.shape[-3], half.shape[-2]
    chroma: torch.Tensor = (half[..., :, None, :, None, :].expand(*lead, rows, 2, columns, 2, 2)
                            .reshape(*lead, rows * 2, columns * 2, 2).float() - 128.0)
    u, v = chroma[..., 0], chroma[..., 1]
    luma: torch.Tensor = (y.float() - 16.0) * 1.164383
    rgb: torch.Tensor = torch.stack((luma + 1.596027 * v, luma - 0.391762 * u - 0.812968 * v, luma + 2.017232 * u), dim=-3)
    return rgb.clamp_(0.0, 255.0).div_(255.0)


def warp_grid(homography: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The `grid_sample` grid (1, height, width, 2) that draws a frame through `homography`, uv to
    uv, on the homography's device: each output pixel looks up where it came from. A pixel from
    behind the camera gets a place outside the frame, which samples to nothing."""
    inverse: torch.Tensor = torch.linalg.inv(homography)
    device: torch.device = homography.device
    u: torch.Tensor = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    v: torch.Tensor = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    source: torch.Tensor = inverse @ torch.stack((grid_u.ravel(), grid_v.ravel(), torch.ones_like(grid_u).ravel()))
    uv: torch.Tensor = (source[:2] / source[2]).T.reshape(height, width, 2) * 2.0 - 1.0
    uv[(source[2] <= 0.0).reshape(height, width)] = 2.0
    return uv.unsqueeze(0)


class ImageUpload:
    """See the module docstring."""

    def __init__(self) -> None:
        # Each stack's pinned staging buffers (by its cameras), the events of the copies queued from them, the one used last
        self._staging: dict[tuple[int, ...], tuple[list[torch.Tensor], list[torch.cuda.Event | None], int]] = {}
        self._warp_grids: dict[int, tuple[np.ndarray, torch.Tensor]] = {}      # each camera's homography and its grid

    def upload(self, frames: list[CameraFrame], stream: torch.cuda.Stream) -> tuple[list[torch.Tensor], torch.cuda.Event]:
        """`frames`, color frames of one size and layout, onto the GPU on `stream`: each one's image, float16 RGB CHW
        in [0, 1], straightened and mirrored, in the frames' order, and the event recorded after them all."""
        first: np.ndarray = frames[0].image
        cam_ids: tuple[int, ...] = tuple(frame.cam_id for frame in frames)
        staging: torch.Tensor = self._staging_buffer(cam_ids, (len(frames), *first.shape), torch.from_numpy(first[:0]).dtype)
        for index, frame in enumerate(frames):
            staging[index].copy_(torch.from_numpy(frame.image))
        with torch.cuda.stream(stream):
            gpu: torch.Tensor = staging.cuda(non_blocking=True)
            rgb: torch.Tensor = self._converted(gpu, frames)
            straightened: list[int] = [index for index, frame in enumerate(frames) if frame.homography is not None]
            if len(straightened) == len(frames):        # every camera (the rig's case): the whole stack, no gather
                rgb = F.grid_sample(rgb.float(), self._grids(frames, rgb.shape[2], rgb.shape[3]),
                                    align_corners=False).to(dtype=torch.float16)
            elif straightened:
                selected: list[CameraFrame] = [frames[i] for i in straightened]
                warped: torch.Tensor = F.grid_sample(rgb[straightened].float(), self._grids(selected, rgb.shape[2], rgb.shape[3]),
                                                     align_corners=False).to(dtype=torch.float16)
                rgb[straightened] = warped
            images: list[torch.Tensor] = [rgb[index].flip(-1) if frame.flip_h else rgb[index] for index, frame in enumerate(frames)]
            ready: torch.cuda.Event = torch.cuda.Event()
            ready.record(stream)
        _, events, slot = self._staging[cam_ids]
        events[slot] = ready            # the buffer is free again once the copy queued from it has run
        return images, ready

    def _converted(self, gpu: torch.Tensor, frames: list[CameraFrame]) -> torch.Tensor:
        """The stack (N, …) as uploaded, to float16 RGB (N, 3, H, W) in [0, 1]."""
        if frames[0].nv12:
            rows: int = gpu.shape[1] * 2 // 3
            return nv12_to_rgb(gpu[:, :rows], gpu[:, rows:]).to(dtype=torch.float16)
        if gpu.ndim == 3:       # mono (N, H, W)
            return gpu.unsqueeze(1).to(dtype=torch.float16).mul_(1.0 / 255.0).expand(-1, 3, -1, -1).contiguous()
        return gpu.permute(0, 3, 1, 2).flip(1).to(dtype=torch.float16).mul_(1.0 / 255.0)       # BGR HWC to RGB CHW

    def _grids(self, frames: list[CameraFrame], height: int, width: int) -> torch.Tensor:
        """The frames' straightening grids (N, H, W, 2); one frame's is its cached grid itself, not a copy."""
        grids: list[torch.Tensor] = [self._grid(frame, height, width) for frame in frames]
        return grids[0] if len(grids) == 1 else torch.cat(grids)

    def _grid(self, frame: CameraFrame, height: int, width: int) -> torch.Tensor:
        """The frame's camera's straightening grid (1, H, W, 2), made once per homography."""
        assert frame.homography is not None
        cached: tuple[np.ndarray, torch.Tensor] | None = self._warp_grids.get(frame.cam_id)
        if cached is None or cached[0] is not frame.homography:
            matrix: torch.Tensor = torch.from_numpy(frame.homography).to('cuda', torch.float32)
            cached = (frame.homography, warp_grid(matrix, height, width))
            self._warp_grids[frame.cam_id] = cached
        return cached[1]

    def _staging_buffer(self, key: tuple[int, ...], shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """The stack's next pinned buffer, (re)made at its shape. From pinned memory the copy to the GPU is queued and
        the caller goes on; from the frame's own pageable memory it would wait until the GPU had taken it."""
        buffers, events, slot = self._staging.get(key, ([], [], -1))
        if not buffers or tuple(buffers[0].shape) != shape or buffers[0].dtype != dtype:
            buffers = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(STAGING_COUNT)]
            events = [None] * STAGING_COUNT
        slot = (slot + 1) % STAGING_COUNT
        event: torch.cuda.Event | None = events[slot]
        if event is not None:
            event.synchronize()
        self._staging[key] = (buffers, events, slot)
        return buffers[slot]

    def reset(self) -> None:
        """Forget the cameras' grids: a new homography makes its own."""
        self._warp_grids.clear()
