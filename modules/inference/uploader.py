import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from modules.camera import CameraFrame, FrameType
from modules.utils import PerformanceTimer
from .source import ImageDict

logger = logging.getLogger(__name__)


def nv12_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """RGB CHW float32 in [0, 1] from NV12's planes, uint8: Y (H, W) and interleaved UV (H / 2, W).

    BT.601 limited range with the chroma repeated, which is how `getCvFrame` reads an OAK4's frame,
    so the colors match the BGR it replaces.
    """
    chroma: torch.Tensor = uv.reshape(uv.shape[0], -1, 2).repeat_interleave(2, 0).repeat_interleave(2, 1).float() - 128.0
    u, v = chroma[..., 0], chroma[..., 1]
    luma: torch.Tensor = (y.float() - 16.0) * 1.164383
    rgb: torch.Tensor = torch.stack((luma + 1.596027 * v, luma - 0.391762 * u - 0.812968 * v, luma + 2.017232 * u))
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


class Uploader:
    """Uploads CPU numpy frames to GPU float16 CHW tensors.

    Maintains a previous-frame buffer per camera for use by downstream
    processors (e.g. optical flow). Call snapshot() once per tick to
    synchronize the CUDA stream and retrieve both buffers.

    Input:  BGR uint8 (H, W, 3), grayscale uint8 (H, W) or packed NV12 uint8 (H * 3 / 2, W)
    Output: float16 RGB CHW [0, 1] tensors keyed by cam_id
    """

    def __init__(self) -> None:
        self._gpu_images: dict[int, torch.Tensor] = {}
        self._prev_gpu_images: dict[int, torch.Tensor] = {}
        self._warp_grids: dict[int, tuple[np.ndarray, torch.Tensor]] = {}   # each camera's homography and its grid
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._accumulated_upload_ms: float = 0.0
        self._timer: PerformanceTimer = PerformanceTimer(
            name="GPU Image Upload  ", sample_count=200, report_interval=100,
            color="green", omit_init=25,
        )

    def set_image(self, frame: CameraFrame) -> None:
        """Upload a camera frame to GPU. Only VIDEO frames are stored.

        Shifts current to previous before uploading the new frame. The image is BGR uint8
        (H, W, 3), grayscale uint8 (H, W) or packed NV12, converted, straightened and flipped here
        on the GPU so the camera's thread does none of it.
        """
        if frame.kind != FrameType.VIDEO:
            return
        cam_id: int = frame.cam_id
        image: np.ndarray = frame.image

        start = time.perf_counter()

        with torch.cuda.stream(self._stream):
            if cam_id in self._gpu_images:
                self._prev_gpu_images[cam_id] = self._gpu_images[cam_id]

            gpu_img = torch.from_numpy(image).cuda(non_blocking=True)
            if frame.nv12:
                rows: int = image.shape[0] * 2 // 3
                rgb: torch.Tensor = nv12_to_rgb(gpu_img[:rows], gpu_img[rows:]).to(dtype=torch.float16)
            elif image.ndim == 2:
                # Grayscale (H, W) -> normalize -> expand to (3, H, W)
                rgb = (
                    gpu_img.unsqueeze(0).to(dtype=torch.float16).mul_(1.0 / 255.0)
                    .expand(3, -1, -1).contiguous()
                )
            else:
                # Color HWC BGR -> CHW RGB, normalize to [0, 1]
                rgb = (
                    gpu_img.permute(2, 0, 1).flip(0)
                    .to(dtype=torch.float16).mul_(1.0 / 255.0)
                )
            if frame.homography is not None:
                rgb = self._straightened(cam_id, rgb, frame.homography)
            self._gpu_images[cam_id] = rgb.flip(-1) if frame.flip_h else rgb

        self._accumulated_upload_ms += (time.perf_counter() - start) * 1000.0

    def _straightened(self, cam_id: int, rgb: torch.Tensor, homography: np.ndarray) -> torch.Tensor:
        """`rgb` drawn through the homography. The grid is made once per homography."""
        cached: tuple[np.ndarray, torch.Tensor] | None = self._warp_grids.get(cam_id)
        if cached is None or cached[0] is not homography:
            matrix: torch.Tensor = torch.from_numpy(homography).to(rgb.device, torch.float32)
            cached = (homography, warp_grid(matrix, rgb.shape[1], rgb.shape[2]))
            self._warp_grids[cam_id] = cached
        return F.grid_sample(rgb.unsqueeze(0).float(), cached[1], align_corners=False)[0].to(dtype=torch.float16)

    def snapshot(self) -> tuple[ImageDict, ImageDict]:
        """Synchronize the upload stream and return current and previous GPU images.

        Returns:
            (current, previous) both as ImageDict keyed by cam_id.
        """
        # Copy references BEFORE synchronizing to avoid a race with concurrent set_image
        # calls. A set_image arriving after the copy updates _gpu_images but those new
        # tensors are not included here; synchronize() then guarantees the captured
        # tensors are fully uploaded and safe to read.
        images_snapshot: ImageDict = dict(self._gpu_images)
        prev_snapshot: ImageDict = dict(self._prev_gpu_images)

        self._stream.synchronize()

        self._timer.add_time(self._accumulated_upload_ms, report=False)
        self._accumulated_upload_ms = 0.0

        return images_snapshot, prev_snapshot

    def reset(self) -> None:
        """Clear all stored GPU images."""
        self._gpu_images.clear()
        self._prev_gpu_images.clear()
        self._warp_grids.clear()
