import time
import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from modules.oak.camera.definitions import FrameType
from modules.utils.PerformanceTimer import PerformanceTimer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullImage:
    """GPU-resident full camera frame.

    Format: float16 RGB CHW [0, 1].

    Attributes:
        cam_id: Camera identifier
        image:  Full source frame on GPU (3, H, W) float16 RGB CHW [0, 1]
    """
    cam_id: int
    image: torch.Tensor


FullImageDict: TypeAlias = dict[int, FullImage]


class ImageUploader:
    """Uploads CPU numpy frames to GPU float16 CHW tensors.

    Maintains a previous-frame buffer per camera for use by downstream
    processors (e.g. optical flow). Call snapshot() once per tick to
    synchronize the CUDA stream and retrieve both buffers.

    Input:  BGR uint8 (H, W, 3) or grayscale uint8 (H, W)
    Output: float16 RGB CHW [0, 1] tensors keyed by cam_id
    """

    def __init__(self) -> None:
        self._gpu_images: dict[int, torch.Tensor] = {}
        self._prev_gpu_images: dict[int, torch.Tensor] = {}
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._accumulated_upload_ms: float = 0.0
        self._timer: PerformanceTimer = PerformanceTimer(
            name="GPU Image Upload  ", sample_count=200, report_interval=100,
            color="green", omit_init=25,
        )

    def set_image(self, cam_id: int, frame_type: FrameType, image: np.ndarray) -> None:
        """Upload a camera frame to GPU. Only VIDEO frames are stored.

        Shifts current to previous before uploading the new frame.

        Args:
            cam_id:     Camera identifier
            frame_type: Type of frame (only VIDEO is processed)
            image:      BGR uint8 (H, W, 3) or grayscale uint8 (H, W)
        """
        if frame_type != FrameType.VIDEO:
            return

        start = time.perf_counter()

        with torch.cuda.stream(self._stream):
            if cam_id in self._gpu_images:
                self._prev_gpu_images[cam_id] = self._gpu_images[cam_id]

            gpu_img = torch.from_numpy(image).cuda(non_blocking=True)
            if image.ndim == 2:
                # Grayscale (H, W) -> normalize -> expand to (3, H, W)
                self._gpu_images[cam_id] = (
                    gpu_img.unsqueeze(0).to(dtype=torch.float16).mul_(1.0 / 255.0)
                    .expand(3, -1, -1).contiguous()
                )
            else:
                # Color HWC BGR -> CHW RGB, normalize to [0, 1]
                self._gpu_images[cam_id] = (
                    gpu_img.permute(2, 0, 1).flip(0)
                    .to(dtype=torch.float16).mul_(1.0 / 255.0)
                )

        self._accumulated_upload_ms += (time.perf_counter() - start) * 1000.0

    def snapshot(self) -> tuple[FullImageDict, dict[int, torch.Tensor]]:
        """Synchronize the upload stream and return current and previous GPU images.

        Returns:
            (current, previous) where current is a FullImageDict keyed by cam_id
            and previous is a dict[int, torch.Tensor] keyed by cam_id.
        """
        self._stream.synchronize()

        self._timer.add_time(self._accumulated_upload_ms)
        self._accumulated_upload_ms = 0.0

        current: FullImageDict = {
            cam_id: FullImage(cam_id=cam_id, image=tensor)
            for cam_id, tensor in self._gpu_images.items()
        }
        return current, dict(self._prev_gpu_images)

    def reset(self) -> None:
        """Clear all stored GPU images."""
        self._gpu_images.clear()
        self._prev_gpu_images.clear()
