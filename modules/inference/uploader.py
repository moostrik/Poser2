"""The uploader: the camera's one hand-off. The camera's thread queues its frames here and returns; this thread
puts each on the GPU and hands it on, in arrival order, so no GPU work and no consumer runs on the camera's thread."""

import time
import logging
from collections import deque
from threading import Condition, Lock, Thread

import numpy as np
import torch
import torch.nn.functional as F

from modules.camera import CameraFrame, CameraFrameCallback, FrameType
from modules.utils import PerformanceTimer
from .source import GpuImage, GpuImageCallback, ImageDict, TimestampDict

logger = logging.getLogger(__name__)

STAGING_COUNT: int = 3      # pinned buffers per camera: frames the GPU may be behind before the uploader's thread waits
QUEUED: int = 2             # frames of a kind waiting per camera; beyond that the oldest is dropped and counted


def nv12_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """RGB CHW float32 in [0, 1] from NV12's planes, uint8: Y (H, W) and interleaved UV (H / 2, W).

    BT.601 limited range with the chroma repeated, which is how `getCvFrame` reads an OAK4's frame,
    so the colors match the BGR it replaces.
    """
    # The chroma repeated by a broadcast view: `repeat_interleave` waits for the GPU to size its output
    half: torch.Tensor = uv.reshape(uv.shape[0], -1, 2)
    rows, columns = half.shape[0], half.shape[1]
    chroma: torch.Tensor = half[:, None, :, None, :].expand(rows, 2, columns, 2, 2).reshape(rows * 2, columns * 2, 2).float() - 128.0
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


class Uploader(Thread):
    """Takes the camera's frames on the camera's thread (`submit_frame`) and does all the rest on its own: the color
    goes to the GPU through a pinned staging buffer, is converted, straightened and flipped there, and is handed on
    as a `GpuImage` with the event that says it is written; the depth is handed on as it came. Neither the camera's
    thread nor this one waits for the GPU while the GPU has room. A frame waits for this thread in a queue of
    `QUEUED` per camera and kind; a frame that finds it full drops the oldest of its kind (`dropped`), so a thread
    that cannot keep up loses frames instead of falling behind.

    Maintains a previous-frame buffer per camera for use by downstream processors (e.g. optical flow). `snapshot()`
    returns both with the current images' capture times.

    Input:  BGR uint8 (H, W, 3), grayscale uint8 (H, W) or packed NV12 uint8 (H * 3 / 2, W)
    Output: float16 RGB CHW [0, 1] tensors keyed by cam_id
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name='Uploader')
        self._gpu_images: dict[int, torch.Tensor] = {}
        self._prev_gpu_images: dict[int, torch.Tensor] = {}
        self._timestamps: TimestampDict = {}
        self._lock: Lock = Lock()       # an image and its time are stored and read together
        self._warp_grids: dict[int, tuple[np.ndarray, torch.Tensor]] = {}   # each camera's homography and its grid
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        # Each camera's pinned staging buffers, the events of the copies queued from them, the one used last
        self._staging: dict[int, tuple[list[torch.Tensor], list[torch.cuda.Event | None], int]] = {}
        self._accumulated_upload_ms: float = 0.0
        self._timer: PerformanceTimer = PerformanceTimer(
            name="GPU Image Upload  ", sample_count=200, report_interval=100,
            color="green", omit_init=25,
        )
        self._gpu_image_callbacks: list[GpuImageCallback] = []     # run in registration order, on this thread
        self._depth_callbacks: list[CameraFrameCallback] = []      # run in registration order, on this thread
        self._queue: deque[CameraFrame] = deque()
        self._wake: Condition = Condition()
        self._running: bool = False
        self._dropped: int = 0

    @property
    def dropped(self) -> int:
        """Frames dropped because this thread had not taken the ones before them yet."""
        return self._dropped

    def start(self) -> None:
        self._running = True
        super().start()

    def stop(self) -> None:
        with self._wake:
            self._running = False
            self._wake.notify()
        if self.is_alive():
            self.join(timeout=2.0)

    def submit_frame(self, frame: CameraFrame) -> None:
        """A camera frame callback, on the camera's thread: queues the frame and returns."""
        with self._wake:
            waiting: list[CameraFrame] = [f for f in self._queue if f.cam_id == frame.cam_id and f.kind == frame.kind]
            if len(waiting) >= QUEUED:
                self._queue.remove(waiting[0])
                self._dropped += 1
            self._queue.append(frame)
            self._wake.notify()

    def run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._queue:
                    self._wake.wait()
                if not self._running:
                    return
                frame: CameraFrame = self._queue.popleft()
            try:
                if frame.kind == FrameType.VIDEO:
                    self._upload(frame)
                elif frame.kind == FrameType.DEPTH:
                    for callback in self._depth_callbacks:
                        callback(frame)
            except Exception:
                logger.exception("Uploader: frame of camera %d failed", frame.cam_id)

    def add_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, with its capture time and the event that says it is written, on this thread."""
        if callback not in self._gpu_image_callbacks:
            self._gpu_image_callbacks.append(callback)

    def add_depth_callback(self, callback: CameraFrameCallback) -> None:
        """Every depth frame, as it came, in its place among the images, on this thread."""
        if callback not in self._depth_callbacks:
            self._depth_callbacks.append(callback)

    def _upload(self, frame: CameraFrame) -> None:
        """A color frame to the GPU: converted, straightened and flipped there, stored as the camera's current
        image (the one before it as its previous) with its capture time, and handed on."""
        cam_id: int = frame.cam_id
        image: np.ndarray = frame.image

        start = time.perf_counter()

        staging: torch.Tensor = self._staging_buffer(cam_id, image)
        staging.copy_(torch.from_numpy(image))
        with torch.cuda.stream(self._stream):
            gpu_img = staging.cuda(non_blocking=True)
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
            if frame.flip_h:
                rgb = rgb.flip(-1)
            ready: torch.cuda.Event = torch.cuda.Event()
            ready.record(self._stream)
        _, events, index = self._staging[cam_id]
        events[index] = ready           # the buffer is free again once the copy queued from it has run
        with self._lock:
            if cam_id in self._gpu_images:
                self._prev_gpu_images[cam_id] = self._gpu_images[cam_id]
            self._gpu_images[cam_id] = rgb
            self._timestamps[cam_id] = frame.timestamp

        self._accumulated_upload_ms += (time.perf_counter() - start) * 1000.0

        gpu_image: GpuImage = GpuImage(cam_id, frame.timestamp, rgb, ready)
        for callback in self._gpu_image_callbacks:
            try:
                callback(gpu_image)
            except Exception:
                logger.exception("Error in GPU image callback")

    def _staging_buffer(self, cam_id: int, image: np.ndarray) -> torch.Tensor:
        """The camera's next pinned buffer for `image`, (re)made at its shape. From pinned memory the copy to the
        GPU is queued and this thread goes on; from the frame's own pageable memory it would wait until the
        GPU had taken it. A buffer is reused once the copy queued from it has run, its event passed; only a GPU
        `STAGING_COUNT` frames behind makes the thread wait."""
        buffers, events, index = self._staging.get(cam_id, ([], [], -1))
        if not buffers or buffers[0].shape != image.shape or buffers[0].dtype != torch.from_numpy(image[:0]).dtype:
            buffers = [torch.empty(image.shape, dtype=torch.from_numpy(image[:0]).dtype, pin_memory=True) for _ in range(STAGING_COUNT)]
            events = [None] * STAGING_COUNT
        index = (index + 1) % STAGING_COUNT
        event: torch.cuda.Event | None = events[index]
        if event is not None:
            event.synchronize()
        self._staging[cam_id] = (buffers, events, index)
        return buffers[index]

    def _straightened(self, cam_id: int, rgb: torch.Tensor, homography: np.ndarray) -> torch.Tensor:
        """`rgb` drawn through the homography. The grid is made once per homography."""
        cached: tuple[np.ndarray, torch.Tensor] | None = self._warp_grids.get(cam_id)
        if cached is None or cached[0] is not homography:
            matrix: torch.Tensor = torch.from_numpy(homography).to(rgb.device, torch.float32)
            cached = (homography, warp_grid(matrix, rgb.shape[1], rgb.shape[2]))
            self._warp_grids[cam_id] = cached
        return F.grid_sample(rgb.unsqueeze(0).float(), cached[1], align_corners=False)[0].to(dtype=torch.float16)

    def snapshot(self) -> tuple[ImageDict, ImageDict, TimestampDict, torch.cuda.Event]:
        """The current and previous GPU images, the current images' capture times, and an event recorded on the
        upload stream after them: a reader on another stream waits on it (`stream.wait_event`) before it reads
        them, and keeps them from being reused (`record_stream`) until its own work on them has run. Nothing
        here waits for the GPU.

        Returns:
            (current, previous, timestamps, ready), the first three keyed by cam_id.
        """
        with self._lock:
            images_snapshot: ImageDict = dict(self._gpu_images)
            prev_snapshot: ImageDict = dict(self._prev_gpu_images)
            timestamps_snapshot: TimestampDict = dict(self._timestamps)
        ready: torch.cuda.Event = torch.cuda.Event()
        ready.record(self._stream)

        self._timer.add_time(self._accumulated_upload_ms, report=False)
        self._accumulated_upload_ms = 0.0

        return images_snapshot, prev_snapshot, timestamps_snapshot, ready

    def reset(self) -> None:
        """Clear all stored GPU images."""
        with self._lock:
            self._gpu_images.clear()
            self._prev_gpu_images.clear()
            self._timestamps.clear()
        self._warp_grids.clear()
