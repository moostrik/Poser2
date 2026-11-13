"""The serial uploader: the camera's one hand-off in deep_flow, each frame onto the GPU on its own, the moment it
comes. The camera's thread queues its frames here and returns; this thread puts each on the GPU and hands it on, in
arrival order, so a color frame goes up at once to start the pose and no GPU work and no consumer runs on the
camera's thread."""

import time
import logging
from collections import deque
from threading import Condition, Lock, Thread

import numpy as np
import torch

from modules.camera import CAPTURE_TOLERANCE, CameraFrame, CameraTracklets, DepthAlignment, FrameType, TrackletCallback, \
    aligned_depth_size, depth_projection, straightened_tracklets
from modules.settings import BaseSettings, Field
from modules.utils import PerformanceTimer
from .depth_align import DepthAligner
from .image_upload import STAGING_COUNT, ImageUpload
from .source import GpuDepth, GpuDepthCallback, GpuImage, GpuImageCallback, ImageSnapshot, ImageSnapshotCallback

logger = logging.getLogger(__name__)

QUEUED: int = 2             # frames of a kind waiting per camera; beyond that the oldest is dropped and counted
NOTED: int = 16             # color frames per camera whose straightening is kept for their tracklets, which come about 2 frames later


class SerialUploaderSettings(BaseSettings):
    """How the serial uploader does its GPU work."""
    depth_kernels: Field[bool] = Field(True, description="Align the depth with CUDA kernels compiled at runtime; off, with torch ops as a CUDA graph")


class SerialUploader(Thread):
    """Takes the camera's frames on the camera's thread (`submit_frame`) and does all the rest on its own: the color
    goes to the GPU (`ImageUpload`, one frame), converted, straightened and flipped there, and is handed on as a
    `GpuImage` with the event that says it is written, and as an `ImageSnapshot` of one (the image, its camera's
    previous one, the event) to the snapshot callbacks; the depth goes the same way, is turned into metres and
    moved onto the color's view (`DepthAligner`, by the camera's calibration, straightened and mirrored as the
    color), and is handed on as a `GpuDepth`. Neither the camera's
    thread nor this one waits for the GPU while the GPU has room. A frame waits for this thread in a queue of
    `QUEUED` per camera and kind; a frame that finds it full drops the oldest of its kind (`dropped`), so a thread
    that cannot keep up loses frames instead of falling behind.

    The camera's tracklets come here too (`submit_tracklets`), as its model found them in the frame as delivered: this
    thread straightens and mirrors them with the frame they were found in, whose homography and mirror `submit_frame`
    notes by capture time (`NOTED` frames, a dropped one included), and hands them on, before any frame that waits.
    Tracklets whose frame is no longer noted take the newest frame's; tracklets `straightened` already pass as they are.

    Each camera's previous color image is kept for the next snapshot, for the optical flow's previous crop; this
    thread alone keeps it.

    Input:  BGR uint8 (H, W, 3), grayscale uint8 (H, W) or packed NV12 uint8 (H * 3 / 2, W)
    Output: float16 RGB CHW [0, 1] tensors keyed by cam_id
    """

    def __init__(self, settings: SerialUploaderSettings) -> None:
        super().__init__(daemon=True, name='SerialUploader')
        self.settings: SerialUploaderSettings = settings
        self._image_upload: ImageUpload = ImageUpload()
        self._previous: dict[int, torch.Tensor] = {}       # each camera's previous color image, on this thread
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        # Each camera's depth staging buffers, the events of the copies queued from them, the one used last
        self._staging: dict[tuple[int, FrameType], tuple[list[torch.Tensor], list[torch.cuda.Event | None], int]] = {}
        self._aligners: dict[int, DepthAligner] = {}
        # Each camera's depth alignment, the depth's size, and what they make: to_ray, to_uv, the aligned size
        self._projections: dict[int, tuple[DepthAlignment | None, tuple[int, int], np.ndarray, np.ndarray, tuple[int, int]]] = {}
        self._timer: PerformanceTimer = PerformanceTimer(
            name="GPU Image Upload  ", sample_count=200, report_interval=100,
            color="green", omit_init=25,
        )
        self._callback_lock: Lock = Lock()                          # the four lists below
        self._gpu_image_callbacks: list[GpuImageCallback] = []     # run in registration order, on this thread
        self._snapshot_callbacks: list[ImageSnapshotCallback] = [] # run in registration order, on this thread
        self._gpu_depth_callbacks: list[GpuDepthCallback] = []     # run in registration order, on this thread
        self._tracklet_callbacks: list[TrackletCallback] = []      # run in registration order, on this thread
        self._queue: deque[tuple[CameraFrame, float]] = deque()            # each with the moment it was queued
        self._tracklets: deque[tuple[CameraTracklets, float]] = deque()    # never dropped: a tracker needs every message
        # How long a frame and a tracklet message wait in their queue for this thread (ms), as timer samples
        self._frame_wait: PerformanceTimer = PerformanceTimer(name="Wait upload")
        self._tracklet_wait: PerformanceTimer = PerformanceTimer(name="Wait upload tracklets")
        self._straightening: dict[int, deque[tuple[float, np.ndarray | None, bool]]] = {}  # per camera: (capture, homography, flip_h)
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
            if frame.kind == FrameType.VIDEO:
                noted = self._straightening.setdefault(frame.cam_id, deque(maxlen=NOTED))
                noted.append((frame.timestamp, frame.homography, frame.flip_h))
            waiting: list[tuple[CameraFrame, float]] = [item for item in self._queue
                                                        if item[0].cam_id == frame.cam_id and item[0].kind == frame.kind]
            if len(waiting) >= QUEUED:
                self._queue.remove(waiting[0])
                self._dropped += 1
            self._queue.append((frame, time.perf_counter()))
            self._wake.notify()

    def submit_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        """A camera's tracklet callback, on its thread: queues the message and returns."""
        with self._wake:
            self._tracklets.append((cam_tracklets, time.perf_counter()))
            self._wake.notify()

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        """The camera's tracklets, straightened and mirrored as the images are, on this thread."""
        with self._callback_lock:
            if callback not in self._tracklet_callbacks:
                self._tracklet_callbacks.append(callback)

    def run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._queue and not self._tracklets:
                    self._wake.wait()
                if not self._running:
                    return
                if self._tracklets:
                    cam_tracklets, queued = self._tracklets.popleft()
                    straightening: tuple[np.ndarray | None, bool] = self._straightening_at(cam_tracklets)
                    frame: CameraFrame | None = None
                    wait: PerformanceTimer = self._tracklet_wait
                else:
                    frame, queued = self._queue.popleft()
                    wait = self._frame_wait
            wait.add_time((time.perf_counter() - queued) * 1000.0, report=False)
            if frame is None:
                self._hand_on_tracklets(cam_tracklets, *straightening)
                continue
            try:
                if frame.kind == FrameType.VIDEO:
                    self._upload(frame)
                elif frame.kind == FrameType.DEPTH:
                    self._upload_depth(frame)
            except Exception:
                logger.exception("SerialUploader: frame of camera %d failed", frame.cam_id)

    def _straightening_at(self, cam_tracklets: CameraTracklets) -> tuple[np.ndarray | None, bool]:
        """The homography and mirror of the frame the tracklets were found in, the newest frame's when it is no
        longer noted; none for tracklets straightened already or a camera without frames yet. Under the lock."""
        noted = self._straightening.get(cam_tracklets.cam_id)
        if cam_tracklets.straightened or not noted:
            return None, False
        for timestamp, homography, flip_h in reversed(noted):
            if abs(timestamp - cam_tracklets.timestamp) <= CAPTURE_TOLERANCE:
                return homography, flip_h
        return noted[-1][1], noted[-1][2]

    def _hand_on_tracklets(self, cam_tracklets: CameraTracklets, homography: np.ndarray | None, flip_h: bool) -> None:
        if not cam_tracklets.straightened:
            cam_tracklets = straightened_tracklets(cam_tracklets, homography, flip_h)
        with self._callback_lock:
            tracklet_callbacks: list[TrackletCallback] = list(self._tracklet_callbacks)
        for callback in tracklet_callbacks:
            try:
                callback(cam_tracklets)
            except Exception:
                logger.exception("Error in tracklet callback")

    def add_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, with its capture time and the event that says it is written, on this thread."""
        with self._callback_lock:
            if callback not in self._gpu_image_callbacks:
                self._gpu_image_callbacks.append(callback)

    def add_snapshot_callback(self, callback: ImageSnapshotCallback) -> None:
        """Every uploaded color image as an `ImageSnapshot` of one: the image, its camera's previous one, its capture
        time and the event that says it is written, on this thread, after the image callbacks."""
        with self._callback_lock:
            if callback not in self._snapshot_callbacks:
                self._snapshot_callbacks.append(callback)

    def add_gpu_depth_callback(self, callback: GpuDepthCallback) -> None:
        """Every uploaded depth frame, in its place among the images, with the event that says it is written, on this thread."""
        with self._callback_lock:
            if callback not in self._gpu_depth_callbacks:
                self._gpu_depth_callbacks.append(callback)

    def _upload(self, frame: CameraFrame) -> None:
        """A color frame to the GPU (`ImageUpload`, one frame), handed on as a `GpuImage` and as an `ImageSnapshot`
        of one with its camera's previous image."""
        cam_id: int = frame.cam_id
        start = time.perf_counter()
        images, ready = self._image_upload.upload([frame], self._stream)
        rgb: torch.Tensor = images[0]
        previous: torch.Tensor | None = self._previous.get(cam_id)
        self._previous[cam_id] = rgb
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=False)

        gpu_image: GpuImage = GpuImage(cam_id, frame.timestamp, rgb, ready)
        snapshot: ImageSnapshot = ({cam_id: rgb}, {cam_id: previous} if previous is not None else {}, {cam_id: frame.timestamp}, ready)
        with self._callback_lock:
            image_callbacks: list[GpuImageCallback] = list(self._gpu_image_callbacks)
            snapshot_callbacks: list[ImageSnapshotCallback] = list(self._snapshot_callbacks)
        for callback in image_callbacks:
            try:
                callback(gpu_image)
            except Exception:
                logger.exception("Error in GPU image callback")
        for snapshot_callback in snapshot_callbacks:
            try:
                snapshot_callback(snapshot)
            except Exception:
                logger.exception("Error in snapshot callback")

    def _upload_depth(self, frame: CameraFrame) -> None:
        """A depth frame to the GPU: metres, moved onto the color's view at its own grid, and handed on."""
        cam_id: int = frame.cam_id
        image: np.ndarray = frame.image
        size: tuple[int, int] = (image.shape[1], image.shape[0])
        projection = self._projections.get(cam_id)
        if projection is None or projection[0] is not frame.alignment or projection[1] != size:
            projection = (frame.alignment, size, *depth_projection(frame.alignment, size), aligned_depth_size(size, frame.alignment))
            self._projections[cam_id] = projection
        _, _, to_ray, to_uv, aligned_size = projection
        aligner: DepthAligner = self._aligners.setdefault(cam_id, DepthAligner())

        staging: torch.Tensor = self._staging_buffer((cam_id, FrameType.DEPTH), image)
        staging.copy_(torch.from_numpy(image))
        with torch.cuda.stream(self._stream):
            depth: torch.Tensor = aligner.align_staged(staging, to_ray, to_uv, frame.homography, frame.flip_h, aligned_size,
                                                       kernels=self.settings.depth_kernels)
            ready: torch.cuda.Event = torch.cuda.Event()
            ready.record(self._stream)
        _, events, index = self._staging[(cam_id, FrameType.DEPTH)]
        events[index] = ready

        gpu_depth: GpuDepth = GpuDepth(cam_id, frame.timestamp, depth, ready)
        with self._callback_lock:
            depth_callbacks: list[GpuDepthCallback] = list(self._gpu_depth_callbacks)
        for callback in depth_callbacks:
            try:
                callback(gpu_depth)
            except Exception:
                logger.exception("Error in GPU depth callback")

    def _staging_buffer(self, key: tuple[int, FrameType], image: np.ndarray) -> torch.Tensor:
        """The camera's next pinned buffer of the frame's kind for `image`, (re)made at its shape. From pinned memory the copy to the
        GPU is queued and this thread goes on; from the frame's own pageable memory it would wait until the
        GPU had taken it. A buffer is reused once the copy queued from it has run, its event passed; only a GPU
        `STAGING_COUNT` frames behind makes the thread wait."""
        buffers, events, index = self._staging.get(key, ([], [], -1))
        if not buffers or buffers[0].shape != image.shape or buffers[0].dtype != torch.from_numpy(image[:0]).dtype:
            buffers = [torch.empty(image.shape, dtype=torch.from_numpy(image[:0]).dtype, pin_memory=True) for _ in range(STAGING_COUNT)]
            events = [None] * STAGING_COUNT
        index = (index + 1) % STAGING_COUNT
        event: torch.cuda.Event | None = events[index]
        if event is not None:
            event.synchronize()
        self._staging[key] = (buffers, events, index)
        return buffers[index]
