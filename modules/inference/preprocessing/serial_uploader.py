"""The serial uploader: the camera's one hand-off, each frame onto the GPU on its own, the moment it
comes. The camera's thread queues its frames here and returns; this thread puts each on the GPU and hands it on, in
arrival order."""

import time
import logging
from collections import deque
from threading import Condition, Thread

import numpy as np
import torch

from modules.camera import CAPTURE_TOLERANCE, CameraFrame, CameraTracklets, DepthAlignment, FrameType, TrackletCallback, \
    aligned_depth_size, depth_projection, straightened_tracklets
from modules.settings import BaseSettings, Field
from modules.utils import Broadcast, PerformanceTimer
from .depth_align import DepthAligner
from .image_upload import ImageUpload
from .source import CaptureCallback, GpuDepth, GpuDepthCallback, GpuImage, GpuImageCallback, ImageSnapshot, ImageSnapshotCallback

logger = logging.getLogger(__name__)

QUEUED: int = 2             # frames of a kind waiting per camera; beyond that the oldest is dropped and counted
NOTED: int = 16             # color frames per camera whose straightening is kept for their tracklets, which come about 2 frames later

# A color frame's way into the image the uploader makes: its homography, turn (`rotate90`), `flip_v` and `flip_h`
Straightening = tuple[np.ndarray | None, bool, bool, bool]
AS_DELIVERED: Straightening = (None, False, False, False)


class SerialUploaderSettings(BaseSettings):
    """How the serial uploader does its GPU work."""
    depth_kernels: Field[bool] = Field(True, description="Align the depth with CUDA kernels compiled at runtime; off, with torch ops as a CUDA graph")


class SerialUploader:
    """Takes the camera's frames on the camera's thread (`submit_frame`) and does all the rest on its own: the color
    goes to the GPU (`ImageUpload`, one frame), converted, straightened and flipped there, and is handed on as a
    `GpuImage` with the event that says it is written, and as an `ImageSnapshot` of one (the image, its capture
    time, the event) to the snapshot callbacks; the depth goes the same way, is turned into meters and
    moved onto the color's view (`DepthAligner`, by the camera's calibration, straightened and mirrored as the
    color), and is handed on as a `GpuDepth`. These go at once, for the readers in the package, which wait on the
    event on their own streams; then each color's capture time goes to the capture callbacks, and then, only when a
    finished callback is registered, this thread waits for its stream and hands the image or the depth on written,
    for the readers outside the package. Neither the camera's thread nor this one waits for the GPU otherwise. A frame waits for this thread in a queue of
    `QUEUED` per camera and kind; a frame that finds it full drops the oldest of its kind (`dropped`), so a thread
    that cannot keep up loses frames instead of falling behind.

    The camera's tracklets come here too (`submit_tracklets`), as its model found them in the frame as delivered: this
    thread straightens, turns and flips them as the frame they were found in, whose homography, turn and flips
    `submit_frame` notes by capture time (`NOTED` frames, a dropped one included), and hands them on, before any frame that waits.
    Tracklets whose frame is no longer noted take the newest frame's; tracklets `straightened` already pass as they are.

    Input:  BGR uint8 (H, W, 3), grayscale uint8 (H, W) or packed NV12 uint8 (H * 3 / 2, W)
    Output: float16 RGB CHW [0, 1] tensors keyed by cam_id
    """

    def __init__(self, settings: SerialUploaderSettings) -> None:
        self._thread: Thread = Thread(target=self._run, daemon=True, name='SerialUploader')
        self.settings: SerialUploaderSettings = settings
        self._image_upload: ImageUpload = ImageUpload()
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._aligners: dict[int, DepthAligner] = {}
        # Each camera's depth alignment, the depth's size, and what they make: to_ray, to_uv, the aligned size
        self._projections: dict[int, tuple[DepthAlignment | None, tuple[int, int], np.ndarray, np.ndarray, tuple[int, int]]] = {}
        self._timer: PerformanceTimer = PerformanceTimer(
            name="GPU Image Upload  ", sample_count=200, report_interval=100,
            color="green", omit_init=25,
        )
        self._gpu_image_callbacks: Broadcast[GpuImage] = Broadcast(name="SerialUploader image")
        self._snapshot_callbacks: Broadcast[ImageSnapshot] = Broadcast(name="SerialUploader snapshot")
        self._capture_callbacks: Broadcast[dict[int, float]] = Broadcast(name="SerialUploader capture")
        self._finished_image_callbacks: Broadcast[GpuImage] = Broadcast(name="SerialUploader finished image")
        self._gpu_depth_callbacks: Broadcast[GpuDepth] = Broadcast(name="SerialUploader depth")
        self._finished_depth_callbacks: Broadcast[GpuDepth] = Broadcast(name="SerialUploader finished depth")
        self._finish_timer: PerformanceTimer = PerformanceTimer(name="Wait upload finished")
        self._tracklet_callbacks: Broadcast[CameraTracklets] = Broadcast(name="SerialUploader tracklet")
        self._queue: deque[tuple[CameraFrame, float]] = deque()            # each with the moment it was queued
        self._tracklets: deque[tuple[CameraTracklets, float]] = deque()    # never dropped: a tracker needs every message
        # How long a frame and a tracklet message wait in their queue for this thread (ms), as timer samples
        self._frame_wait: PerformanceTimer = PerformanceTimer(name="Wait upload")
        self._tracklet_wait: PerformanceTimer = PerformanceTimer(name="Wait upload tracklets")
        self._straightening: dict[int, deque[tuple[float, Straightening]]] = {}  # per camera: (capture, its straightening)
        self._refused_depth: bool = False           # a depth frame still to be turned or flipped top to bottom was refused
        self._wake: Condition = Condition()
        self._running: bool = False
        self._dropped: int = 0

    @property
    def dropped(self) -> int:
        """Frames dropped because this thread had not taken the ones before them yet."""
        return self._dropped

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._running = False
            self._wake.notify()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit_frame(self, frame: CameraFrame) -> None:
        """A camera frame callback, on the camera's thread: queues the frame and returns."""
        with self._wake:
            if frame.kind == FrameType.VIDEO:
                noted = self._straightening.setdefault(frame.cam_id, deque(maxlen=NOTED))
                noted.append((frame.timestamp, (frame.homography, frame.rotate90, frame.flip_v, frame.flip_h)))
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
        self._tracklet_callbacks.add_callback(callback)

    def _run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._queue and not self._tracklets:
                    self._wake.wait()
                if not self._running:
                    return
                if self._tracklets:
                    cam_tracklets, queued = self._tracklets.popleft()
                    straightening: Straightening = self._straightening_at(cam_tracklets)
                    frame: CameraFrame | None = None
                    wait: PerformanceTimer = self._tracklet_wait
                else:
                    frame, queued = self._queue.popleft()
                    wait = self._frame_wait
            wait.add_time((time.perf_counter() - queued) * 1000.0, report=False)
            if frame is None:
                self._hand_on_tracklets(cam_tracklets, straightening)
                continue
            try:
                if frame.kind == FrameType.VIDEO:
                    self._upload(frame)
                elif frame.kind == FrameType.DEPTH:
                    self._upload_depth(frame)
            except Exception:
                logger.exception("SerialUploader: frame of camera %d failed", frame.cam_id)

    def _straightening_at(self, cam_tracklets: CameraTracklets) -> Straightening:
        """The straightening of the frame the tracklets were found in, the newest frame's when it is no longer noted;
        none for tracklets straightened already or a camera without frames yet. Under the lock."""
        noted = self._straightening.get(cam_tracklets.cam_id)
        if cam_tracklets.straightened or not noted:
            return AS_DELIVERED
        for timestamp, straightening in reversed(noted):
            if abs(timestamp - cam_tracklets.timestamp) <= CAPTURE_TOLERANCE:
                return straightening
        return noted[-1][1]

    def _hand_on_tracklets(self, cam_tracklets: CameraTracklets, straightening: Straightening) -> None:
        if not cam_tracklets.straightened:
            cam_tracklets = straightened_tracklets(cam_tracklets, *straightening)
        self._tracklet_callbacks(cam_tracklets)

    def add_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, with its capture time and the event that says it is written, on this thread."""
        self._gpu_image_callbacks.add_callback(callback)

    def add_snapshot_callback(self, callback: ImageSnapshotCallback) -> None:
        """Every uploaded color image as an `ImageSnapshot` of one: the image, its capture
        time and the event that says it is written, on this thread, after the image callbacks."""
        self._snapshot_callbacks.add_callback(callback)

    def add_capture_callback(self, callback: CaptureCallback) -> None:
        """Every uploaded color image's capture time by cam_id, a set of one, on this thread, after the snapshot callbacks."""
        self._capture_callbacks.add_callback(callback)

    def add_finished_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, written, on this thread, last."""
        self._finished_image_callbacks.add_callback(callback)

    def add_finished_gpu_depth_callback(self, callback: GpuDepthCallback) -> None:
        """Every uploaded depth frame, written, on this thread, after the depth callbacks."""
        self._finished_depth_callbacks.add_callback(callback)

    def _finish(self) -> None:
        """This thread waits for its stream: what it queued is written."""
        start: float = time.perf_counter()
        self._stream.synchronize()
        self._finish_timer.add_time((time.perf_counter() - start) * 1000.0, report=False)

    def add_gpu_depth_callback(self, callback: GpuDepthCallback) -> None:
        """Every uploaded depth frame, in its place among the images, with the event that says it is written, on this thread."""
        self._gpu_depth_callbacks.add_callback(callback)

    def _upload(self, frame: CameraFrame) -> None:
        """A color frame to the GPU (`ImageUpload`, one frame), handed on as a `GpuImage` and as an `ImageSnapshot`
        of one."""
        cam_id: int = frame.cam_id
        start = time.perf_counter()
        images, ready = self._image_upload.upload([frame], self._stream)
        rgb: torch.Tensor = images[0]
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=False)

        gpu_image: GpuImage = GpuImage(cam_id, frame.timestamp, rgb, ready, frame)
        snapshot: ImageSnapshot = ({cam_id: rgb}, {cam_id: frame.timestamp}, ready)
        self._gpu_image_callbacks(gpu_image)
        self._snapshot_callbacks(snapshot)
        self._capture_callbacks({cam_id: frame.timestamp})
        if self._finished_image_callbacks:
            self._finish()
            self._finished_image_callbacks(gpu_image)

    def _upload_depth(self, frame: CameraFrame) -> None:
        """A depth frame to the GPU: meters and its confidence, moved onto the color's view at its own grid, and handed on.
        The alignment straightens and mirrors only: a depth frame still to be turned or flipped top to bottom is
        refused, with an error once."""
        if frame.rotate90 or frame.flip_v:
            if not self._refused_depth:
                self._refused_depth = True
                logger.error("SerialUploader: camera %d's depth frames carry rotate90 or flip_v, which the depth's "
                             "alignment does not apply; they are not uploaded", frame.cam_id)
            return
        cam_id: int = frame.cam_id
        image: np.ndarray = frame.image
        size: tuple[int, int] = (image.shape[1], image.shape[0])
        projection = self._projections.get(cam_id)
        if projection is None or projection[0] is not frame.alignment or projection[1] != size:
            projection = (frame.alignment, size, *depth_projection(frame.alignment, size), aligned_depth_size(size, frame.alignment))
            self._projections[cam_id] = projection
        _, _, to_ray, to_uv, aligned_size = projection
        aligner: DepthAligner = self._aligners.setdefault(cam_id, DepthAligner())

        key: tuple[int, str] = (cam_id, 'depth')       # beside the color stacks' keys in the upload's ring
        staging: torch.Tensor = self._image_upload.staging.buffer(key, image.shape, torch.from_numpy(image[:0]).dtype)
        staging.copy_(torch.from_numpy(image))
        confidence_key: tuple[int, str] = (cam_id, 'confidence')
        confidence: torch.Tensor | None = None
        if frame.confidence is not None:
            confidence = self._image_upload.staging.buffer(confidence_key, frame.confidence.shape, torch.uint8)
            confidence.copy_(torch.from_numpy(frame.confidence))
        with torch.cuda.stream(self._stream):
            depth: torch.Tensor = aligner.align_staged(staging, confidence, to_ray, to_uv, frame.homography, frame.flip_h,
                                                       aligned_size, kernels=self.settings.depth_kernels)
            ready: torch.cuda.Event = torch.cuda.Event()
            ready.record(self._stream)
        self._image_upload.staging.queued(key, ready)
        if confidence is not None:
            self._image_upload.staging.queued(confidence_key, ready)

        gpu_depth: GpuDepth = GpuDepth(cam_id, frame.timestamp, depth, ready)
        self._gpu_depth_callbacks(gpu_depth)
        if self._finished_depth_callbacks:
            self._finish()
            self._finished_depth_callbacks(gpu_depth)
