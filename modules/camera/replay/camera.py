"""A camera fed from recordings without a device: the `Player`'s clip frames, stamped at hand-over
and emitted as they were recorded, and the people in them found by a detector and followed by a
tracker on the host, so `-sim` needs no OAK plugged in.

The detector and the tracker are the app's to give (`modules/inference/detection` and the
`Tracker` beside this file); without a detector the camera is frames only, as a rig without
`yolo` is. The device's warp is not applied here: a replay plays what was recorded.
"""

import logging
import time
from typing import Protocol

import numpy as np

from modules.utils import FPS, Rect

from .. import CameraFrame, CameraTracklet, FrameCallback, FrameType, Player, RigCameraReadings, RigCameraSettings, TrackletCallback
from .tracker import Track

logger = logging.getLogger(__name__)


class Detection(Protocol):
    """What the detector reports for one frame: `modules.inference.detection.Detection`."""
    cam_id: int
    boxes: list[Rect]


class Detector(Protocol):
    """A person detector the camera submits frames to and hears back from, on the detector's
    thread: `modules.inference.detection.Detector`. Named here so the contract does not import
    the inference package."""
    def submit(self, cam_id: int, image: np.ndarray) -> None: ...
    def register_callback(self, callback) -> None: ...


class BoxTracker(Protocol):
    """A tracker that turns the boxes of one frame into tracks with ids and statuses."""
    def update(self, boxes: list[Rect]) -> list[Track]: ...


class Camera:
    def __init__(self, player: Player, settings: RigCameraSettings, detector: Detector | None, tracker: BoxTracker | None, id: int = 0) -> None:
        self.id: int = id
        self.settings: RigCameraSettings = settings
        self.player: Player = player
        self.detector: Detector | None = detector
        self.tracker: BoxTracker | None = tracker if detector is not None else None
        self.running: bool = False

        self.video_fps: FPS = FPS(120)
        self.tracklet_fps: FPS = FPS(120)
        self.frame_callbacks: list[FrameCallback] = []
        self.tracklet_callbacks: list[TrackletCallback] = []

    # ── CameraSource ───────────────────────────────────────────────────

    @property
    def readings(self) -> RigCameraReadings:
        return self.settings.readings

    @property
    def connected(self) -> bool:
        return self.running

    def start(self) -> None:
        """Hooks into the player (before it runs) and the detector; there is no thread of its own:
        the frames come on the player's threads, the tracklets on the detector's."""
        self.player.add_frame_callback(self._clip_callback)
        if self.detector is not None:
            self.detector.register_callback(self._on_detection)
        self.running = True

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        pass

    def add_frame_callback(self, callback: FrameCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.tracklet_callbacks:
            self.tracklet_callbacks.append(callback)

    # ── Hand-over ──────────────────────────────────────────────────────

    def _clip_callback(self, cam_id: int, frame_type: FrameType, image: np.ndarray) -> None:
        """A decoded clip frame, on the player's thread: stamped here, to the callbacks and the detector."""
        if not self.running or cam_id != self.id or frame_type != FrameType.VIDEO:
            return
        self.video_fps.processed()
        self.settings.readings.video_fps = self.video_fps.get_rate_average()
        self.settings.readings.video_min_fps = self.video_fps.get_rate_minimum()
        frame = CameraFrame(cam_id=self.id, kind=frame_type, image=image, time=time.perf_counter())
        for c in self.frame_callbacks:
            c(frame)
        if self.detector is not None:
            self.detector.submit(self.id, image)

    def _on_detection(self, detection: Detection) -> None:
        """The detector's boxes for one frame, on its thread: through the tracker, out as the
        contract's tracklets with the time of this message, an empty list included."""
        if not self.running or detection.cam_id != self.id or self.tracker is None:
            return
        now: float = time.perf_counter()
        self.tracklet_fps.processed()
        self.settings.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        tracklets: list[CameraTracklet] = [
            CameraTracklet(cam_id=self.id, id=t.id, status=t.status, roi=t.roi, age=t.age, time=now)
            for t in self.tracker.update(detection.boxes)]
        self.settings.readings.tracklets = len(tracklets)
        for c in self.tracklet_callbacks:
            c(self.id, tracklets, now)
