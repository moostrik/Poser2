"""A camera fed from recordings without a device: the `Player`'s clip frames, stamped at hand-over
and emitted as they were recorded, and the people in them found by a detector and followed by a
tracker on the host, so `-sim` needs no OAK plugged in.

The detector and the tracker are the app's to give (`modules/inference/detection` and the
`Tracker` beside this file); the app feeds the detector the uploaded images, and the camera hears its
boxes. Without a detector the camera is frames only, as a rig without `yolo` is. The device's warp is
not applied here: a replay plays what was recorded. A recording with depth and metadata plays as the
camera did: depth frames too, with the recorded alignment, and with `tilt` the frames straightened from
the recorded tilt plus the live offset; the detector looks at the straightened frames, so its boxes are.
"""

import logging
import time
from typing import Protocol

import numpy as np

from modules.utils import FPS, Rect

from .. import CameraFrame, CameraTracklet, CameraTracklets, CameraFrameCallback, ChunkInfo, DepthAlignment, FrameMetadata, FrameReadings, FrameType, \
    Player, RigCameraReadings, RigCameraSettings, TiltCorrection, TiltCorrectionSettings, TrackletCallback
from .tracker import Track

logger = logging.getLogger(__name__)


class Detection(Protocol):
    """What the detector reports for one frame: `modules.inference.detection.Detection`."""
    cam_id: int
    timestamp: float
    boxes: list[Rect]


class PersonDetector(Protocol):
    """A person detector the camera hears back from, on the detector's thread:
    `modules.inference.detection.PersonDetector`. The app feeds it the uploaded images, straightened and
    mirrored, so its boxes are too. Named here so the contract does not import the inference package."""
    def add_detection_callback(self, callback) -> None: ...


class BoxTracker(Protocol):
    """A tracker that turns the boxes of one frame into tracks with ids and statuses."""
    def update(self, boxes: list[Rect]) -> list[Track]: ...


class Camera:
    def __init__(self, player: Player, settings: RigCameraSettings, detector: PersonDetector | None, tracker: BoxTracker | None, id: int = 0,
                 tilt: TiltCorrectionSettings | None = None) -> None:
        self.id: int = id
        self.settings: RigCameraSettings = settings
        self.player: Player = player
        self.detector: PersonDetector | None = detector
        self.tracker: BoxTracker | None = tracker if detector is not None else None
        self.running: bool = False
        self.tilt_correction: TiltCorrection | None = TiltCorrection(tilt) if tilt is not None else None
        self._video_timestamp: float | None = None     # the stamp of the recorded frame being handed over

        self.video_fps: FPS = FPS(120)
        self.tracklet_fps: FPS = FPS(120)
        self.frame_callbacks: list[CameraFrameCallback] = []
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
            self.detector.add_detection_callback(self._on_detection)
        self.running = True

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        pass

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
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

    def _clip_callback(self, cam_id: int, frame_type: FrameType, image: np.ndarray, metadata: FrameMetadata | None,
                       captured: float | None) -> None:
        """A clip frame, on the player's thread, to the callbacks: stamped with its capture on the play's clock,
        or its hand-over for a recording without metadata, and its hand-over as its arrival. The player hands
        a recorded frame's video first, so its depth takes the video's stamp: one capture, one time. The frame
        goes on as the camera delivered it: `flip_h` as recorded, and the homography of the recorded tilt and
        the live offset on the recorded lens, as the camera made it."""
        if not self.running or cam_id != self.id:
            return
        arrival: float = time.perf_counter()
        info: ChunkInfo | None = self.player.get_chunk_info(self.id)
        if frame_type == FrameType.VIDEO:
            self._video_timestamp = captured if captured is not None else arrival
            self.video_fps.processed()
            self.settings.readings.video_fps = self.video_fps.get_rate_average()
            self.settings.readings.video_min_fps = self.video_fps.get_rate_minimum()
            self._update_tilt(metadata.readings if metadata is not None else None, info)
        timestamp: float = self._video_timestamp if self._video_timestamp is not None else arrival
        homography: np.ndarray | None = self.tilt_correction.homography if self.tilt_correction is not None else None
        alignment: DepthAlignment | None = info.depth_alignment if frame_type == FrameType.DEPTH and info is not None else None
        flip_h: bool = info.flip_h if info is not None else False
        frame = CameraFrame(cam_id=self.id, kind=frame_type, image=image, timestamp=timestamp, flip_h=flip_h,
                            homography=homography, alignment=alignment, arrival=arrival)
        for c in self.frame_callbacks:
            c(frame)

    def _update_tilt(self, readings: FrameReadings | None, info: ChunkInfo | None) -> None:
        """The recorded tilt on the recorded lens."""
        if readings is None or info is None:
            return
        self.settings.readings.tilt_measured = readings.tilt
        self.settings.readings.roll_measured = readings.roll
        if self.tilt_correction is not None and info.intrinsics is not None:
            self.tilt_correction.update(readings.tilt, info.intrinsics, (info.width, info.height))

    def _on_detection(self, detection: Detection) -> None:
        """The detector's boxes for one frame, on its thread: through the tracker, out as the
        contract's tracklets with that frame's time, an empty list included. The boxes were found in the
        straightened frame, so they are straightened already."""
        if not self.running or detection.cam_id != self.id or self.tracker is None:
            return
        timestamp: float = detection.timestamp
        self.tracklet_fps.processed()
        self.settings.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        tracklets: list[CameraTracklet] = [
            CameraTracklet(cam_id=self.id, id=t.id, status=t.status, roi=t.roi, age=t.age, timestamp=timestamp)
            for t in self.tracker.update(detection.boxes)]
        self.settings.readings.tracklets = len(tracklets)
        message = CameraTracklets(cam_id=self.id, timestamp=timestamp, tracklets=tracklets, arrival=time.perf_counter(), straightened=True)
        for c in self.tracklet_callbacks:
            c(message)
