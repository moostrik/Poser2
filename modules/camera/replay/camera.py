"""A camera fed from recordings without a device: the `Player`'s clip frames, handed over as the camera
delivered them, and the players in them as the camera's tracklets, so `-sim` needs no OAK plugged in.

A frame is stamped with its capture on the play's clock (`PlayClock`), arriving as late after it as it
came live, and goes on as the camera delivered it: `flip_h` as recorded, depth frames with the recorded
alignment, and with `tilt` the homography of the recorded tilt plus the live offset on the recorded lens. It
carries the recorded readings and lens, so a recording played can be recorded again as it was.
The device's warp is not applied here: a replay plays what was recorded.

The tracklets come one of two ways. A recording that keeps the camera's tracklets plays them: each
frame's after its recorded latency, stamped with the frame's capture, as the camera gave them, so the
uploader straightens them as it does live. They go out from this camera's own thread, so nothing but
the hand-over runs on the player's. With a person detector enabled (`PersonDetector.enabled`), the
detector's boxes play instead: the app feeds it the uploaded images, straightened and mirrored, its
boxes go through a host tracker (`Tracker`) and out as tracklets marked `straightened`.
"""

import heapq
import logging
import time
from threading import Condition, Thread
from typing import Protocol

import numpy as np

from modules.utils import FPS, Rect

from .. import CameraFrame, CameraTracklet, CameraTracklets, CameraFrameCallback, ChunkInfo, DepthAlignment, FrameMetadata, FrameReadings, FrameType, \
    Player, RigCameraReadings, RigCameraSettings, TiltCorrection, TiltCorrectionSettings, TrackletCallback
from .tracker import Track

logger = logging.getLogger(__name__)

JUMP: float = 1.0       # s of recorded time between two frames beyond which the play jumped: what waits is dropped


class Detection(Protocol):
    """What the detector reports for one frame: `modules.inference.detection.Detection`."""
    cam_id: int
    timestamp: float
    boxes: list[Rect]


class PersonDetector(Protocol):
    """A person detector the camera hears back from, on the detector's thread:
    `modules.inference.detection.PersonDetector`. The app feeds it the uploaded images, straightened and
    mirrored, so its boxes are too. Named here so the contract does not import the inference package."""
    @property
    def enabled(self) -> bool: ...
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
        self._recorded_time: float | None = None       # its recorded capture time

        # The recorded tracklets waiting for their moment: (due, order, message), on this camera's thread
        self._waiting: list[tuple[float, int, CameraTracklets]] = []
        self._order: int = 0
        self._wake: Condition = Condition()
        self._thread: Thread | None = None

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
        """Hooks into the player (before it runs) and the detector, and starts the thread the recorded
        tracklets go out on; the frames come on the player's threads, the detector's tracklets on its own."""
        self.player.add_frame_callback(self._clip_callback)
        if self.detector is not None:
            self.detector.add_detection_callback(self._on_detection)
        self.running = True
        self._thread = Thread(target=self._send_recorded, daemon=True, name=f'ReplayTracklets{self.id}')
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self.running = False
            self._waiting.clear()
            self._wake.notify()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

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
        a recorded frame's video first, so its depth takes the video's stamp: one capture, one time."""
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
        readings: FrameReadings | None = metadata.readings if frame_type == FrameType.VIDEO and metadata is not None else None
        intrinsics: list[list[float]] | None = info.intrinsics if info is not None else None
        frame = CameraFrame(cam_id=self.id, kind=frame_type, image=image, timestamp=timestamp, flip_h=flip_h,
                            homography=homography, readings=readings, intrinsics=intrinsics, alignment=alignment, arrival=arrival)
        for c in self.frame_callbacks:
            c(frame)
        if frame_type == FrameType.VIDEO and metadata is not None:
            self._queue_recorded(metadata, timestamp)

    def _update_tilt(self, readings: FrameReadings | None, info: ChunkInfo | None) -> None:
        """The recorded tilt on the recorded lens."""
        if readings is None or info is None:
            return
        self.settings.readings.tilt_measured = readings.tilt
        self.settings.readings.roll_measured = readings.roll
        if self.tilt_correction is not None and info.intrinsics is not None:
            self.tilt_correction.update(readings.tilt, info.intrinsics, (info.width, info.height))

    # ── The recorded tracklets ─────────────────────────────────────────

    def _queue_recorded(self, metadata: FrameMetadata, timestamp: float) -> None:
        """The frame's recorded tracklets, due their recorded latency after its stamp; none while the detector
        plays, or for a frame no tracklets came for. A jump in the recorded time (a loop, a seek) drops what
        still waits: it belongs to frames before the jump."""
        jumped: bool = self._recorded_time is not None and not 0.0 < metadata.timestamp - self._recorded_time <= JUMP
        self._recorded_time = metadata.timestamp
        playing_detector: bool = self.detector is not None and self.detector.enabled
        with self._wake:
            if jumped:
                self._waiting.clear()
            if metadata.tracklets is None or playing_detector:
                return
            tracklets: list[CameraTracklet] = [CameraTracklet(cam_id=self.id, id=t.id, status=t.status, roi=t.roi, age=t.age,
                                                              timestamp=timestamp, keypoints=t.keypoints) for t in metadata.tracklets]
            message = CameraTracklets(cam_id=self.id, timestamp=timestamp, tracklets=tracklets)
            due: float = timestamp + (metadata.tracklet_latency or 0.0)
            heapq.heappush(self._waiting, (due, self._order, message))
            self._order += 1
            self._wake.notify()

    def _send_recorded(self) -> None:
        """This camera's thread: each waiting message out at its moment, stamped with its arrival."""
        while True:
            with self._wake:
                while self.running and (not self._waiting or self._waiting[0][0] > time.perf_counter()):
                    self._wake.wait(self._waiting[0][0] - time.perf_counter() if self._waiting else None)
                if not self.running:
                    return
                _, _, message = heapq.heappop(self._waiting)
            self._emit_tracklets(CameraTracklets(message.cam_id, message.timestamp, message.tracklets, arrival=time.perf_counter()))

    # ── The detector's tracklets ───────────────────────────────────────

    def _on_detection(self, detection: Detection) -> None:
        """The detector's boxes for one frame, on its thread: through the tracker, out as the
        contract's tracklets with that frame's time, an empty list included. The boxes were found in the
        straightened frame, so they are straightened already."""
        if not self.running or detection.cam_id != self.id or self.tracker is None:
            return
        timestamp: float = detection.timestamp
        tracklets: list[CameraTracklet] = [
            CameraTracklet(cam_id=self.id, id=t.id, status=t.status, roi=t.roi, age=t.age, timestamp=timestamp)
            for t in self.tracker.update(detection.boxes)]
        self._emit_tracklets(CameraTracklets(cam_id=self.id, timestamp=timestamp, tracklets=tracklets, arrival=time.perf_counter(), straightened=True))

    def _emit_tracklets(self, message: CameraTracklets) -> None:
        self.tracklet_fps.processed()
        self.settings.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        self.settings.readings.tracklets = len(message.tracklets)
        for c in self.tracklet_callbacks:
            c(message)
