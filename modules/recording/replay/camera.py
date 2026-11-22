"""A camera fed from recordings without a device: the `Player`'s clip frames, handed over as the camera
delivered them, and the players in them as the camera's tracklets, so `-sim` needs no OAK plugged in.

A frame is stamped with its capture on the play's clock (`PlayClock`), arriving as late after it as it
came live, and goes on as the camera delivered it: the turn and the flips as recorded, depth frames with the recorded
alignment, and with `tilt` the homography of the recorded tilt plus the live offset on the recorded lens. It
carries the recorded readings and lens, so a recording played can be recorded again as it was.
The device's warp is not applied here: a replay plays what was recorded.

The tracklets come one of two ways. A recording that keeps the camera's tracklets plays them: each
frame's after its recorded latency, stamped with the frame's capture, as the camera gave them, so the
uploader straightens them as it does live. They go out from this camera's own thread, so nothing but
the hand-over runs on the player's. With a person detector enabled (`PersonDetector.enabled`), the
detector's boxes play instead: the app feeds it the uploaded images, straightened and mirrored, its
boxes go through a host tracker (`Tracker`) on this camera's thread and out as tracklets marked `straightened`.
"""

import heapq
import logging
import time
from collections import deque
from threading import Lock, Thread
from typing import Callable, Protocol

import numpy as np

from modules.camera import CameraFrame, CameraTracklet, ClipFrame, ClipFrameCallback, CameraTracklets, CameraFrameCallback, DepthAlignment, FrameReadings, FrameType, \
    RigCameraReadings, TiltCorrection, TiltCorrectionSettings, TrackletCallback
from modules.utils import FPS, Broadcast, Rect, Waker

from ..chunk import ChunkInfo, FrameMetadata
from .tracker import Track

logger = logging.getLogger(__name__)

DETECTIONS_KEPT: int = 4        # detections waiting for the tracker, per camera


class Detection(Protocol):
    """What the detector reports for one frame: `modules.inference.person.Detection`."""
    @property
    def cam_id(self) -> int: ...
    @property
    def timestamp(self) -> float: ...
    @property
    def boxes(self) -> list[Rect]: ...


class PersonDetector(Protocol):
    """A person detector the camera hears back from, on the detector's thread:
    `modules.inference.person.PersonDetector`. The app feeds it the uploaded images, straightened and
    mirrored, so its boxes are too. Named here so the contract does not import the inference package."""
    @property
    def enabled(self) -> bool: ...
    def add_detection_callback(self, callback: Callable[[Detection], None]) -> None: ...


class BoxTracker(Protocol):
    """A tracker that turns the boxes of one frame into tracks with ids and statuses."""
    def update(self, boxes: list[Rect]) -> list[Track]: ...


class PlayerFeed(Protocol):
    """What the replay camera needs of a player: its frames, its jumps, a chunk's info, and whether video comes as
    NV12 (`modules.recording.Player`)."""
    nv12: bool
    def add_frame_callback(self, callback: ClipFrameCallback) -> None: ...
    def add_jump_callback(self, callback: Callable[[], None]) -> None: ...
    def get_chunk_info(self, cam_id: int) -> ChunkInfo | None: ...


class Camera:
    def __init__(self, player: PlayerFeed, readings: RigCameraReadings, detector: PersonDetector | None, tracker: BoxTracker | None, cam_id: int = 0,
                 tilt: TiltCorrectionSettings | None = None) -> None:
        self.cam_id: int = cam_id
        self.readings: RigCameraReadings = readings     # its only settings: the mount and the lens come with the recording
        self.player: PlayerFeed = player
        self.detector: PersonDetector | None = detector
        self.tracker: BoxTracker | None = tracker if detector is not None else None
        self.running: bool = False
        self.tilt_correction: TiltCorrection | None = TiltCorrection(tilt) if tilt is not None else None
        self._video_timestamp: float | None = None     # the stamp of the recorded frame being handed over

        # The recorded tracklets waiting for their moment: (due, order, message), on this camera's thread
        self._waiting: list[tuple[float, int, CameraTracklets]] = []
        self._order: int = 0
        self._detections: deque[Detection] = deque()    # the detector's, for the tracker on this camera's thread
        self.dropped_detections: int = 0                # dropped, oldest first, while the tracker fell behind
        self._lock: Lock = Lock()
        self._waker: Waker = Waker()
        self._thread: Thread | None = None

        self.video_fps: FPS = FPS()
        self.tracklet_fps: FPS = FPS()
        self.frame_callbacks: Broadcast[[CameraFrame]] = Broadcast(name='replay frame')
        self.tracklet_callbacks: Broadcast[[CameraTracklets]] = Broadcast(name='replay tracklets')

    # ── CameraSource ───────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self.running

    def start(self) -> None:
        """Hooks into the player (before it runs) and the detector, and starts the thread the tracklets go out
        on, the recorded and the detector's; the frames come on the player's threads."""
        self.player.add_frame_callback(self._clip_callback)
        self.player.add_jump_callback(self._on_jump)
        if self.detector is not None:
            self.detector.add_detection_callback(self._on_detection)
        self.running = True
        self._thread = Thread(target=self._send_recorded, daemon=True, name=f'ReplayTracklets{self.cam_id}')
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self._waiting.clear()
            self._detections.clear()
        self._waker.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                return
        self._waker.close()

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        self.frame_callbacks.add_callback(callback)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        self.tracklet_callbacks.add_callback(callback)

    # ── Hand-over ──────────────────────────────────────────────────────

    def _clip_callback(self, clip: ClipFrame) -> None:
        """A clip frame, on the player's thread, to the callbacks: stamped with its capture on the play's clock,
        or its hand-over for a recording without metadata, and its hand-over as its arrival; a depth with its
        confidence. The player hands a recorded frame's video first, so its depth takes the video's stamp: one
        capture, one time."""
        if not self.running or clip.cam_id != self.cam_id:
            return
        frame_type: FrameType = clip.kind
        captured: float | None = clip.timestamp
        metadata: FrameMetadata | None = clip.metadata if isinstance(clip.metadata, FrameMetadata) else None
        arrival: float = time.perf_counter()
        info: ChunkInfo | None = self.player.get_chunk_info(self.cam_id)
        if frame_type == FrameType.VIDEO:
            self._video_timestamp = captured if captured is not None else arrival
            self.video_fps.processed()
            self.readings.video_fps = self.video_fps.get_rate_average()
            self.readings.video_min_fps = self.video_fps.get_rate_minimum()
            self._update_tilt(metadata.readings if metadata is not None else None, info)
        # The player hands a frame's streams with one stamp; without metadata the depth takes its video's
        timestamp: float = captured if captured is not None else self._video_timestamp if self._video_timestamp is not None else arrival
        homography: np.ndarray | None = self.tilt_correction.homography if self.tilt_correction is not None else None
        alignment: DepthAlignment | None = info.depth_alignment if frame_type == FrameType.DEPTH and info is not None else None
        flip_h: bool = info.flip_h if info is not None else False
        readings: FrameReadings | None = metadata.readings if frame_type == FrameType.VIDEO and metadata is not None else None
        intrinsics: list[list[float]] | None = info.intrinsics if info is not None else None
        nv12: bool = frame_type == FrameType.VIDEO and self.player.nv12
        frame = CameraFrame(cam_id=self.cam_id, kind=frame_type, image=clip.image, timestamp=timestamp, nv12=nv12, flip_h=flip_h,
                            homography=homography, readings=readings, intrinsics=intrinsics, alignment=alignment, arrival=arrival,
                            confidence=clip.confidence, rotate90=info.rotate90 if info is not None else False,
                            flip_v=info.flip_v if info is not None else False)
        self.frame_callbacks(frame)
        if frame_type == FrameType.VIDEO and metadata is not None:
            self._queue_recorded(metadata, timestamp)

    def _update_tilt(self, readings: FrameReadings | None, info: ChunkInfo | None) -> None:
        """The recorded tilt on the recorded lens."""
        if readings is None or info is None:
            return
        self.readings.tilt_measured = readings.tilt
        self.readings.roll_measured = readings.roll
        if self.tilt_correction is not None and info.intrinsics is not None:
            self.tilt_correction.update(readings.tilt, info.intrinsics, (info.width, info.height))

    # ── The recorded tracklets ─────────────────────────────────────────

    def _on_jump(self) -> None:
        """The player starts somewhere that does not follow what played (a loop, a seek): what still waits
        belongs to the frames before it, and is dropped."""
        with self._lock:
            self._waiting.clear()

    def _queue_recorded(self, metadata: FrameMetadata, timestamp: float) -> None:
        """The frame's recorded tracklets, due their recorded latency after its stamp; none while the detector
        plays, or for a frame no tracklets came for."""
        playing_detector: bool = self.detector is not None and self.detector.enabled
        with self._lock:
            if metadata.tracklets is None or playing_detector:
                return
            tracklets: list[CameraTracklet] = [CameraTracklet(cam_id=self.cam_id, id=t.id, status=t.status, roi=t.roi, age=t.age,
                                                              timestamp=timestamp, keypoints=t.keypoints) for t in metadata.tracklets]
            message = CameraTracklets(cam_id=self.cam_id, timestamp=timestamp, tracklets=tracklets)
            due: float = timestamp + (metadata.tracklet_latency or 0.0)
            heapq.heappush(self._waiting, (due, self._order, message))
            self._order += 1
        self._waker.set()

    def _send_recorded(self) -> None:
        """This camera's thread: each waiting message out at its moment, stamped with its arrival, and each
        detection through the tracker."""
        while True:
            with self._lock:
                if not self.running:
                    return
                detection: Detection | None = self._detections.popleft() if self._detections else None
                due: float | None = self._waiting[0][0] if self._waiting else None
                message: CameraTracklets | None = None
                if due is not None and due <= time.perf_counter():
                    _, _, message = heapq.heappop(self._waiting)
            try:
                if detection is not None:
                    self._track(detection)
                if message is not None:
                    self._emit_tracklets(CameraTracklets(message.cam_id, message.timestamp, message.tracklets, arrival=time.perf_counter()))
            except Exception:
                logger.exception("Replay camera %d: tracklets not sent", self.cam_id)
            if detection is None and message is None:
                self._waker.wait(due)

    # ── The detector's tracklets ───────────────────────────────────────

    def _on_detection(self, detection: Detection) -> None:
        """The detector's boxes for one frame, on its thread, queued for this camera's thread; the oldest goes,
        counted, past `DETECTIONS_KEPT`."""
        if not self.running or detection.cam_id != self.cam_id or self.tracker is None:
            return
        with self._lock:
            self._detections.append(detection)
            if len(self._detections) > DETECTIONS_KEPT:
                self._detections.popleft()
                self.dropped_detections += 1
                if self.dropped_detections == 1:
                    logger.warning("Replay camera %d: the tracker falls behind the detector, oldest detections dropped", self.cam_id)
        self._waker.set()

    def _track(self, detection: Detection) -> None:
        """A detection through the tracker, out as the contract's tracklets with that frame's time, an empty list
        included. The boxes were found in the straightened frame, so they are straightened already."""
        assert self.tracker is not None
        timestamp: float = detection.timestamp
        tracklets: list[CameraTracklet] = [
            CameraTracklet(cam_id=self.cam_id, id=t.id, status=t.status, roi=t.roi, age=t.age, timestamp=timestamp)
            for t in self.tracker.update(detection.boxes)]
        self._emit_tracklets(CameraTracklets(cam_id=self.cam_id, timestamp=timestamp, tracklets=tracklets, arrival=time.perf_counter(), straightened=True))

    def _emit_tracklets(self, message: CameraTracklets) -> None:
        self.tracklet_fps.processed()
        self.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        self.readings.tracklets = len(message.tracklets)
        self.tracklet_callbacks(message)
