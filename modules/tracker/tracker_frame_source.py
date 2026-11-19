"""TrackerFrameSource: the pose pipeline's frame source fed by the tracker, a pose per player from the tracker's
newest boxes, at each frame bang.

A frame bang is a set: each camera's capture time in it (`cam_timestamps`, one camera's for a set of one).
`submit(cam_timestamps)` queues it for this one's own thread and returns: at most
`WAITING` wait, beyond that the oldest is dropped, counted as `dropped`. The thread makes the poses (`process`): a
player is posed when its camera is in the set and its last detection is within `detection_timeout` of that camera's
capture time. It hands them with the capture times to the frames callbacks (the pipeline's push, which carries them
to the crop, where they find the images), every bang. A bang before
`start` is an error; one after `stop` is dropped."""

import logging
import math
import time
from collections import deque
from threading import Condition, Lock, Thread

import numpy as np

from modules.pose import Frame, FrameDict, BBox, BBoxAzimuth, Distance, Points2D, SourceFramesCallback
from modules.settings import BaseSettings, Field
from modules.utils import PerformanceTimer, Rect
from .tracklet import Tracklet
from .panoramic.annotation import Annotation as PanoramicAnnotation

logger = logging.getLogger(__name__)

WAITING: int = 2        # bangs waiting for the thread; a newer image makes an older one's poses late


class TrackerFrameSourceSettings(BaseSettings):
    # How long after their last detection a person is still cropped from their last box. Judged here
    # rather than in the tracker, which emits everyone it still remembers: inside this window a
    # person the detector dropped keeps their pose, so a missed detection neither interrupts it nor
    # resets the filters downstream. Not below a few frames: a fresh detection is already some
    # milliseconds old by the time the frame bang crops it, so at 0 nobody would ever be posed.
    detection_timeout: Field[float] = Field(2.0, min=0.1, max=5.0, step=0.05,
                                            description="Seconds after the last detection a person still gets a pose")
    # A camera that runs the pose model itself (OAK4) hands its keypoints over in the frame; the pose
    # carries them in the host crop's convention, normalised to a window of this aspect around the
    # box, padded like the host crop, and the box becomes that window (`modules.inference.preprocessing.crop`).
    crop_aspect:       Field[float] = Field(0.75, min=0.25, max=4.0, step=0.05,
                                            description="Width over height of the window the camera's keypoints are normalised to (the host crop's 384/512)")
    crop_padding:      Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05,
                                            description="Margin around the box before the window is fitted (fraction; the host crop's padding)")
    dropped:           Field[int]   = Field(0, access=Field.READ, description="Frame bangs dropped while the pose step fell behind")


class TrackerFrameSource(Thread):
    """The pipeline's frame source fed by the tracker; see the module docstring."""

    def __init__(self, settings: TrackerFrameSourceSettings, max_players: int, frame_aspect: float = 1.0) -> None:
        """`frame_aspect` is the delivered frame's width over its height, the one thing the keypoint
        window needs to know about the image; it is 1.0 for the square frames of the rigs."""
        super().__init__(daemon=True, name='TrackerFrameSource')
        self.settings: TrackerFrameSourceSettings = settings
        self.frame_aspect: float = frame_aspect
        # The incoming dict keys are used directly as slot indices below, so the feeding tracker
        # must key by a slot under `max_players` (the panoramic tracker's player slots; cam ids for
        # one-per-cam rigs): a key at or above it would vanish here without a trace. Both come from
        # `max_players` in main.py — an invariant spanning two modules, so it is worth stating.
        if max_players <= 0:
            raise ValueError(f"TrackerFrameSource needs at least one track slot, got {max_players}")
        self._max_players = max_players
        # Store submitted tracklets per track ID
        self._tracklets: dict[int, Tracklet | None] = {
            track_id: None for track_id in range(max_players)
        }
        self._lock = Lock()
        self._batch_id_counter = 0
        # The bangs waiting for the thread: (cam_timestamps, the moment it was queued)
        self._wake: Condition = Condition()
        self._running: bool = False
        self._bangs: deque[tuple[dict[int, float], float]] = deque()
        self._dropped: int = 0
        self._wait: PerformanceTimer = PerformanceTimer(name="Wait pose step")
        self._stopped: bool = False
        self._callback_lock: Lock = Lock()
        self._frames_callbacks: list[SourceFramesCallback] = []     # run in registration order

    def add_frames_callback(self, callback: SourceFramesCallback) -> None:
        """Each bang's poses with its capture times (the pipeline's push)."""
        with self._callback_lock:
            if callback not in self._frames_callbacks:
                self._frames_callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        with self._wake:
            self._running = True
        super().start()

    def stop(self) -> None:
        """The thread ends; the bangs still waiting are dropped and logged."""
        with self._wake:
            self._running = False
            self._stopped = True
            dropped: int = len(self._bangs)
            self._bangs.clear()
            self._wake.notify()
        if dropped:
            logger.info("TrackerFrameSource: %d frame bangs dropped at stop", dropped)
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._bangs:
                    self._wake.wait()
                if not self._running:
                    return
                cam_timestamps, queued = self._bangs.popleft()
            self._wait.add_time((time.perf_counter() - queued) * 1000.0, report=False)
            try:
                self.process(cam_timestamps)
            except Exception:
                logger.exception("TrackerFrameSource: a frame bang failed")

    def submit(self, cam_timestamps: dict[int, float]) -> None:
        """A frame bang: each camera's capture time in the set (an uploader's capture callback), queued for the
        thread, the oldest dropped beyond `WAITING`."""
        dropped: int | None = None
        with self._wake:
            if not self._running:
                if self._stopped:
                    return
                raise RuntimeError("TrackerFrameSource: a frame bang before start")
            if len(self._bangs) >= WAITING:
                self._bangs.popleft()
                self._dropped += 1
                dropped = self._dropped
            self._bangs.append((dict(cam_timestamps), time.perf_counter()))
            self._wake.notify()
        if dropped is not None:
            self.settings.dropped = dropped

    @property
    def track_ids(self) -> list[int]:
        """The ids this source can emit: one slot per player."""
        return list(range(self._max_players))

    def set_tracklets(self, tracklet_dict: dict[int, Tracklet]) -> None:
        """Set tracklets for pose generation.

        Args:
            tracklet_dict: Dictionary of track_id -> Tracklet
        """
        dropped: list[int] = [k for k in tracklet_dict if k >= self._max_players]
        if dropped:
            logger.warning(
                f"TrackerFrameSource: track ids {dropped} are beyond {self._max_players} slots "
                f"and produce no pose — the feeding tracker has more slots than max_players."
            )
        with self._lock:
            # Update all track slots
            for track_id in range(self._max_players):
                if track_id in tracklet_dict:
                    self._tracklets[track_id] = tracklet_dict[track_id]
                else:
                    self._tracklets[track_id] = None

    def process(self, cam_timestamps: dict[int, float]) -> FrameDict:
        """Generate poses from all ready tracklets and hand them, with `cam_timestamps`, to the frames callbacks. A
        player is posed when its camera is in `cam_timestamps`, and its latched box's age is judged on that camera's
        capture time.

        Returns:
            Dictionary of track_id -> Frame for tracks with valid tracklets
        """
        # Copy tracklets under lock to avoid holding lock during processing
        with self._lock:
            tracklets_snapshot = self._tracklets.copy()
            self._batch_id_counter += 1

        generated_poses: FrameDict = {}
        # Checked on every frame, not when the tracker last published: a latched box goes stale
        # between tracker ticks, and a stale one is left out so downstream filters reset for it.
        detection_timeout: float = self.settings.detection_timeout

        for track_id, tracklet in tracklets_snapshot.items():
            if tracklet is None or tracklet.cam_id not in cam_timestamps:
                continue                    # its camera has no image in this set to crop it from
            if tracklet.is_expired(cam_timestamps[tracklet.cam_id], detection_timeout):
                continue

            try:
                bounding_box = BBox.from_rect(tracklet.roi)
                annotation = tracklet.annotation if isinstance(tracklet.annotation, PanoramicAnnotation) else None
                features: dict = {BBox: bounding_box}
                if tracklet.keypoints is not None:
                    window, points = self.keypoint_window(tracklet.roi, tracklet.keypoints)
                    features[BBox] = BBox.from_rect(window)
                    features[Points2D] = points
                if annotation is not None:
                    # SingleAngle.from_value wraps degrees-as-radians to [-π, π).
                    features[BBoxAzimuth] = BBoxAzimuth.from_value(math.radians(float(annotation.world_angle)))
                    # Without a reading the feature stays absent (NaN, score 0.0).
                    if not math.isnan(annotation.zone_distance):
                        features[Distance] = Distance.from_value(annotation.zone_distance)

                generated_poses[track_id] = Frame(
                    track_id=tracklet.id,
                    cam_id=tracklet.cam_id,
                    timestamp=tracklet.timestamp,
                    features=features,
                )
            except Exception as e:
                logger.error(
                    f"TrackerFrameSource: Error generating pose "
                    f"{track_id}: {e}"
                )
        with self._callback_lock:
            frames_callbacks: list[SourceFramesCallback] = list(self._frames_callbacks)
        for frames_callback in frames_callbacks:
            try:
                frames_callback(generated_poses, cam_timestamps)
            except Exception:
                logger.exception("TrackerFrameSource: a frames callback failed")

        return generated_poses

    def keypoint_window(self, roi: Rect, keypoints: np.ndarray) -> tuple[Rect, Points2D]:
        """The window the host crop would cut for this box, normalised to the frame, and the camera's
        keypoints normalised to it: the box padded, then grown to `crop_aspect` about its centre, the
        aspect judged in the frame's own proportions (`frame_aspect`)."""
        aspect: float = self.settings.crop_aspect
        padding: float = self.settings.crop_padding
        w: float = self.frame_aspect
        box = Rect(roi.x * w, roi.y, roi.width * w, roi.height).zoom(1.0 + padding)
        window = Rect(0.0, 0.0, aspect, 1.0).aspect_fill(box)
        x = (keypoints[:, 0] * w - window.x) / window.width
        y = (keypoints[:, 1] - window.y) / window.height
        points = Points2D.from_xy_arrays(x, y, keypoints[:, 2].astype(np.float32))
        return Rect(window.x / w, window.y, window.width / w, window.height), points

    def is_ready(self, now: float) -> bool:
        """Return True if at least one tracklet is ready for generation at `now`."""
        detection_timeout: float = self.settings.detection_timeout
        with self._lock:
            return any(tracklet is not None and not tracklet.is_expired(now, detection_timeout)
                       for tracklet in self._tracklets.values())

    def reset(self) -> None:
        """Reset all tracklets."""
        with self._lock:
            for track_id in range(self._max_players):
                self._tracklets[track_id] = None

    def reset_at(self, id_: int) -> None:
        """Reset tracklet for a specific track ID.

        Args:
            id_: Track ID to reset
        """
        with self._lock:
            if id_ in self._tracklets:
                self._tracklets[id_] = None