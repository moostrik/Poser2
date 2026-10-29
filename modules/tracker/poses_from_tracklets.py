import logging
import math
from threading import Lock

import numpy as np

from modules.pose import Frame, FrameDict, FrameDictCallbackMixin, BBox, BBoxAzimuth, Distance, Points2D
from modules.settings import BaseSettings, Field
from modules.utils import Rect
from .tracklet import Tracklet
from .panoramic.annotation import Annotation as PanoramicAnnotation

logger = logging.getLogger(__name__)


class PosesFromTrackletsSettings(BaseSettings):
    # How long after their last detection a person is still cropped from their last box. Judged here
    # rather than in the tracker, which emits everyone it still remembers: inside this window a
    # person the detector dropped keeps their pose, so a missed detection neither interrupts it nor
    # resets the filters downstream. Not below a few frames: a fresh detection is already some
    # milliseconds old by the time the frame bang crops it, so at 0 nobody would ever be posed.
    detection_timeout: Field[float] = Field(2.0, min=0.1, max=5.0, step=0.05,
                                            description="Seconds after the last detection a person still gets a pose")
    # A camera that runs the pose model itself (OAK4) hands its keypoints over in the frame; the pose
    # carries them in the host crop's convention, normalised to a window of this aspect around the
    # box, padded like the host crop, and the box becomes that window (`modules.inference.crop`).
    crop_aspect:       Field[float] = Field(0.75, min=0.25, max=4.0, step=0.05,
                                            description="Width over height of the window the camera's keypoints are normalised to (the host crop's 384/512)")
    crop_padding:      Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05,
                                            description="Margin around the box before the window is fitted (fraction; the host crop's padding)")


class PosesFromTracklets(FrameDictCallbackMixin):
    """Generates poses from tracklets, maintaining state per track."""

    def __init__(self, settings: PosesFromTrackletsSettings, max_players: int, frame_aspect: float = 1.0) -> None:
        """`frame_aspect` is the delivered frame's width over its height, the one thing the keypoint
        window needs to know about the image; it is 1.0 for the square frames of the rigs."""
        super().__init__()
        self.settings: PosesFromTrackletsSettings = settings
        self.frame_aspect: float = frame_aspect
        # The incoming dict keys are used directly as slot indices below, so the feeding tracker
        # must key by a slot under `max_players` (the panoramic tracker's player slots; cam ids for
        # one-per-cam rigs): a key at or above it would vanish here without a trace. Both come from
        # `max_players` in main.py — an invariant spanning two modules, so it is worth stating.
        if max_players <= 0:
            raise ValueError(f"PosesFromTracklets needs at least one track slot, got {max_players}")
        self._max_players = max_players
        # Store submitted tracklets per track ID
        self._tracklets: dict[int, Tracklet | None] = {
            track_id: None for track_id in range(max_players)
        }
        self._lock = Lock()
        self._batch_id_counter = 0

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
                f"PosesFromTracklets: track ids {dropped} are beyond {self._max_players} slots "
                f"and produce no pose — the feeding tracker has more slots than max_players."
            )
        with self._lock:
            # Update all track slots
            for track_id in range(self._max_players):
                if track_id in tracklet_dict:
                    self._tracklets[track_id] = tracklet_dict[track_id]
                else:
                    self._tracklets[track_id] = None

    def process(self, now: float) -> FrameDict:
        """Generate poses from all ready tracklets; `now` is the time of the frame bang that asks,
        the clock a latched box's age is judged on.

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
            if tracklet is None or tracklet.is_expired(now, detection_timeout):
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
                    f"PosesFromTracklets: Error generating pose "
                    f"{track_id}: {e}"
                )
        self._notify_frames_callbacks(generated_poses)

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