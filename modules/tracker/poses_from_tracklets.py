import logging
import math
from threading import Lock

from modules.pose.frame import Frame, FrameDict, FrameDictCallbackMixin
from modules.pose.features import BBox, BBoxAzimuth
from modules.settings import BaseSettings, Field
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


class PosesFromTracklets(FrameDictCallbackMixin):
    """Generates poses from tracklets, maintaining state per track."""

    def __init__(self, config: PosesFromTrackletsSettings, num_tracks: int) -> None:
        super().__init__()
        self._config: PosesFromTrackletsSettings = config
        # World ids are used directly as slot indices below, so this must match the tracker's
        # id pool exactly: a world id at or above `num_tracks` would vanish here without a
        # trace. Both come from `num_players` in main.py — an invariant spanning two modules,
        # so it is worth stating.
        if num_tracks <= 0:
            raise ValueError(f"PosesFromTracklets needs at least one track slot, got {num_tracks}")
        self._num_tracks = num_tracks
        # Store submitted tracklets per track ID
        self._tracklets: dict[int, Tracklet | None] = {
            track_id: None for track_id in range(num_tracks)
        }
        self._lock = Lock()
        self._batch_id_counter = 0

    def set_tracklets(self, tracklet_dict: dict[int, Tracklet]) -> None:
        """Set tracklets for pose generation.

        Args:
            tracklet_dict: Dictionary of track_id -> Tracklet
        """
        dropped: list[int] = [k for k in tracklet_dict if k >= self._num_tracks]
        if dropped:
            logger.warning(
                f"PosesFromTracklets: world ids {dropped} are beyond {self._num_tracks} slots "
                f"and produce no pose — the tracker's id pool is larger than num_tracks."
            )
        with self._lock:
            # Update all track slots
            for track_id in range(self._num_tracks):
                if track_id in tracklet_dict:
                    self._tracklets[track_id] = tracklet_dict[track_id]
                else:
                    self._tracklets[track_id] = None

    def process(self) -> FrameDict:
        """Generate poses from all ready tracklets.

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
        detection_timeout: float = self._config.detection_timeout

        for track_id, tracklet in tracklets_snapshot.items():
            if tracklet is None or tracklet.is_expired(detection_timeout):
                continue

            try:
                bounding_box = BBox.from_rect(tracklet.roi)
                world_angle = tracklet.annotation.world_angle if isinstance(tracklet.annotation, PanoramicAnnotation) else None
                features: dict = {BBox: bounding_box}
                if world_angle is not None:
                    # SingleAngle.from_value wraps degrees-as-radians to [-π, π).
                    features[BBoxAzimuth] = BBoxAzimuth.from_value(math.radians(float(world_angle)))

                generated_poses[track_id] = Frame(
                    track_id=tracklet.id,
                    cam_id=tracklet.cam_id,
                    time_stamp=tracklet.time_stamp,
                    features=features,
                )
            except Exception as e:
                logger.error(
                    f"PoseFromTrackletGenerator: Error generating pose "
                    f"{track_id}: {e}"
                )
        self._notify_frames_callbacks(generated_poses)

        return generated_poses

    def is_ready(self) -> bool:
        """Return True if at least one tracklet is ready for generation."""
        detection_timeout: float = self._config.detection_timeout
        with self._lock:
            return any(tracklet is not None and not tracklet.is_expired(detection_timeout)
                       for tracklet in self._tracklets.values())

    def reset(self) -> None:
        """Reset all tracklets."""
        with self._lock:
            for track_id in range(self._num_tracks):
                self._tracklets[track_id] = None

    def reset_at(self, id_: int) -> None:
        """Reset tracklet for a specific track ID.

        Args:
            id_: Track ID to reset
        """
        with self._lock:
            if id_ in self._tracklets:
                self._tracklets[id_] = None