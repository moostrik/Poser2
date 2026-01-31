from threading import Lock
from traceback import print_exc

from modules.pose.callback import PoseDictCallbackMixin
from modules.pose.Frame import Frame, FrameDict
from modules.pose.features import BBox
from modules.tracker.Tracklet import Tracklet


class PosesFromTracklets(PoseDictCallbackMixin):
    """Generates poses from tracklets, maintaining state per track."""

    def __init__(self, num_tracks: int) -> None:
        super().__init__()
        self._num_tracks = num_tracks
        # Store submitted tracklets per track ID
        self._tracklets: dict[int, Tracklet | None] = {
            track_id: None for track_id in range(num_tracks)
        }
        self._lock = Lock()
        self._batch_id_counter = 0

    def submit_tracklets(self, tracklet_dict: dict[int, Tracklet]) -> None:
        """Set tracklets for pose generation.

        Args:
            tracklet_dict: Dictionary of track_id -> Tracklet
        """
        with self._lock:
            # Update all track slots
            for track_id in range(self._num_tracks):
                if track_id in tracklet_dict:
                    self._tracklets[track_id] = tracklet_dict[track_id]
                else:
                    self._tracklets[track_id] = None

    def generate(self) -> FrameDict:
        """Generate poses from all ready tracklets.

        Returns:
            Dictionary of track_id -> Frame for tracks with valid tracklets
        """
        # Copy tracklets under lock to avoid holding lock during processing
        with self._lock:
            tracklets_snapshot = self._tracklets.copy()
            batch_id = self._batch_id_counter
            self._batch_id_counter += 1

        generated_poses: FrameDict = {}

        for track_id, tracklet in tracklets_snapshot.items():
            if tracklet is None:
                continue

            try:
                bounding_box = BBox.from_rect(tracklet.roi)

                generated_poses[track_id] = Frame(
                    track_id=tracklet.id,
                    cam_id=tracklet.cam_id,
                    batch_id=batch_id,
                    bbox=bounding_box,
                    time_stamp=tracklet.time_stamp,
                    is_removed=tracklet.is_removed
                )
            except Exception as e:
                print(f"PoseFromTrackletGenerator: Error generating pose {track_id}: {e}")
                print_exc()

        self._notify_poses_callbacks(generated_poses)

        return generated_poses

    def is_ready(self) -> bool:
        """Return True if at least one tracklet is ready for generation."""
        with self._lock:
            return any(tracklet is not None for tracklet in self._tracklets.values())

    def reset(self) -> None:
        """Reset all tracklets."""
        with self._lock:
            for track_id in range(self._num_tracks):
                self._tracklets[track_id] = None

    def reset_at(self, id: int) -> None:
        """Reset tracklet for a specific track ID.

        Args:
            id: Track ID to reset
        """
        with self._lock:
            if id in self._tracklets:
                self._tracklets[id] = None