"""
Provides smoothed pose data for rendering at output framerate.

Manages pose tracking components that smooth noisy input and interpolate between
samples, enabling higher framerate output than input (e.g., 24 FPS input → 60 FPS output).

Counterpart to CaptureDataHub:
- CaptureDataHub: Stores raw pose data at capture rate (input FPS)
- RenderDataHub: Provides smoothed pose data at render rate (output FPS)

Key capabilities:
1. Coordinates viewport, angle, and correlation tracking
2. Provides thread-safe access to smoothed pose data
3. Enables higher framerate output through interpolation
4. Manages pose correlation analysis between tracklets

Note: Interpolation introduces latency of approximately 1 input frame.
"""

from threading import Lock
from collections.abc import Mapping

from modules.pose.Pose import PoseDict
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.features.PoseAngleSymmetry import PoseAngleSymmetryData
from modules.pose.trackers.PoseViewportTracker import PoseViewportTracker, PoseViewportTrackerSettings
from modules.pose.trackers.PoseAngleTracker import PoseAngleTracker, PoseAngleTrackerSettings
from modules.Settings import Settings
from modules.utils.depricated.SmoothedInterpolator import OneEuroSettings, SmoothedInterpolator
from modules.utils.PointsAndRects import Rect
from modules.pose.features.PoseAngleSimilarity import PoseSimilarityBatch

from modules.utils.HotReloadMethods import HotReloadMethods

class RenderDataHub_Old:
    def __init__(self, settings: Settings) -> None:
        self._num_players: int = settings.num_players
        fps: float = settings.camera_fps

        self.one_euro_settings: OneEuroSettings = OneEuroSettings(fps, 1.0, 0.1)
        self.viewport_settings: PoseViewportTrackerSettings = PoseViewportTrackerSettings(
            smooth_settings=self.one_euro_settings,
            center_dest_x=0.5,
            centre_dest_y=0.2,
            height_dest=0.95,
            dst_aspectratio=9/16
        )
        self.angle_settings: PoseAngleTrackerSettings = PoseAngleTrackerSettings(
            smooth_settings=self.one_euro_settings,
            motion_threshold=0.003
        )

        # Dictionaries to store smoothers for each tracklet ID
        self._viewport_trackers: dict[int, PoseViewportTracker] = {}
        self._angle_trackers: dict[int, PoseAngleTracker] = {}

        for i in range(self._num_players):
            self._viewport_trackers[i] = PoseViewportTracker(self.viewport_settings)
            self._angle_trackers[i] = PoseAngleTracker(self.angle_settings)

        # Correlation smoothing (automatically managed)
        self._pose_correlation_smoothers: dict[tuple[int, int], SmoothedInterpolator] = {}
        self._motion_correlation_smoothers: dict[tuple[int, int], SmoothedInterpolator] = {}

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self) -> None:
        """Update all active trackers and smoothers."""
        with self._lock:
            for tracker in self._viewport_trackers.values():
                if tracker.is_active:
                    tracker.update()
            for tracker in self._angle_trackers.values():
                if tracker.is_active:
                    tracker.update()

            for smoother in self._pose_correlation_smoothers.values():
                smoother.update()
            for smoother in self._motion_correlation_smoothers.values():
                smoother.update()

    def reset(self) -> None:
        """Reset all trackers and smoothers."""
        with self._lock:
            for tracker in self._viewport_trackers.values():
                tracker.reset()
            for tracker in self._angle_trackers.values():
                tracker.reset()

            self._pose_correlation_smoothers.clear()
            self._motion_correlation_smoothers.clear()

    def add_poses(self, poses: PoseDict) -> None:
        """ Add a new pose data point for processing."""
        with self._lock:
            for pose in poses.values():
                tracklet_id: int = pose.tracklet.id

                self._viewport_trackers[tracklet_id].add_pose(pose)
                self._angle_trackers[tracklet_id].add_pose(pose)

    def _add_correlation(self, batch: PoseSimilarityBatch , smoothers: dict[tuple[int, int], SmoothedInterpolator]
    ) -> None:
        """Helper to add correlation data and manage smoother lifecycle."""
        batch_pair_ids: set[tuple[int, int]] = {pair.pair_id for pair in batch}
        pairs_to_remove: set[tuple[int, int]] = set(smoothers.keys()) - batch_pair_ids

        for pair_id in pairs_to_remove:
            del smoothers[pair_id]

        for pair in batch:
            if pair.pair_id not in smoothers:
                smoothers[pair.pair_id] = SmoothedInterpolator(self.one_euro_settings)
            smoothers[pair.pair_id].add_sample(pair.geometric_mean)

    def add_pose_correlation(self, batch: PoseSimilarityBatch ) -> None:
        """Add new pose correlation and manage smoothers.
           Smoothers for pairs not in batch are automatically removed.
        """

        with self._lock:
            self._add_correlation(batch, self._pose_correlation_smoothers)

    def add_motion_correlation(self, batch: PoseSimilarityBatch ) -> None:
        """Add new motion correlation and manage smoothers.
           Smoothers for pairs not in batch are automatically removed.
        """
        with self._lock:
            self._add_correlation(batch, self._motion_correlation_smoothers)

    # ACTIVE
    def get_is_active(self, tracklet_id: int) -> bool:
        """Check if the smoother for the specified tracklet ID is active."""
        with self._lock:
            return self._viewport_trackers[tracklet_id].is_active

    # RECT
    def get_viewport(self, tracklet_id: int) -> Rect:
        """Get smoothed rectangle for the specified tracklet ID."""
        with self._lock:
            return self._viewport_trackers[tracklet_id].smoothed_rect

    #  BODY JOINT ANGLES
    def get_angles(self, tracklet_id: int) -> PoseAngleData:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_trackers[tracklet_id].angles

    def get_velocities(self, tracklet_id: int) -> PoseAngleData:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_trackers[tracklet_id].velocities

    def get_motions(self, tracklet_id: int) -> PoseAngleData:
        """Get smoothed angle change for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_trackers[tracklet_id].motions

    def get_symmetries(self, tracklet_id: int) -> PoseAngleSymmetryData:
        """Get the synchrony value for the specified symmetric joint type."""
        with self._lock:
            return self._angle_trackers[tracklet_id].symmetries

    # TIME
    def get_cumulative_motion(self, tracklet_id: int) -> float:
        """Get combined motion (body + head) for the specified tracklet ID."""
        with self._lock:
            return self._angle_trackers[tracklet_id].cumulative_total_motion

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        with self._lock:
            return self._viewport_trackers[tracklet_id].age

    # CORRELATION
    def _get_correlation(self, id1: int, id2: int, smoothers: dict[tuple[int, int], SmoothedInterpolator]) -> float:
        """Helper to get smoothed correlation value."""
        pair_id: tuple[int, int] = (min(id1, id2), max(id1, id2))
        smoother: SmoothedInterpolator | None = smoothers.get(pair_id)
        if smoother is None:
            return 0.0
        value: float = smoother.smooth_value
        return value if value is not None else 0.0

    def get_pose_correlation(self, id1: int, id2: int) -> float:
        with self._lock:
            return self._get_correlation(id1, id2, self._pose_correlation_smoothers)

    def get_motion_correlation(self, id1: int, id2: int) -> float:
        with self._lock:
            return self._get_correlation(id1, id2, self._motion_correlation_smoothers)
