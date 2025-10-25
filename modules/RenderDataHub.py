"""
Manages multiple pose smoothing components for different poses and tracklets.
Uses tracklet IDs as keys to track different poses.

Key capabilities:
1. Coordinates rect, angle, and head smoothing operations
2. Provides thread-safe access to smoothed pose data
3. Coordinates underlying smoothers that enable higher framerate output than input
   through interpolation (queryable at higher rates than the input frequency)
4. Manages pose correlation analysis
5. Note: Interpolation introduces a latency of approximately 1 input frame
"""

from threading import Lock
from collections.abc import Mapping

from modules.pose.Pose import Pose, PoseDict
from modules.pose.features.PoseAngles import AngleJoint
from modules.pose.interpolation.PoseInterpolationBase import PoseInterpolationBase
from modules.pose.interpolation.PoseViewportInterpolator import PoseViewportInterpolator, PoseViewportInterpolatorSettings
from modules.pose.interpolation.PoseKinematicsInterpolator import PoseKinematicsInterpolator, PoseKinematicsInterpolatorSettings, SymmetricJointType
from modules.Settings import Settings
from modules.utils.OneEuroInterpolation import OneEuroSettings, OneEuroInterpolator
from modules.utils.PointsAndRects import Rect
from modules.pose.correlation.PairCorrelation import PairCorrelationBatch, SimilarityMetric

from modules.utils.HotReloadMethods import HotReloadMethods

class RenderDataHub:
    def __init__(self, settings: Settings) -> None:
        self._num_players: int = settings.num_players

        self.OneEuro_settings: OneEuroSettings = OneEuroSettings(25, 1.0, 0.1)
        self.rect_settings: PoseViewportInterpolatorSettings = PoseViewportInterpolatorSettings(
            smooth_settings=self.OneEuro_settings,
            center_dest_x=0.5,
            centre_dest_y=0.2,
            height_dest=0.95,
            dst_aspectratio=9/16
        )
        self.angle_settings: PoseKinematicsInterpolatorSettings = PoseKinematicsInterpolatorSettings(
            smooth_settings=self.OneEuro_settings,
            motion_threshold=0.002
        )

        # Dictionaries to store smoothers for each tracklet ID
        self._rect_smoothers: dict[int, PoseViewportInterpolator] = {}
        self._angle_smoothers: dict[int, PoseKinematicsInterpolator] = {}

        for i in range(self._num_players):
            self._rect_smoothers[i] = PoseViewportInterpolator(self.rect_settings)
            self._angle_smoothers[i] = PoseKinematicsInterpolator(self.angle_settings)
        self._all_smoothers: list[Mapping[int, PoseInterpolationBase]] = [self._rect_smoothers, self._angle_smoothers]

        # Correlation smoothing (automatically managed)
        self._pose_correlation_smoothers: dict[tuple[int, int], OneEuroInterpolator] = {}
        self._motion_correlation_smoothers: dict[tuple[int, int], OneEuroInterpolator] = {}

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self) -> None:
        """Update all active smoothers."""
        with self._lock:
            # Update pose/rect smoothers
            for smoothers in self._all_smoothers:
                for smoother in smoothers.values():
                    if smoother.is_active:
                        smoother.update()

            # Update correlation smoothers
            for smoother in self._pose_correlation_smoothers.values():
                smoother.update()
            for smoother in self._motion_correlation_smoothers.values():
                smoother.update()

    def reset(self) -> None:
        """Reset all smoothers."""
        with self._lock:
            for smoothers in self._all_smoothers:
                for smoother in smoothers.values():
                    smoother.reset()

            # Reset correlation smoothers
            for smoother in self._pose_correlation_smoothers.values():
                smoother.reset()
            for smoother in self._motion_correlation_smoothers.values():
                smoother.reset()

    def add_poses(self, poses: PoseDict) -> None:
        """ Add a new pose data point for processing."""
        with self._lock:
            for pose in poses.values():
                tracklet_id: int = pose.tracklet.id
                for smoothers in self._all_smoothers:
                    smoothers[tracklet_id].add_pose(pose)

    def add_pose_correlation(self, batch: PairCorrelationBatch) -> None:
        """Add new pose correlation and  manage smoothers.
           Smoothers for pairs not in batch are automatically removed.
        """

        with self._lock:
            batch_pair_ids: set[tuple[int, int]] = {pair.pair_id for pair in batch}
            pairs_to_remove: set[tuple[int, int]] = set(self._pose_correlation_smoothers.keys()) - batch_pair_ids

            for pair_id in pairs_to_remove:
                del self._pose_correlation_smoothers[pair_id]

            for pair in batch:
                pair_id: tuple[int, int] = pair.pair_id
                if pair_id not in self._pose_correlation_smoothers:
                    self._pose_correlation_smoothers[pair_id] = OneEuroInterpolator(self.OneEuro_settings)
                self._pose_correlation_smoothers[pair_id].add_sample(pair.geometric_mean)

    def add_motion_correlation(self, batch: PairCorrelationBatch) -> None:
        """Add new motion correlation and manage smoothers.
           Smoothers for pairs not in batch are automatically removed.
        """
        with self._lock:
            batch_pair_ids: set[tuple[int, int]] = {pair.pair_id for pair in batch}
            pairs_to_remove: set[tuple[int, int]] = set(self._motion_correlation_smoothers.keys()) - batch_pair_ids

            for pair_id in pairs_to_remove:
                del self._motion_correlation_smoothers[pair_id]

            for pair in batch:
                pair_id: tuple[int, int] = pair.pair_id
                if pair_id not in self._motion_correlation_smoothers:
                    self._motion_correlation_smoothers[pair_id] = OneEuroInterpolator(self.OneEuro_settings)
                self._motion_correlation_smoothers[pair_id].add_sample(pair.geometric_mean)

    # ACTIVE
    def get_is_active(self, tracklet_id: int) -> bool:
        """Check if the smoother for the specified tracklet ID is active."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].is_active

    # RECT
    def get_rect(self, tracklet_id: int) -> Rect:
        """Get smoothed rectangle for the specified tracklet ID."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].smoothed_rect

    #  BODY JOINT ANGLES
    def get_angle(self, tracklet_id: int, joint: AngleJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_angle(joint, symmetric=True)

    def get_velocity(self, tracklet_id: int, joint: AngleJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_velocity(joint, symmetric=True)

    def get_motion(self, tracklet_id: int, joint: AngleJoint) -> float:
        """Get smoothed angle change for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_motion(joint)

    # BODY JOINT SYMMETRY
    def get_symmetry(self, tracklet_id: int, type: SymmetricJointType) -> float:
        """Get the synchrony value for the specified symmetric joint type."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_symmetry(type)

    def get_mean_symmetry(self, tracklet_id: int) -> float:
        """Get the mean synchrony value across all symmetric joint types."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].mean_symmetry

    # TIME
    def get_cumulative_motion(self, tracklet_id: int) -> float:
        """Get combined motion (body + head) for the specified tracklet ID."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].cumulative_total_motion

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].age

    # CORRELATION
    def get_pose_correlation(self, id1: int, id2: int) -> float:
        """Get smoothed pose correlation between two tracklets."""
        with self._lock:
            pair_id: tuple[int, int] = (min(id1, id2), max(id1, id2))
            smoother: OneEuroInterpolator | None = self._pose_correlation_smoothers.get(pair_id)
            if smoother is None:
                return 0.0
            value: float | None = smoother.smooth_value
            return value if value is not None else 0.0

    def get_motion_correlation(self, id1: int, id2: int) -> float:
        """Get smoothed motion correlation between two tracklets."""
        with self._lock:
            pair_id: tuple[int, int] = (min(id1, id2), max(id1, id2))
            smoother: OneEuroInterpolator | None = self._motion_correlation_smoothers.get(pair_id)
            if smoother is None:
                return 0.0
            value: float | None = smoother.smooth_value
            return value if value is not None else 0.0
