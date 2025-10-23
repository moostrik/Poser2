"""
Manages multiple pose smoothing components for different poses and tracklets.
Uses tracklet IDs as keys to track different poses.

Key capabilities:
1. Coordinates rect, angle, and head smoothing operations
2. Provides thread-safe access to smoothed pose data
3. Coordinates underlying smoothers that enable higher framerate output than input
   through interpolation (queryable at higher rates than the input frequency)
4. Note: Interpolation introduces a latency of approximately 1 input frame
"""

from threading import Lock
from collections.abc import Mapping

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.smooth.PoseSmoothBase import PoseSmoothBase
from modules.pose.smooth.PoseSmoothRect import PoseSmoothRect, PoseSmoothRectSettings
from modules.pose.smooth.PoseSmoothAngles import PoseSmoothAngles, PoseSmoothAngleSettings, SymmetricJointType
from modules.pose.smooth.PoseSmoothHead import PoseSmoothHead, PoseSmoothHeadSettings
from modules.utils.OneEuroInterpolation import OneEuroSettings
from modules.utils.PointsAndRects import Rect
from modules.pose.correlation.PairCorrelation import PairCorrelationBatch

from modules.utils.HotReloadMethods import HotReloadMethods

class PoseSmoothDataManager:
    def __init__(self, num_players: int) -> None:
        self._num_players: int = num_players

        self.OneEuro_settings: OneEuroSettings = OneEuroSettings(25, 1.0, 0.1)
        self.rect_settings: PoseSmoothRectSettings = PoseSmoothRectSettings(
            smooth_settings=self.OneEuro_settings,
            center_dest_x=0.5,
            centre_dest_y=0.2,
            height_dest=0.95,
            dst_aspectratio=9/16
        )
        self.angle_settings: PoseSmoothAngleSettings = PoseSmoothAngleSettings(
            smooth_settings=self.OneEuro_settings,
            motion_threshold=0.002
        )
        self.head_settings: PoseSmoothHeadSettings = PoseSmoothHeadSettings(
            smooth_settings=self.OneEuro_settings,
            motion_threshold=0.002
        )

        # Dictionaries to store smoothers for each tracklet ID
        self._rect_smoothers: dict[int, PoseSmoothRect] = {}
        self._angle_smoothers: dict[int, PoseSmoothAngles] = {}
        self._head_smoothers: dict[int, PoseSmoothHead] = {}

        for i in range(num_players):
            self._rect_smoothers[i] = PoseSmoothRect(self.rect_settings)
            self._angle_smoothers[i] = PoseSmoothAngles(self.angle_settings)
            self._head_smoothers[i] = PoseSmoothHead(self.head_settings)
        self._all_smoothers: list[Mapping[int, PoseSmoothBase]] = [self._rect_smoothers, self._angle_smoothers, self._head_smoothers]

        self.correlation: PairCorrelationBatch = PairCorrelationBatch([])

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_pose(self, pose: Pose) -> None:
        """ Add a new pose data point for processing."""
        with self._lock:
            tracklet_id: int = pose.tracklet.id
            for smoothers in self._all_smoothers:
                smoothers[tracklet_id].add_pose(pose)

    def set_correlation_batch(self, batch: PairCorrelationBatch) -> None:
        """ Set the correlation data for synchrony calculations."""
        with self._lock:
            self.correlation = batch
            # print(f"Set correlation: {self.correlation}")

    def update(self) -> None:
        """Update all active smoothers."""
        with self._lock:
            for smoothers in self._all_smoothers:
                for smoother in smoothers.values():
                    if smoother.is_active:
                        smoother.update()

    def reset(self) -> None:
        """Reset all smoothers."""
        with self._lock:
            for smoothers in self._all_smoothers:
                for smoother in smoothers.values():
                    smoother.reset()

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
    def get_angle(self, tracklet_id: int, joint: PoseJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_angle(joint, symmetric=True)

    def get_velocity(self, tracklet_id: int, joint: PoseJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_velocity(joint, symmetric=True)

    def get_motion(self, tracklet_id: int, joint: PoseJoint) -> float:
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

    # HEAD ANGLE
    def get_head(self, tracklet_id: int) -> float:
        """Get smoothed head angles for the specified tracklet ID."""
        with self._lock:
            return self._head_smoothers[tracklet_id].angle

    def get_head_velocity(self, tracklet_id: int) -> float:
        """Get smoothed head angle changes for the specified tracklet ID."""
        with self._lock:
            return self._head_smoothers[tracklet_id].velocity

    def get_head_motion(self, tracklet_id: int) -> float:
        """Get smoothed head motion for the specified tracklet ID."""
        with self._lock:
            return self._head_smoothers[tracklet_id].motion

    # TIME
    def get_cumulative_motion(self, tracklet_id: int) -> float:
        """Get combined motion (body + head) for the specified tracklet ID."""
        with self._lock:
            body_motion: float = self._angle_smoothers[tracklet_id].cumulative_total_motion
            head_motion: float = self._head_smoothers[tracklet_id].cumulative_motion
            return body_motion + head_motion

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].age

    # CORRELATION (SHOULD NOT BE HERE)
    def get_pose_correlation(self, id1: int, id2: int) -> float:
        return 0.0

    def get_motion_correlation(self, id1: int, id2: int) -> float:
        """Get the correlation value between two tracklet IDs."""
        with self._lock:
            pair_id = (id1, id2) if id1 <= id2 else (id2, id1)
            return self.correlation.get_similarity(pair_id)