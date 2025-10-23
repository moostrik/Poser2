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
from modules.pose.correlation.PoseSmoothCorrelator import PoseSmoothCorrelator, PairCorrelation, PairCorrelationBatch
from modules.pose.correlation.PairCorrelationStream import PairCorrelationStream, PairCorrelationStreamData
from modules.pose.features.PoseAngles import AngleJoint
from modules.pose.smooth.PoseSmoothBase import PoseSmoothBase
from modules.pose.smooth.PoseSmoothRect import PoseSmoothRect, PoseSmoothRectSettings
from modules.pose.smooth.PoseSmoothAngles import PoseSmoothAngles, PoseSmoothAngleSettings, SymmetricJointType
from modules.Settings import Settings
from modules.utils.OneEuroInterpolation import OneEuroSettings
from modules.utils.PointsAndRects import Rect
from modules.pose.correlation.PairCorrelation import PairCorrelationBatch

from modules.utils.HotReloadMethods import HotReloadMethods

class PoseSmoothDataManager:
    def __init__(self, settings: Settings) -> None:
        self._num_players: int = settings.num_players

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

        # Dictionaries to store smoothers for each tracklet ID
        self._rect_smoothers: dict[int, PoseSmoothRect] = {}
        self._angle_smoothers: dict[int, PoseSmoothAngles] = {}

        for i in range(self._num_players):
            self._rect_smoothers[i] = PoseSmoothRect(self.rect_settings)
            self._angle_smoothers[i] = PoseSmoothAngles(self.angle_settings)
        self._all_smoothers: list[Mapping[int, PoseSmoothBase]] = [self._rect_smoothers, self._angle_smoothers]

        self.smooth_correlator: PoseSmoothCorrelator = PoseSmoothCorrelator(settings)

        self.smooth_correlation_stream: PairCorrelationStream = PairCorrelationStream(60 * 10, 0.5)
        self.smooth_correlator.add_correlation_callback(self.smooth_correlation_stream.add_correlation)
        self.motion_correlation: PairCorrelationBatch = PairCorrelationBatch([])

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def start(self) -> None:
        self.smooth_correlator.start()
        self.smooth_correlation_stream.start()

    def stop(self) -> None:
        self.smooth_correlator.stop()
        self.smooth_correlation_stream.stop()

    def update(self) -> None:
        """Update all active smoothers."""
        with self._lock:
            for smoothers in self._all_smoothers:
                for smoother in smoothers.values():
                    if smoother.is_active:
                        smoother.update()
        self.smooth_correlator.set_input_data(self.get_active_angles())

    def reset(self) -> None:
        """Reset all smoothers."""
        with self._lock:
            for smoothers in self._all_smoothers:
                for smoother in smoothers.values():
                    smoother.reset()

    def add_pose(self, pose: Pose) -> None:
        """ Add a new pose data point for processing."""
        with self._lock:
            tracklet_id: int = pose.tracklet.id
            for smoothers in self._all_smoothers:
                smoothers[tracklet_id].add_pose(pose)

    def get_correlation_streams(self) -> PairCorrelationStreamData | None:
        """ Get the correlation stream data."""
        with self._lock:
            return self.smooth_correlation_stream.get_stream_data()

    def set_correlation_batch(self, batch: PairCorrelationBatch) -> None:
        """ Set the correlation data for synchrony calculations."""
        with self._lock:
            self.motion_correlation = batch
            # print(f"Set correlation: {self.correlation}")

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
    def get_pose_correlation_batch(self) -> PairCorrelationBatch | None:
        return self.smooth_correlator.get_output_data()

    def get_pose_correlation(self, id1: int, id2: int) -> float:
        """Get the correlation value between two tracklet IDs."""
        batch: PairCorrelationBatch | None = self.get_pose_correlation_batch()
        if batch is None:
            return 0.0
        return batch.get_mean_correlation_for_pair((id1, id2))

    def get_motion_correlation(self, id1: int, id2: int) -> float:
        """Get the correlation value between two tracklet IDs."""
        with self._lock:
            return self.motion_correlation.get_mean_correlation_for_pair((id1, id2))

    # CONVENIENCE METHODS
    def get_active_angles(self) -> dict[int, dict[AngleJoint, float]]:
        """Get smoothed angles for all active tracklets."""
        active_angles: dict[int, dict[AngleJoint, float]] = {}
        with self._lock:
            for tracklet_id, smoother in self._angle_smoothers.items():
                if smoother.is_active:
                    active_angles[tracklet_id] = smoother.angles
        return active_angles