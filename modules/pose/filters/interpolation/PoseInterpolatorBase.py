from abc import abstractmethod
from typing import Any
from modules.pose.Pose import Pose, PoseDict
from modules.pose.filters.PoseFilterBase import PoseFilterBase

from modules.Settings import Settings


class PoseInterpolaterBase(PoseFilterBase):
    """Base class for pose data interpolation with separated input and output rates.

    Separates input sampling (at camera FPS) from output generation (at display/processing FPS):
    - add_poses(): Feeds input samples to interpolators (called at input_rate)
    - update(): Generates interpolated output (called at output_rate)

    Handles:
    - Per-tracklet filter state management
    - Tracklet lifecycle (initialization and cleanup)
    - Dual-rate processing (input vs output)

    Subclasses implement:
    - Filter initialization for new tracklets
    - Sample addition logic for specific data types
    - Interpolation/update logic for generating output
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        # Per-tracklet state: tracklet_id -> filter state (managed by subclasses)
        self._tracklets: dict[int, Any] = {}

        # Store last input poses for reconstruction during update
        self._last_poses: PoseDict = {}

        self.input_rate: float = settings.camera_fps
        self.alpha_v: float = 0.45  # Default velocity smoothing factor

    def add_poses(self, poses: PoseDict) -> None:
        """Add input samples to interpolators (called at input rate).

        Args:
            poses: Dictionary of poses to add as samples

        Note:
            This does NOT generate output or notify callbacks.
            Call update() separately to generate interpolated output.
        """
        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Initialize tracklet filters if needed
            if tracklet_id not in self._tracklets:
                self._tracklets[tracklet_id] = self._create_tracklet_state()

            # Add sample to interpolator
            self._add_sample(pose, tracklet_id)

            # Store for reconstruction during update
            self._last_poses[pose_id] = pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]
                if pose_id in self._last_poses:
                    del self._last_poses[pose_id]

    def update(self, current_time: float | None = None) -> None:
        """Generate interpolated output at current time (called at output rate).

        Args:
            current_time: Optional explicit time for interpolation.
                         If None, interpolators use their internal time tracking.

        Note:
            Generates interpolated poses for all active tracklets and
            notifies callbacks with the results.
        """
        interpolated_poses: PoseDict = {}

        for pose_id, pose in self._last_poses.items():
            tracklet_id: int = pose.tracklet.id

            # Skip if tracklet state was cleaned up
            if tracklet_id not in self._tracklets:
                continue

            # Generate interpolated pose
            interpolated_pose: Pose = self._interpolate(pose, tracklet_id, current_time)
            interpolated_poses[pose_id] = interpolated_pose

        # Notify callbacks with interpolated output
        if interpolated_poses:
            self._notify_callbacks(interpolated_poses)

    @abstractmethod
    def _create_tracklet_state(self) -> Any:
        """Create initial filter state for a new tracklet.

        Returns:
            Tracklet-specific state (filters, validity masks, etc.)
        """
        pass

    @abstractmethod
    def _add_sample(self, pose: Pose, tracklet_id: int) -> None:
        """Add input sample to tracklet's interpolators.

        Args:
            pose: Input pose to sample
            tracklet_id: ID of the tracklet (for accessing filter state)
        """
        pass

    @abstractmethod
    def _interpolate(self, pose: Pose, tracklet_id: int, current_time: float | None) -> Pose:
        """Generate interpolated pose at current time.

        Args:
            pose: Last input pose (used as template for reconstruction)
            tracklet_id: ID of the tracklet (for accessing filter state)
            current_time: Optional explicit time for interpolation

        Returns:
            Interpolated pose with updated data
        """
        pass
