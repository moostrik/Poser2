from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from modules.pose.Pose import Pose, PoseDict
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.Settings import Settings


@dataclass
class SmootherSettings:
    """Configuration for OneEuroFilter-based smoothing."""
    frequency: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.025
    d_cutoff: float = 1.0
    reset_on_reappear: bool = False


class PoseSmoothFilterBase(PoseFilterBase):
    """Base class for pose data smoothing using OneEuroFilter.

    Handles:
    - Per-tracklet filter state management
    - Tracklet lifecycle (initialization and cleanup)
    - Common pose processing loop

    Subclasses implement:
    - Filter initialization for new tracklets
    - Smoothing logic for specific data types (points, angles, bbox)
    - Settings update propagation to filters
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = SmootherSettings(
            frequency=settings.camera_fps,
            min_cutoff=2.0,
            beta=0.05,
            d_cutoff=1.0,
            reset_on_reappear=False
        )
        # Per-tracklet state: tracklet_id -> filter state (managed by subclasses)
        self._tracklets: dict[int, Any] = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth data for all poses."""
        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Initialize tracklet filters if needed
            if tracklet_id not in self._tracklets:
                self._tracklets[tracklet_id] = self._create_tracklet_state()

            # Smooth the pose data
            smoothed_pose = self._smooth_pose(pose, tracklet_id)
            smoothed_poses[pose_id] = smoothed_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]

        self._notify_callbacks(smoothed_poses)

    @abstractmethod
    def _create_tracklet_state(self) -> Any:
        """Create initial filter state for a new tracklet.

        Returns:
            Tracklet-specific state (filters, validity masks, etc.)
        """
        pass

    @abstractmethod
    def _smooth_pose(self, pose: Pose, tracklet_id: int) -> Pose:
        """Apply smoothing to a single pose.

        Args:
            pose: Input pose to smooth
            tracklet_id: ID of the tracklet (for accessing filter state)

        Returns:
            Smoothed pose with updated data
        """
        pass

    @abstractmethod
    def _update_tracklet_filters(self, tracklet_state: Any) -> None:
        """Update filter parameters for a tracklet's filters.

        Args:
            tracklet_state: The tracklet's filter state
        """
        pass

    def update_settings(self, settings: SmootherSettings) -> None:
        """Update filter parameters for all tracklets."""
        self.settings = settings
        for tracklet_state in self._tracklets.values():
            self._update_tracklet_filters(tracklet_state)