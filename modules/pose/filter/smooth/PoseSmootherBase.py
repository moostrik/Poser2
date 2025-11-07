# Standard library imports
from abc import abstractmethod
from typing import Any

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.Pose import Pose


class PoseSmootherConfig(PoseFilterConfigBase):
    """Configuration for OneEuroFilter-based smoothing with automatic change notification."""

    def __init__(self,
                 frequency: float = 30.0,
                 min_cutoff: float = 1.0,
                 beta: float = 0.025,
                 d_cutoff: float = 1.0,
                 reset_on_reappear: bool = False) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff
        self.reset_on_reappear: bool = reset_on_reappear


class PoseSmootherBase(PoseFilterBase):
    """
    Base class for single-pose smoothing filters using OneEuroFilter.

    Handles:
    - Filter state management for a single pose
    - Config change notifications

    Subclasses implement:
    - Filter initialization for a new pose
    - Smoothing logic for specific data types (points, angles, bbox)
    - Config change handling via _on_config_changed()
    """

    def __init__(self, config: PoseSmootherConfig) -> None:
        super().__init__(config)
        self._config: PoseSmootherConfig = config
        # State for the current pose (managed by subclasses)
        self._state: Any = None

    def process(self, pose: Pose) -> Pose:
        """Smooth data for a single pose."""
        # Initialize filter state if needed
        if self._state is None:
            self._state = self._create_state()

        # Smooth the pose data
        smoothed_pose: Pose = self._smooth(pose, self._state)

        return smoothed_pose

    @abstractmethod
    def _create_state(self) -> Any:
        """Create initial filter state for a new pose."""
        pass

    @abstractmethod
    def _smooth(self, pose: Pose, state: Any) -> Pose:
        """Apply smoothing to a single pose."""
        pass

    def reset(self) -> None:
        """Reset the filter's internal state."""
        self._state = None

    # Note: Subclasses should override _on_config_changed() to update their state