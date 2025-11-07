# Standard library imports
from abc import abstractmethod
from typing import Any
from enum import Enum

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase
from modules.pose.filter.prediction.PosePredictorConfig import PosePredictorConfig

from modules.pose.Pose import Pose

class Features(Enum):
    """Enumeration for different pose features."""
    float_array = 1
    angle_array = 2
    float = 3




class PosePredictorBase(PoseFilterBase):

    def __init__(self, config: PosePredictorConfig) -> None:
        super().__init__(config)
        self._config: PosePredictorConfig = config
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