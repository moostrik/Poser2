
from modules.pose.filters.PoseFilterBase import PoseFilterConfigBase

class PoseInterpolatorConfig(PoseFilterConfigBase):
    """Configuration for pose interpolators with automatic change notification."""

    def __init__(self) -> None:
        super().__init__()
        self.frequency: float = 30.0
        self.responsiveness: float = 0.2
        self.friction: float = 0.03