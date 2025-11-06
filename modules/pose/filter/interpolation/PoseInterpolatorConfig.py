
from modules.pose.filter.PoseFilterBase import PoseFilterConfigBase

class PoseInterpolatorConfig(PoseFilterConfigBase):
    """Configuration for pose interpolators with automatic change notification."""

    def __init__(self) -> None:
        super().__init__()
        self._frequency: float = 30.0
        self._alpha_v: float = 0.2
        self._friction: float = 0.03

    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = value
        self._notify()

    @property
    def alpha_v(self) -> float:
        return self._alpha_v

    @alpha_v.setter
    def alpha_v(self, value: float) -> None:
        self._alpha_v = value
        self._notify()

    @property
    def friction(self) -> float:
        return self._friction

    @friction.setter
    def friction(self, value: float) -> None:
        self._friction = value
        self._notify()