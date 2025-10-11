import numpy as np

from modules.pose.Pose import Pose
from modules.pose.smooth.PoseSmoothAngles import PoseSmoothAngleSettings

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings

from modules.utils.HotReloadMethods import HotReloadMethods

# CLASSES
class PoseSmoothHead():
    def __init__(self, settings: PoseSmoothAngleSettings) -> None:
        self._active: bool = False
        self.settings: PoseSmoothAngleSettings = settings
        self._head_smoother: AngleEuroInterpolator = AngleEuroInterpolator(settings.smooth_settings)

        self._orientation: float = 0.0
        self._delta: float = 0.0
        self._motion: float = 0.0

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def orientation(self) -> float:
        return self._orientation

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def motion(self) -> float:
        return self._motion

    def add_pose(self, pose: Pose) -> None:
        if pose.tracklet.is_removed:
            self._active = False
            self.reset()
            return

        if pose.tracklet.is_active and not self._active:
            self._active = True

        if not self._active:
            return

        # Always add data, OneEuroInterpolator will handle missing data
        if pose.head_data is None:
            self._head_smoother.add_sample(np.nan)
        else:
            self._head_smoother.add_sample(pose.head_data.yaw)

    def update(self) -> None:
        if not self._active:
            return

        self._head_smoother.update()
        self._orientation = self._head_smoother.smooth_value if self._head_smoother.smooth_value is not None else 0.0
        delta: float | None = self._head_smoother._smooth_delta
        self._delta = delta if delta is not None else 0.0
        motion: float = abs(self._delta)
        if motion < self.settings.motion_threshold:
            motion = 0.0
        self._motion += motion

    def reset(self) -> None:
        self._head_smoother.reset()
        self._orientation = 0.0
        self._delta = 0.0
        self._motion = 0.0
