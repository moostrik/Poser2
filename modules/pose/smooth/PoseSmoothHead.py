import numpy as np
from dataclasses import dataclass

from modules.pose.Pose import Pose
from modules.pose.smooth.PoseSmoothBase import PoseSmoothBase
from modules.pose.smooth.PoseSmoothAngles import PoseSmoothAngleSettings

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class PoseSmoothHeadSettings:
    smooth_settings: OneEuroSettings
    motion_threshold: float = 0.002
    motion_weight: float = 2.0 # based on PoseSmoothAngleSettings motion weights

# CLASSES
class PoseSmoothHead(PoseSmoothBase):
    def __init__(self, settings: PoseSmoothHeadSettings) -> None:
        self._active: bool = False
        self.settings: PoseSmoothHeadSettings = settings
        self._head_smoother: AngleEuroInterpolator = AngleEuroInterpolator(settings.smooth_settings)

        self._angle: float = 0.0
        self._velocity: float = 0.0
        self._acceleration: float = 0.0

        self._motion: float = 0.0

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def acceleration(self) -> float:
        return self._acceleration
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


        self._angle = self._head_smoother.smooth_value if self._head_smoother.smooth_value is not None else 0.0
        self._velocity = self._head_smoother._smooth_velocity if self._head_smoother._smooth_velocity is not None else 0.0
        self._acceleration = self._head_smoother._smooth_acceleration if self._head_smoother._smooth_acceleration is not None else 0.0

        motion: float = abs(self._velocity)
        if motion < self.settings.motion_threshold:
            motion = 0.0
        self._motion += motion * self.settings.motion_weight

    def reset(self) -> None:
        self._head_smoother.reset()
        self._angle = 0.0
        self._velocity = 0.0
        self._motion = 0.0
