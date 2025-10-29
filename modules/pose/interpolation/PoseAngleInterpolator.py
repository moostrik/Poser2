import numpy as np
from dataclasses import dataclass, field

from modules.pose.Pose import Pose
from modules.pose.features.PoseAngles import AngleJoint, PoseAngleData, PoseAngles
from modules.pose.features.PoseAngleSymmetry import PoseAngleSymmetry, PoseSymmetryData
from modules.pose.interpolation.PoseInterpolationBase import PoseInterpolationBase

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings
from modules.utils.HotReloadMethods import HotReloadMethods

POSE_ANGLE_MOTION_WEIGHTS: dict[AngleJoint, float] = {
    AngleJoint.left_elbow:       0.5,
    AngleJoint.right_elbow:      0.5,
    AngleJoint.left_shoulder:    0.6,
    AngleJoint.right_shoulder:   0.6,
    AngleJoint.left_hip:         0.8,
    AngleJoint.right_hip:        0.8,
    AngleJoint.left_knee:        1.2,
    AngleJoint.right_knee:       1.2,
    AngleJoint.torso:            2.0,
    AngleJoint.head:             2.0
}

@dataclass
class PoseAngleInterpolatorSettings:
    smooth_settings: OneEuroSettings
    symmetry_exponent: float = 1.0
    motion_threshold: float = 0.002
    motion_weights: dict[AngleJoint, float] = field(default_factory=lambda: POSE_ANGLE_MOTION_WEIGHTS)


class PoseAngleInterpolator(PoseInterpolationBase):
    def __init__(self, settings: PoseAngleInterpolatorSettings) -> None:
        self.settings: PoseAngleInterpolatorSettings = settings
        self._active: bool = False
        self._angle_smoothers: dict[AngleJoint, AngleEuroInterpolator] = {}
        self._motions: dict[AngleJoint, float] = {}
        self._total_motion: float = 0.0
        self._cumulative_motions: dict[AngleJoint, float] = {}
        self._cumulative_total_motion: float = 0.0

        for joint in AngleJoint:
            self._angle_smoothers[joint] = AngleEuroInterpolator(settings.smooth_settings)
            self._motions[joint] = 0.0
            self._cumulative_motions[joint] = 0.0

        self._symmetry_data: PoseSymmetryData | None = None
        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_pose(self, pose: Pose) -> None:
        if pose.tracklet.is_removed:
            self._active = False
            self.reset()
            return

        if pose.tracklet.is_active and not self._active:
            self._active = True
            self.reset()

        if not self._active:
            return

        # Add angle data to smoothers (OneEuroInterpolator handles NaN values)
        for joint in AngleJoint:
            self._angle_smoothers[joint].add_sample(pose.angle_data.values[joint.value])

    def update(self) -> None:
        if not self._active:
            return

        total_motion: float = 0.0
        for joint in AngleJoint:
            self._angle_smoothers[joint].update()
            velocity: float | None = self._angle_smoothers[joint].smooth_velocity

            if velocity is None or np.isnan(velocity):
                motion = 0.0
            else:
                motion: float = abs(velocity)
                if motion < self.settings.motion_threshold:
                    motion = 0.0
                motion *= self.settings.motion_weights.get(joint, 1.0)

            self._motions[joint] = motion
            self._cumulative_motions[joint] += motion
            total_motion += motion

        self._total_motion = total_motion
        self._cumulative_total_motion += total_motion

        # Compute symmetry
        self._symmetry_data = PoseAngleSymmetry.compute(self.angles, self.settings.symmetry_exponent)

    def reset(self) -> None:
        """Reset all smoothers and accumulated state"""
        for joint in AngleJoint:
            self._angle_smoothers[joint].reset()
            self._motions[joint] = 0.0
            self._cumulative_motions[joint] = 0.0
        self._total_motion = 0.0
        self._cumulative_total_motion = 0.0
        self._symmetry_data = None

    # PROPERTIES
    @property
    def is_active(self) -> bool:
        """Whether the interpolator is actively tracking a pose"""
        return self._active

    @property
    def angles(self) -> PoseAngleData:
        """Current smoothed angles (radians)"""
        values = {joint: self._angle_smoothers[joint].smooth_value for joint in AngleJoint}
        return PoseAngles.from_values(values)

    @property
    def velocities(self) -> PoseAngleData:
        """Current angular velocities (rad/s)"""
        values = {joint: self._angle_smoothers[joint].smooth_velocity for joint in AngleJoint}
        return PoseAngles.from_values(values)

    @property
    def accelerations(self) -> PoseAngleData:
        """Current angular accelerations (rad/s²)"""
        values = {joint: self._angle_smoothers[joint].smooth_acceleration for joint in AngleJoint}
        return PoseAngles.from_values(values)

    @property
    def motions(self) -> PoseAngleData:
        """Current weighted motion values"""
        return PoseAngles.from_values(self._motions)

    @property
    def cumulative_motions(self) -> PoseAngleData:
        """Cumulative motion values since tracking started"""
        return PoseAngles.from_values(self._cumulative_motions)

    @property
    def total_motion(self) -> float:
        """Total motion across all joints for current frame"""
        return self._total_motion

    @property
    def cumulative_total_motion(self) -> float:
        """Cumulative total motion across all joints since tracking started"""
        return self._cumulative_total_motion

    @property
    def symmetry_data(self) -> PoseSymmetryData | None:
        """Current pose symmetry metrics, or None if no data available"""
        return self._symmetry_data
