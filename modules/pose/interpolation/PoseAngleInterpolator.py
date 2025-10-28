import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.features.PoseAngles import AngleJoint, ANGLE_JOINT_NAMES, ANGLE_NUM_JOINTS
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

class SymmetricJointType(Enum):
    elbow = auto()
    hip = auto()
    knee = auto()
    shoulder = auto()

SYMMETRIC_JOINT_PAIRS: dict[SymmetricJointType, tuple[AngleJoint, AngleJoint]] = {
    SymmetricJointType.shoulder: (AngleJoint.left_shoulder, AngleJoint.right_shoulder),
    SymmetricJointType.elbow: (AngleJoint.left_elbow, AngleJoint.right_elbow),
    SymmetricJointType.hip: (AngleJoint.left_hip, AngleJoint.right_hip),
    SymmetricJointType.knee: (AngleJoint.left_knee, AngleJoint.right_knee)
}

SYMMETRIC_JOINT_TYPE_MAP: dict[AngleJoint, SymmetricJointType] = {}
for joint_type, (left_joint, right_joint) in SYMMETRIC_JOINT_PAIRS.items():
    SYMMETRIC_JOINT_TYPE_MAP[left_joint] = joint_type
    SYMMETRIC_JOINT_TYPE_MAP[right_joint] = joint_type

@dataclass
class PoseAngleInterpolatorSettings:
    smooth_settings: OneEuroSettings
    motion_threshold: float = 0.002
    motion_weights: dict[AngleJoint, float] = field(default_factory=lambda: POSE_ANGLE_MOTION_WEIGHTS)

# CLASSES
class PoseAngleInterpolator(PoseInterpolationBase):
    def __init__(self, settings: PoseAngleInterpolatorSettings) -> None:
        self._active: bool = False
        self.settings: PoseAngleInterpolatorSettings = settings
        self._angle_smoothers: dict[AngleJoint, AngleEuroInterpolator] = {}
        self._motions: dict[AngleJoint, float] = {}
        self._total_motion: float = 0.0
        self._cumulative_motions: dict[AngleJoint, float] = {}
        self._cumulative_total_motion: float = 0.0
        for joint in AngleJoint:
            self._angle_smoothers[joint] = AngleEuroInterpolator(settings.smooth_settings)
            self._cumulative_motions[joint] = 0.0

        self._symmetries: dict[SymmetricJointType, float] = {}
        self._mean_symmetry: float = 0.0
        for joint_type in SYMMETRIC_JOINT_PAIRS.keys():
            self._symmetries[joint_type] = 0.0

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

        # Always add data, OneEuroInterpolator will handle missing data
        if pose.angle_data is None:
            for joint in AngleJoint:
                self._angle_smoothers[joint].add_sample(np.nan)
        else:
            for joint in AngleJoint:
                self._angle_smoothers[joint].add_sample(pose.angle_data.angles[joint.value])

    def update(self) -> None:
        if not self._active:
            return

        total_motion: float = 0.0
        for joint in AngleJoint:
            self._angle_smoothers[joint].update()
            delta: float | None = self._angle_smoothers[joint]._smooth_velocity
            motion: float = abs(delta) if delta is not None else 0.0
            if motion < self.settings.motion_threshold:
                motion = 0.0
            motion *= self.settings.motion_weights.get(joint, 1.0)
            self._motions[joint] = motion
            self._cumulative_motions[joint] += motion
            total_motion += motion
        self._total_motion = total_motion
        self._cumulative_total_motion += total_motion

        total_symmetry: float = 0.0
        for sym_type, (left_joint, right_joint) in SYMMETRIC_JOINT_PAIRS.items():
            left_angle: float | None = self.get_angle(left_joint, symmetric=True)
            right_angle: float | None = self.get_angle(right_joint, symmetric=True)

            if left_angle is not None and right_angle is not None:
                angle_diff: float = abs(left_angle - right_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                symmetry: float = 1.0 - angle_diff / np.pi
                self._symmetries[sym_type] = symmetry
                total_symmetry += symmetry
            else:
                self._symmetries[sym_type] = 0.0
        self._mean_symmetry = total_symmetry / len(SYMMETRIC_JOINT_PAIRS)

    def reset(self) -> None:
        for smoother in self._angle_smoothers.values():
            smoother.reset()
        for key in self._motions.keys():
            self._motions[key] = 0.0
        self._total_motion = 0.0
        for key in self._cumulative_motions.keys():
            self._cumulative_motions[key] = 0.0
        self._cumulative_total_motion = 0.0
        for key in self._symmetries.keys():
            self._symmetries[key] = 0.0
        self._mean_symmetry = 0.0

    # GETTERS
    def get_angle(self, joint: AngleJoint, symmetric: bool = True) -> float:
        angle: float | None = self._angle_smoothers[joint].smooth_value
        if angle is None:
            return 0.0
        if symmetric and "right_" in joint.name and angle is not None:
           angle = -angle
        return angle

    def get_velocity(self, joint: AngleJoint, symmetric: bool = True) -> float:
        velocity: float | None = self._angle_smoothers[joint].smooth_velocity
        if velocity is None:
            return 0.0
        if symmetric and "right_" in joint.name and velocity is not None:
           velocity = -velocity
        return velocity

    def get_acceleration(self, joint: AngleJoint, symmetric: bool = True) -> float:
        acceleration: float | None = self._angle_smoothers[joint].smooth_acceleration
        if acceleration is None:
            return 0.0
        if symmetric and "right_" in joint.name and acceleration is not None:
           acceleration = -acceleration
        return acceleration

    def get_motion(self, joint: AngleJoint) -> float:
        return self._motions.get(joint, 0.0)

    def get_cumulative_motion(self, joint: AngleJoint) -> float:
        return self._cumulative_motions.get(joint, 0.0)

    def get_total_motion(self) -> float:
        return self._total_motion

    def get_cumulative_total_motion(self) -> float:
        return self._cumulative_total_motion

    def get_symmetry(self, joint_type: SymmetricJointType) -> float:
        return self._symmetries.get(joint_type, 0.0)

    def get_mean_symmetry(self) -> float:
        return self._mean_symmetry

    # PROPERTIES
    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def angles(self) -> dict[AngleJoint, float]:
        return {joint: self.get_angle(joint, symmetric=True) for joint in AngleJoint}

    @property
    def velocities(self) -> dict[AngleJoint, float]:
        return {joint: self.get_velocity(joint, symmetric=True) for joint in AngleJoint}

    @property
    def accelerations(self) -> dict[AngleJoint, float]:
        return {joint: self.get_acceleration(joint, symmetric=True) for joint in AngleJoint}

    @property
    def motions(self) -> dict[AngleJoint, float]:
        return self._motions

    @property
    def total_motion(self) -> float:
        return self._total_motion

    @property
    def cumulative_motions(self) -> dict[AngleJoint, float]:
        return self._cumulative_motions

    @property
    def cumulative_total_motion(self) -> float:
        return self._cumulative_total_motion

    @property
    def symmetries(self) -> dict[SymmetricJointType, float]:
        return self._symmetries

    @property
    def mean_symmetry(self) -> float:
        return self._mean_symmetry
