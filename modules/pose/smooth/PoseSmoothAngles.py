import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.features.PoseAngles import POSE_ANGLE_JOINTS, POSE_ANGLE_JOINT_IDXS
from modules.pose.smooth.PoseSmoothBase import PoseSmoothBase

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings

from modules.utils.HotReloadMethods import HotReloadMethods

POSE_ANGLE_MOTION_WEIGHTS: dict[PoseJoint, float] = {
    PoseJoint.left_elbow:       0.5,
    PoseJoint.right_elbow:      0.5,
    PoseJoint.left_shoulder:    0.6,
    PoseJoint.right_shoulder:   0.6,
    PoseJoint.left_hip:         0.8,
    PoseJoint.right_hip:        0.8,
    PoseJoint.left_knee:        1.2,
    PoseJoint.right_knee:       1.2
}

class SymmetricJointType(Enum):
    elbow = auto()
    hip = auto()
    knee = auto()
    shoulder = auto()

SYMMETRIC_JOINT_PAIRS: dict[SymmetricJointType, tuple[PoseJoint, PoseJoint]] = {
    SymmetricJointType.shoulder: (PoseJoint.left_shoulder, PoseJoint.right_shoulder),
    SymmetricJointType.elbow: (PoseJoint.left_elbow, PoseJoint.right_elbow),
    SymmetricJointType.hip: (PoseJoint.left_hip, PoseJoint.right_hip),
    SymmetricJointType.knee: (PoseJoint.left_knee, PoseJoint.right_knee)
}

SYMMETRIC_JOINT_TYPE_MAP: dict[PoseJoint, SymmetricJointType] = {}
for joint_type, (left_joint, right_joint) in SYMMETRIC_JOINT_PAIRS.items():
    SYMMETRIC_JOINT_TYPE_MAP[left_joint] = joint_type
    SYMMETRIC_JOINT_TYPE_MAP[right_joint] = joint_type

@dataclass
class PoseSmoothAngleSettings:
    smooth_settings: OneEuroSettings
    motion_threshold: float = 0.002
    motion_weights: dict[PoseJoint, float] = field(default_factory=lambda: POSE_ANGLE_MOTION_WEIGHTS)

# CLASSES
class PoseSmoothAngles(PoseSmoothBase):
    def __init__(self, settings: PoseSmoothAngleSettings) -> None:
        self._active: bool = False
        self.settings: PoseSmoothAngleSettings = settings
        self._angle_smoothers: dict[PoseJoint, AngleEuroInterpolator] = {}
        self._cumulative_joint_motion: dict[PoseJoint, float] = {}
        self._cumulative_total_motion: float = 0.0
        for joint in POSE_ANGLE_JOINTS:
            self._angle_smoothers[joint] = AngleEuroInterpolator(settings.smooth_settings)
            self._cumulative_joint_motion[joint] = 0.0

        self._synchronies: dict[SymmetricJointType, float] = {}
        self._mean_synchrony: float = 0.0
        for joint_type in SYMMETRIC_JOINT_PAIRS.keys():
            self._synchronies[joint_type] = 0.0


        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def angles(self) -> dict[PoseJoint, float]:
        return {joint: self.get_angle(joint, symmetric=True) for joint in POSE_ANGLE_JOINTS}

    @property
    def deltas(self) -> dict[PoseJoint, float]:
        return {joint: self.get_delta(joint, symmetric=True) for joint in POSE_ANGLE_JOINTS}

    @property
    def motions(self) -> dict[PoseJoint, float]:
        return self._cumulative_joint_motion

    @property
    def total_motion(self) -> float:
        return self._cumulative_total_motion

    @property
    def synchronies(self) -> dict[SymmetricJointType, float]:
        return self._synchronies

    @property
    def mean_synchrony(self) -> float:
        return self._mean_synchrony

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
            for joint in POSE_ANGLE_JOINTS:
                self._angle_smoothers[joint].add_sample(np.nan)
        else:
            for joint in POSE_ANGLE_JOINTS:
                self._angle_smoothers[joint].add_sample(pose.angle_data.angles[POSE_ANGLE_JOINT_IDXS[joint]])

    def update(self) -> None:
        if not self._active:
            return

        total_movement: float = 0.0
        for joint in POSE_ANGLE_JOINTS:
            self._angle_smoothers[joint].update()
            delta: float | None = self._angle_smoothers[joint]._smooth_delta
            movement: float = abs(delta) if delta is not None else 0.0
            if movement < self.settings.motion_threshold:
                movement = 0.0
            movement *= self.settings.motion_weights.get(joint, 1.0)
            self._cumulative_joint_motion[joint] += movement
            total_movement += movement
        self._cumulative_total_motion += total_movement

        total_synchrony: float = 0.0
        for sym_type, (left_joint, right_joint) in SYMMETRIC_JOINT_PAIRS.items():
            left_angle: float | None = self.get_angle(left_joint, symmetric=True)
            right_angle: float | None = self.get_angle(right_joint, symmetric=True)

            if left_angle is not None and right_angle is not None:
                angle_diff: float = abs(left_angle - right_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                symmetry: float = 1.0 - angle_diff / np.pi
                self._synchronies[sym_type] = symmetry
                total_synchrony += symmetry
            else:
                self._synchronies[sym_type] = 0.0
        self._mean_synchrony = total_synchrony / len(SYMMETRIC_JOINT_PAIRS)

    def reset(self) -> None:
        for smoother in self._angle_smoothers.values():
            smoother.reset()
        for motion_time in self._cumulative_joint_motion:
            self._cumulative_joint_motion[motion_time] = 0.0
        self._cumulative_total_motion = 0.0

    def get_angle(self, joint: PoseJoint, symmetric: bool = True) -> float:
        angle: float | None = self._angle_smoothers[joint].smooth_value
        if angle is None:
            return 0.0
        if symmetric and "right_" in joint.name and angle is not None:
           angle = -angle
        return angle

    def get_delta(self, joint: PoseJoint, symmetric: bool = True) -> float:
        delta: float | None = self._angle_smoothers[joint].smooth_delta
        if delta is None:
            return 0.0
        if symmetric and "right_" in joint.name and delta is not None:
           delta = -delta
        return delta

    # SYMMETRY METHODS
    def get_symmetry(self, joint_type: SymmetricJointType) -> float:
        return self._synchronies.get(joint_type, 0.0)
