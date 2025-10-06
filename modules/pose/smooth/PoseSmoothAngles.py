import numpy as np

from enum import Enum, auto
from threading import Lock


from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.PoseAngles import POSE_ANGLE_JOINTS, POSE_ANGLE_JOINT_IDXS

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings

from modules.utils.HotReloadMethods import HotReloadMethods

# DEFINITIONS
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

# CLASSES
class PoseSmoothAngles():
    def __init__(self, one_euro_settings: OneEuroSettings) -> None:
        self.angle_smoothers: dict[PoseJoint, AngleEuroInterpolator] = {}
        for joint in POSE_ANGLE_JOINTS:
            self.angle_smoothers[joint] = AngleEuroInterpolator(one_euro_settings)

        self.active: bool = False
        # self._lock = Lock()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def reset(self) -> None:
        # with self._lock:
        for smoother in self.angle_smoothers.values():
            smoother.reset()

    def add_pose(self, pose: Pose) -> None:
        # with self._lock:
        if pose.tracklet.is_removed:
            self.active = False
            self.reset()
            return

        if pose.tracklet.is_active and not self.active:
            self.active = True
            self.reset()

        if not self.active:
            return

        # Always add data, OneEuroInterpolator will handle missing data
        if pose.angle_data is None:
            for joint in POSE_ANGLE_JOINTS:
                self.angle_smoothers[joint].add_sample(np.nan)
        else:
            for joint in POSE_ANGLE_JOINTS:
                self.angle_smoothers[joint].add_sample(pose.angle_data.angles[POSE_ANGLE_JOINT_IDXS[joint]])

    def get_smoothed_angle(self, joint: PoseJoint, symmetric: bool = False) -> float | None:
        # with self._lock:
        if not self.active:
            return None

        if joint not in self.angle_smoothers:
            print(f"Warning: Joint {joint} not in angle smoothers.")
            return None

        angle: float | None = self.angle_smoothers[joint].get()

        if not symmetric or angle is None:
            return angle

        if joint in SYMMETRIC_JOINT_TYPE_MAP:
            joint_type: SymmetricJointType = SYMMETRIC_JOINT_TYPE_MAP[joint]
            left_joint, right_joint = SYMMETRIC_JOINT_PAIRS[joint_type]

            # Invert angle for right joints to maintain symmetry
            if joint == right_joint:
                return -angle

        return angle

    # SYMMETRY METHODS
    def get_joint_symmetry(self, joint_type: SymmetricJointType) -> float | None:
        """Get symmetry value based on left and right angles for a specific joint type."""
        if not self.active:
            return None

        left_joint, right_joint = SYMMETRIC_JOINT_PAIRS[joint_type]

        # Get the symmetrized angles for both left and right joints
        left_angle: float | None = self.get_smoothed_angle(left_joint, symmetric=True)
        right_angle: float | None = self.get_smoothed_angle(right_joint, symmetric=True)

        # Calculate symmetry if both angles are available
        if left_angle is not None and right_angle is not None:
            # Symmetry is highest when difference is smallest
            # Normalize by pi (assuming angles are in radians)
            symmetry: float = 1.0 - abs(left_angle - right_angle) / np.pi
            return symmetry

        return None

    def get_average_symmetry(self) -> float | None:
        """Get average symmetry value across all symmetric joint pairs."""
        if not self.active:
            return None

        symmetry_values: list[float] = []

        # Use SYMMETRIC_JOINT_PAIRS to iterate through all joint types
        for joint_type in SYMMETRIC_JOINT_PAIRS.keys():
            joint_symmetry: float | None = self.get_joint_symmetry(joint_type)
            if joint_symmetry is not None:
                symmetry_values.append(joint_symmetry)

        # Return average if we have values, otherwise None
        if not symmetry_values:
            return None

        return sum(symmetry_values) / len(symmetry_values)
