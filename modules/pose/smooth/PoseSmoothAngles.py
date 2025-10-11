import numpy as np

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.PoseAngles import POSE_ANGLE_JOINTS, POSE_ANGLE_JOINT_IDXS

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings

from modules.utils.HotReloadMethods import HotReloadMethods

# DEFINITIONS
CUMULATIVE_CHANGE_JOINTS: list[PoseJoint] = [
    PoseJoint.left_elbow,
    PoseJoint.right_elbow,
    PoseJoint.left_shoulder,
    PoseJoint.right_shoulder
]

# CLASSES
class PoseSmoothAngles():
    def __init__(self, one_euro_settings: OneEuroSettings) -> None:
        self._active: bool = False
        self._angle_smoothers: dict[PoseJoint, AngleEuroInterpolator] = {}
        self._motion_times: dict[PoseJoint, float] = {}
        self._total_motion_time: float = 0.0
        for joint in POSE_ANGLE_JOINTS:
            self._angle_smoothers[joint] = AngleEuroInterpolator(one_euro_settings)
            self._motion_times[joint] = 0.0

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
        return self._motion_times

    @property
    def total_motion(self) -> float:
        return self._total_motion_time

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
            self._motion_times[joint] += movement
            if joint in CUMULATIVE_CHANGE_JOINTS:
                total_movement += movement
        self._total_motion_time += total_movement / len(CUMULATIVE_CHANGE_JOINTS)

    def reset(self) -> None:
        for smoother in self._angle_smoothers.values():
            smoother.reset()
        for motion_time in self._motion_times:
            self._motion_times[motion_time] = 0.0
        self._total_motion_time = 0.0

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
