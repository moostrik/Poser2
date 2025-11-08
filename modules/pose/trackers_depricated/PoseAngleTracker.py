import numpy as np
from dataclasses import dataclass, field
from time import time

from modules.pose.Pose import Pose
from modules.pose.features.PoseAngles import AngleJoint, PoseAngleData
from modules.pose.features.PoseAngleSymmetry import PoseAngleSymmetryFactory, PoseAngleSymmetryData
from modules.pose.trackers_depricated.PoseTrackerBase import PoseTrackerBase

from modules.utils.depricated.SmoothedInterpolator import SmoothedAngleInterpolator, OneEuroSettings
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
    # AngleJoint.torso:            2.0,
    AngleJoint.head:             2.0
}

@dataclass
class PoseAngleTrackerSettings:
    smooth_settings: OneEuroSettings
    symmetry_exponent: float = 1.0
    motion_threshold: float = 0.002
    motion_weights: dict[AngleJoint, float] = field(default_factory=lambda: POSE_ANGLE_MOTION_WEIGHTS)


class PoseAngleTracker(PoseTrackerBase):
    def __init__(self, settings: PoseAngleTrackerSettings) -> None:
        self.settings: PoseAngleTrackerSettings = settings
        self._active: bool = False
        self._angle_smoothers: dict[AngleJoint, SmoothedAngleInterpolator] = {}
        self._motions: dict[AngleJoint, float] = {}
        self._total_motion: float = 0.0
        self._cumulative_motions: dict[AngleJoint, float] = {}
        self._cumulative_total_motion: float = 0.0

        for joint in AngleJoint:
            self._angle_smoothers[joint] = SmoothedAngleInterpolator(settings.smooth_settings)
            self._motions[joint] = 0.0
            self._cumulative_motions[joint] = 0.0


        self._angle_data: PoseAngleData = PoseAngleData.create_empty()
        self._velocity_data: PoseAngleData = PoseAngleData.create_empty()
        self._acceleration_data: PoseAngleData = PoseAngleData.create_empty()
        self._motion_data: PoseAngleData = PoseAngleData.create_empty()
        self._cumulative_motion_data: PoseAngleData = PoseAngleData.create_empty()
        self._symmetry_data: PoseAngleSymmetryData = PoseAngleSymmetryData.create_empty()

        self._angles_dirty = False
        self._velocities_dirty = False
        self._accelerations_dirty = False
        self._motions_dirty = False
        self._cumulative_motions_dirty = False

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
        add_time: float = time()
        for joint in AngleJoint:
            self._angle_smoothers[joint].add_sample(pose.angle_data.values[joint.value], add_time)

    def update(self) -> None:
        if not self._active:
            return

        update_time: float = time()
        total_motion: float = 0.0
        for joint in AngleJoint:
            self._angle_smoothers[joint].update(update_time)
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

        self._angles_dirty = True
        self._velocities_dirty = True
        self._accelerations_dirty = True
        self._motions_dirty = True
        self._cumulative_motions_dirty = True

        # Compute symmetry
        self._symmetry_data = PoseAngleSymmetryFactory.from_angles(self.angles, self.settings.symmetry_exponent)

    def reset(self) -> None:
        """Reset all smoothers and accumulated state"""
        for joint in AngleJoint:
            self._angle_smoothers[joint].reset()
            self._motions[joint] = 0.0
            self._cumulative_motions[joint] = 0.0
        self._total_motion = 0.0
        self._cumulative_total_motion = 0.0

        self._angle_data = PoseAngleData.create_empty()
        self._velocity_data = PoseAngleData.create_empty()
        self._acceleration_data = PoseAngleData.create_empty()
        self._motion_data = PoseAngleData.create_empty()
        self._cumulative_motion_data = PoseAngleData.create_empty()
        self._symmetry_data = PoseAngleSymmetryData.create_empty()

        self._angles_dirty = False
        self._velocities_dirty = False
        self._accelerations_dirty = False
        self._motions_dirty = False
        self._cumulative_motions_dirty = False
        self._symmetries_dirty = False

    # ========== PROPERTIES ==========

    @property
    def is_active(self) -> bool:
        """Whether the interpolator is actively tracking a pose"""
        return self._active

    @property
    def angles(self) -> PoseAngleData:
        """Current smoothed angles (radians)"""
        if self._angles_dirty:
            angle_values: np.ndarray = np.array([self._angle_smoothers[joint].smooth_value for joint in AngleJoint], dtype=np.float32)
            self._angle_data = PoseAngleData.from_values(angle_values)
            self._angles_dirty = False
        return self._angle_data

    @property
    def velocities(self) -> PoseAngleData:
        """Current angular velocities (rad/s)"""
        if self._velocities_dirty:
            velocity_values: np.ndarray = np.array([self._angle_smoothers[joint].smooth_velocity for joint in AngleJoint], dtype=np.float32)
            self._velocity_data = PoseAngleData.from_values(velocity_values)
            self._velocities_dirty = False
        return self._velocity_data

    @property
    def accelerations(self) -> PoseAngleData:
        """Current angular accelerations (rad/s²)"""
        if self._accelerations_dirty:
            acceleration_values: np.ndarray = np.array([self._angle_smoothers[joint].smooth_acceleration for joint in AngleJoint], dtype=np.float32)
            self._acceleration_data = PoseAngleData.from_values(acceleration_values)
            self._accelerations_dirty = False
        return self._acceleration_data

    @property
    def motions(self) -> PoseAngleData:
        """Current weighted motion values"""
        if self._motions_dirty:
            motion_values: np.ndarray = np.array([self._motions[joint] for joint in AngleJoint], dtype=np.float32)
            self._motion_data = PoseAngleData.from_values(motion_values)
            self._motions_dirty = False
        return self._motion_data

    @property
    def cumulative_motions(self) -> PoseAngleData:
        """Cumulative motion values since tracking started"""
        if self._cumulative_motions_dirty:
            cumulative_values: np.ndarray = np.array([self._cumulative_motions[joint] for joint in AngleJoint], dtype=np.float32)
            self._cumulative_motion_data = PoseAngleData.from_values(cumulative_values)
            self._cumulative_motions_dirty = False
        return self._cumulative_motion_data

    @property
    def total_motion(self) -> float:
        """Total motion across all joints for current frame"""
        return self._total_motion

    @property
    def cumulative_total_motion(self) -> float:
        """Cumulative total motion across all joints since tracking started"""
        return self._cumulative_total_motion

    @property
    def symmetries(self) -> PoseAngleSymmetryData:
        """Current pose symmetry metrics"""
        return self._symmetry_data
