import numpy as np
from dataclasses import dataclass, field
from time import time

from modules.pose.Pose import Pose
from modules.pose.features import AngleLandmark, AngleFeature, SymmetryFeature, SymmetryFactory
from modules.data.depricated.PoseTrackerBase import PoseTrackerBase

from modules.utils.depricated.SmoothedInterpolator import SmoothedAngleInterpolator, OneEuroSettings
from modules.utils.HotReloadMethods import HotReloadMethods

POSE_ANGLE_MOTION_WEIGHTS: dict[AngleLandmark, float] = {
    AngleLandmark.left_elbow:       0.5,
    AngleLandmark.right_elbow:      0.5,
    AngleLandmark.left_shoulder:    0.6,
    AngleLandmark.right_shoulder:   0.6,
    AngleLandmark.left_hip:         0.8,
    AngleLandmark.right_hip:        0.8,
    AngleLandmark.left_knee:        1.2,
    AngleLandmark.right_knee:       1.2,
    # AngleJoint.torso:            2.0,
    AngleLandmark.head:             2.0
}

@dataclass
class PoseAngleTrackerSettings:
    smooth_settings: OneEuroSettings
    symmetry_exponent: float = 1.0
    motion_threshold: float = 0.002
    motion_weights: dict[AngleLandmark, float] = field(default_factory=lambda: POSE_ANGLE_MOTION_WEIGHTS)


class PoseAngleTracker(PoseTrackerBase):
    def __init__(self, settings: PoseAngleTrackerSettings) -> None:
        self.settings: PoseAngleTrackerSettings = settings
        self._active: bool = False
        self._angle_smoothers: dict[AngleLandmark, SmoothedAngleInterpolator] = {}
        self._motions: dict[AngleLandmark, float] = {}
        self._total_motion: float = 0.0
        self._cumulative_motions: dict[AngleLandmark, float] = {}
        self._cumulative_total_motion: float = 0.0

        for joint in AngleLandmark:
            self._angle_smoothers[joint] = SmoothedAngleInterpolator(settings.smooth_settings)
            self._motions[joint] = 0.0
            self._cumulative_motions[joint] = 0.0


        self._angles: AngleFeature = AngleFeature.create_dummy()
        self._velocity_data: AngleFeature = AngleFeature.create_dummy()
        self._acceleration_data: AngleFeature = AngleFeature.create_dummy()
        self._motion_data: AngleFeature = AngleFeature.create_dummy()
        self._cumulative_motion_data: AngleFeature = AngleFeature.create_dummy()
        self._symmetry_data: SymmetryFeature = SymmetryFeature.create_dummy()

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
        for joint in AngleLandmark:
            self._angle_smoothers[joint].add_sample(pose.angles.values[joint.value], add_time)

    def update(self) -> None:
        if not self._active:
            return

        update_time: float = time()
        total_motion: float = 0.0
        for joint in AngleLandmark:
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
        self._symmetry_data = SymmetryFactory.from_angles(self.angles, self.settings.symmetry_exponent)

    def reset(self) -> None:
        """Reset all smoothers and accumulated state"""
        for joint in AngleLandmark:
            self._angle_smoothers[joint].reset()
            self._motions[joint] = 0.0
            self._cumulative_motions[joint] = 0.0
        self._total_motion = 0.0
        self._cumulative_total_motion = 0.0

        self._angles = AngleFeature.create_dummy()
        self._velocity_data = AngleFeature.create_dummy()
        self._acceleration_data = AngleFeature.create_dummy()
        self._motion_data = AngleFeature.create_dummy()
        self._cumulative_motion_data = AngleFeature.create_dummy()
        self._symmetry_data = SymmetryFeature.create_dummy()

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
    def angles(self) -> AngleFeature:
        """Current smoothed angles (radians)"""
        if self._angles_dirty:
            angle_values: np.ndarray = np.array([self._angle_smoothers[joint].smooth_value for joint in AngleLandmark], dtype=np.float32)
            self._angles = AngleFeature.from_values(angle_values)
            self._angles_dirty = False
        return self._angles

    @property
    def velocities(self) -> AngleFeature:
        """Current angular velocities (rad/s)"""
        if self._velocities_dirty:
            velocity_values: np.ndarray = np.array([self._angle_smoothers[joint].smooth_velocity for joint in AngleLandmark], dtype=np.float32)
            self._velocity_data = AngleFeature.from_values(velocity_values)
            self._velocities_dirty = False
        return self._velocity_data

    @property
    def accelerations(self) -> AngleFeature:
        """Current angular accelerations (rad/s²)"""
        if self._accelerations_dirty:
            acceleration_values: np.ndarray = np.array([self._angle_smoothers[joint].smooth_acceleration for joint in AngleLandmark], dtype=np.float32)
            self._acceleration_data = AngleFeature.from_values(acceleration_values)
            self._accelerations_dirty = False
        return self._acceleration_data

    @property
    def motions(self) -> AngleFeature:
        """Current weighted motion values"""
        if self._motions_dirty:
            motion_values: np.ndarray = np.array([self._motions[joint] for joint in AngleLandmark], dtype=np.float32)
            self._motion_data = AngleFeature.from_values(motion_values)
            self._motions_dirty = False
        return self._motion_data

    @property
    def cumulative_motions(self) -> AngleFeature:
        """Cumulative motion values since tracking started"""
        if self._cumulative_motions_dirty:
            cumulative_values: np.ndarray = np.array([self._cumulative_motions[joint] for joint in AngleLandmark], dtype=np.float32)
            self._cumulative_motion_data = AngleFeature.from_values(cumulative_values)
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
    def symmetries(self) -> SymmetryFeature:
        """Current pose symmetry metrics"""
        return self._symmetry_data
