from dataclasses import replace, dataclass, field
from traceback import print_exc
from typing import Callable

import numpy as np

from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS, PoseJoint
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS, AngleJoint

from modules.utils.Smoothing import OneEuroFilter, OneEuroFilterAngular
from modules.Settings import Settings


@dataclass
class OneEuroSettings:
    """Configuration parameters for the 1€ Filter.

    Attributes:
        frequency: INPUT data frequency in Hz (default: 30.0)
        min_cutoff: Minimum cutoff frequency for position smoothing (default: 1.0)
        beta: Speed coefficient - higher values reduce lag but increase jitter (default: 0.025)
        d_cutoff: Cutoff frequency for derivative smoothing (default: 1.0)
    """
    frequency: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.025       # recommended: 0.007 to 0.05 for generative a/v
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        """Initialize observer list and validate parameters."""
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        if self.min_cutoff < 0:
            raise ValueError(f"min_cutoff must be non-negative, got {self.min_cutoff}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.d_cutoff < 0:
            raise ValueError(f"d_cutoff must be non-negative, got {self.d_cutoff}")

        self._observers: list[Callable[[], None]] = []

    def __setattr__(self, name: str, value: object) -> None:
        """Notify observers when settings change."""
        super().__setattr__(name, value)
        if name != '_observers' and hasattr(self, '_observers') and len(self._observers) > 0:
            self._notify()

    def add_observer(self, callback: Callable[[], None]) -> None:
        """Add observer callback to be notified of setting changes."""
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[], None]) -> None:
        """Remove observer callback."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify(self) -> None:
        """Notify all observers of setting changes."""
        for callback in self._observers:
            try:
                callback()
            except Exception as e:
                print(f"OneEuroSettings: Error in observer callback: {str(e)}")
                print_exc()


@dataclass
class TrackletSmoothingState:
    """All smoothing state for one tracked person/tracklet.

    Encapsulates filters and validity tracking to ensure they stay synchronized.
    """
    point_filters: list[tuple[OneEuroFilter, OneEuroFilter]] = field(default_factory=list)  # (x, y) per joint
    angle_filters: list[OneEuroFilterAngular] = field(default_factory=list)
    prev_point_valid: np.ndarray = field(default_factory=lambda: np.zeros(POSE_NUM_JOINTS, dtype=bool))
    prev_angle_valid: np.ndarray = field(default_factory=lambda: np.zeros(ANGLE_NUM_JOINTS, dtype=bool))

    @classmethod
    def create(cls, settings: OneEuroSettings) -> 'TrackletSmoothingState':
        """Create new tracklet state with all filters initialized from settings."""
        # Point filters (x, y for each joint)
        point_filters = [
            (   OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff),
                OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)    )
            for _ in range(POSE_NUM_JOINTS)
        ]

        # Angle filters
        angle_filters = [
            OneEuroFilterAngular(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
            for _ in range(ANGLE_NUM_JOINTS)
        ]

        return cls(
            point_filters=point_filters,
            angle_filters=angle_filters,
            prev_point_valid=np.zeros(POSE_NUM_JOINTS, dtype=bool),
            prev_angle_valid=np.zeros(ANGLE_NUM_JOINTS, dtype=bool)
        )

    def update_all_filters(self, settings: OneEuroSettings) -> None:
        """Update all filter parameters from settings."""
        for x_filter, y_filter in self.point_filters:
            x_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
            y_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)

        for angle_filter in self.angle_filters:
            angle_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)


class PoseSmoother:
    """Smooths pose data using OneEuroFilter, maintaining per-tracklet filter state.

    Note: Not thread-safe. Call add_poses() from a single thread only.
    Settings can be modified from any thread and will be applied on the next add_poses() call.

    Features:
    - Resets filters when joints reappear after occlusion (prevents temporal artifacts)
    - Computes velocity data from OneEuroFilter's internal derivative
    - Handles NaN values gracefully (skips filtering, preserves NaN)
    """

    def __init__(self, settings: Settings) -> None:
        # Smoothing settings with observer pattern
        self.settings = OneEuroSettings(
            frequency=settings.camera_fps,
            min_cutoff=1.0,
            beta=0.025,
            d_cutoff=1.0
        )
        self.settings.add_observer(self._on_settings_changed)

        # Per-tracklet smoothing state
        self._tracklets: dict[int, TrackletSmoothingState] = {}

        # Callbacks
        self.pose_output_callbacks: set[PoseDictCallback] = set()

    def _on_settings_changed(self) -> None:
        """Called automatically when settings change - updates all active filters."""
        for state in self._tracklets.values():
            state.update_all_filters(self.settings)

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth all poses in the dictionary.

        For each pose:
        - Creates filters if needed for new tracklets
        - Resets filters when joints reappear after occlusion
        - Smooths valid points and angles (skips NaN values)
        - Computes velocity data from filter derivatives
        - Cleans up filters for lost tracklets
        """
        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id = pose.tracklet.id

            # Get or create tracklet state
            if tracklet_id not in self._tracklets:
                self._tracklets[tracklet_id] = TrackletSmoothingState.create(self.settings)

            state = self._tracklets[tracklet_id]
            timestamp = pose.time_stamp.timestamp()

            # Smooth points with reset detection
            smoothed_values = pose.point_data.values.copy()
            smoothed_values.flags.writeable = True
            point_velocities = np.full((POSE_NUM_JOINTS, 2), np.nan)

            for joint in PoseJoint:
                is_valid = pose.point_data.valid_mask[joint]
                was_valid = state.prev_point_valid[joint]

                if is_valid:
                    x, y = pose.point_data.values[joint]
                    x_filter, y_filter = state.point_filters[joint]

                    # Reset if joint just reappeared
                    if not was_valid:
                        x_filter.reset()
                        y_filter.reset()

                    # Smooth
                    smoothed_values[joint] = [x_filter(x, timestamp), y_filter(y, timestamp)]

                    # Extract velocity using public property
                    point_velocities[joint] = [x_filter.velocity, y_filter.velocity]

                state.prev_point_valid[joint] = is_valid

            # Smooth angles with reset detection
            smoothed_angles = pose.angle_data.values.copy()
            smoothed_angles.flags.writeable = True
            angle_velocities = np.full(ANGLE_NUM_JOINTS, np.nan)

            for angle_joint in AngleJoint:
                is_valid = pose.angle_data.valid_mask[angle_joint]
                was_valid = state.prev_angle_valid[angle_joint]

                if is_valid:
                    angle = float(pose.angle_data.values[angle_joint])
                    angle_filter = state.angle_filters[angle_joint]

                    # Reset if angle just reappeared
                    if not was_valid:
                        angle_filter.reset()

                    # Smooth
                    smoothed_angles[angle_joint] = angle_filter(angle, timestamp)

                    # Extract angular velocity using public property
                    angle_velocities[angle_joint] = angle_filter.velocity

                state.prev_angle_valid[angle_joint] = is_valid

            # Create smoothed pose
            smoothed_point_data = PosePointData(smoothed_values, pose.point_data.scores)
            smoothed_angle_data = PoseAngleData(smoothed_angles, pose.angle_data.scores)
            velocity_point_data = PosePointData.create_empty()
            velocity_angle_data = PoseAngleData.create_empty()

            smoothed_pose = replace(
                pose,
                point_data=smoothed_point_data,
                angle_data=smoothed_angle_data,
                pose_velocity_data=velocity_point_data,
                angle_velocity_data=velocity_angle_data
            )

            smoothed_poses[pose_id] = smoothed_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]

        self._notify_pose_callbacks(smoothed_poses)

    # CALLBACK METHODS
    def add_pose_callback(self, callback: PoseDictCallback) -> None:
        """Register a callback to be invoked when pose smoothing completes."""
        self.pose_output_callbacks.add(callback)

    def _notify_pose_callbacks(self, poses: PoseDict) -> None:
        """Invoke all registered pose output callbacks."""
        for callback in self.pose_output_callbacks:
            try:
                callback(poses)
            except Exception as e:
                print(f"PoseSmoother: Error in pose output callback: {str(e)}")
                print_exc()