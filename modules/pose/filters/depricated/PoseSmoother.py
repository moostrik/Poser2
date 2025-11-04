from dataclasses import replace, dataclass, field
from traceback import print_exc
from typing import Callable

import numpy as np

from modules.pose.Pose import Pose, PoseDict, PoseDictCallback, Rect
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS, PoseJoint
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS, AngleJoint

from modules.utils.Smoothing import OneEuroFilter, OneEuroFilterAngular
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class PoseSmootherSettings:
    """Configuration parameters for the 1€ Filter.

    Attributes:
        frequency: INPUT data frequency in Hz (default: 30.0)
        min_cutoff: Minimum cutoff frequency for position smoothing (default: 1.0)
        beta: Speed coefficient - higher values reduce lag but increase jitter (default: 0.025)
        d_cutoff: Cutoff frequency for derivative smoothing (default: 1.0)
        smooth_points: Enable point smoothing (default: True)
        smooth_angles: Enable angle smoothing (default: True)
        smooth_bbox: Enable bounding box smoothing (default: True)
        reset_on_reappear: Reset filters when joints reappear after occlusion (default: False)
    """
    frequency: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.025       # recommended: 0.007 to 0.05 for generative a/v
    d_cutoff: float = 1.0
    smooth_points: bool = True
    smooth_angles: bool = True
    smooth_bbox: bool = True  # NEW
    reset_on_reappear: bool = False

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

@dataclass
class PoseSmoothingState:
    """All smoothing state for one tracked person/tracklet.

    Encapsulates filters and validity tracking to ensure they stay synchronized.
    """
    point_filters: list[tuple[OneEuroFilter, OneEuroFilter]] = field(default_factory=list)  # (x, y) per joint
    angle_filters: list[OneEuroFilterAngular] = field(default_factory=list)
    bbox_filters: tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter] | None = None  # NEW: (x, y, w, h)
    prev_point_valid: np.ndarray = field(default_factory=lambda: np.zeros(POSE_NUM_JOINTS, dtype=bool))
    prev_angle_valid: np.ndarray = field(default_factory=lambda: np.zeros(ANGLE_NUM_JOINTS, dtype=bool))

    @classmethod
    def create(cls, settings: PoseSmootherSettings) -> 'PoseSmoothingState':
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

        # Bounding box filters (x, y, width, height)
        bbox_filters = (
            OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff),  # x
            OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff),  # y
            OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff),  # width
            OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)   # height
        )

        return cls(
            point_filters=point_filters,
            angle_filters=angle_filters,
            bbox_filters=bbox_filters,
            prev_point_valid=np.zeros(POSE_NUM_JOINTS, dtype=bool),
            prev_angle_valid=np.zeros(ANGLE_NUM_JOINTS, dtype=bool)
        )

    def update_all_filters(self, settings: PoseSmootherSettings) -> None:
        """Update all filter parameters from settings."""
        for x_filter, y_filter in self.point_filters:
            x_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
            y_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)

        for angle_filter in self.angle_filters:
            angle_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)

        # Update bbox filters
        if self.bbox_filters:
            for bbox_filter in self.bbox_filters:
                bbox_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)


class PoseSmoother:
    """Smooths pose data using OneEuroFilter, maintaining per-tracklet filter state.

    Note: Not thread-safe. Call add_poses() from a single thread only.

    Features:
    - Resets filters when joints reappear after occlusion (prevents temporal artifacts)
    - Computes velocity data from OneEuroFilter's internal derivative
    - Handles NaN values gracefully (skips filtering, preserves NaN)
    """

    def __init__(self, settings: Settings) -> None:
        # Smoothing settings
        self.settings: PoseSmootherSettings = PoseSmootherSettings(
            frequency = settings.camera_fps,
            min_cutoff= 2.0,
            beta =      0.05,
            d_cutoff =  1.0,
            smooth_points = True,
            smooth_angles = True,
            smooth_bbox = True,
            reset_on_reappear = False
        )

        # Per-tracklet smoothing state
        self._tracklets: dict[int, PoseSmoothingState] = {}

        # Callbacks
        self.pose_output_callbacks: set[PoseDictCallback] = set()

        self.hotreload = HotReloadMethods(self.__class__, True, True)
        self.hotreload.add_file_changed_callback(self._on_hot_reload)
        self._on_hot_reload()

    def _on_hot_reload(self) -> None:
        self._uds()

    def _uds(self) -> None:
        settings: PoseSmootherSettings = PoseSmootherSettings(
            frequency = self.settings.frequency,
            min_cutoff= 2.0,
            beta =      0.05,
            d_cutoff =  1.0,
            smooth_points = True,
            smooth_angles = True,
            smooth_bbox = True,
            reset_on_reappear = False
        )
        settings: PoseSmootherSettings = PoseSmootherSettings(
            frequency = 24,
            min_cutoff= 0.2,
            beta =      0.2,
            d_cutoff =  1.0,
            smooth_points = True,
            smooth_angles = True,
            smooth_bbox = True,
            reset_on_reappear = False
        )
        self.update_settings(settings)

    def update_settings(self, settings: PoseSmootherSettings) -> None:
        """Update all filter settings from a new settings object.

        Args:
            settings: New OneEuroSettings to apply

        Example:
            new_settings = OneEuroSettings(frequency=60, beta=0.05)
            smoother.update_settings(new_settings)
        """
        self.settings = settings

        # Apply to all active filters
        for state in self._tracklets.values():
            state.update_all_filters(self.settings)

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth all poses in the dictionary.

        For each pose:
        - Creates filters if needed for new tracklets
        - Resets filters when joints reappear after occlusion
        - Smooths valid points and angles (skips NaN values)
        - Cleans up filters for lost tracklets
        """

        self._add_poses(poses)

    def _add_poses(self, poses: PoseDict) -> None:

        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Get or create tracklet state
            if tracklet_id not in self._tracklets:
                self._tracklets[tracklet_id] = PoseSmoothingState.create(self.settings)

            state: PoseSmoothingState = self._tracklets[tracklet_id]
            timestamp: float = pose.time_stamp.timestamp()

            # Smooth points with reset detection (or skip if disabled)
            if self.settings.smooth_points:
                smoothed_values: np.ndarray = pose.point_data.values.copy()
                smoothed_values.flags.writeable = True

                for joint in PoseJoint:
                    is_valid = pose.point_data.valid_mask[joint]
                    was_valid = state.prev_point_valid[joint]

                    if is_valid:
                        x, y = pose.point_data.values[joint]
                        x_filter, y_filter = state.point_filters[joint]

                        # Reset if joint just reappeared (only if reset enabled)
                        if not was_valid and self.settings.reset_on_reappear:
                            x_filter.reset()
                            y_filter.reset()

                        # Smooth
                        smoothed_values[joint] = [x_filter(x, timestamp), y_filter(y, timestamp)]

                    state.prev_point_valid[joint] = is_valid

                smoothed_point_data = PosePointData(smoothed_values, pose.point_data.scores)
            else:
                # Use original points without smoothing
                smoothed_point_data: PosePointData = pose.point_data

            # Smooth angles with reset detection (or skip if disabled)
            if self.settings.smooth_angles:
                smoothed_angles: np.ndarray = pose.angle_data.values.copy()
                smoothed_angles.flags.writeable = True

                for angle_joint in AngleJoint:
                    is_valid = pose.angle_data.valid_mask[angle_joint]
                    was_valid = state.prev_angle_valid[angle_joint]

                    if is_valid:
                        angle = float(pose.angle_data.values[angle_joint])
                        angle_filter = state.angle_filters[angle_joint]

                        # Reset if angle just reappeared (only if reset enabled)
                        if not was_valid and self.settings.reset_on_reappear:
                            angle_filter.reset()

                        # Smooth
                        smoothed_angles[angle_joint] = angle_filter(angle, timestamp)

                    state.prev_angle_valid[angle_joint] = is_valid

                smoothed_angle_data = PoseAngleData(smoothed_angles, pose.angle_data.scores)
            else:
                # Use original angles without smoothing
                smoothed_angle_data: PoseAngleData = pose.angle_data

            # Smooth bounding box (or skip if disabled)
            if self.settings.smooth_bbox and state.bbox_filters:
                x_filter, y_filter, w_filter, h_filter = state.bbox_filters

                smoothed_x = x_filter(pose.bounding_box.x, timestamp)
                smoothed_y = y_filter(pose.bounding_box.y, timestamp)
                smoothed_w = w_filter(pose.bounding_box.width, timestamp)
                smoothed_h = h_filter(pose.bounding_box.height, timestamp)

                smoothed_bbox = Rect(smoothed_x, smoothed_y, smoothed_w, smoothed_h)
            else:
                # Use original bounding box without smoothing
                smoothed_bbox: Rect = pose.bounding_box

            # Create smoothed pose
            smoothed_pose: Pose = replace(
                pose,
                point_data=smoothed_point_data,
                angle_data=smoothed_angle_data,
                bounding_box=smoothed_bbox
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