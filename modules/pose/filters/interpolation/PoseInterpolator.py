# Standard library imports
from dataclasses import dataclass, replace
from threading import Lock
from typing import Optional

# Pose imports
from modules.pose.filters.interpolation.FeatureAngleInterpolator import FeatureAngleInterpolator
from modules.pose.filters.interpolation.FeatureFloatInterpolator import FeatureFloatInterpolator
from modules.pose.filters.interpolation.FeaturePointInterpolator import FeaturePointInterpolator
from modules.pose.filters.interpolation.FeatureRectInterpolator import FeatureRectInterpolator
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.features.PosePoints import PosePointData
from modules.pose.Pose import Pose, PoseDict

# Local application imports
from modules.utils.PointsAndRects import Rect
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class PoseInterpolatorState:
    """All feature interpolators for a single tracklet."""
    point: FeaturePointInterpolator
    angle: FeatureAngleInterpolator
    delta: FeatureAngleInterpolator
    b_box: FeatureRectInterpolator
    mtime: FeatureFloatInterpolator
    pose_template: Pose  # Last known pose for metadata

    def set_alpha_v(self, value: float) -> None:
        """Set alpha_v for all interpolators in this state."""
        self.point.alpha_v = value
        self.angle.alpha_v = value
        self.delta.alpha_v = value
        self.b_box.alpha_v = value
        self.mtime.alpha_v = value


class PoseInterpolator(PoseFilterBase):
    """Combines multiple pose feature interpolators for complete pose interpolation.

    Manages per-tracklet interpolators for all pose features:
    - Points (keypoints)
    - Angles (joint angles)
    - Delta angles (angle velocities)
    - Bounding box (spatial bounds)
    - Motion time (temporal metric)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize pose interpolator with per-tracklet feature interpolators."""
        super().__init__()
        self._settings: Settings = settings
        self._alpha_v: float = 0.45  # Default velocity smoothing factor

        # Single lock protects all mutable state
        self._lock = Lock()

        # Per-tracklet state: tracklet_id -> all interpolators + template
        self._tracklet_states: dict[int, PoseInterpolatorState] = {}

        # Thread-safe storage for interpolated poses
        self._interpolated_poses: PoseDict = {}

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def alpha_v(self) -> float:
        """Get velocity smoothing factor."""
        return self._alpha_v

    @alpha_v.setter
    def alpha_v(self, value: float) -> None:
        """Set velocity smoothing factor for all interpolators."""
        value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
        self._alpha_v = value

        with self._lock:
            for state in self._tracklet_states.values():
                state.set_alpha_v(value)

    def add_poses(self, poses: PoseDict) -> None:
        """Add input samples to all interpolators (called at input rate)."""
        with self._lock:
            # Detect lost tracklets: those that exist in state but not in new input
            current_tracklet_ids: set[int] = set(poses.keys())
            existing_tracklet_ids: set[int] = set(self._tracklet_states.keys())
            lost_tracklet_ids: set[int] = existing_tracklet_ids - current_tracklet_ids

            # Remove state for lost tracklets
            for tracklet_id in lost_tracklet_ids:
                del self._tracklet_states[tracklet_id]
                self._interpolated_poses.pop(tracklet_id, None)

            for pose in poses.values():
                tracklet_id: int = pose.tracklet.id

                # Create state for new tracklets
                if tracklet_id not in self._tracklet_states:
                    self._tracklet_states[tracklet_id] = self._create_tracklet_state(pose)

                state: PoseInterpolatorState = self._tracklet_states[tracklet_id]

                # Update pose template
                state.pose_template = pose

                # Add samples to all feature interpolators
                state.point.add_feature(pose.point_data)
                state.angle.add_feature(pose.angle_data)
                state.delta.add_feature(pose.delta_data)
                state.b_box.add_feature(pose.bounding_box)
                state.mtime.add_feature(pose.motion_time)

    def update(self, current_time: float | None = None) -> None:
        """Update output for all poses (called at output rate)."""
        with self._lock:
            # Collect interpolated features for all tracklets
            interpolated_poses: PoseDict = {}

            for tracklet_id, state in self._tracklet_states.items():
                # Generate interpolated features
                point_data: PosePointData = state.point.update(current_time)
                angle_data: PoseAngleData = state.angle.update(current_time)
                delta_data: PoseAngleData = state.delta.update(current_time)
                b_box_data: Rect = state.b_box.update(current_time)
                mtime_data: float = state.mtime.update(current_time)

                # Create new pose with interpolated features (preserve metadata)
                interpolated_pose: Pose = replace(
                    state.pose_template,
                    point_data=point_data,
                    angle_data=angle_data,
                    delta_data=delta_data,
                    bounding_box=b_box_data,
                    motion_time=mtime_data
                )

                interpolated_poses[tracklet_id] = interpolated_pose

            # Update stored results
            self._interpolated_poses = interpolated_poses

        # Notify callbacks outside lock to avoid potential deadlock
        self._notify_callbacks(interpolated_poses)

    def _create_tracklet_state(self, pose: Pose) -> PoseInterpolatorState:
        """Create interpolator state for a new tracklet."""
        state = PoseInterpolatorState(
            point=FeaturePointInterpolator(self._settings),
            angle=FeatureAngleInterpolator(self._settings),
            delta=FeatureAngleInterpolator(self._settings),
            b_box=FeatureRectInterpolator(self._settings),
            mtime=FeatureFloatInterpolator(self._settings),
            pose_template=pose
        )
        # Apply current alpha_v to new state
        state.set_alpha_v(self._alpha_v)
        return state

    def reset(self) -> None:
        """Reset all interpolators and clear stored poses."""
        with self._lock:
            self._tracklet_states.clear()
            self._interpolated_poses.clear()

    def get_poses(self) -> PoseDict:
        """Get all currently interpolated poses."""
        with self._lock:
            return self._interpolated_poses.copy()

