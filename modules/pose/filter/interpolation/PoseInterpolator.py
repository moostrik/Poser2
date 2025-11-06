# Standard library imports
from dataclasses import dataclass, replace

# Pose imports
from modules.pose.filter.interpolation.FeatureAngleInterpolator import FeatureAngleInterpolator
from modules.pose.filter.interpolation.FeatureFloatInterpolator import FeatureFloatInterpolator
from modules.pose.filter.interpolation.FeaturePointInterpolator import FeaturePointInterpolator
from modules.pose.filter.interpolation.FeatureRectInterpolator import FeatureRectInterpolator
from modules.pose.filter.PoseFilterBase import PoseFilterBase
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.features.PosePoints import PosePointData
from modules.pose.Pose import Pose
from modules.utils.PointsAndRects import Rect
from modules.Settings import Settings


@dataclass
class PoseInterpolatorState:
    """All feature interpolators for a single pose."""
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
    """Interpolates pose features over time for smooth animation.

    Manages interpolators for all pose features:
    - Points (keypoints)
    - Angles (joint angles)
    - Delta angles (angle velocities)
    - Bounding box (spatial bounds)
    - Motion time (temporal metric)

    This filter is typically called at a higher rate than input poses
    (e.g., 60 FPS output vs 30 FPS input) to generate interpolated frames.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize pose interpolator with feature interpolators."""
        super().__init__()
        self._settings: Settings = settings
        self._alpha_v: float = 0.45  # Default velocity smoothing factor
        self._state: PoseInterpolatorState | None = None

    @property
    def alpha_v(self) -> float:
        """Get velocity smoothing factor."""
        return self._alpha_v

    @alpha_v.setter
    def alpha_v(self, value: float) -> None:
        """Set velocity smoothing factor for all interpolators."""
        value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
        self._alpha_v = value
        if self._state is not None:
            self._state.set_alpha_v(value)

    def process(self, pose: Pose) -> Pose:
        """Add input sample to interpolators (called at input rate).

        This doesn't return interpolated data yet - call update() for that.
        Returns the input pose unchanged.
        """
        # Initialize state on first pose
        if self._state is None:
            self._state = self._create_state(pose)

        # Update pose template
        self._state.pose_template = pose

        # Add samples to all feature interpolators
        self._state.point.add_feature(pose.point_data)
        self._state.angle.add_feature(pose.angle_data)
        self._state.delta.add_feature(pose.delta_data)
        self._state.b_box.add_feature(pose.bounding_box)
        self._state.mtime.add_feature(pose.motion_time)

        return pose

    def update(self, current_time: float | None = None) -> Pose:
        """Generate interpolated pose at current time (called at output rate).

        Args:
            current_time: Timestamp for interpolation (if None, uses current time)

        Returns:
            Pose with interpolated features, or last known pose if no state
        """
        if self._state is None:
            raise RuntimeError("PoseInterpolator: No pose data available for interpolation")

        # Generate interpolated features
        point_data: PosePointData = self._state.point.update(current_time)
        angle_data: PoseAngleData = self._state.angle.update(current_time)
        delta_data: PoseAngleData = self._state.delta.update(current_time)
        b_box_data: Rect = self._state.b_box.update(current_time)
        mtime_data: float = self._state.mtime.update(current_time)

        # Create new pose with interpolated features (preserve metadata)
        interpolated_pose: Pose = replace(
            self._state.pose_template,
            point_data=point_data,
            angle_data=angle_data,
            delta_data=delta_data,
            bounding_box=b_box_data,
            motion_time=mtime_data
        )

        return interpolated_pose

    def _create_state(self, pose: Pose) -> PoseInterpolatorState:
        """Create interpolator state for a new pose."""
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
        """Reset interpolator state."""
        self._state = None

