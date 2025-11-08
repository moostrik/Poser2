# Standard library imports
from dataclasses import dataclass, replace

# Pose imports
from modules.pose.filter.interpolation.FeatureAngleInterpolator import FeatureAngleInterpolator
from modules.pose.filter.interpolation.FeatureFloatInterpolator import FeatureFloatInterpolator
from modules.pose.filter.interpolation.FeaturePointInterpolator import FeaturePointInterpolator
from modules.pose.filter.interpolation.FeatureRectInterpolator import FeatureRectInterpolator
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.features.PosePoints import PosePointData
from modules.pose.Pose import Pose
from modules.utils.PointsAndRects import Rect

from modules.pose.filter.interpolation.PoseInterpolatorConfig import PoseInterpolatorConfig

@dataclass
class PoseInterpolatorState:
    """All feature interpolators for a single pose."""
    point: FeaturePointInterpolator
    angle: FeatureAngleInterpolator
    delta: FeatureAngleInterpolator
    b_box: FeatureRectInterpolator
    mtime: FeatureFloatInterpolator
    pose_template: Pose


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

    def __init__(self, config: PoseInterpolatorConfig) -> None:
        """Initialize pose interpolator with feature interpolators."""
        super().__init__()
        self._config: PoseInterpolatorConfig = config
        self._state: PoseInterpolatorState | None = None

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
            point=FeaturePointInterpolator(self._config),
            angle=FeatureAngleInterpolator(self._config),
            delta=FeatureAngleInterpolator(self._config),
            b_box=FeatureRectInterpolator(self._config),
            mtime=FeatureFloatInterpolator(self._config),
            pose_template=pose
        )
        return state

    def reset(self) -> None:
        """Reset interpolator state."""
        self._state = None

