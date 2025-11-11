"""Pose smoothing filters using OneEuroFilter for noise reduction.

Provides smoothing for angles, points, and deltas with proper handling
of circular values and coordinate clamping.
"""

# Standard library imports
from abc import abstractmethod
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose
from modules.pose.filters.general.algorithms.VectorSmooth import Smooth, AngleSmooth, PointSmooth
from modules.pose.features import PoseFeatureData, ANGLE_NUM_LANDMARKS, POINT_NUM_LANDMARKS, POINT2D_COORD_RANGE


class SmootherConfig(NodeConfigBase):
    """Configuration for pose smoothing with automatic change notification."""

    def __init__(self, frequency: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.025, d_cutoff: float = 1.0) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff


class SmootherBase(FilterNode):
    """Base class for pose smoothers.

    Handles common smoothing logic. Subclasses only need to specify:
    - Which smoother instance to create (_initialize_smoother)
    - Which feature to extract from pose (_get_feature_data)
    - How to replace feature data in pose (_replace_feature_data)

    Score handling: Smoothing never introduces NaN for valid inputs, it only
    preserves NaN from input or filters valid values. Therefore, original scores
    are preserved by default. Subclasses can override _create_smoothed_data for
    custom score handling if needed.
    """

    def __init__(self, config: SmootherConfig) -> None:
        self._config: SmootherConfig = config
        self._smoother: Smooth
        self._initialize_smoother()
        self._config.add_listener(self._on_config_changed)

    @property
    def config(self) -> SmootherConfig:
        """Access the smoother's configuration."""
        return self._config

    @abstractmethod
    def _initialize_smoother(self) -> None:
        """Create the appropriate smoother instance."""
        pass

    @abstractmethod
    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        """Extract the feature data to smooth from the pose."""
        pass

    def _create_smoothed_data(self, original_data: PoseFeatureData, smoothed_values: np.ndarray) -> PoseFeatureData:
        """Create new feature data with smoothed values and original scores.

        Default implementation preserves original scores since smoothing never
        introduces NaN for valid inputs - it only preserves NaN from input or
        filters valid values (always produces valid output).

        Subclasses can override for custom score handling if needed.
        """
        return type(original_data)(values=smoothed_values, scores=original_data.scores)

    @abstractmethod
    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        """Create new pose with replaced feature data."""
        pass

    def process(self, pose: Pose) -> Pose:
        """Add current feature data to smoother and return pose with smoothed values."""

        # Get feature data
        feature_data = self._get_feature_data(pose)

        # Add sample and get smoothed values
        self._smoother.add_sample(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value

        # Create new feature data with smoothed values
        smoothed_data = self._create_smoothed_data(feature_data, smoothed_values)

        # Return new pose with smoothed feature
        return self._replace_feature_data(pose, smoothed_data)

    def reset(self) -> None:
        """Reset the smoother's internal state (clear filter history)."""
        self._smoother.reset()

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating smoother parameters."""
        self._smoother.frequency = self._config.frequency
        self._smoother.min_cutoff = self._config.min_cutoff
        self._smoother.beta = self._config.beta
        self._smoother.d_cutoff = self._config.d_cutoff


class AngleSmoother(SmootherBase):
    """Smooths angle data using OneEuroFilter with angular wrapping.

    Uses AngleSmoother which handles circular wrapping of angle values
    by filtering sin/cos components separately.
    """

    def _initialize_smoother(self) -> None:
        self._smoother = AngleSmooth(
            vector_size=ANGLE_NUM_LANDMARKS,
            frequency=self._config.frequency,
            min_cutoff=self._config.min_cutoff,
            beta=self._config.beta,
            d_cutoff=self._config.d_cutoff
        )

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.angle_data

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, angle_data=new_data)


class PointSmoother(SmootherBase):
    """Smooths point data using OneEuroFilter with coordinate clamping.

    Uses PointSmoother which clamps coordinates to [0, 1] range and handles 2D data.
    """

    def _initialize_smoother(self) -> None:
        self._smoother = PointSmooth(
            num_points=POINT_NUM_LANDMARKS,
            frequency=self._config.frequency,
            min_cutoff=self._config.min_cutoff,
            beta=self._config.beta,
            d_cutoff=self._config.d_cutoff,
            clamp_range=POINT2D_COORD_RANGE
        )

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.point_data

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, point_data=new_data)


class DeltaSmoother(SmootherBase):
    """Smooths delta data using OneEuroFilter with angular wrapping.

    Uses AngleSmoother since delta represents angle changes (circular values).
    """

    def _initialize_smoother(self) -> None:
        self._smoother = AngleSmooth(
            vector_size=ANGLE_NUM_LANDMARKS,
            frequency=self._config.frequency,
            min_cutoff=self._config.min_cutoff,
            beta=self._config.beta,
            d_cutoff=self._config.d_cutoff
        )

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.delta_data

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, delta_data=new_data)


class PoseSmoother(FilterNode):
    """Smooths all pose features (angles, points, and deltas) using OneEuroFilter.

    Applies the same smoothing configuration to all features. For independent
    control of each feature, use PoseAngleSmoother, PosePointSmoother, and
    PoseDeltaSmoother separately.
    """

    def __init__(self, config: SmootherConfig) -> None:
        self._config: SmootherConfig = config

        # Create individual smoothers for each feature
        self._angle_smoother = AngleSmoother(config)
        self._point_smoother = PointSmoother(config)
        self._delta_smoother = DeltaSmoother(config)

    @property
    def config(self) -> SmootherConfig:
        """Access the smoother's configuration."""
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Smooth all features in the pose."""
        pose = self._angle_smoother.process(pose)
        pose = self._point_smoother.process(pose)
        pose = self._delta_smoother.process(pose)
        return pose

    def reset(self) -> None:
        """Reset all smoothers' internal state."""
        self._angle_smoother.reset()
        self._point_smoother.reset()
        self._delta_smoother.reset()