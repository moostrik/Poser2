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
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose
from modules.pose.nodes.filters.algorithms.VectorSmooth import Smooth, VectorSmooth, AngleSmooth, PointSmooth
from modules.pose.features import PoseFeature, ANGLE_NUM_LANDMARKS, POINT_NUM_LANDMARKS, POINT2D_COORD_RANGE
from modules.pose.features import BaseFeature, AngleFeature, BBoxFeature, Point2DFeature, SymmetryFeature


class SmootherConfig(NodeConfigBase):
    """Configuration for pose smoothing with automatic change notification."""

    def __init__(self, frequency: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.025, d_cutoff: float = 1.0) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff

SMOOTHER_LOOKUP = {
    AngleFeature:       AngleSmooth,
    BBoxFeature:        VectorSmooth,
    Point2DFeature:     PointSmooth,
    SymmetryFeature:    VectorSmooth
}

class GenericSmoother(FilterNode):
    """Generic pose feature smoother using OneEuroFilter.

    Handles smoothing for any pose feature by specifying the feature type and attribute name.
    """

    def __init__(self, config: SmootherConfig, feature_type: type, attr_name: str):
        self._config: SmootherConfig = config
        self._feature_type: type = feature_type
        self._attr_name: str = attr_name

        if feature_type not in SMOOTHER_LOOKUP:
            raise ValueError(f"Smoother not implemented for feature type: {feature_type.__name__}")

        self._smoother: Smooth = SMOOTHER_LOOKUP[feature_type](
            vector_size=feature_type.default_range(),
            frequency=self._config.frequency,
            min_cutoff=self._config.min_cutoff,
            beta=self._config.beta,
            d_cutoff=self._config.d_cutoff
        )

        self._config.add_listener(self._on_config_changed)

    @property
    def config(self) -> SmootherConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        feature_data = getattr(pose, self._attr_name)
        self._smoother.add_sample(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, **{self._attr_name: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()

    def _on_config_changed(self) -> None:
        self._smoother.frequency = self._config.frequency
        self._smoother.min_cutoff = self._config.min_cutoff
        self._smoother.beta = self._config.beta
        self._smoother.d_cutoff = self._config.d_cutoff


class AngleSmoother(GenericSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, AngleFeature, "angles")


class BboxSmoother(GenericSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, BBoxFeature, "bbox")


class DeltaSmoother(GenericSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, AngleFeature, "deltas")


class Point2DSmoother(GenericSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, Point2DFeature, "points")


class SymmetrySmoother(GenericSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, SymmetryFeature, "symmetry")


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
        self._point_smoother = Point2DSmoother(config)
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