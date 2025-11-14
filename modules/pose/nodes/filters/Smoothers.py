"""Pose smoothing filters using OneEuroFilter for noise reduction.

Provides smoothing for angles, points, and deltas with proper handling
of circular values and coordinate clamping.
"""

# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, Symmetry
from modules.pose.nodes._utils.VectorSmooth import VectorSmooth, AngleSmooth, PointSmooth
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose


class SmootherConfig(NodeConfigBase):
    """Configuration for pose smoothing with automatic change notification."""

    def __init__(self, frequency: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.025, d_cutoff: float = 1.0) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff


class FeatureSmoother(FilterNode):
    """Generic pose feature smoother using OneEuroFilter.

    Args:
        config: Smoother configuration
        feature_class: Feature class type (e.g., AngleFeature, Point2DFeature)
        attr_name: Name of the pose attribute to smooth

    Example:
        smoother = GenericSmoother(config, AngleFeature, "angles")
        smoother = GenericSmoother(config, Point2DFeature, "points")
    """

    # Registry mapping feature classes to smoother classes
    SMOOTHER_REGISTRY = {
        Angles: AngleSmooth,
        BBox: VectorSmooth,
        Points2D: PointSmooth,
        Symmetry: VectorSmooth,
    }

    def __init__(self, config: SmootherConfig, feature_class: type, attr_name: str):
        if feature_class not in self.SMOOTHER_REGISTRY:
            valid_classes = [cls.__name__ for cls in self.SMOOTHER_REGISTRY.keys()]
            raise ValueError(
                f"Unknown feature class '{feature_class.__name__}'. "
                f"Must be one of: {valid_classes}"
            )

        self._config = config
        self._attr_name = attr_name
        self._feature_class = feature_class

        smoother_cls = self.SMOOTHER_REGISTRY[feature_class]
        self._smoother = smoother_cls(
            vector_size=len(feature_class.feature_enum()),
            frequency=config.frequency,
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
            clamp_range=feature_class.default_range()
        )
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass  # Config already cleaned up or listener not found

    def _on_config_changed(self) -> None:
        self._smoother.frequency = self._config.frequency
        self._smoother.min_cutoff = self._config.min_cutoff
        self._smoother.beta = self._config.beta
        self._smoother.d_cutoff = self._config.d_cutoff

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


# Convenience classes
class AngleSmoother(FeatureSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, Angles, "angles")


class DeltaSmoother(FeatureSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, Angles, "deltas")


class BBoxSmoother(FeatureSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, BBox, "bbox")


class PointSmoother(FeatureSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, Points2D, "points")


class SymmetrySmoother(FeatureSmoother):
    def __init__(self, config: SmootherConfig) -> None:
        super().__init__(config, Symmetry, "symmetry")