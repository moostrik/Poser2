# Third-party imports
import numpy as np

# Pose imports
from ..Nodes import FilterNode
from ...features import BBox, Distance
from ...frame import Frame, replace

from modules.settings import BaseSettings, Field


class DistanceExtractorSettings(BaseSettings):
    """Configuration for DistanceExtractor."""
    near_y: Field[float] = Field(0.9, min=0.0, max=1.0, step=0.01,
                                 description="bbox bottom y at closest position → distance 0.0")
    far_y:  Field[float] = Field(0.3, min=0.0, max=1.0, step=0.01,
                                 description="bbox bottom y at farthest position → distance 1.0")


class DistanceExtractor(FilterNode):
    """Estimates a person's distance from their vertical position.

    Maps the bbox bottom edge (feet) between two y thresholds: ``near_y`` → 0.0
    (closest, screen bottom) and ``far_y`` → 1.0 (farthest). The result is
    stored as the normalized Distance feature.
    """

    def __init__(self, config: DistanceExtractorSettings | None = None) -> None:
        self._config = config if config is not None else DistanceExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        bottom: float = pose[BBox].to_rect().bottom
        if np.isnan(bottom):
            return pose

        span: float = self._config.near_y - self._config.far_y
        if span == 0.0:
            distance: float = 0.0
        else:
            distance = float(np.clip((self._config.near_y - bottom) / span, 0.0, 1.0))

        return replace(pose, {Distance: Distance.from_value(distance)})
