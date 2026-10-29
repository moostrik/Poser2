"""A sequence of InterpolatorNodes for processing a single track."""

from ..nodes import InterpolatorNode
from ..frame import Frame, replace

import logging
logger = logging.getLogger(__name__)


class InterpolatorPipeline:
    """Runs a sequence of InterpolatorNodes for a single track.

    Each node interpolates a different feature at its own rate.
    set() stores targets from the input clock; update() produces
    merged interpolated frames at the output (render) clock.
    """

    __slots__ = ('_nodes',)

    def __init__(self, nodes: list[InterpolatorNode]) -> None:
        if not nodes:
            raise ValueError("InterpolatorPipeline: node list must not be empty.")
        self._nodes = nodes

    def set(self, pose: Frame | None) -> None:
        """Forward target pose to all interpolator nodes."""
        for node in self._nodes:
            node.set(pose)

    def update(self) -> Frame | None:
        """Collect and merge interpolated features from all nodes."""
        pose: Frame | None = None
        for node in self._nodes:
            interpolated: Frame | None = node.update()
            if interpolated is not None:
                if pose is None:
                    pose = interpolated
                else:
                    ft = node.feature_type
                    pose = replace(pose, {ft: interpolated[ft]})
        return pose

    def reset(self) -> None:
        """Reset all nodes in the pipeline."""
        for node in self._nodes:
            node.reset()

    @property
    def nodes(self) -> list[InterpolatorNode]:
        return self._nodes
