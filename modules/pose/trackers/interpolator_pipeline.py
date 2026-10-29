"""A sequence of InterpolatorNodes for processing a single track."""

from ..nodes import InterpolatorNode
from ..frame import Frame, replace

import logging
logger = logging.getLogger(__name__)


class InterpolatorPipeline:
    """Runs a sequence of InterpolatorNodes for a single track.

    Each node interpolates a different feature at its own rate.
    set() stores targets from the input clock; update(now) produces
    merged interpolated frames at the output (render) clock, stamped with the tick's time.
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

    def update(self, now: float) -> Frame | None:
        """Collect and merge interpolated features from all nodes into one frame stamped ``now``."""
        pose: Frame | None = None
        for node in self._nodes:
            interpolated: Frame | None = node.update(now)
            if interpolated is not None:
                if pose is None:
                    pose = interpolated
                else:
                    ft = node.feature_type
                    pose = replace(pose, {ft: interpolated[ft]})
        return None if pose is None else replace(pose, {}, timestamp=now)

    def reset(self) -> None:
        """Reset all nodes in the pipeline."""
        for node in self._nodes:
            node.reset()

    @property
    def nodes(self) -> list[InterpolatorNode]:
        return self._nodes
