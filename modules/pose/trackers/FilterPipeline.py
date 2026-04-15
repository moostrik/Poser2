"""A sequence of FilterNodes for processing a single track."""

from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame

import logging
logger = logging.getLogger(__name__)


class FilterPipeline:
    """Runs a sequence of FilterNodes on a single pose frame.

    Each pipeline instance holds independent per-track state via its nodes.
    Shared (stateless or externally-fed) nodes may appear in multiple pipelines.
    """

    __slots__ = ('_nodes',)

    def __init__(self, nodes: list[FilterNode]) -> None:
        if not nodes:
            raise ValueError("FilterPipeline: node list must not be empty.")
        self._nodes = nodes

    def process(self, pose: Frame) -> Frame:
        """Run all nodes in sequence, returning the enriched frame."""
        for node in self._nodes:
            pose = node.process(pose)
        return pose

    def reset(self) -> None:
        """Reset all nodes in the pipeline."""
        for node in self._nodes:
            node.reset()

    @property
    def nodes(self) -> list[FilterNode]:
        return self._nodes
