"""Tracks feature windows for all scalar features across multiple tracks."""

from .WindowNode import WindowNode, WindowNodeSettings
from ..features import SCALAR_FEATURES, BaseFeature
from ..frame import FrameDict, FrameWindowDict, FrameWindowDictCallbackMixin

import logging
logger = logging.getLogger(__name__)


class WindowTracker(FrameWindowDictCallbackMixin):
    """Maintains independent WindowNode ring buffers per (feature_type, track_id).

    Fans out each incoming FrameDict across all scalar feature types,
    collects per-track FeatureWindows, and emits FrameWindowDict.
    Automatically resets buffers when a track disappears.
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        super().__init__()

        if config is None:
            config = WindowNodeSettings()

        self._nodes: dict[type[BaseFeature], dict[int, WindowNode]] = {
            ft: {i: WindowNode(ft, config) for i in range(num_tracks)}
            for ft in SCALAR_FEATURES
        }
        self._track_ids = set(range(num_tracks))

    def process(self, poses: FrameDict) -> None:
        """Process poses through all window nodes and emit FrameWindowDict."""

        # Reset buffers for tracks that are no longer present
        missing = self._track_ids - poses.keys()
        if missing:
            for ft_nodes in self._nodes.values():
                for track_id in missing:
                    ft_nodes[track_id].reset()

        result: FrameWindowDict = {}
        for ft, ft_nodes in self._nodes.items():
            windows = {}
            for track_id, pose in poses.items():
                try:
                    window = ft_nodes[track_id].process(pose)
                    if window is not None:
                        windows[track_id] = window
                except Exception as e:
                    logger.error(f"WindowTracker: Error processing track {track_id} feature {ft.__name__}: {e}")
            result[ft] = windows

        self._notify_windows_callbacks(result)

    def reset(self) -> None:
        """Reset all window buffers."""
        for ft_nodes in self._nodes.values():
            for node in ft_nodes.values():
                node.reset()
