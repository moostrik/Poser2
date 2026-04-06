"""Convenience tracker for WindowNode-based tracking."""

from __future__ import annotations

from ..WindowTracker import WindowTracker
from modules.pose.nodes.windows.WindowNode import WindowNode, WindowNodeSettings
from modules.pose.features import SCALAR_FEATURES, BaseFeature
from modules.pose.frame import FrameDict, FeatureWindowDictCallback, FrameWindowDict, FrameWindowDictCallbackMixin


class FrameWindowTracker(FrameWindowDictCallbackMixin):
    """Creates a WindowTracker for each scalar feature type.

    By default tracks all scalar features (from SCALAR_FEATURES).
    Results are emitted as:
        {type[BaseFeature]: {track_id: FeatureWindow}}

    Usage:
        tracker = FrameWindowTracker(num_tracks, config)
        some_filter.add_frames_callback(tracker.process)
        tracker.add_frame_window_callback(data_hub.set_frame_windows)
        tracker.add_field_window_callback(Angles, similarity_callback)
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        super().__init__()

        if config is None:
            config = WindowNodeSettings()

        self._trackers: dict[type[BaseFeature], WindowTracker] = {}
        for ft in SCALAR_FEATURES:
            self._trackers[ft] = WindowTracker(
                num_tracks=num_tracks,
                window_factory=lambda f=ft: WindowNode(f, config)
            )

    @property
    def feature_types(self) -> list[type[BaseFeature]]:
        """Return all tracked feature types."""
        return list(self._trackers.keys())

    def process(self, poses: FrameDict) -> None:
        """Process poses through all inner WindowTrackers and emit result."""
        result: FrameWindowDict = {}
        for feature_type, tracker in self._trackers.items():
            result[feature_type] = tracker.process(poses)

        self._notify_frame_window_callbacks(result)

    def get_tracker(self, feature_type: type[BaseFeature]) -> WindowTracker:
        """Get the WindowTracker for a specific feature type."""
        return self._trackers[feature_type]

    # Per-field window callbacks (for selective consumers)
    def add_field_window_callback(self, feature_type: type[BaseFeature], callback: FeatureWindowDictCallback) -> None:
        """Register a window callback for a specific feature."""
        self._trackers[feature_type].add_windows_callback(callback)

    def remove_field_window_callback(self, feature_type: type[BaseFeature], callback: FeatureWindowDictCallback) -> None:
        """Unregister a window callback for a specific feature."""
        self._trackers[feature_type].remove_windows_callback(callback)

    def reset(self) -> None:
        """Reset all window trackers."""
        for tracker in self._trackers.values():
            tracker.reset()