"""Convenience tracker for WindowNode-based tracking."""

from __future__ import annotations

from ..WindowTracker import WindowTracker
from modules.pose.nodes.windows.WindowNode import WindowNode, WindowNodeSettings
from modules.pose.frame import FrameDict, FrameField, FeatureWindowDict, FeatureWindowDictCallback, FrameWindowDict, FrameWindowDictCallbackMixin


class FrameWindowTracker(FrameWindowDictCallbackMixin):
    """Auto-creates a WindowTracker for every scalar FrameField.

    Discovers all scalar features via FrameField.get_scalar_fields() and creates
    one WindowTracker per feature. Results are emitted as:
        {FrameField: {track_id: FeatureWindow}}

    Usage:
        tracker = FrameWindowTracker(num_tracks, config)
        some_filter.add_frames_callback(tracker.process)
        tracker.add_frame_window_callback(data_hub.set_frame_windows)
        tracker.add_field_window_callback(FrameField.angles, similarity_callback)
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        super().__init__()

        if config is None:
            config = WindowNodeSettings()

        self._trackers: dict[FrameField, WindowTracker] = {}
        for ff in FrameField.get_scalar_fields():
            self._trackers[ff] = WindowTracker(
                num_tracks=num_tracks,
                window_factory=lambda f=ff: WindowNode(f, config)
            )

    @property
    def fields(self) -> list[FrameField]:
        """Return all tracked FrameFields."""
        return list(self._trackers.keys())

    def process(self, poses: FrameDict) -> None:
        """Process poses through all inner WindowTrackers and emit result."""
        result: FrameWindowDict = {}
        for field, tracker in self._trackers.items():
            result[field] = tracker.process(poses)

        self._notify_frame_window_callbacks(result)

    def get_tracker(self, field: FrameField) -> WindowTracker:
        """Get the WindowTracker for a specific feature field."""
        return self._trackers[field]

    # Per-field window callbacks (for selective consumers)
    def add_field_window_callback(self, field: FrameField, callback: FeatureWindowDictCallback) -> None:
        """Register a window callback for a specific feature."""
        self._trackers[field].add_windows_callback(callback)

    def remove_field_window_callback(self, field: FrameField, callback: FeatureWindowDictCallback) -> None:
        """Unregister a window callback for a specific feature."""
        self._trackers[field].remove_windows_callback(callback)

    def reset(self) -> None:
        """Reset all window trackers."""
        for tracker in self._trackers.values():
            tracker.reset()