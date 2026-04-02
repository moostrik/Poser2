"""Convenience tracker factories for WindowNode-based tracking."""

from __future__ import annotations
from threading import Lock
from traceback import print_exc

from ..WindowTracker import WindowTracker
from modules.pose.nodes import AngleMotionWindowNode, AngleSymmetryWindowNode, AngleVelocityWindowNode, AngleWindowNode, BBoxWindowNode, SimilarityWindowNode, WindowNodeSettings
from modules.pose.nodes.windows.WindowNode import WindowNode
from modules.pose.frame import FrameDict, FrameField, FeatureWindow, FeatureWindowDict, FeatureWindowDictCallback, FrameWindowDict, FrameWindowDictCallback


class AllWindowTracker:
    """Auto-creates a WindowTracker for every BaseScalarFeature field in Frame.

    Discovers all scalar features via FrameField.get_scalar_fields() and creates
    one WindowTracker per feature. Results are emitted as:
        {FrameField: {track_id: FeatureWindow}}

    Usage:
        tracker = AllWindowTracker(num_tracks, config)
        some_filter.add_poses_callback(tracker.process)
        tracker.add_callback(data_hub.set_frame_windows)
        tracker.add_window_callback(FrameField.angles, similarity_callback)
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()

        self._trackers: dict[FrameField, WindowTracker] = {}
        for ff in FrameField.get_scalar_fields():
            self._trackers[ff] = WindowTracker(
                num_tracks=num_tracks,
                window_factory=lambda f=ff: WindowNode(f, config)
            )

        self._callbacks: set[FrameWindowDictCallback] = set()
        self._callback_lock = Lock()

    @property
    def fields(self) -> list[FrameField]:
        """Return all tracked FrameFields."""
        return list(self._trackers.keys())

    def process(self, poses: FrameDict) -> None:
        """Process poses through all inner WindowTrackers and emit result."""
        result: FrameWindowDict = {}
        for field, tracker in self._trackers.items():
            result[field] = tracker.process(poses)

        self._notify_callbacks(result)

    def get_tracker(self, field: FrameField) -> WindowTracker:
        """Get the WindowTracker for a specific feature field."""
        return self._trackers[field]

    # Per-field window callbacks (for selective consumers like WindowSimilarity)
    def add_window_callback(self, field: FrameField, callback: FeatureWindowDictCallback) -> None:
        """Register a window callback for a specific feature."""
        self._trackers[field].add_window_callback(callback)

    def remove_window_callback(self, field: FrameField, callback: FeatureWindowDictCallback) -> None:
        """Unregister a window callback for a specific feature."""
        self._trackers[field].remove_window_callback(callback)

    # Combined callbacks (for DataHub storage)
    def add_callback(self, callback: FrameWindowDictCallback) -> None:
        """Register a callback for the {FrameField: {track_id: FeatureWindow}} dict."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def remove_callback(self, callback: FrameWindowDictCallback) -> None:
        """Unregister a combined callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

    def _notify_callbacks(self, combined: FrameWindowDict) -> None:
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(combined)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def reset(self) -> None:
        """Reset all window trackers."""
        for tracker in self._trackers.values():
            tracker.reset()


# DEPRECATED: Use AllWindowTracker instead.
# These classes are kept for backward compatibility.

class AngleWindowTracker(WindowTracker):
    """Convenience tracker for Angles feature windows.

    Buffers angle values over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleWindowNode(config)
        )


class AngleVelocityWindowTracker(WindowTracker):
    """Convenience tracker for AngleVelocity feature windows.

    Buffers angular velocity values over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleVelocityWindowNode(config)
        )


class AngleMotionWindowTracker(WindowTracker):
    """Convenience tracker for AngleMotion feature windows.

    Buffers angular motion magnitude over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleMotionWindowNode(config)
        )


class AngleSymmetryWindowTracker(WindowTracker):
    """Convenience tracker for AngleSymmetry feature windows.

    Buffers angular symmetry metrics over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleSymmetryWindowNode(config)
        )


class BBoxWindowTracker(WindowTracker):
    """Convenience tracker for BBox feature windows.

    Buffers bounding box trajectories over time and returns windows with shape (time, 4).
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: BBoxWindowNode(config)
        )


class SimilarityWindowTracker(WindowTracker):
    """Convenience tracker for Similarity feature windows.

    Buffers similarity trajectories over time and returns windows with shape (time, 1).
    """

    def __init__(self, num_tracks: int, config: WindowNodeSettings | None = None) -> None:
        if config is None:
            config = WindowNodeSettings()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: SimilarityWindowNode(config)
        )