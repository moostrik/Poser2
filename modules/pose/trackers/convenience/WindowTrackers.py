"""Convenience tracker factories for WindowNode-based tracking."""

from ..WindowTracker import WindowTracker
from modules.pose.nodes.windows import AngleMotionWindowNode, AngleSymmetryWindowNode, AngleVelocityWindowNode, AngleWindowNode, BBoxWindowNode, Points2DWindowNode, WindowNodeConfig


class AngleWindowTracker(WindowTracker):
    """Convenience tracker for Angles feature windows.

    Buffers angle values over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeConfig | None = None) -> None:
        if config is None:
            config = WindowNodeConfig()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleWindowNode(config)
        )


class AngleVelocityWindowTracker(WindowTracker):
    """Convenience tracker for AngleVelocity feature windows.

    Buffers angular velocity values over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeConfig | None = None) -> None:
        if config is None:
            config = WindowNodeConfig()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleVelocityWindowNode(config)
        )


class Points2DWindowTracker(WindowTracker):
    """Convenience tracker for Points2D feature windows.

    Buffers 2D keypoint trajectories over time and returns windows with shape (time, 17).
    """

    def __init__(self, num_tracks: int, config: WindowNodeConfig | None = None) -> None:
        if config is None:
            config = WindowNodeConfig()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: Points2DWindowNode(config)
        )


class AngleMotionWindowTracker(WindowTracker):
    """Convenience tracker for AngleMotion feature windows.

    Buffers angular motion magnitude over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeConfig | None = None) -> None:
        if config is None:
            config = WindowNodeConfig()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleMotionWindowNode(config)
        )


class AngleSymmetryWindowTracker(WindowTracker):
    """Convenience tracker for AngleSymmetry feature windows.

    Buffers angular symmetry metrics over time and returns windows with shape (time, 9).
    """

    def __init__(self, num_tracks: int, config: WindowNodeConfig | None = None) -> None:
        if config is None:
            config = WindowNodeConfig()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: AngleSymmetryWindowNode(config)
        )


class BBoxWindowTracker(WindowTracker):
    """Convenience tracker for BBox feature windows.

    Buffers bounding box trajectories over time and returns windows with shape (time, 4).
    """

    def __init__(self, num_tracks: int, config: WindowNodeConfig | None = None) -> None:
        if config is None:
            config = WindowNodeConfig()
        super().__init__(
            num_tracks=num_tracks,
            window_factory=lambda: BBoxWindowNode(config)
        )
