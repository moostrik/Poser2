from ..InterpolatorTracker import InterpolatorTracker

from modules.pose.nodes.interpolators.ChaseInterpolators import (
    ChaseInterpolatorConfig,
    AngleChaseInterpolator,
    BBoxChaseInterpolator,
    PointChaseInterpolator,
    DeltaChaseInterpolator,
)

class AngleChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for angle chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: AngleChaseInterpolator(config)
        )

class BBoxChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for bounding box chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: BBoxChaseInterpolator(config)
        )


class PointChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for point chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: PointChaseInterpolator(config)
        )


class DeltaChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for delta chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: DeltaChaseInterpolator(config)
        )


class APChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for delta chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory= [
                lambda: AngleChaseInterpolator(config),
                lambda: PointChaseInterpolator(config)
            ]
        )


