from ..InterpolatorTracker import InterpolatorTracker

from modules.pose.nodes.interpolators.ChaseInterpolators import (
    ChaseInterpolatorConfig,
    AngleChaseInterpolator,
    BBoxChaseInterpolator,
    PointChaseInterpolator,
    AngleVelChaseInterpolator,
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


class AngleChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for angle chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: AngleChaseInterpolator(config)
        )


class AngleVelChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for delta chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: AngleVelChaseInterpolator(config)
        )


class BPAChaseInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for delta chase interpolation."""

    def __init__(self, num_tracks: int, config: ChaseInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory= [
                lambda: AngleChaseInterpolator(config),
                lambda: PointChaseInterpolator(config),
                lambda: BBoxChaseInterpolator(config),
            ]
        )


from modules.pose.nodes.interpolators.LerpInterpolators import (
    LerpInterpolatorConfig,
    AngleLerpInterpolator,
    BBoxLerpInterpolator,
    PointLerpInterpolator,
    AngleVelLerpInterpolator,
)


class BBoxLerpInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for bounding box lerp interpolation."""

    def __init__(self, num_tracks: int, config: LerpInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: BBoxLerpInterpolator(config)
        )


class PointLerpInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for point lerp interpolation."""

    def __init__(self, num_tracks: int, config: LerpInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: PointLerpInterpolator(config)
        )


class AngleLerpInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for angle lerp interpolation."""

    def __init__(self, num_tracks: int, config: LerpInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: AngleLerpInterpolator(config)
        )


class DeltaLerpInterpolatorTracker(InterpolatorTracker):
    """Convenience tracker for delta lerp interpolation."""

    def __init__(self, num_tracks: int, config: LerpInterpolatorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            interpolator_factory=lambda: AngleVelLerpInterpolator(config)
        )