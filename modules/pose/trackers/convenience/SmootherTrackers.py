from ..FilterTracker import FilterTracker
from modules.pose.nodes import (
    EuroSmootherConfig,
    AngleEuroSmoother,
    BBoxEuroSmoother,
    PointEuroSmoother,
    DeltaEuroSmoother
)

class AngleSmootherTracker(FilterTracker):
    """Convenience tracker for angle smoothing interpolation."""

    def __init__(self, num_tracks: int, config: EuroSmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: AngleEuroSmoother(config)
        )


class BboxSmootherTracker(FilterTracker):
    """Convenience tracker for bounding box smoothing interpolation."""

    def __init__(self, num_tracks: int, config: EuroSmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: BBoxEuroSmoother(config)
        )


class PointSmootherTracker(FilterTracker):
    """Convenience tracker for point smoothing interpolation."""

    def __init__(self, num_tracks: int, config: EuroSmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: PointEuroSmoother(config)
        )


class DeltaSmootherTracker(FilterTracker):
    """Convenience tracker for delta smoothing interpolation."""

    def __init__(self, num_tracks: int, config: EuroSmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: DeltaEuroSmoother(config)
        )
