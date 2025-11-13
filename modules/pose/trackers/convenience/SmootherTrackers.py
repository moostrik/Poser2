from ..FilterTracker import FilterTracker
from modules.pose.nodes import (
    SmootherConfig,
    AngleSmoother,
    BBoxSmoother,
    PointSmoother,
    DeltaSmoother
)

class AngleSmootherTracker(FilterTracker):
    """Convenience tracker for angle smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: AngleSmoother(config)
        )


class BboxSmootherTracker(FilterTracker):
    """Convenience tracker for bounding box smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: BBoxSmoother(config)
        )


class PointSmootherTracker(FilterTracker):
    """Convenience tracker for point smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: PointSmoother(config)
        )


class DeltaSmootherTracker(FilterTracker):
    """Convenience tracker for delta smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: DeltaSmoother(config)
        )
