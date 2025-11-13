from ..FilterTracker import FilterTracker
from modules.pose.nodes import (
    SmootherConfig,
    AngleSmoother,
    BboxSmoother,
    Point2DSmoother,
    DeltaSmoother,
    PoseSmoother
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
            filter_factory=lambda: BboxSmoother(config)
        )

class PointSmootherTracker(FilterTracker):
    """Convenience tracker for point smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: Point2DSmoother(config)
        )

class DeltaSmootherTracker(FilterTracker):
    """Convenience tracker for delta smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: DeltaSmoother(config)
        )

class PoseSmootherTracker(FilterTracker):
    """Convenience tracker for full pose smoothing interpolation."""

    def __init__(self, num_tracks: int, config: SmootherConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: PoseSmoother(config)
        )