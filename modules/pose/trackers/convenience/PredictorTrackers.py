from ..FilterTracker import FilterTracker
from modules.pose.nodes.filters.Predictors import (
    PredictorConfig,
    AnglePredictor,
    BBoxPredictor,
    PointPredictor,
    AngleVelPredictor,
    AngleSymPredictor
)


class BBoxPredictorTracker(FilterTracker):
    """Convenience tracker for bounding box prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: BBoxPredictor(config)
        )


class PointPredictorTracker(FilterTracker):
    """Convenience tracker for point prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: PointPredictor(config)
        )


class AnglePredictorTracker(FilterTracker):
    """Convenience tracker for angle prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: AnglePredictor(config)
        )


class AngleVelPredictorTracker(FilterTracker):
    """Convenience tracker for delta prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: AngleVelPredictor(config)
        )


class AngleSymPredictorTracker(FilterTracker):
    """Convenience tracker for symmetry prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: AngleSymPredictor(config)
        )