from ..FilterTracker import FilterTracker
from modules.pose.nodes.filters.Predictors import (
    PredictorConfig,
    AnglePredictor,
    BBoxPredictor,
    PointPredictor,
    DeltaPredictor,
    SymmetryPredictor
)

class AnglePredictorTracker(FilterTracker):
    """Convenience tracker for angle prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: AnglePredictor(config)
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


class DeltaPredictorTracker(FilterTracker):
    """Convenience tracker for delta prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: DeltaPredictor(config)
        )


class SymmetryPredictorTracker(FilterTracker):
    """Convenience tracker for symmetry prediction."""

    def __init__(self, num_tracks: int, config: PredictorConfig) -> None:
        super().__init__(
            num_tracks=num_tracks,
            filter_factory=lambda: SymmetryPredictor(config)
        )