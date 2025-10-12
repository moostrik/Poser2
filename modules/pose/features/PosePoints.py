import numpy as np
from dataclasses import dataclass, field


# POINT DATA
@dataclass(frozen=True)
class PosePointData():
    raw_points: np.ndarray = field(repr=False)     # shape (17, 2)
    raw_scores: np.ndarray = field(repr=False)     # shape (17, 1) - original unmodified scores
    score_threshold: float

    points: np.ndarray = field(init=False)         # filtered points (NaN where score < threshold)
    scores: np.ndarray = field(init=False)         # normalized scores (0 where < threshold, else normalized)

    def __post_init__(self) -> None:
        s_t: float = max(0.0, min(0.99, self.score_threshold))
        object.__setattr__(self, 'score_threshold', s_t)

        # Filter points based on threshold
        filtered = self.raw_scores >= self.score_threshold
        object.__setattr__(self, 'points', np.where(filtered[:, np.newaxis], self.raw_points, np.nan))

        # Normalize scores based on threshold
        normalized: np.ndarray = np.zeros_like(self.raw_scores)
        above_threshold = self.raw_scores >= self.score_threshold
        normalized[above_threshold] = (self.raw_scores[above_threshold] - self.score_threshold) / (1.0 - self.score_threshold)
        object.__setattr__(self, 'scores', normalized)
