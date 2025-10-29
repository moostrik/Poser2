import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator

from modules.pose.PoseTypes import PoseJoint, POSE_NUM_JOINTS

@dataclass(frozen=True)
class PosePointData:
    """Keypoint data with convenient access methods and summary statistics.

    Stores raw and filtered points/scores. Points/scores can be NaN/0.0
    to indicate missing or low-confidence detections.
    """
    raw_points: np.ndarray = field(repr=False)  # shape (17, 2)
    raw_scores: np.ndarray = field(repr=False)  # shape (17,)
    score_threshold: float = field(default=0.5)

    points: np.ndarray = field(init=False, repr=False)   # filtered points (NaN where score < threshold)
    scores: np.ndarray = field(init=False, repr=False)   # normalized scores (0 where < threshold)

    def __post_init__(self) -> None:
        # Validate shapes
        if self.raw_points.shape != (POSE_NUM_JOINTS, 2):
            raise ValueError(f"raw_points must have shape ({POSE_NUM_JOINTS}, 2), got {self.raw_points.shape}")

        if self.raw_scores.shape != (POSE_NUM_JOINTS,):
            raise ValueError(f"raw_scores must have shape ({POSE_NUM_JOINTS},), got {self.raw_scores.shape}")

        # Clamp threshold to valid range
        s_t: float = max(0.0, min(0.99, self.score_threshold))
        object.__setattr__(self, 'score_threshold', s_t)

        # Filter points based on threshold
        filtered = self.raw_scores >= self.score_threshold
        filtered_points: np.ndarray = np.where(filtered[:, np.newaxis], self.raw_points, np.nan)
        object.__setattr__(self, 'points', filtered_points)

        # Normalize scores based on threshold
        normalized: np.ndarray = np.zeros_like(self.raw_scores)
        above_threshold = self.raw_scores >= self.score_threshold
        denominator: float = max(1.0 - self.score_threshold, 1e-6)
        normalized[above_threshold] = (self.raw_scores[above_threshold] - self.score_threshold) / denominator
        object.__setattr__(self, 'scores', normalized)

    def __repr__(self) -> str:
        """Readable string representation."""

        if not self.any_valid:
            return f"PosePointData(0/{POSE_NUM_JOINTS})"

        mean_score = float(np.mean(self.scores[self.valid_mask]))
        return f"PosePointData({self.valid_count}/{POSE_NUM_JOINTS}, μ_score={mean_score:.2f})"

    # ========== ACCESS ==========

    def get(self, joint: PoseJoint, default: np.ndarray | None = None) -> np.ndarray:
        """Get point with default for NaN."""
        point = self.points[joint]
        if np.isnan(point).any():
            return default if default is not None else np.array([np.nan, np.nan])
        return point

    def get_score(self, joint: PoseJoint, default: float = 0.0) -> float:
        """Get score with default."""
        score = self.scores[joint]
        return score if score > 0.0 else default

    # ========== ITERATION ==========

    def items(self) -> Iterator[tuple[PoseJoint, np.ndarray]]:
        """Iterate all (joint, point) pairs."""
        for joint in PoseJoint:
            yield joint, self.points[joint]

    def items_with_scores(self) -> Iterator[tuple[PoseJoint, np.ndarray, float]]:
        """Iterate all (joint, point, score) tuples."""
        for joint in PoseJoint:
            yield joint, self.points[joint], self.scores[joint]

    # ========== VALIDATION ==========

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating which joints have valid (non-NaN) points."""
        return ~np.isnan(self.points).any(axis=1)

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-NaN) points."""
        return int(np.sum(self.valid_mask))

    @cached_property
    def any_valid(self) -> bool:
        """True if at least one valid (non-NaN) angle is available."""
        return self.valid_count > 0

    # ========== CONVERSION ==========

    def to_dict(self) -> dict[PoseJoint, tuple[float, float]]:
        """Convert to dictionary mapping joints to (x, y) coordinate tuples (includes NaN)."""
        return {joint: (float(point[0]), float(point[1])) for joint, point in self.items()}

    def safe(self, default: np.ndarray | None = None) -> 'PosePointData':
        """Return copy with NaN replaced by default value."""
        if default is None:
            default = np.array([0.0, 0.0])

        safe_points: np.ndarray = self.points.copy()
        nan_mask: np.ndarray = np.isnan(safe_points).any(axis=1)
        safe_points[nan_mask] = default

        # Return new instance with safe points
        result = PosePointData(
            raw_points=self.raw_points,
            raw_scores=self.raw_scores,
            score_threshold=self.score_threshold
        )
        object.__setattr__(result, 'points', safe_points)
        object.__setattr__(result, 'scores', self.scores)

        return result