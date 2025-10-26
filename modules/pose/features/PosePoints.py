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
    raw_scores: np.ndarray = field(repr=False)  # shape (17,) - flattened
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
        denominator: float = max(1.0 - self.score_threshold, 1e-6)  # Avoid division by zero
        normalized[above_threshold] = (self.raw_scores[above_threshold] - self.score_threshold) / denominator
        object.__setattr__(self, 'scores', normalized)

    def __len__(self) -> int:
        """Total number of keypoints"""
        return POSE_NUM_JOINTS

    def __contains__(self, joint: PoseJoint) -> bool:
        """Check if joint has valid (non-NaN) point. Both x and y coordinates must be valid. Supports 'joint in point_data' syntax."""
        return not np.isnan(self.points[joint]).any()

    def __iter__(self) -> Iterator[tuple[PoseJoint, np.ndarray, float]]:
        """Iterate over (joint, point, score) tuples. Supports 'for joint, point, score in point_data' syntax."""
        for joint in PoseJoint:
            yield joint, self.points[joint], self.scores[joint]

    def __repr__(self) -> str:
        """Readable string representation"""
        return f"PosePointData(valid={self.valid_count}/{POSE_NUM_JOINTS}, mean_score={self.mean_score:.2f})"

    def __getitem__(self, joint: PoseJoint) -> np.ndarray:
        """Dict-like access: point_data[PoseJoint.nose]"""
        return self.points[joint]

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating which joints have valid (non-NaN) points"""
        return ~np.isnan(self.points).any(axis=1)

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-NaN) points"""
        return int(np.sum(self.valid_mask))

    @cached_property
    def has_data(self) -> bool:
        """True if at least one valid point available (opposite of is_empty)"""
        return self.valid_count > 0

    @cached_property
    def mean_score(self) -> float:
        """Mean confidence score of all valid joints. Returns 0.0 if no valid points."""
        if not self.has_data:
            return 0.0
        return float(np.mean(self.scores[self.valid_mask]))

    @cached_property
    def std_score(self) -> float:
        """Standard deviation of valid scores. Returns 0.0 if < 2 valid points."""
        if self.valid_count < 2:
            return 0.0
        return float(np.std(self.scores[self.valid_mask]))

    def to_dict(self) -> dict[PoseJoint, tuple[float, float]]:
        """Convert to dict with joint names as keys.Note: Includes NaN values for invalid/low-confidence joints."""
        return {joint: (float(self.points[joint, 0]), float(self.points[joint, 1])) for joint in PoseJoint}
