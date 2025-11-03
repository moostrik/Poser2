import numpy as np
from dataclasses import dataclass, field
from functools import cached_property

from modules.pose.PoseTypes import PoseJoint, POSE_NUM_JOINTS

@dataclass(frozen=True)
class PosePointData:
    """Keypoint data with convenient access methods and summary statistics."""

    values: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        """Initialize filtered data and freeze arrays for immutability."""
        # Validate shapes
        if self.values.shape != (POSE_NUM_JOINTS, 2):
            raise ValueError(f"raw_values must have shape ({POSE_NUM_JOINTS}, 2), got {self.values.shape}")

        if self.scores.shape != (POSE_NUM_JOINTS,):
            raise ValueError(f"raw_scores must have shape ({POSE_NUM_JOINTS},), got {self.scores.shape}")

        self.values.flags.writeable = False
        self.scores.flags.writeable = False

        # Validate: NaN values must have 0.0 scores
        nan_mask = np.isnan(self.values).any(axis=1)  # Any NaN in (x, y)
        valid_mask = self.scores > 0.0
        invalid = nan_mask & valid_mask

        if np.any(invalid):
            invalid_joints = [PoseJoint(i).name for i in np.where(invalid)[0]]
            raise ValueError(
                f"Data integrity violation: NaN values must have 0.0 scores. "
                f"Invalid joints: {', '.join(invalid_joints)}"
            )


    def __repr__(self) -> str:
        """Readable string representation."""
        if not self.any_valid:
            return f"PosePointData(0/{POSE_NUM_JOINTS})"

        mean_score = float(np.mean(self.scores[self.valid_mask]))
        return f"PosePointData({self.valid_count}/{POSE_NUM_JOINTS}, μ_score={mean_score:.2f})"

    @classmethod
    def joint_enum(cls) -> type[PoseJoint]:
        """Get the joint enum type for this class."""
        return PoseJoint

    @classmethod
    def from_values(cls, values: np.ndarray, scores: np.ndarray | None = None, score_threshold: float = 0.0) -> 'PosePointData':
        """Create from already-filtered values, auto-generating scores if not provided."""
        if scores is None:
            has_nan = np.isnan(values).any(axis=1)
            scores = np.where(~has_nan, 1.0, 0.0).astype(np.float32)

        return cls(values=values, scores=scores)

    @classmethod
    def create_empty(cls) -> 'PosePointData':
        """Create instance with all joints marked as invalid (NaN values, zero scores)."""
        values = np.full((POSE_NUM_JOINTS, 2), np.nan, dtype=np.float32)
        scores = np.zeros(POSE_NUM_JOINTS, dtype=np.float32)
        return cls(values=values, scores=scores)

    # ========== ACCESS ==========

    def __len__(self) -> int:
        """Return total number of joints (including invalid)."""
        return len(self.values)

    def __getitem__(self, joint: PoseJoint | int) -> np.ndarray:
        """Get point for a joint (may be NaN)."""
        return self.values[joint]

    def get(self, joint: PoseJoint | int, default: np.ndarray | None = None) -> np.ndarray:
        """Get point with default for NaN."""
        point = self.values[joint]
        if np.isnan(point).any():
            return default if default is not None else np.array([np.nan, np.nan])
        return point

    def get_score(self, joint: PoseJoint | int) -> float:
        """Get score for a joint (always between 0.0 and 1.0)."""
        return float(self.scores[joint])

    # ========== VALIDATION ==========

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating which joints have valid (non-zero score) values."""
        return self.scores > 0.0

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-zero score) values."""
        return int(np.sum(self.valid_mask))

    @cached_property
    def any_valid(self) -> bool:
        """True if at least one valid value is available."""
        return self.valid_count > 0

    @cached_property
    def valid_values(self) -> np.ndarray:
        """Array of valid (non-zero score) points, shape (N, 2)."""
        return self.values[self.valid_mask]

    @cached_property
    def valid_joints(self) -> list[PoseJoint]:
        """List of joints with valid (non-zero score) values."""
        return [PoseJoint(i) for i in np.where(self.valid_mask)[0]]  # ← More efficient

    # ========== STATISTICS (2D spatial) ==========

    @cached_property
    def center_of_mass(self) -> np.ndarray:
        """Mean position of valid points, or [NaN, NaN] if none valid."""
        if not self.any_valid:
            return np.full(2, np.nan, dtype=np.float32)  # More explicit
        return np.nanmean(self.values, axis=0)

    @cached_property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box (min_point, max_point) of valid points.

        Returns ([min_x, min_y], [max_x, max_y]) or NaN if no valid points.
        """
        if not self.any_valid:
            nan_point = np.array([np.nan, np.nan])
            return nan_point, nan_point

        valid_points = self.valid_values
        return np.min(valid_points, axis=0), np.max(valid_points, axis=0)

    @cached_property
    def spatial_std(self) -> np.ndarray:
        """Standard deviation of valid points in x and y directions."""
        if not self.any_valid:
            return np.array([np.nan, np.nan])
        return np.nanstd(self.values, axis=0)

    # ========== CONVERSION ==========

    def to_dict(self, include_invalid: bool = True) -> dict[str, tuple[float, float]]:
        """Convert to dictionary mapping joint names to (x, y) tuples.

        Args:
            include_invalid: If True, includes NaN values. If False, only valid joints.

        Returns:
            Dictionary mapping joint names to coordinate tuples
        """
        if include_invalid:
            return {PoseJoint(i).name: (float(p[0]), float(p[1]))
                    for i, p in enumerate(self.values)}
        else:
            return {joint.name: (float(self.values[joint][0]), float(self.values[joint][1]))
                    for joint in self.valid_joints}

