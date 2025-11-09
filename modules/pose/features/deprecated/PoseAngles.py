# Standard library imports
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Optional

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features.PoseAngleFeatureBase import PoseAngleFeatureBase
from modules.pose.features.Point2DFeature import Point2DFeature, PointLandmark


class AngleJoint(IntEnum):
    left_shoulder =  0
    right_shoulder = 1
    left_elbow =     2
    right_elbow =    3
    left_hip =       4
    right_hip =      5
    left_knee =      6
    right_knee =     7
    # torso =          8  # Torso rotation (special calculation)
    head =           8  # Head yaw (special calculation)

ANGLE_JOINT_NAMES: list[str] = [e.name for e in AngleJoint]
ANGLE_NUM_JOINTS: int = len(AngleJoint)

# Keypoint requirements for each angle measurement
# Triplets (3 points) use standard angle calculation
# Quads (4 points) require special calculation functions
ANGLE_JOINT_KEYPOINTS: dict[AngleJoint, tuple[PointLandmark, ...]] = {
    # Standard 3-point angles
    AngleJoint.left_shoulder:  (PointLandmark.left_hip,       PointLandmark.left_shoulder,  PointLandmark.left_elbow),
    AngleJoint.right_shoulder: (PointLandmark.right_hip,      PointLandmark.right_shoulder, PointLandmark.right_elbow),
    AngleJoint.left_elbow:     (PointLandmark.left_shoulder,  PointLandmark.left_elbow,     PointLandmark.left_wrist),
    AngleJoint.right_elbow:    (PointLandmark.right_shoulder, PointLandmark.right_elbow,    PointLandmark.right_wrist),
    AngleJoint.left_hip:       (PointLandmark.left_shoulder,  PointLandmark.left_hip,       PointLandmark.left_knee),
    AngleJoint.right_hip:      (PointLandmark.right_shoulder, PointLandmark.right_hip,      PointLandmark.right_knee),
    AngleJoint.left_knee:      (PointLandmark.left_hip,       PointLandmark.left_knee,      PointLandmark.left_ankle),
    AngleJoint.right_knee:     (PointLandmark.right_hip,      PointLandmark.right_knee,     PointLandmark.right_ankle),
    # Special 4-point measurements
    AngleJoint.head:           (PointLandmark.left_eye,       PointLandmark.right_eye,      PointLandmark.left_shoulder, PointLandmark.right_shoulder),
    # AngleJoint.torso:          (PoseJoint.left_shoulder,  PoseJoint.right_shoulder, PoseJoint.left_hip,      PoseJoint.right_hip),
}

NEUTRAL_ROTATIONS: dict[AngleJoint, float] = {
    AngleJoint.left_shoulder:   0.15 * np.pi,
    AngleJoint.right_shoulder: -0.15 * np.pi,
    AngleJoint.left_elbow:      0.9 * np.pi,
    AngleJoint.right_elbow:    -0.9 * np.pi,
    AngleJoint.left_hip:       -0.95 * np.pi,
    AngleJoint.right_hip:       0.95 * np.pi,
    AngleJoint.left_knee:       np.pi,
    AngleJoint.right_knee:      np.pi,
    AngleJoint.head:            0.0,
    # AngleJoint.torso:           0.0
}

# Right-side joints that should be mirrored for symmetric representation
# When mirrored, symmetric poses (e.g., arms both at 45°) have similar values for left/right
ANGLE_JOINT_SYMMETRIC_MIRROR: set[AngleJoint] = {
    AngleJoint.right_shoulder,
    AngleJoint.right_elbow,
    AngleJoint.right_hip,
    AngleJoint.right_knee,
}

ANGLE_RANGE: tuple[float, float] = (-np.pi, np.pi)

@dataclass(frozen=True)
class PoseAngleData(PoseAngleFeatureBase[AngleJoint]):
    """Container for joint angle measurements with convenient access and statistics.

    Stores angle values (in radians, [-π, π]) and confidence scores for body joints.
    Provides dict-like access, iteration, and summary statistics over valid angles.

    Individual angles can be NaN to indicate missing/uncomputable joints.
    Scores represent detection confidence: 0.0 (missing/invalid) to 1.0 (full confidence).
    For computed angles, score is the minimum confidence of the three constituent keypoints.

    Note: Right-side angles are mirrored in compute(), so symmetric poses have similar left/right values.
    """

    # ========== CLASS-LEVEL PROPERTIES ==========

    @classmethod
    def joint_enum(cls) -> type[AngleJoint]:
        """Return the AngleJoint enum class."""
        return AngleJoint

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        """Return the default range for angle joints."""
        return ANGLE_RANGE

    # ========== ANGLE-SPECIFIC OPERATIONS =========

    def subtract(self, other: 'PoseAngleData') -> 'PoseAngleData':
        """Computes angular differences with proper wrapping to [-π, π] range."""
        diff: np.ndarray = self.values - other.values
        # Wrap angles to [-π, π] range (shortest angular distance)
        wrapped_diff: np.ndarray = np.arctan2(np.sin(diff), np.cos(diff))
        min_scores: np.ndarray = np.minimum(self.scores, other.scores)
        return PoseAngleData(values=wrapped_diff, scores=min_scores)

    def similarity(self, other: 'PoseAngleData', exponent: float) -> 'PoseAngleData':
        """Computes similarity scores between two PoseAngleData objects.

        Similarity for each joint is calculated as:
            similarity = (1 - (|angle_1 - angle_2| / π)) ^ exponent
        """
        diff_data: PoseAngleData = self.subtract(other)
        similarity_values: np.ndarray = np.power(1.0 - np.abs(diff_data.values) / np.pi, exponent)
        return PoseAngleData(values=similarity_values, scores=diff_data.scores)

    # ========== CIRCULAR STATISTICS ==========

    @cached_property
    def mean(self) -> float:
        """Circular mean of valid angle values in radians, or NaN if none valid.

        Uses circular statistics to properly handle angle wrapping.
        For example, the mean of [-π, π] is correctly computed as ±π, not 0.
        """
        valid = self.values[self.valid_mask]
        if valid.size == 0:
            return np.nan

        # Convert to unit circle coordinates
        sin_mean = np.mean(np.sin(valid))
        cos_mean = np.mean(np.cos(valid))

        # Compute circular mean
        return float(np.arctan2(sin_mean, cos_mean))

    @cached_property
    def std(self) -> float:
        """Circular standard deviation of valid angle values, or NaN if none valid.

        Measures angular dispersion. Returns values in [0, ~1.48] radians.
        Higher values indicate more spread around the circle.
        A value near 0 means angles are clustered, near 1.48 means uniformly distributed.
        """
        valid = self.values[self.valid_mask]
        if valid.size == 0:
            return np.nan

        # Compute resultant length R
        sin_mean = np.mean(np.sin(valid))
        cos_mean = np.mean(np.cos(valid))
        R = np.sqrt(sin_mean**2 + cos_mean**2)

        # Circular standard deviation
        # Returns NaN if R is too close to 0 (uniform distribution)
        if R < 1e-10:
            return np.nan

        return float(np.sqrt(-2 * np.log(R)))

    @cached_property
    def median(self) -> float:
        """Circular median approximation of valid angle values, or NaN if none valid.

        Returns the angle with minimum circular distance to the circular mean.
        This is an approximation; true circular median requires more complex algorithms.
        """
        if not self.any_valid:
            return np.nan

        valid = self.values[self.valid_mask]
        circ_mean = self.mean

        if np.isnan(circ_mean):
            return np.nan

        # Find angle with minimum circular distance to mean
        # Use complex exponential to handle wrapping correctly
        diffs = np.abs(np.angle(np.exp(1j * (valid - circ_mean))))
        return float(valid[np.argmin(diffs)])

    # Override to return NaN - these don't make sense for angles
    @cached_property
    def harmonic_mean(self) -> float:
        """Not applicable for angular data. Always returns NaN.

        Harmonic mean is mathematically undefined for angles in [-π, π].
        Use circular_mean instead.
        """
        return np.nan

    @cached_property
    def geometric_mean(self) -> float:
        """Not applicable for angular data. Always returns NaN.

        Geometric mean is mathematically undefined for angles in [-π, π].
        Use circular_mean instead.
        """
        return np.nan

    # min_value and max_value inherited from base class
    # They work but have limited meaning for circular data
    # (e.g., min of [-π, π] doesn't mean they're "far apart")
