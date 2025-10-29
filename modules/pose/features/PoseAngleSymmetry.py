import numpy as np
import math
from functools import cached_property
from dataclasses import dataclass, field
from enum import IntEnum

from modules.pose.features.PoseAngles import PoseAngleData

from modules.pose.features.PoseAngles import AngleJoint

class SymmetricJointType(IntEnum):
    shoulder = 0
    elbow = 1
    hip = 2
    knee = 3

# Source of truth for symmetric joint pairs
SYMMETRIC_JOINT_PAIRS: dict[SymmetricJointType, tuple[AngleJoint, AngleJoint]] = {
    SymmetricJointType.shoulder: (AngleJoint.left_shoulder, AngleJoint.right_shoulder),
    SymmetricJointType.elbow: (AngleJoint.left_elbow, AngleJoint.right_elbow),
    SymmetricJointType.hip: (AngleJoint.left_hip, AngleJoint.right_hip),
    SymmetricJointType.knee: (AngleJoint.left_knee, AngleJoint.right_knee)
}


@dataclass(frozen=True)
class PoseSymmetryData:
    """Symmetry metrics for symmetric joint pairs.

    Measures how similar left/right joint angles are after mirroring. Multiple mean
    types provide different sensitivity levels to asymmetric outliers.

    Attributes:
        symmetries: Dict of symmetry scores [0, 1] per joint type, NaN if missing data
    """

    symmetries: dict[SymmetricJointType, float] = field(default_factory=lambda: {joint_type: math.nan for joint_type in SymmetricJointType})

    def __repr__(self) -> str:
        valid_count = sum(1 for s in self.symmetries.values() if not np.isnan(s))
        return f"PoseSymmetryData(valid={valid_count}/4, mean={self.mean:.3f})"

    def __getitem__(self, joint_type: SymmetricJointType) -> float:
        """Dict-like access: symmetry_data[SymmetricJointType.elbow]"""
        return self.symmetries[joint_type]

    def safe(self, default: float = 0.0) -> 'PoseSymmetryData':
        """Return copy with NaN replaced by default value."""
        safe_symmetries: dict[SymmetricJointType, float] = {
            joint_type: (symmetry if not np.isnan(symmetry) else default) for joint_type, symmetry in self.symmetries.items()
        }
        return PoseSymmetryData(safe_symmetries)

    @cached_property
    def _valid_symmetries(self) -> np.ndarray:
        """Cached array of valid (non-NaN) symmetry values for efficient computation."""
        symmetries_array = np.array(list(self.symmetries.values()), dtype=np.float32)
        return symmetries_array[~np.isnan(symmetries_array)]

    @cached_property
    def mean(self) -> float:
        """Arithmetic mean symmetry - tolerant of individual asymmetric joints.

        Use for: General assessment, UI display, beginners

        Returns: Mean in [0, 1], or NaN if no valid pairs

        Example: (1.0, 1.0, 1.0, 0.1) → 0.775
        """
        if len(self._valid_symmetries) == 0:
            return np.nan
        return float(np.mean(self._valid_symmetries))

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean symmetry - strict, requires ALL joints symmetric.

        Use for: Balanced strictness, exercise grading, intermediate users

        Returns: Mean in [0, 1], or NaN if no valid pairs

        Example: (1.0, 1.0, 1.0, 0.1) → 0.562 (vs arithmetic 0.775)
        """
        if len(self._valid_symmetries) == 0:
            return np.nan

        epsilon = 1e-10
        log_mean = np.mean(np.log(self._valid_symmetries + epsilon))
        geometric_mean = np.exp(log_mean) - epsilon

        return float(np.clip(geometric_mean, 0.0, 1.0))

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean symmetry - very strict, dominated by worst joint.

        Use for: Critical applications, competition standards, professionals

        Returns: Mean in [0, 1], or NaN if no valid pairs

        Example: (1.0, 1.0, 1.0, 0.1) → 0.308 (vs geometric 0.562, arithmetic 0.775)
        """
        if len(self._valid_symmetries) == 0:
            return np.nan

        epsilon = 1e-10
        reciprocal_sum = np.sum(1.0 / (self._valid_symmetries + epsilon))
        harmonic_mean = len(self._valid_symmetries) / reciprocal_sum

        return float(np.clip(harmonic_mean, 0.0, 1.0))


class PoseAngleSymmetry:
    """Utility class for computing symmetry metrics from angle data."""

    @staticmethod
    def from_angles(angle_data: 'PoseAngleData', symmetry_exponent: float = 1.0) -> PoseSymmetryData:
        """Calculate symmetry metrics from angle data.

        Args:
            angle_data: Angle measurements (angles should already be mirrored)
            symmetry_exponent: Exponent to emphasize symmetry differences (e.g., 2.0 for quadratic).
                              Higher values penalize asymmetries more severely.

        Returns:
            PoseSymmetryData with individual joint pair scores and aggregate metrics
        """
        symmetries = {}

        for joint_type, (left_joint, right_joint) in SYMMETRIC_JOINT_PAIRS.items():
            left_angle = angle_data.values[left_joint]
            right_angle = angle_data.values[right_joint]

            if np.isnan(left_angle) or np.isnan(right_angle):
                symmetries[joint_type] = np.nan
            else:
                # After mirroring, symmetric poses have similar angles
                angle_diff = abs(left_angle - right_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                # Normalize to [0, 1] and apply exponent for emphasis
                normalized_diff = angle_diff / np.pi
                symmetry = (1.0 - normalized_diff) ** symmetry_exponent
                symmetries[joint_type] = float(max(0.0, min(1.0, symmetry)))

        return PoseSymmetryData(symmetries)