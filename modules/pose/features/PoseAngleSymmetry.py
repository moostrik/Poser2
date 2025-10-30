import numpy as np
import math
from functools import cached_property
from dataclasses import dataclass, field
from enum import IntEnum

from numpy._typing._array_like import NDArray

from modules.pose.features.PoseAngles import PoseAngleData, AngleJoint
from modules.pose.features.PoseFeatureBase import PoseFeatureBase

class SymmetricJoint(IntEnum):
    shoulder = 0
    elbow = 1
    hip = 2
    knee = 3

SYMM_NUM_JOINTS: int = len(SymmetricJoint)

# Source of truth for symmetric joint pairs
SYMMETRIC_JOINT_PAIRS: dict[SymmetricJoint, tuple[AngleJoint, AngleJoint]] = {
    SymmetricJoint.shoulder: (AngleJoint.left_shoulder, AngleJoint.right_shoulder),
    SymmetricJoint.elbow: (AngleJoint.left_elbow, AngleJoint.right_elbow),
    SymmetricJoint.hip: (AngleJoint.left_hip, AngleJoint.right_hip),
    SymmetricJoint.knee: (AngleJoint.left_knee, AngleJoint.right_knee)
}


@dataclass(frozen=True)
class PoseSymmetryData(PoseFeatureBase[SymmetricJoint]):
    """Symmetry metrics for symmetric joint pairs.

    Measures how similar left/right joint angles are after mirroring. Multiple mean
    types provide different sensitivity levels to asymmetric outliers.

    Values are symmetry scores [0, 1] per joint type, where 1.0 is perfect symmetry.
    Scores represent confidence in the measurement.
    """

    @classmethod
    def joint_enum(cls) -> type[SymmetricJoint]:
        """The enum class used for joint indexing."""
        return SymmetricJoint

    def validate(self) -> None:
        """Validate that all symmetry values are in [0, 1] range."""
        valid_values = self.values[~np.isnan(self.values)]
        if np.any((valid_values < 0.0) | (valid_values > 1.0)):
            raise ValueError("Symmetry values must be in range [0, 1]")


class PoseAngleSymmetryFactory:
    """Utility class for computing symmetry metrics from angle data."""

    @staticmethod
    def from_dict(symmetries: dict[SymmetricJoint, float]) -> PoseSymmetryData:
        """Create symmetry data from dictionary of symmetry values.

        Args:
            symmetries: Dict mapping joint types to symmetry scores [0, 1], or NaN

        Returns:
            PoseSymmetryData instance
        """
        values: np.ndarray = np.array([symmetries.get(jt, np.nan) for jt in SymmetricJoint], dtype=np.float32)
        return PoseSymmetryData.from_values(values)

    @staticmethod
    def from_angles(angle_data: PoseAngleData, symmetry_exponent: float = 1.0) -> PoseSymmetryData:
        """Calculate symmetry metrics from angle data.

        Args:
            angle_data: Angle measurements (angles should already be mirrored)
            symmetry_exponent: Exponent to emphasize symmetry differences (e.g., 2.0 for quadratic).
                              Higher values penalize asymmetries more severely.

        Returns:
            PoseSymmetryData with individual joint pair scores and aggregate metrics
        """
        values: np.ndarray = np.empty(len(SymmetricJoint), dtype=np.float32)

        for joint_type in SymmetricJoint:
            left_joint, right_joint = SYMMETRIC_JOINT_PAIRS[joint_type]
            left_angle = angle_data.values[left_joint]
            right_angle = angle_data.values[right_joint]

            if np.isnan(left_angle) or np.isnan(right_angle):
                values[joint_type] = np.nan
            else:
                # After mirroring, symmetric poses have similar angles
                angle_diff = abs(left_angle - right_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                # Normalize to [0, 1] and apply exponent for emphasis
                normalized_diff = angle_diff / np.pi
                symmetry = (1.0 - normalized_diff) ** symmetry_exponent
                values[joint_type] = float(max(0.0, min(1.0, symmetry)))

        return PoseSymmetryData.from_values(values)