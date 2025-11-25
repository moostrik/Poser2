import numpy as np

from modules.pose.features.Angles import Angles, AngleLandmark
from modules.pose.features.AngleSymmetry import AngleSymmetry, SymmetryElement

# Maps each symmetric joint type to its left/right AngleLandmark pair
_SYMMETRY_PAIRS: dict[SymmetryElement, tuple[AngleLandmark, AngleLandmark]] = {
    SymmetryElement.shoulder: (AngleLandmark.left_shoulder, AngleLandmark.right_shoulder),
    SymmetryElement.elbow: (AngleLandmark.left_elbow, AngleLandmark.right_elbow),
    SymmetryElement.hip: (AngleLandmark.left_hip, AngleLandmark.right_hip),
    SymmetryElement.knee: (AngleLandmark.left_knee, AngleLandmark.right_knee),
}

class SymmetryUtils:
    """Utility class for computing symmetry metrics from angle data."""

    @staticmethod
    def from_angles(angles: Angles, symmetry_exponent: float = 1.0) -> AngleSymmetry:
        """Calculate symmetry metrics from angle data.

        Args:
            angles: Angle measurements (angles should already be mirrored)
            symmetry_exponent: Exponent to emphasize symmetry differences (e.g., 2.0 for quadratic).
                              Higher values penalize asymmetries more severely.

        Returns:
            SymmetryFeature with individual joint pair scores and aggregate metrics
        """
        values: np.ndarray = np.empty(len(SymmetryElement), dtype=np.float32)

        for joint_type in SymmetryElement:
            left_joint, right_joint = _SYMMETRY_PAIRS[joint_type]
            left_angle = angles.values[left_joint]
            right_angle = angles.values[right_joint]

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

        return AngleSymmetry.from_values(values)