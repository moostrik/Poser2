import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from modules.pose.PoseTypes import PoseAngleJointTriplets, PoseAngleRotations, POSE_NUM_ANGLES
from modules.pose.PosePoints import PosePointData

@dataclass(frozen=True)
class PoseAngleData:
    angles: np.ndarray = field(default_factory=lambda: np.full(POSE_NUM_ANGLES, np.nan, dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(POSE_NUM_ANGLES, dtype=np.float32))


class PoseAngles:
    @staticmethod
    def compute(point_data: Optional['PosePointData']) -> Optional[PoseAngleData]:
        """
        Calculate angle data from point data.
        Returns PoseAngleData with calculated angles or None if point_data is None.
        """
        if point_data is None:
            return PoseAngleData()

        point_values: np.ndarray = point_data.points
        point_scores: np.ndarray = point_data.scores

        angle_values: np.ndarray = np.full(POSE_NUM_ANGLES, np.nan, dtype=np.float32)
        angle_scores: np.ndarray = np.zeros(POSE_NUM_ANGLES, dtype=np.float32)

        for i, (joint, (kp1, kp2, kp3)) in enumerate(PoseAngleJointTriplets.items()):
            idx1, idx2, idx3 = kp1.value, kp2.value, kp3.value
            p1, p2, p3 = point_values[idx1], point_values[idx2], point_values[idx3]

            if not (np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any()):
                # All points are valid (not NaN), calculate the angle
                rotate_by: float = PoseAngleRotations[joint]
                angle: float = PoseAngles.calculate_angle(p1, p2, p3, rotate_by)
                angle_values[i] = angle

                s1, s2, s3 = point_scores[idx1], point_scores[idx2], point_scores[idx3]
                score: float = min(s1, s2, s3)
                angle_scores[i] = score

        return PoseAngleData(angle_values, angle_scores)

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, rotate_by: float = 0) -> float:
        """
        Calculate the signed angle between three points in the 2D plane (0 to 2π radians).

        Args:
            p1: First point coordinates [x, y]
            p2: Second point (vertex) coordinates [x, y]
            p3: Third point coordinates [x, y]

        Returns:
            Angle in radians, in the range [-π, π)
        """
        v1: np.ndarray = p1 - p2
        v2: np.ndarray = p3 - p2

        dot: float = np.dot(v1, v2)
        det: float = v1[0] * v2[1] - v1[1] * v2[0]  # 2D cross product
        angle: float = np.arctan2(det, dot)

        # Rotate the angle by a specified amount (in radians)
        angle += rotate_by

        # Normalize to [-π, π) range
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        return angle