import numpy as np
import math
from dataclasses import dataclass, field
from functools import cached_property
from enum import IntEnum
from typing import Optional, Iterator

from modules.pose.PoseTypes import PoseJoint
from modules.pose.features.PosePoints import PosePointData

class AngleJoint(IntEnum):
    left_shoulder =  0
    right_shoulder = 1
    left_elbow =     2
    right_elbow =    3
    left_hip =       4
    right_hip =      5
    left_knee =      6
    right_knee =     7
    head =           8

ANGLE_JOINT_NAMES: list[str] = [e.name for e in AngleJoint]
ANGLE_NUM_JOINTS: int = len(AngleJoint)

# DEFINITIONS
ANGLE_JOINT_TRIPLETS: dict[AngleJoint, tuple[PoseJoint, PoseJoint, PoseJoint]] = {
    AngleJoint.left_shoulder:  ( PoseJoint.left_hip,       PoseJoint.left_shoulder,  PoseJoint.left_elbow  ),
    AngleJoint.right_shoulder: ( PoseJoint.right_hip,      PoseJoint.right_shoulder, PoseJoint.right_elbow ),
    AngleJoint.left_elbow:     ( PoseJoint.left_shoulder,  PoseJoint.left_elbow,     PoseJoint.left_wrist  ),
    AngleJoint.right_elbow:    ( PoseJoint.right_shoulder, PoseJoint.right_elbow,    PoseJoint.right_wrist ),
    AngleJoint.left_hip:       ( PoseJoint.left_shoulder,  PoseJoint.left_hip,       PoseJoint.left_knee   ),
    AngleJoint.right_hip:      ( PoseJoint.right_shoulder, PoseJoint.right_hip,      PoseJoint.right_knee  ),
    AngleJoint.left_knee:      ( PoseJoint.left_hip,       PoseJoint.left_knee,      PoseJoint.left_ankle  ),
    AngleJoint.right_knee:     ( PoseJoint.right_hip,      PoseJoint.right_knee,     PoseJoint.right_ankle ),
    AngleJoint.head:           ( PoseJoint.nose,           PoseJoint.left_eye,       PoseJoint.right_eye   )
}

POSE_JOINT_TO_ANGLE_IDX: dict[PoseJoint, int] = {
    PoseJoint.left_shoulder:  AngleJoint.left_shoulder.value,
    PoseJoint.right_shoulder: AngleJoint.right_shoulder.value,
    PoseJoint.left_elbow:     AngleJoint.left_elbow.value,
    PoseJoint.right_elbow:    AngleJoint.right_elbow.value,
    PoseJoint.left_hip:       AngleJoint.left_hip.value,
    PoseJoint.right_hip:      AngleJoint.right_hip.value,
    PoseJoint.left_knee:      AngleJoint.left_knee.value,
    PoseJoint.right_knee:     AngleJoint.right_knee.value,
    PoseJoint.nose:           AngleJoint.head.value,
    PoseJoint.left_eye:       AngleJoint.head.value,
    PoseJoint.right_eye:      AngleJoint.head.value
}

ANGLE_JOINT_ROTATIONS: dict[AngleJoint, float] = {
    AngleJoint.left_shoulder:    0.0,
    AngleJoint.right_shoulder:   0.0,
    AngleJoint.left_elbow:       np.pi,
    AngleJoint.right_elbow:      np.pi,
    AngleJoint.left_hip:         np.pi,
    AngleJoint.right_hip:        np.pi,
    AngleJoint.left_knee:        np.pi,
    AngleJoint.right_knee:       np.pi,
    AngleJoint.head:             0.0
}


@dataclass(frozen=True)
class PoseAngleData:
    """Joint angle data with convenient access methods and summary statistics.

    Stores angles and confidence scores for all joints. Individual angles/scores
    can be NaN/0.0 to indicate missing or low-confidence detections.
    """
    angles: np.ndarray = field(default_factory=lambda: np.full(ANGLE_NUM_JOINTS, np.nan, dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32))

    def __len__(self) -> int:
        """Total number of joint angles"""
        return ANGLE_NUM_JOINTS

    def __contains__(self, joint: AngleJoint) -> bool:
        """Check if joint has valid (non-NaN) angle. Supports 'joint in angle_data' syntax."""
        return not np.isnan(self.angles[joint])

    def __iter__(self) -> Iterator[tuple[AngleJoint, float, float]]:
        """Iterate over (joint, angle, score) tuples. Supports 'for joint, angle, score in angle_data'."""
        for joint in AngleJoint:
            yield joint, self.angles[joint], self.scores[joint]

    def __repr__(self) -> str:
        """Readable string representation"""
        return f"PoseAngleData(valid={self.valid_count}/{ANGLE_NUM_JOINTS}, mean={self.mean_angle:.2f}°)"

    def __getitem__(self, joint: AngleJoint) -> float:
        """Dict-like access: angle_data[AngleJoint.LEFT_ELBOW]"""
        return self.angles[joint]

    def get_angle(self, joint: AngleJoint) -> float:
        """Get angle for specific joint (alias for __getitem__)"""
        return self[joint]

    def get_score(self, joint: AngleJoint) -> float:
        """Get confidence score for specific joint"""
        return self.scores[joint]

    def is_valid(self, joint: AngleJoint) -> bool:
        """Check if joint has valid (non-NaN) angle"""
        return joint in self

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-NaN) angles"""
        return int(np.count_nonzero(~np.isnan(self.angles)))

    @cached_property
    def is_empty(self) -> bool:
        """True if no valid angles available"""
        return self.valid_count == 0

    @cached_property
    def mean_angle(self) -> float:
        """Mean of all valid angles (in degrees). Returns 0.0 if no valid angles."""
        if self.is_empty:
            return 0.0
        return float(np.rad2deg(np.nanmean(self.angles)))

    @cached_property
    def mean_score(self) -> float:
        """Mean confidence score of all valid joints. Returns 0.0 if no valid angles."""
        if self.is_empty:
            return 0.0
        # Only average scores where angle is valid
        valid_mask = ~np.isnan(self.angles)
        return float(np.mean(self.scores[valid_mask]))

    @cached_property
    def std_angle(self) -> float:
        """Standard deviation of valid angles (in degrees). Returns 0.0 if < 2 valid angles."""
        if self.valid_count < 2:
            return 0.0
        return float(np.rad2deg(np.nanstd(self.angles)))

    def get_valid_angles(self) -> dict[AngleJoint, float]:
        """Get dict of only valid (non-NaN) angles"""
        return {
            joint: angle
            for joint, angle, _ in self
            if not np.isnan(angle)
        }

    def get_valid_scores(self) -> dict[AngleJoint, float]:
        """Get dict of scores for valid joints"""
        return {
            joint: score
            for joint, angle, score in self
            if not np.isnan(angle)
        }

    def to_degrees(self) -> np.ndarray:
        """Convert all angles to degrees (NaN preserved)"""
        return np.rad2deg(self.angles)

    def to_dict(self) -> dict[str, float]:
        """Convert to dict with joint names as keys (includes NaN values)"""
        return {joint.name: self.angles[joint] for joint in AngleJoint}


class PoseAngles:
    @staticmethod
    def compute(point_data: Optional['PosePointData']) -> PoseAngleData:
        """Calculate angle data from point data. Returns NaN-filled PoseAngleData if point_data is None."""
        if point_data is None:
            return PoseAngleData()

        point_values: np.ndarray = point_data.points
        point_scores: np.ndarray = point_data.scores

        angle_values: np.ndarray = np.full(ANGLE_NUM_JOINTS, np.nan, dtype=np.float32)
        angle_scores: np.ndarray = np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32)

        for i, (joint, (kp1, kp2, kp3)) in enumerate(ANGLE_JOINT_TRIPLETS.items()):
            idx1, idx2, idx3 = kp1.value, kp2.value, kp3.value
            p1, p2, p3 = point_values[idx1], point_values[idx2], point_values[idx3]

            if not (np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any()):
                rotate_by: float = ANGLE_JOINT_ROTATIONS[joint]

                if joint == AngleJoint.head:
                    angle: float = PoseAngles.calculate_head_yaw(p1, p2, p3, rotate_by)
                else:
                    angle: float = PoseAngles.calculate_angle(p1, p2, p3, rotate_by)

                if not np.isnan(angle):  # Only set if calculation succeeded
                    angle_values[i] = angle
                    s1, s2, s3 = point_scores[idx1], point_scores[idx2], point_scores[idx3]
                    angle_scores[i] = min(s1, s2, s3)

        return PoseAngleData(angle_values, angle_scores)

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, rotate_by: float = 0) -> float:
        """Calculate the signed angle between three points in the 2D plane.

        Args:
            p1: First point coordinates [x, y]
            p2: Second point (vertex) coordinates [x, y]
            p3: Third point coordinates [x, y]
            rotate_by: Rotation offset in radians

        Returns:
            Angle in radians in range [-π, π)
        """
        v1: np.ndarray = p1 - p2
        v2: np.ndarray = p3 - p2

        dot: float = np.dot(v1, v2)
        det: float = v1[0] * v2[1] - v1[1] * v2[0]
        angle: float = np.arctan2(det, dot)

        angle += rotate_by
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        return angle

    @staticmethod
    def calculate_head_yaw(nose: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray, rotate_by: float = 0) -> float:
        """Calculate yaw angle from nose and eye positions.

        Args:
            nose: Nose point coordinates [x, y]
            left_eye: Left eye point coordinates [x, y]
            right_eye: Right eye point coordinates [x, y]
            rotate_by: Rotation offset in radians

        Returns:
            Yaw angle in radians, or NaN if eye_width is invalid
        """
        eye_midpoint = (left_eye + right_eye) / 2
        eye_width = float(np.linalg.norm(right_eye - left_eye))

        if eye_width > 0:
            nose_offset_x = (nose[0] - eye_midpoint[0]) / eye_width
            yaw = np.arctan(nose_offset_x * 2)

            yaw += rotate_by
            yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi

            return float(yaw)
        return np.nan
