import numpy as np
from functools import cached_property
from dataclasses import dataclass, field
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
    torso =          8  # Torso rotation (special calculation)
    head =           9  # Head yaw (special calculation)

ANGLE_JOINT_NAMES: list[str] = [e.name for e in AngleJoint]
ANGLE_NUM_JOINTS: int = len(AngleJoint)

# Keypoint requirements for each angle measurement
# Triplets (3 points) use standard angle calculation
# Quads (4 points) require special calculation functions
ANGLE_JOINT_KEYPOINTS: dict[AngleJoint, tuple[PoseJoint, ...]] = {
    # Standard 3-point angles
    AngleJoint.left_shoulder:  (PoseJoint.left_hip,       PoseJoint.left_shoulder,  PoseJoint.left_elbow),
    AngleJoint.right_shoulder: (PoseJoint.right_hip,      PoseJoint.right_shoulder, PoseJoint.right_elbow),
    AngleJoint.left_elbow:     (PoseJoint.left_shoulder,  PoseJoint.left_elbow,     PoseJoint.left_wrist),
    AngleJoint.right_elbow:    (PoseJoint.right_shoulder, PoseJoint.right_elbow,    PoseJoint.right_wrist),
    AngleJoint.left_hip:       (PoseJoint.left_shoulder,  PoseJoint.left_hip,       PoseJoint.left_knee),
    AngleJoint.right_hip:      (PoseJoint.right_shoulder, PoseJoint.right_hip,      PoseJoint.right_knee),
    AngleJoint.left_knee:      (PoseJoint.left_hip,       PoseJoint.left_knee,      PoseJoint.left_ankle),
    AngleJoint.right_knee:     (PoseJoint.right_hip,      PoseJoint.right_knee,     PoseJoint.right_ankle),
    # Special 4-point measurements
    AngleJoint.head:           (PoseJoint.left_eye,       PoseJoint.right_eye,      PoseJoint.left_shoulder, PoseJoint.right_shoulder),
    AngleJoint.torso:          (PoseJoint.left_shoulder,  PoseJoint.right_shoulder, PoseJoint.left_hip,      PoseJoint.right_hip),
}

ANGLE_JOINT_ROTATIONS: dict[AngleJoint, float] = {
    AngleJoint.left_shoulder:  0.0,
    AngleJoint.right_shoulder: 0.0,
    AngleJoint.left_elbow:     np.pi,
    AngleJoint.right_elbow:    np.pi,
    AngleJoint.left_hip:       np.pi,
    AngleJoint.right_hip:      np.pi,
    AngleJoint.left_knee:      np.pi,
    AngleJoint.right_knee:     np.pi,
    AngleJoint.head:           0.0,
    AngleJoint.torso:          0.0
}

# Right-side joints that should be mirrored for symmetric representation
# When mirrored, symmetric poses (e.g., arms both at 45°) have similar values for left/right
ANGLE_JOINT_SYMMETRIC_MIRROR: set[AngleJoint] = {
    AngleJoint.right_shoulder,
    AngleJoint.right_elbow,
    AngleJoint.right_hip,
    AngleJoint.right_knee,
}


@dataclass(frozen=True)
class PoseAngleData:
    """Container for joint angle measurements with convenient access and statistics.

    Stores angle values (in radians) and confidence scores for body joints.
    Provides dict-like access, iteration, and summary statistics over valid angles.

    All angles are stored in radians and normalized to range [-π, π).

    Individual angles can be NaN to indicate missing/uncomputable joints.
    Scores represent detection confidence: 0.0 (missing/invalid) to 1.0 (full confidence).
    For computed angles, score is the minimum confidence of the three constituent keypoints.

    Note: Right-side angles are mirrored in compute(), so symmetric poses have similar left/right values.
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
        return f"PoseAngleData(valid={self.valid_count}/{ANGLE_NUM_JOINTS}, mean={self.mean_angle:.2f} rad)"

    def __getitem__(self, joint: AngleJoint) -> float:
        """Dict-like access: angle_data[AngleJoint.left_elbow]"""
        return self.angles[joint]

    @cached_property
    def valid_joints(self) -> list[AngleJoint]:
        """List of joints with valid angles"""
        return [joint for joint, is_valid in zip(AngleJoint, self.valid_mask) if is_valid]

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating which joints have valid (non-NaN) angles"""
        return ~np.isnan(self.angles)

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-NaN) angles"""
        return int(np.sum(self.valid_mask))

    @cached_property
    def has_data(self) -> bool:
        """True if at least one valid angle available"""
        return self.valid_count > 0

    @cached_property
    def mean_angle(self) -> float:
        """Mean of all valid angles. Returns 0.0 if no valid angles."""
        if not self.has_data:
            return 0.0
        return float(np.nanmean(self.angles))

    @cached_property
    def mean_score(self) -> float:
        """Mean confidence score of all valid joints. Returns 0.0 if no valid angles."""
        if not self.has_data:
            return 0.0
        return float(np.mean(self.scores[self.valid_mask]))

    @cached_property
    def std_angle(self) -> float:
        """Standard deviation of valid angles. Returns 0.0 if < 2 valid angles."""
        if self.valid_count < 2:
            return 0.0
        return float(np.nanstd(self.angles))

    def to_dict(self) -> dict[AngleJoint, float]:
        """Convert to dict with AngleJoint enum as keys."""
        return {joint: float(self.angles[joint]) for joint in AngleJoint}


class PoseAngles:
    @staticmethod
    def compute(point_data: Optional['PosePointData']) -> PoseAngleData:
        """Calculate angle data from point data with transformations applied.

        Applies rotation offsets and mirrors right-side angles for symmetric representation.
        Returns NaN-filled PoseAngleData if point_data is None.
        """
        if point_data is None:
            return PoseAngleData()

        angle_values: np.ndarray = np.full(ANGLE_NUM_JOINTS, np.nan, dtype=np.float32)
        angle_scores: np.ndarray = np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32)

        # Compute all angle measurements
        for joint, keypoints in ANGLE_JOINT_KEYPOINTS.items():
            # Check if all required keypoints are valid using PosePointData's API
            if not all(kp in point_data for kp in keypoints):
                continue

            # Extract points and scores
            points = [point_data.points[kp.value] for kp in keypoints]
            rotate_by = ANGLE_JOINT_ROTATIONS[joint]

            # Compute angle based on number of keypoints
            if len(keypoints) == 3:
                # Standard 3-point angle
                angle = PoseAngles.calculate_angle(points[0], points[1], points[2], rotate_by)
            elif joint == AngleJoint.head:
                # Special: head yaw
                angle = PoseAngles.calculate_head_yaw(points[0], points[1], points[2], points[3], rotate_by)
            elif joint == AngleJoint.torso:
                # Special: torso rotation
                angle = PoseAngles.calculate_torso_rotation(points[0], points[1], points[2], points[3], rotate_by)
            else:
                continue

            if not np.isnan(angle):
                # Mirror right-side angles for symmetric representation
                if joint in ANGLE_JOINT_SYMMETRIC_MIRROR:
                    angle = -angle

                angle_values[joint.value] = angle
                scores = [point_data.scores[kp.value] for kp in keypoints]
                angle_scores[joint.value] = min(scores)

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
    def calculate_head_yaw(left_eye: np.ndarray, right_eye: np.ndarray,
                          left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                          rotate_by: float = 0) -> float:
        """Calculate head yaw from eye positions relative to shoulder base.

        Measures how much the head is rotated left/right relative to the torso.

        Args:
            left_eye: Left eye coordinates [x, y]
            right_eye: Right eye coordinates [x, y]
            left_shoulder: Left shoulder coordinates [x, y]
            right_shoulder: Right shoulder coordinates [x, y]
            rotate_by: Rotation offset in radians

        Returns:
            Yaw angle in radians [-π, π), or NaN if not computable
        """
        eye_midpoint = (left_eye + right_eye) / 2
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        eye_width = float(np.linalg.norm(right_eye - left_eye))

        if eye_width > 0:
            # Normalize horizontal offset by eye width to get rotation angle
            offset_x = (eye_midpoint[0] - shoulder_midpoint[0]) / eye_width
            yaw = np.arctan(offset_x)

            yaw += rotate_by
            yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi

            return float(yaw)
        return np.nan

    @staticmethod
    def calculate_torso_rotation(left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                                 left_hip: np.ndarray, right_hip: np.ndarray,
                                 rotate_by: float = 0) -> float:
        """Calculate torso rotation from shoulder and hip alignment.

        Measures the twist/rotation of the upper body relative to lower body.

        Args:
            left_shoulder: Left shoulder coordinates [x, y]
            right_shoulder: Right shoulder coordinates [x, y]
            left_hip: Left hip coordinates [x, y]
            right_hip: Right hip coordinates [x, y]
            rotate_by: Rotation offset in radians

        Returns:
            Rotation angle in radians [-π, π), or NaN if not computable
        """
        shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
        hip_width = float(np.linalg.norm(right_hip - left_hip))

        if shoulder_width > 0 and hip_width > 0:
            # Calculate the angle of shoulder line
            shoulder_angle = np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            )

            # Calculate the angle of hip line
            hip_angle = np.arctan2(
                right_hip[1] - left_hip[1],
                right_hip[0] - left_hip[0]
            )

            # Rotation is the difference between shoulder and hip orientation
            rotation = shoulder_angle - hip_angle

            rotation += rotate_by
            rotation = ((rotation + np.pi) % (2 * np.pi)) - np.pi

            return float(rotation)
        return np.nan
