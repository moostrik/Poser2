import numpy as np
from functools import cached_property
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Iterator, Callable, Mapping

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

    Stores angle values (in radians) or its derivative values and confidence scores for body joints.
    Provides dict-like access, iteration, and summary statistics over valid angles.

    Individual angles can be NaN to indicate missing/uncomputable joints.
    Scores represent detection confidence: 0.0 (missing/invalid) to 1.0 (full confidence).
    For computed angles, score is the minimum confidence of the three constituent keypoints.

    Note: Right-side angles are mirrored in compute(), so symmetric poses have similar left/right values.
    """
    values: np.ndarray = field(default_factory=lambda: np.full(ANGLE_NUM_JOINTS, np.nan, dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32))

    def __post_init__(self) -> None:
        """Validate data integrity and freeze arrays for immutability."""
        # Make arrays immutable
        self.values.flags.writeable = False
        self.scores.flags.writeable = False

        # Validate data integrity: NaN values must have 0.0 scores."""
        nan_mask: np.ndarray = np.isnan(self.values)
        score_mask = self.scores > 0.0
        invalid = nan_mask & score_mask

        if np.any(invalid):
            invalid_joints: list[str] = [AngleJoint(i).name for i in np.where(invalid)[0]]
            raise ValueError(
                f"Data integrity violation: NaN values must have 0.0 scores. "
                f"Invalid joints: {', '.join(invalid_joints)}"
            )
    def __repr__(self) -> str:
        """Readable string representation."""
        if not self.any_valid:
            return f"PoseAngleData(0/{ANGLE_NUM_JOINTS})"

        mean_val = float(np.mean(self.values[self.valid_mask]))
        return f"PoseAngleData({self.valid_count}/{ANGLE_NUM_JOINTS}, μ_score={mean_val:.2f})"

    # ========== ACCESS ==========

    def get(self, joint: AngleJoint, default: float = np.nan) -> float:
        """Get value with default for NaN."""
        value = self.values[joint]
        return value if not np.isnan(value) else default

    def get_score(self, joint: AngleJoint, default: float = 0.0) -> float:
        """Get score with default."""
        score = self.scores[joint]
        return score if score > 0.0 else default

    # ========== ITERATION ==========

    def items(self) -> Iterator[tuple[AngleJoint, float]]:
        """Iterate all (joint, value) pairs."""
        for joint in AngleJoint:
            yield joint, self.values[joint]

    def items_with_scores(self) -> Iterator[tuple[AngleJoint, float, float]]:
        """Iterate all (joint, value, score) tuples."""
        for joint in AngleJoint:
            yield joint, self.values[joint], self.scores[joint]

    # ========== CONVERSION ==========

    def to_dict(self) -> dict[AngleJoint, float]:
        """Convert to dictionary mapping joints to angle values (includes NaN)."""
        return dict(self.items())

    def safe(self, default: float = 0.0) -> 'PoseAngleData':
        """Return copy with NaN replaced by default value."""
        safe_values: np.ndarray = self.values.copy()
        safe_values[np.isnan(safe_values)] = default
        return PoseAngleData(values=safe_values, scores=self.scores)

    # ========== VALIDATION ==========

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean array indicating which joints have valid (non-zero score) angles."""
        return self.scores > 0.0

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-zero score) angles."""
        return int(np.sum(self.valid_mask))

    @cached_property
    def any_valid(self) -> bool:
        """True if at least one valid (non-zero score) angle is available."""
        return self.valid_count > 0

    @cached_property
    def valid_values(self) -> np.ndarray:
        """Array of valid (non-zero score) angle values."""
        return self.values[self.valid_mask]

    @cached_property
    def valid_joints(self) -> list[AngleJoint]:
        """List of joints with valid (non-zero score) angles."""
        return [joint for joint in AngleJoint if self.scores[joint] > 0.0]

    # ========== STATISTICS ==========

    @cached_property
    def mean(self) -> float:
        """Mean of valid angle values, or NaN if none are valid."""
        return float(np.nanmean(self.values))

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of valid angle values, or NaN if none are valid."""
        valid: np.ndarray = self.values[~np.isnan(self.values)]
        if valid.size == 0:
            return np.nan
        return float(np.exp(np.mean(np.log(np.abs(valid)))))

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of valid angle values, or NaN if none are valid."""
        valid_values: np.ndarray = self.values[self.valid_mask]
        if valid_values.size == 0:
            return np.nan
        reciprocal_sum = np.sum(1.0 / valid_values)
        harm_mean = valid_values.size / reciprocal_sum
        return float(harm_mean)

    @cached_property
    def std(self) -> float:
        """Standard deviation of valid angle values, or NaN if none are valid."""
        return float(np.nanstd(self.values))

    @cached_property
    def median(self) -> float:
        """Median of valid angle values, or NaN if none are valid."""
        return float(np.nanmedian(self.values))

    # ========== COMPARISON ==========

    def subtract(self, other: 'PoseAngleData') -> 'PoseAngleData':
        """Computes angular differences with proper wrapping to [-π, π] range. """
        diff: np.ndarray = self.values - other.values
        # Wrap angles to [-π, π] range (shortest angular distance)
        wrapped_diff: np.ndarray = np.arctan2(np.sin(diff), np.cos(diff))
        min_scores: np.ndarray = np.minimum(self.scores, other.scores)

        return PoseAngleData(values=wrapped_diff, scores=min_scores)

    def similarity(self, other: 'PoseAngleData', exponent: float) -> 'PoseAngleData':
        """Computes similarity scores between two PoseAngleData objects.
        Similarity for each joint is: (1 - (|angle_1 - angle_2| / π)) ^ exponent
        """
        diff_data: PoseAngleData = self.subtract(other)
        similarity_values: np.ndarray = np.power(1.0 - np.abs(diff_data.values) / np.pi, exponent)
        return PoseAngleData(values=similarity_values, scores=diff_data.scores)


class PoseAngleFactory:

    @staticmethod
    def from_dicts(value_dict: dict[AngleJoint, float], score_dict: dict[AngleJoint, float] | None = None) -> PoseAngleData:
        """Create PoseAngleData from joint value and optional score dictionaries.

        Args:
            values: Dictionary mapping joints to angle values (radians, velocities, etc.)
                   None values are converted to NaN
            scores: Optional dictionary mapping joints to quality scores [0, 1]
                   If None, assigns 1.0 to all valid (non-NaN) values, 0.0 to NaN
        """

        # Build values array
        values: np.ndarray = np.array([value if (value := value_dict.get(joint)) is not None else np.nan for joint in AngleJoint], dtype=np.float32)

        # Build scores array
        if score_dict is None:
            # Default: 1.0 for valid values, 0.0 for NaN
            scores: np.ndarray = np.where(~np.isnan(values), 1.0, 0.0).astype(np.float32)
        else:
            scores = np.array([score_dict.get(joint, 0.0) for joint in AngleJoint], dtype=np.float32)

        return PoseAngleData(values=values, scores=scores)

    @staticmethod
    def from_points(point_data: Optional[PosePointData]) -> PoseAngleData:
        """Create angle measurements from keypoint data.

        Computes joint angles from 2D keypoint positions, applies rotation offsets,
        and mirrors right-side angles for symmetric representation.

        Args:
            point_data: Keypoint data or None

        Returns:
            PoseAngleData with computed angles and confidence scores
        """
        if point_data is None:
            return PoseAngleData()

        angle_values: np.ndarray = np.full(ANGLE_NUM_JOINTS, np.nan, dtype=np.float32)
        angle_scores: np.ndarray = np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32)

        # Compute all angle measurements
        for joint, keypoints in ANGLE_JOINT_KEYPOINTS.items():
            # Extract points using .get() method (returns NaN array if invalid)
            points = [point_data.get(kp) for kp in keypoints]

            rotate_by = ANGLE_JOINT_ROTATIONS[joint]

            # Compute angle based on number of keypoints
            if len(keypoints) == 3:
                # Standard 3-point angle
                angle = PoseAngleFactory._calculate_angle(points[0], points[1], points[2], rotate_by)
            elif joint == AngleJoint.head:
                # Special: head yaw
                angle = PoseAngleFactory._calculate_head_yaw(points[0], points[1], points[2], points[3], rotate_by)
            elif joint == AngleJoint.torso:
                # Special: torso rotation
                angle = PoseAngleFactory._calculate_torso_rotation(points[0], points[1], points[2], points[3], rotate_by)
            else:
                continue

            if not np.isnan(angle):
                # Mirror right-side angles for symmetric representation
                if joint in ANGLE_JOINT_SYMMETRIC_MIRROR:
                    angle = -angle

                angle_values[joint.value] = angle
                scores = [point_data.scores[kp.value] for kp in keypoints]
                scores = [point_data.get_score(kp) for kp in keypoints]
                angle_scores[joint.value] = min(scores)

        return PoseAngleData(values=angle_values, scores=angle_scores)

    @staticmethod
    def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, rotate_by: float = 0) -> float:
        """Calculate the signed angle between three points in the 2D plane.

        Args:
            p1: First point coordinates [x, y]
            p2: Second point (vertex) coordinates [x, y]
            p3: Third point coordinates [x, y]
            rotate_by: Rotation offset in radians

        Returns:
            Angle in radians in range [-π, π), or NaN if points contain NaN
        """
        # Early return if any coordinate is NaN
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)):
            return np.nan

        v1: np.ndarray = p1 - p2
        v2: np.ndarray = p3 - p2

        dot: float = np.dot(v1, v2)
        det: float = v1[0] * v2[1] - v1[1] * v2[0]
        angle: float = np.arctan2(det, dot)

        angle += rotate_by
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        return angle

    @staticmethod
    def _calculate_head_yaw(left_eye: np.ndarray, right_eye: np.ndarray,
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
    def _calculate_torso_rotation(left_shoulder: np.ndarray, right_shoulder: np.ndarray,
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
