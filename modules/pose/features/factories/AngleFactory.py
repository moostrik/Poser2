import numpy as np

from modules.pose.features.AngleFeature import AngleFeature, AngleLandmark
from modules.pose.features.Point2DFeature import Point2DFeature, PointLandmark

"""Maps angle joints to the point landmarks needed to compute them"""
ANGLE_KEYPOINTS: dict[AngleLandmark, tuple[PointLandmark, ...]] = {
    # Standard 3-point angles
    AngleLandmark.left_shoulder:  (PointLandmark.left_hip,       PointLandmark.left_shoulder,  PointLandmark.left_elbow),
    AngleLandmark.right_shoulder: (PointLandmark.right_hip,      PointLandmark.right_shoulder, PointLandmark.right_elbow),
    AngleLandmark.left_elbow:     (PointLandmark.left_shoulder,  PointLandmark.left_elbow,     PointLandmark.left_wrist),
    AngleLandmark.right_elbow:    (PointLandmark.right_shoulder, PointLandmark.right_elbow,    PointLandmark.right_wrist),
    AngleLandmark.left_hip:       (PointLandmark.left_shoulder,  PointLandmark.left_hip,       PointLandmark.left_knee),
    AngleLandmark.right_hip:      (PointLandmark.right_shoulder, PointLandmark.right_hip,      PointLandmark.right_knee),
    AngleLandmark.left_knee:      (PointLandmark.left_hip,       PointLandmark.left_knee,      PointLandmark.left_ankle),
    AngleLandmark.right_knee:     (PointLandmark.right_hip,      PointLandmark.right_knee,     PointLandmark.right_ankle),
    # Special 4-point measurements
    AngleLandmark.head:           (PointLandmark.left_eye,       PointLandmark.right_eye,      PointLandmark.left_shoulder, PointLandmark.right_shoulder),
}

"""Rotation offsets to normalize angles to neutral body position"""
_ANGLE_OFFSET: dict[AngleLandmark, float] = {
    AngleLandmark.left_shoulder:   0.15 * np.pi,
    AngleLandmark.right_shoulder: -0.15 * np.pi,
    AngleLandmark.left_elbow:      0.9 * np.pi,
    AngleLandmark.right_elbow:    -0.9 * np.pi,
    AngleLandmark.left_hip:       -0.95 * np.pi,
    AngleLandmark.right_hip:       0.95 * np.pi,
    AngleLandmark.left_knee:       np.pi,
    AngleLandmark.right_knee:      np.pi,
    AngleLandmark.head:            0.0,
}

"""Right-side angles that get negated for left-right symmetry"""
_ANGLE_MIRRORED: set[AngleLandmark] = {
    AngleLandmark.right_shoulder,
    AngleLandmark.right_elbow,
    AngleLandmark.right_hip,
    AngleLandmark.right_knee,
}

class AngleFactory:

    @staticmethod
    def from_points(points: Point2DFeature) -> AngleFeature:
        """Create angle measurements from keypoint data.

        Computes joint angles from 2D keypoint positions, applies rotation offsets,
        and mirrors right-side angles for symmetric representation.

        Args:
            points: Keypoint data

        Returns:
            AngleFeature with computed angles and confidence scores
        """
        if points.valid_count == 0:
            return AngleFeature.create_empty()

        angle_values = np.full(len(AngleLandmark), np.nan, dtype=np.float32)
        angle_scores = np.zeros(len(AngleLandmark), dtype=np.float32)

        # Compute all angle measurements
        for landmark, keypoints in ANGLE_KEYPOINTS.items():
            # ✅ NEW: Use are_valid() - cleaner batch validation
            if not points.are_valid(list(keypoints)):
                continue  # Skip if any required keypoint is invalid

            # Extract points (guaranteed valid - no NaN)
            P = [points[kp] for kp in keypoints]
            rotate_by = _ANGLE_OFFSET[landmark]

            # Compute angle based on number of keypoints (no NaN checks needed)
            if len(keypoints) == 3:
                angle = AngleFactory._calculate_angle(P[0], P[1], P[2], rotate_by)
            elif landmark == AngleLandmark.head:
                angle = AngleFactory._calculate_head_yaw(P[0], P[1], P[2], P[3], rotate_by)
            else:
                continue

            # Mirror right-side angles for symmetric representation
            if landmark in _ANGLE_MIRRORED:
                angle = -angle

            angle_values[landmark] = angle

            # ✅ Batch score retrieval
            scores = points.get_scores(list(keypoints))
            angle_scores[landmark] = min(scores)

        return AngleFeature(values=angle_values, scores=angle_scores)

    @staticmethod
    def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, rotate_by: float = 0) -> float:
        """Calculate signed angle between three points (assumes valid input).

        Args:
            p1: First point coordinates [x, y] (must be valid)
            p2: Second point (vertex) coordinates [x, y] (must be valid)
            p3: Third point coordinates [x, y] (must be valid)
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

        return float(angle)

    @staticmethod
    def _calculate_head_yaw(left_eye: np.ndarray, right_eye: np.ndarray,
                                   left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                                   rotate_by: float = 0) -> float:
        """Calculate head yaw (assumes valid input).

        Args:
            left_eye: Left eye coordinates [x, y] (must be valid)
            right_eye: Right eye coordinates [x, y] (must be valid)
            left_shoulder: Left shoulder coordinates [x, y] (must be valid)
            right_shoulder: Right shoulder coordinates [x, y] (must be valid)
            rotate_by: Rotation offset in radians

        Returns:
            Yaw angle in radians [-π, π), or NaN if eye_width is 0
        """
        eye_midpoint = (left_eye + right_eye) / 2
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        eye_width = float(np.linalg.norm(right_eye - left_eye))

        if eye_width > 0:
            offset_x = (eye_midpoint[0] - shoulder_midpoint[0]) / eye_width
            yaw = np.arctan(offset_x)

            yaw += rotate_by
            yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi

            return float(yaw)

        return np.nan  # Only case: eye_width == 0 (shouldn't happen with valid data)
