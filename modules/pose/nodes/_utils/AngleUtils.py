import numpy as np

from modules.pose.features.Angles import Angles, AngleLandmark
from modules.pose.features.Points2D import Points2D, PointLandmark

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
    AngleLandmark.head:           (PointLandmark.left_eye,       PointLandmark.right_eye,      PointLandmark.nose),
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

class AngleUtils:

    @staticmethod
    def from_points(points: Points2D, min_dist: float = 0.02) -> Angles:
        """Create angle measurements from keypoint data.

        Computes joint angles from 2D keypoint positions, applies rotation offsets,
        and mirrors right-side angles for symmetric representation.

        Args:
            points: Keypoint data
            min_dist: Minimum distance between points to consider them valid
                        0.02 is ~2% of image size or about 4 pixels in 192 x 256 image crops

        Returns:
            AngleFeature with computed angles and confidence scores
        """
        if points.valid_count == 0:
            return Angles.create_dummy()

        angle_values = np.full(len(AngleLandmark), np.nan, dtype=np.float32)
        angle_scores = np.zeros(len(AngleLandmark), dtype=np.float32)

        # Compute all angle measurements
        for landmark, keypoints in ANGLE_KEYPOINTS.items():
            if not points.are_valid(list(keypoints)):
                continue  # Skip if any required keypoint is invalid

            # Extract points (guaranteed valid - no NaN)
            P = [points[kp] for kp in keypoints]
            if AngleUtils._points_too_close(P, min_dist):
                continue  # Skip if any points are too close

            rotate_by = _ANGLE_OFFSET[landmark]

            # Compute angle based on number of keypoints (no NaN checks needed)
            if landmark == AngleLandmark.head:
                angle = AngleUtils._calculate_head_yaw(P[0], P[1], P[2], rotate_by)
            else:
                angle = AngleUtils._calculate_angle(P[0], P[1], P[2], rotate_by)

            # Mirror right-side angles for symmetric representation
            if landmark in _ANGLE_MIRRORED:
                angle = -angle

            angle_values[landmark] = angle

            scores = points.get_scores(list(keypoints))
            angle_scores[landmark] = min(scores) if not np.isnan(angle) else 0.0

        return Angles(values=angle_values, scores=angle_scores)

    @staticmethod
    def _points_too_close(points: list[np.ndarray], min_dist: float = 0.02) -> bool:
        """Return True if any pair of points are closer than min_dist."""
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < min_dist:
                    return True
        return False

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
                            nose: np.ndarray,
                            rotate_by: float = 0) -> float:
        """Calculate head yaw using eyes and nose (assumes valid input).

        Args:
            left_eye: Left eye coordinates [x, y] (must be valid)
            right_eye: Right eye coordinates [x, y] (must be valid)
            nose: Nose coordinates [x, y] (must be valid)
            rotate_by: Rotation offset in radians

        Returns:
            Yaw angle in radians [-π, π), or NaN if eye_width is 0
        """
        eye_midpoint = (left_eye + right_eye) / 2
        eye_width = float(np.linalg.norm(right_eye - left_eye))

        if eye_width > 0:
            offset_x = (nose[0] - eye_midpoint[0]) / eye_width
            yaw = np.arctan(offset_x)

            yaw += rotate_by
            yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi

            return float(yaw)

        return np.nan  # Only case: eye_width == 0 (shouldn't happen with valid data)
