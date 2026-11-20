import numpy as np

from ...features import Angles, AngleLandmark, Points2D, PointLandmark

"""Maps angle joints to the point landmarks needed to compute them"""
ANGLE_KEYPOINTS: dict[AngleLandmark, tuple[PointLandmark, PointLandmark, PointLandmark]] = {
    # Standard 3-point angles
    AngleLandmark.left_shoulder:  (PointLandmark.left_hip,       PointLandmark.left_shoulder,  PointLandmark.left_elbow),
    AngleLandmark.right_shoulder: (PointLandmark.right_hip,      PointLandmark.right_shoulder, PointLandmark.right_elbow),
    AngleLandmark.left_elbow:     (PointLandmark.left_shoulder,  PointLandmark.left_elbow,     PointLandmark.left_wrist),
    AngleLandmark.right_elbow:    (PointLandmark.right_shoulder, PointLandmark.right_elbow,    PointLandmark.right_wrist),
    AngleLandmark.left_hip:       (PointLandmark.left_shoulder,  PointLandmark.left_hip,       PointLandmark.left_knee),
    AngleLandmark.right_hip:      (PointLandmark.right_shoulder, PointLandmark.right_hip,      PointLandmark.right_knee),
    AngleLandmark.left_knee:      (PointLandmark.left_hip,       PointLandmark.left_knee,      PointLandmark.left_ankle),
    AngleLandmark.right_knee:     (PointLandmark.right_hip,      PointLandmark.right_knee,     PointLandmark.right_ankle),
    # Special 3-point measurement: the head's yaw from the eyes and the nose
    AngleLandmark.head:           (PointLandmark.left_eye,       PointLandmark.right_eye,      PointLandmark.nose),
}

"""Right-side angles that get negated for left-right symmetry"""
_ANGLE_MIRRORED: set[AngleLandmark] = {
    AngleLandmark.right_shoulder,
    AngleLandmark.right_elbow,
    AngleLandmark.right_hip,
    AngleLandmark.right_knee,
}

# The three points of every angle as index arrays, in AngleLandmark order, and the sign each angle gets
_ORDER: list[AngleLandmark] = list(AngleLandmark)
_P0: np.ndarray = np.array([ANGLE_KEYPOINTS[a][0] for a in _ORDER], dtype=np.intp)
_P1: np.ndarray = np.array([ANGLE_KEYPOINTS[a][1] for a in _ORDER], dtype=np.intp)
_P2: np.ndarray = np.array([ANGLE_KEYPOINTS[a][2] for a in _ORDER], dtype=np.intp)
_SIGN: np.ndarray = np.array([-1.0 if a in _ANGLE_MIRRORED else 1.0 for a in _ORDER], dtype=np.float32)
_HEAD: int = int(AngleLandmark.head)


class AngleUtils:

    @staticmethod
    def from_points(points: Points2D, aspect_ratio: float = 0.75, min_dist: float = 0.02) -> Angles:
        """Create angle measurements from keypoint data.

        Computes the geometric joint angles from 2D keypoint positions and mirrors right-side
        angles for symmetric representation. Nothing is corrected here: the neutral pose reads
        as the body's geometry has it, and ``AngleCalibrator`` maps it to 0.

        Points are scaled to isotropic space before angle computation to correct
        for non-square crop aspect ratios (e.g. 192x256 → 3:4).

        Args:
            points: Keypoint data
            aspect_ratio: Width/height ratio of the crop (e.g. 0.75 for 192x256).
                          Used to correct y-axis scaling so angles are geometrically accurate.
            min_dist: Minimum distance between points to consider them valid
                        0.02 is ~2% of image size or about 4 pixels in 192 x 256 image crops

        Returns:
            AngleFeature with computed angles and confidence scores
        """
        if points.valid_count == 0:
            return Angles.create_dummy()

        # Scale factor to correct normalized coords to isotropic (square-pixel) space.
        # For a 192x256 crop (ar=0.75), y is stretched by 1/0.75 = 1.333 so that
        # equal normalized distances correspond to equal physical distances.
        ar_scale = np.array([1.0, 1.0 / aspect_ratio], dtype=np.float32)
        P = points.values * ar_scale
        a, b, c = P[_P0], P[_P1], P[_P2]

        valid = points.valid_mask
        ok = valid[_P0] & valid[_P1] & valid[_P2]

        with np.errstate(invalid='ignore', divide='ignore'):
            # every pair of an angle's points at least min_dist apart
            d_ab = np.linalg.norm(a - b, axis=1)
            ok &= (d_ab >= min_dist) & (np.linalg.norm(a - c, axis=1) >= min_dist) & (np.linalg.norm(b - c, axis=1) >= min_dist)

            # signed angle at the middle point, from the first to the third
            v1 = a - b
            v2 = c - b
            dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
            det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            angles = np.arctan2(det, dot)

            # the head: the nose's offset from the eyes' midpoint, over the eye width
            eye_width = d_ab[_HEAD]
            if eye_width > 0:
                offset_x = (c[_HEAD, 0] - (a[_HEAD, 0] + b[_HEAD, 0]) / 2) / eye_width
                angles[_HEAD] = np.arctan(offset_x)
            else:
                angles[_HEAD] = np.nan

            angles = (((angles + np.pi) % (2 * np.pi)) - np.pi) * _SIGN

        ok &= ~np.isnan(angles)
        angle_values = np.where(ok, angles, np.nan).astype(np.float32)
        scores = points.scores
        angle_scores = np.where(ok, np.minimum(np.minimum(scores[_P0], scores[_P1]), scores[_P2]), 0.0).astype(np.float32)

        return Angles(values=angle_values, scores=angle_scores)
