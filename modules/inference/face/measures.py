"""What a face's points measure: the head pose from the detector's five keypoints, the eyes' openness from the 106
landmarks, the distance between the eyes. Pure arithmetic: no model, no thread, testable with synthetic points."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HeadPose:
    yaw: float          # degrees, positive when the nose moves to the image right
    pitch: float        # degrees, positive when looking up
    roll: float         # degrees, the eye line's angle, positive clockwise in the image
    nose_ratio: float   # the nose tip between the eye line (0) and the mouth line (1), the pitch's raw value


# Approximate face proportions, for the 5 keypoints. Nose tip depth in front of the eye plane relative
# to the eye distance; nose tip depth in front of the eye-mouth plane relative to the eye-mouth
# distance; where the nose tip sits between the eye line (0) and the mouth line (1) when looking at
# the camera: the median of 301 visitor captures (p25 0.43, p75 0.53). It depends on the camera
# height, so the panel can set it (`FaceDetectorSettings.pitch_zero`).
NOSE_DEPTH_YAW: float = 0.5
NOSE_DEPTH_PITCH: float = 0.45
NOSE_NEUTRAL: float = 0.48

# Eye contour indices of the 106 landmarks (insightface 2d106det)
LEFT_EYE: tuple[int, ...] = tuple(range(33, 43))
RIGHT_EYE: tuple[int, ...] = tuple(range(87, 97))


def _derolled(points: np.ndarray, roll: float, origin: np.ndarray) -> np.ndarray:
    """Points rotated about origin so a line at `roll` degrees becomes horizontal."""
    angle: float = math.radians(-roll)
    c, s = math.cos(angle), math.sin(angle)
    matrix: np.ndarray = np.array([[c, -s], [s, c]])
    return (points - origin) @ matrix.T + origin


def head_pose(kps: np.ndarray, neutral: float = NOSE_NEUTRAL) -> HeadPose:
    """Yaw, pitch and roll from the 5 keypoints: the eye line gives the roll; with the roll undone,
    the nose tip's offset between the eyes gives the yaw and its height between the eye line and
    the mouth line gives the pitch, zero at `neutral`. Approximate degrees: the face proportions
    are assumed."""
    eye_a, eye_b = kps[0], kps[1]
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a
    roll: float = math.degrees(math.atan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))

    eye_mid: np.ndarray = (eye_a + eye_b) * 0.5
    points: np.ndarray = _derolled(np.asarray(kps, dtype=np.float64), roll, eye_mid)
    eye_dist: float = max(1e-6, float(abs(points[1][0] - points[0][0])))
    nose: np.ndarray = points[2]
    mouth_mid: np.ndarray = (points[3] + points[4]) * 0.5

    yaw_ratio: float = float(nose[0] - eye_mid[0]) / eye_dist
    yaw: float = math.degrees(math.atan(yaw_ratio / NOSE_DEPTH_YAW))

    face_height: float = max(1e-6, float(mouth_mid[1] - eye_mid[1]))
    nose_ratio: float = float(nose[1] - eye_mid[1]) / face_height
    pitch: float = math.degrees(math.atan((neutral - nose_ratio) / NOSE_DEPTH_PITCH))
    return HeadPose(yaw, pitch, roll, nose_ratio)


def eye_openness(landmarks: np.ndarray, roll: float) -> float:
    """Openness of the less open eye: height over width of each eye's contour, with the roll undone."""
    points: np.ndarray = _derolled(np.asarray(landmarks, dtype=np.float64), roll, np.zeros(2))
    openness: list[float] = []
    for indices in (LEFT_EYE, RIGHT_EYE):
        eye: np.ndarray = points[list(indices)]
        width: float = max(1e-6, float(eye[:, 0].max() - eye[:, 0].min()))
        height: float = float(eye[:, 1].max() - eye[:, 1].min())
        openness.append(height / width)
    return min(openness)


def eye_distance(kps: np.ndarray) -> float:
    """Distance between the two eye keypoints, in pixels: the face's scale, whatever the roll."""
    return float(np.linalg.norm(np.asarray(kps[1], dtype=np.float64) - np.asarray(kps[0], dtype=np.float64)))
