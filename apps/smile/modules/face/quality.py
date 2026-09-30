"""Is a detected face a good portrait? Pose from the detector's 5 keypoints, eye openness from the
106 landmarks, sharpness from the pixels; a rejection reason and a quality per face. Pure arithmetic:
no model, no thread, no settings object, so it is testable with synthetic points."""

from dataclasses import dataclass
from enum import IntEnum, auto
import math

import cv2
import numpy as np


class Rejection(IntEnum):
    """Why a face does not qualify; the first failing check, in this order."""
    LOW_SCORE = auto()  # detector score below confidence
    SMALL     = auto()  # face height below size_filter
    TURNED    = auto()  # yaw past max_yaw
    PITCHED   = auto()  # pitch past max_pitch
    TILTED    = auto()  # roll past max_roll
    EYES      = auto()  # an eye less open than eye_open


REJECTION_LABELS: dict[Rejection, str] = {
    Rejection.LOW_SCORE: 'score',
    Rejection.SMALL:     'small',
    Rejection.TURNED:    'turned',
    Rejection.PITCHED:   'pitched',
    Rejection.TILTED:    'tilted',
    Rejection.EYES:      'eyes',
}


@dataclass(frozen=True)
class Pose:
    yaw: float      # degrees, positive when the nose moves to the image right
    pitch: float    # degrees, positive when looking up
    roll: float     # degrees, the eye line's angle, positive clockwise in the image


@dataclass(frozen=True)
class FaceObservation:
    """One face of one frame, as the overlay shows it. Coordinates are normalised to the image."""
    bbox: tuple[float, float, float, float]     # x0, y0, x1, y1
    kps: tuple[tuple[float, float], ...]        # eyes, nose, mouth corners
    score: float
    size: float                                 # face height relative to the image height
    pose: Pose
    eyes: float | None                          # openness of the less open eye, None if not measured
    quality: float
    rejection: Rejection | None
    subject: bool                               # the face the sequence uses

    @property
    def accepted(self) -> bool:
        return self.rejection is None


# Approximate face proportions, for the 5 keypoints. Nose tip depth in front of the eye plane relative
# to the eye distance; nose tip depth in front of the eye-mouth plane relative to the eye-mouth
# distance; where the nose tip sits between the eye line (0) and the mouth line (1) when frontal.
NOSE_DEPTH_YAW: float = 0.5
NOSE_DEPTH_PITCH: float = 0.45
NOSE_NEUTRAL: float = 0.58

# Eye contour indices of the 106 landmarks (insightface 2d106det)
LEFT_EYE: tuple[int, ...] = tuple(range(33, 43))
RIGHT_EYE: tuple[int, ...] = tuple(range(87, 97))

# Laplacian variance of a 128 px face at which the sharpness term is 0.5
SHARPNESS_HALF: float = 150.0
SHARPNESS_SIZE: int = 128


def _derolled(points: np.ndarray, roll: float, origin: np.ndarray) -> np.ndarray:
    """Points rotated about origin so a line at `roll` degrees becomes horizontal."""
    angle: float = math.radians(-roll)
    c, s = math.cos(angle), math.sin(angle)
    matrix: np.ndarray = np.array([[c, -s], [s, c]])
    return (points - origin) @ matrix.T + origin


def head_pose(kps: np.ndarray) -> Pose:
    """Yaw, pitch and roll from the 5 keypoints: the eye line gives the roll; with the roll undone,
    the nose tip's offset between the eyes gives the yaw and its height between the eye line and
    the mouth line gives the pitch. Approximate degrees: the face proportions are assumed."""
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
    pitch: float = math.degrees(math.atan((NOSE_NEUTRAL - nose_ratio) / NOSE_DEPTH_PITCH))
    return Pose(yaw, pitch, roll)


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


def sharpness(image: np.ndarray, bbox: np.ndarray) -> float:
    """Laplacian variance of the face box, at a fixed size so it does not depend on the face size."""
    height, width = image.shape[:2]
    x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x1, y1 = min(width, int(bbox[2])), min(height, int(bbox[3]))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    face: np.ndarray = cv2.resize(image[y0:y1, x0:x1], (SHARPNESS_SIZE, SHARPNESS_SIZE), interpolation=cv2.INTER_AREA)
    gray: np.ndarray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def gate(score: float, size: float, pose: Pose, confidence: float, size_filter: float,
         max_yaw: float, max_pitch: float, max_roll: float) -> Rejection | None:
    """The checks before the eyes, in order; None when the face passes them all."""
    if score < confidence:
        return Rejection.LOW_SCORE
    if size < size_filter:
        return Rejection.SMALL
    if abs(pose.yaw) > max_yaw:
        return Rejection.TURNED
    if abs(pose.pitch) > max_pitch:
        return Rejection.PITCHED
    if abs(pose.roll) > max_roll:
        return Rejection.TILTED
    return None


def quality(pose: Pose, eyes: float | None, sharp: float, max_yaw: float, max_pitch: float, eye_open: float) -> float:
    """0..1, higher is a better portrait. Each pose axis halves the quality at its limit and falls
    off smoothly past it; the eyes count fully open at twice eye_open; an unmeasured eye counts half."""
    pose_term: float = 1.0 / (1.0 + (pose.yaw / max(1e-6, max_yaw)) ** 2)
    pose_term *= 1.0 / (1.0 + (pose.pitch / max(1e-6, max_pitch)) ** 2)
    eye_term: float = 0.5 if eyes is None else min(1.0, eyes / max(1e-6, 2.0 * eye_open))
    sharp_term: float = sharp / (sharp + SHARPNESS_HALF)
    return pose_term * eye_term * sharp_term
