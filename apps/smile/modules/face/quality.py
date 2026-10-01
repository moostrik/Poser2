"""Does a detected face qualify for a portrait? Pose from the detector's 5 keypoints, eye openness
from the 106 landmarks; the first failing check is the face's rejection. Pure arithmetic: no model,
no thread, no settings object, so it is testable with synthetic points."""

from dataclasses import dataclass
from enum import IntEnum, auto
import math

import numpy as np


class Rejection(IntEnum):
    """Why a face does not qualify; the first failing check, in this order."""
    LOW_SCORE = auto()  # detector score below confidence
    SMALL     = auto()  # eye distance below eye_min: too few pixels for the portrait
    LARGE     = auto()  # eye distance past eye_max: the portrait crop would leave the frame
    OFF_CENTRE = auto() # further from the image centre than max_offset
    TURNED    = auto()  # yaw past max_yaw
    PITCHED   = auto()  # pitch past max_pitch
    TILTED    = auto()  # roll past max_roll
    EYES      = auto()  # an eye less open than eye_open


@dataclass(frozen=True)
class Pose:
    yaw: float          # degrees, positive when the nose moves to the image right
    pitch: float        # degrees, positive when looking up
    roll: float         # degrees, the eye line's angle, positive clockwise in the image
    nose_ratio: float   # the nose tip between the eye line (0) and the mouth line (1), the pitch's raw value


@dataclass(frozen=True)
class FaceObservation:
    """One face of one frame, as the overlay shows it. Coordinates are normalised to the image."""
    bbox: tuple[float, float, float, float]     # x0, y0, x1, y1
    kps: tuple[tuple[float, float], ...]        # eyes, nose, mouth corners
    score: float
    eye_dist: float                             # distance between the eyes relative to the image width
    centre: float                               # horizontal distance of the eyes' midpoint from the image centre, relative to the image width
    pose: Pose
    eyes: float | None                          # openness of the less open eye, None if not measured
    rejection: Rejection | None
    limit: float                                # the setting the failed check compared to; 0 when accepted
    subject: bool                               # the face the sequence uses

    @property
    def accepted(self) -> bool:
        return self.rejection is None


def readout(face: FaceObservation) -> str:
    """The measured values, in the panel's names: `conf 0.92 eyes 0.06 centre 0.10 yaw 4 pitch -6 roll 2 open 0.34`."""
    p: Pose = face.pose
    open_: str = '-' if face.eyes is None else f'{face.eyes:.2f}'
    return f'conf {face.score:.2f} eyes {face.eye_dist:.2f} centre {face.centre:.2f} yaw {p.yaw:.0f} pitch {p.pitch:.0f} roll {p.roll:.0f} open {open_}'


def rejection_label(face: FaceObservation) -> str:
    """Why the face was rejected, in the panel's names: the value measured and the limit it missed,
    e.g. `yaw 32 > 20` (an angle keeps its sign: `yaw -32 > 20` is beyond 20 the other way)."""
    p: Pose = face.pose
    if face.rejection is None:
        return ''
    if face.rejection == Rejection.LOW_SCORE:
        return f'conf {face.score:.2f} < {face.limit:.2f}'
    if face.rejection == Rejection.SMALL:
        return f'eyes {face.eye_dist:.2f} < {face.limit:.2f}'
    if face.rejection == Rejection.LARGE:
        return f'eyes {face.eye_dist:.2f} > {face.limit:.2f}'
    if face.rejection == Rejection.OFF_CENTRE:
        return f'centre {face.centre:.2f} > {face.limit:.2f}'
    if face.rejection == Rejection.TURNED:
        return f'yaw {p.yaw:.0f} > {face.limit:.0f}'
    if face.rejection == Rejection.PITCHED:
        return f'pitch {p.pitch:.0f} > {face.limit:.0f}'
    if face.rejection == Rejection.TILTED:
        return f'roll {p.roll:.0f} > {face.limit:.0f}'
    open_: str = '-' if face.eyes is None else f'{face.eyes:.2f}'
    return f'open {open_} < {face.limit:.2f}'


# Approximate face proportions, for the 5 keypoints. Nose tip depth in front of the eye plane relative
# to the eye distance; nose tip depth in front of the eye-mouth plane relative to the eye-mouth
# distance; where the nose tip sits between the eye line (0) and the mouth line (1) when looking at
# the camera: the median of 301 visitor captures (p25 0.43, p75 0.53). It depends on the camera
# height, so the panel can set it (FaceSettings.pitch_zero).
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


def head_pose(kps: np.ndarray, neutral: float = NOSE_NEUTRAL) -> Pose:
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
    return Pose(yaw, pitch, roll, nose_ratio)


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


def centre_offset(kps: np.ndarray, image_width: int) -> float:
    """Horizontal distance of the point between the eyes from the image centre, relative to the
    image width: 0 in the centre, 0.5 at the edge."""
    mid_x: float = (float(kps[0][0]) + float(kps[1][0])) * 0.5
    return abs(mid_x / float(image_width) - 0.5)


def gate(score: float, eye_dist: float, centre: float, pose: Pose, confidence: float, eye_min: float, eye_max: float,
         max_offset: float, max_yaw: float, max_pitch: float, max_roll: float) -> Rejection | None:
    """The checks before the eyes, in order; None when the face passes them all."""
    if score < confidence:
        return Rejection.LOW_SCORE
    if eye_dist < eye_min:
        return Rejection.SMALL
    if eye_dist > eye_max:
        return Rejection.LARGE
    if centre > max_offset:
        return Rejection.OFF_CENTRE
    if abs(pose.yaw) > max_yaw:
        return Rejection.TURNED
    if abs(pose.pitch) > max_pitch:
        return Rejection.PITCHED
    if abs(pose.roll) > max_roll:
        return Rejection.TILTED
    return None
