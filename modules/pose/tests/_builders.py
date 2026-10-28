"""Shared builders for pose tests: frames, features and a bendable skeleton with known joint geometry."""

import math
from typing import Callable

import numpy as np

from modules.pose.features import BaseScalarFeature, PointLandmark, Points2D, configure_features
from modules.pose.frame import Frame

# Track-indexed features (Similarity, LeaderScore, MotionGate) are configured process-wide and the
# first call wins, so every pose test must agree on one count. 4 matches the applicators' defaults.
NUM_POSES = 4
configure_features(NUM_POSES)

FPS = 30.0


def frame(track_id: int = 0, t: float = 0.0, cam_id: int = 0, features: dict | None = None) -> Frame:
    """A pose frame with an explicit timestamp."""
    return Frame(track_id=track_id, cam_id=cam_id, time_stamp=t, features=features)


def scalar(feature_type: type[BaseScalarFeature], values: dict[int, float], score: float = 1.0) -> BaseScalarFeature:
    """A scalar feature with the given elements set; every other element NaN with score 0."""
    n = feature_type.length()
    v = np.full(n, np.nan, dtype=np.float32)
    s = np.zeros(n, dtype=np.float32)
    for element, value in values.items():
        v[element] = value
        s[element] = 0.0 if math.isnan(value) else score
    return feature_type(v, s)


def points(coords: dict[PointLandmark, tuple[float, float]], scores: dict[PointLandmark, float] | None = None) -> Points2D:
    """Points2D with the given landmarks set (score 1 unless overridden); the rest NaN with score 0."""
    n = len(PointLandmark)
    v = np.full((n, 2), np.nan, dtype=np.float32)
    s = np.zeros(n, dtype=np.float32)
    for lm, xy in coords.items():
        v[lm] = xy
        s[lm] = 1.0 if scores is None else scores.get(lm, 1.0)
    return Points2D(v, s)


# A standing figure in physical (square-pixel) units, facing the camera, centred on x = 0.5.
_P = PointLandmark
_STANDING: dict[PointLandmark, tuple[float, float]] = {
    _P.nose:           (0.50, 0.16),
    _P.left_eye:       (0.47, 0.13),
    _P.right_eye:      (0.53, 0.13),
    _P.left_ear:       (0.44, 0.15),
    _P.right_ear:      (0.56, 0.15),
    _P.left_shoulder:  (0.40, 0.33),
    _P.right_shoulder: (0.60, 0.33),
    _P.left_elbow:     (0.38, 0.53),
    _P.right_elbow:    (0.62, 0.53),
    _P.left_wrist:     (0.37, 0.71),
    _P.right_wrist:    (0.63, 0.71),
    _P.left_hip:       (0.44, 0.73),
    _P.right_hip:      (0.56, 0.73),
    _P.left_knee:      (0.44, 0.96),
    _P.right_knee:     (0.56, 0.96),
    _P.left_ankle:     (0.44, 1.17),
    _P.right_ankle:    (0.56, 1.17),
}

# (joint, the distal point rotated about it, whether the joint is on the left side)
_BENDS: dict[str, tuple[PointLandmark, PointLandmark, bool]] = {
    'left_elbow':  (PointLandmark.left_elbow,  PointLandmark.left_wrist,  True),
    'right_elbow': (PointLandmark.right_elbow, PointLandmark.right_wrist, False),
    'left_knee':   (PointLandmark.left_knee,   PointLandmark.left_ankle,  True),
    'right_knee':  (PointLandmark.right_knee,  PointLandmark.right_ankle, False),
}


def skeleton_coords(aspect_ratio: float = 0.75, **bends: float) -> dict[PointLandmark, tuple[float, float]]:
    """Normalised keypoints of the standing figure with joints bent by the given radians.

    The distal point is rotated about the joint in physical space: counter-clockwise (in image
    coordinates) on the left, mirrored on the right, so equal bends give a mirror-symmetric pose.
    Physical y is squeezed by ``aspect_ratio`` into crop-normalised y, as a non-square crop would.
    """
    phys = dict(_STANDING)
    for name, angle in bends.items():
        joint, distal, left = _BENDS[name]
        a = angle if left else -angle
        jx, jy = phys[joint]
        dx, dy = phys[distal][0] - jx, phys[distal][1] - jy
        c, s = math.cos(a), math.sin(a)
        phys[distal] = (jx + c * dx - s * dy, jy + s * dx + c * dy)
    return {lm: (x, y * aspect_ratio) for lm, (x, y) in phys.items()}


def skeleton(aspect_ratio: float = 0.75, **bends: float) -> Points2D:
    """Points2D of the standing figure with the given joint bends (see ``skeleton_coords``)."""
    return points(skeleton_coords(aspect_ratio, **bends))


def sequence(n: int, build: Callable[[int, float], Frame]) -> list[Frame]:
    """``n`` frames built by ``build(index, timestamp)`` at 30 Hz."""
    return [build(i, i / FPS) for i in range(n)]


def wrap(angle: float) -> float:
    """Wrap an angle to [-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))
