"""AngleCalibrator — the joint angles corrected from two reference poses per joint.

The angle extractor measures the geometric angle between body segments; what a body reads at
its neutral pose is its geometry, not 0. The calibrator maps each joint from two reference poses
held in settings as raw readings: the body standing with the arms hanging (neutral) and with the
arms raised. With ``neutral`` on, every joint's neutral reading becomes 0. With ``raised`` on, the
shoulder's raised reading becomes π in magnitude, so "0 hanging → π straight up" holds by
calibration for every consumer, sound and light alike; and the elbow, straight in both poses but
relaxed the other way once the arm is up, gets its zero from both: its zero slides from the
neutral reading to the raised reading as the arm lifts, so a relaxed straight elbow reads 0
hanging or raised, and its π stays the geometric fold from there. The legs are neutralised only;
their full poses are the leg deviation's. The sign is kept: it is the side of the body the limb
passes. The readings are tuned in the panel, read against a person or the dummy.
"""

import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field, Widget


class AngleCalibratorSettings(BaseSettings):
    """The raw readings at the reference poses, in degrees: the neutral row, then the raised row. The
    calibrator converts them to radians where it reads them; the angles themselves stay radians."""
    neutral:          Field[bool]  = Field(True, description="Map each joint's neutral reading to 0")
    shoulder_neutral: Field[float] = Field(-27.0,  min=-180.0, max=180.0, step=0.5, widget=Widget.number, description="Shoulder angle read with the arm hanging (°)")
    elbow_neutral:    Field[float] = Field(-162.0, min=-180.0, max=180.0, step=0.5, widget=Widget.number, description="Elbow angle read with the arm straight (°)")
    hip_neutral:      Field[float] = Field(171.0,  min=-180.0, max=180.0, step=0.5, widget=Widget.number, description="Hip angle read standing (°)")
    knee_neutral:     Field[float] = Field(-180.0, min=-180.0, max=180.0, step=0.5, widget=Widget.number, description="Knee angle read standing (°)")
    raised:           Field[bool]  = Field(True, description="Use the arms-raised readings: the shoulder's is 180°, the elbow's is straight with the arm up", newline=True)
    shoulder_raised:  Field[float] = Field(153.0,  min=-180.0, max=180.0, step=0.5, widget=Widget.number, description="Shoulder angle read with the arm raised (°)")
    elbow_raised:     Field[float] = Field(-162.0, min=-180.0, max=180.0, step=0.5, widget=Widget.number, description="Elbow angle read with the arm raised (°)")


# The joint kinds a setting covers; the right side is mirrored by the extractor, so one value serves both.
_NEUTRAL: dict[str, tuple[AngleLandmark, ...]] = {
    'shoulder_neutral': (AngleLandmark.left_shoulder, AngleLandmark.right_shoulder),
    'elbow_neutral':    (AngleLandmark.left_elbow,    AngleLandmark.right_elbow),
    'hip_neutral':      (AngleLandmark.left_hip,      AngleLandmark.right_hip),
    'knee_neutral':     (AngleLandmark.left_knee,     AngleLandmark.right_knee),
}
_SHOULDERS: tuple[AngleLandmark, ...] = _NEUTRAL['shoulder_neutral']
_ARMS: tuple[tuple[AngleLandmark, AngleLandmark], ...] = (
    (AngleLandmark.left_shoulder,  AngleLandmark.left_elbow),
    (AngleLandmark.right_shoulder, AngleLandmark.right_elbow),
)


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _wrap1(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class AngleCalibrator(FilterNode):
    """Maps the angles from the calibration's reference poses; see the module docstring."""

    def __init__(self, config: AngleCalibratorSettings | None = None) -> None:
        self._config = config if config is not None else AngleCalibratorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        if angles.valid_count == 0:
            return pose
        C = self._config
        neutral, scale = self._tables()
        raw = angles.values.astype(np.float64)
        values = _wrap(_wrap(raw - neutral) * scale)
        if C.neutral and C.raised:
            # The elbow's zero slides from its neutral reading to its raised reading with the arm's
            # lift (the calibrated shoulder, 0 hanging to π raised), so a straight elbow reads 0 in
            # both poses; its π is the geometric fold from there.
            elbow_neutral = math.radians(C.elbow_neutral)
            slide = _wrap1(math.radians(C.elbow_raised) - elbow_neutral)
            for shoulder, elbow in _ARMS:
                lift = 0.0 if np.isnan(values[shoulder]) else min(abs(float(values[shoulder])) / math.pi, 1.0)
                values[elbow] = _wrap1(raw[elbow] - (elbow_neutral + lift * slide))
        return replace(pose, {Angles: Angles(values.astype(np.float32), angles.scores)})

    def _tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Per landmark: the neutral to subtract and the scale to apply (radians), from the live settings."""
        C = self._config
        neutral = np.zeros(len(AngleLandmark), dtype=np.float64)
        scale = np.ones(len(AngleLandmark), dtype=np.float64)
        if C.neutral:
            for name, landmarks in _NEUTRAL.items():
                neutral[list(landmarks)] = math.radians(getattr(C, name))
        if C.raised:
            travel = abs(_wrap1(math.radians(C.shoulder_raised - C.shoulder_neutral)))
            if travel > 1e-6:
                scale[list(_SHOULDERS)] = math.pi / travel
        return neutral, scale
