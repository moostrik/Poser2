"""Joint selection — which ``AngleLandmark`` joints a posture comparison compares."""

from __future__ import annotations

import numpy as np

from modules.settings import BaseSettings, Field
from ..features import AngleLandmark


class JointSelectSettings(BaseSettings):
    """Which joints a posture comparison compares; an unchecked joint is neither compared nor counted.
    The fields are the ``AngleLandmark`` members, in their order."""
    left_shoulder:  Field[bool] = Field(True, description="Compare the left shoulder")
    right_shoulder: Field[bool] = Field(True, description="Compare the right shoulder")
    left_elbow:     Field[bool] = Field(True, description="Compare the left elbow")
    right_elbow:    Field[bool] = Field(True, description="Compare the right elbow")
    left_hip:       Field[bool] = Field(True, description="Compare the left hip")
    right_hip:      Field[bool] = Field(True, description="Compare the right hip")
    left_knee:      Field[bool] = Field(True, description="Compare the left knee")
    right_knee:     Field[bool] = Field(True, description="Compare the right knee")
    head:           Field[bool] = Field(True, description="Compare the head")


def joint_mask(joints: JointSelectSettings) -> np.ndarray:
    """The joints selected, as a bool ``(F,)`` in ``AngleLandmark`` order."""
    return np.array([bool(getattr(joints, landmark.name)) for landmark in AngleLandmark], dtype=bool)
