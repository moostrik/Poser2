import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum

from modules.pose.PoseTypes import PoseJoint
from modules.pose.PosePoints import PosePointData


class PoseHeadOrientation(IntEnum):
    yaw =       0
    pitch =     1
    roll =      2
    combined =  3

@dataclass(frozen=True)
class PoseHeadData:
    yaw: float =        np.nan  # left/right rotation
    pitch: float =      np.nan  # up/down tilt
    roll: float =       np.nan  # side tilt

    def __getitem__(self, orientation: PoseHeadOrientation) -> float:
        if orientation ==   PoseHeadOrientation.yaw:
            return self.yaw
        elif orientation == PoseHeadOrientation.pitch:
            return self.pitch
        elif orientation == PoseHeadOrientation.roll:
            return self.roll
        else:
            raise KeyError(f"Unknown orientation: {orientation}")

class PoseHead:
    @staticmethod
    def compute(point_data: Optional['PosePointData']) -> PoseHeadData:
        """
        Calculate head orientation data from point data.
        Returns HeadPoseData with calculated angles or default values if calculation not possible.
        """
        if point_data is None:
            return PoseHeadData()

        points: np.ndarray = point_data.points
        # Check if we have all necessary points (eyes and nose)
        left_eye = points[PoseJoint.left_eye.value]
        right_eye = points[PoseJoint.right_eye.value]
        nose = points[PoseJoint.nose.value]

        if (np.isnan(left_eye).any() or
            np.isnan(right_eye).any() or
            np.isnan(nose).any()):
            return PoseHeadData()  # NaN values if points are missing

        # Calculate eye midpoint
        eye_midpoint = (left_eye + right_eye) / 2

        roll: float = PoseHead.calculate_roll(left_eye, right_eye)
        yaw: float = PoseHead.calculate_yaw(nose, eye_midpoint, float(np.linalg.norm(right_eye - left_eye)))
        pitch: float = PoseHead.calculate_pitch(nose, eye_midpoint, float(np.linalg.norm(right_eye - left_eye)))

        return PoseHeadData(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
        )

    @staticmethod
    def calculate_roll(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Calculate roll angle from eye positions."""
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = (np.arctan2(dy, dx) +  2 * np.pi) % (2 * np.pi) - np.pi
        return float(roll)

    @staticmethod
    def calculate_yaw(nose: np.ndarray, eye_midpoint: np.ndarray, eye_width: float) -> float:
        """Calculate yaw angle from nose and eye midpoint positions."""
        if eye_width > 0:
            nose_offset_x = (nose[0] - eye_midpoint[0]) / eye_width
            yaw = np.arctan(nose_offset_x * 2)
            return float(yaw)
        return np.nan

    @staticmethod
    def calculate_pitch(nose: np.ndarray, eye_midpoint: np.ndarray, eye_width: float) -> float:
        """Calculate pitch angle from nose and eye midpoint positions."""
        if eye_width > 0:
            nose_offset_y = (nose[1] - eye_midpoint[1]) / eye_width
            nose_offset_y -= 0.55  # Empirical neutral position offset
            pitch = np.arctan(nose_offset_y * 2)
            return float(pitch)
        return np.nan
