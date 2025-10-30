import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum

from modules.pose.PoseTypes import PoseJoint
from modules.pose.features.PosePoints import PosePointData


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
    yaw_score: float =  0.0     # confidence for yaw
    pitch_score: float=0.0     # confidence for pitch
    roll_score: float = 0.0     # confidence for roll

    def __getitem__(self, orientation: PoseHeadOrientation) -> float:
        if orientation ==   PoseHeadOrientation.yaw:
            return self.yaw
        elif orientation == PoseHeadOrientation.pitch:
            return self.pitch
        elif orientation == PoseHeadOrientation.roll:
            return self.roll
        else:
            raise KeyError(f"Unknown orientation: {orientation}")

class PoseHeadFactory:
    @staticmethod
    def from_points(point_data: Optional['PosePointData']) -> PoseHeadData:
        """
        Calculate head orientation data from point data.
        Returns HeadPoseData with calculated angles or default values if calculation not possible.
        """
        if point_data is None:
            return PoseHeadData()

        points: np.ndarray = point_data.values
        scores: np.ndarray = point_data.scores

        # Check if we have all necessary points (eyes and nose)
        left_eye = points[PoseJoint.left_eye.value]
        right_eye = points[PoseJoint.right_eye.value]
        nose = points[PoseJoint.nose.value]

        if (np.isnan(left_eye).any() or
            np.isnan(right_eye).any() or
            np.isnan(nose).any()):
            return PoseHeadData()  # NaN values if points are missing

        # Get scores for the points
        left_eye_score = scores[PoseJoint.left_eye.value]
        right_eye_score = scores[PoseJoint.right_eye.value]
        nose_score = scores[PoseJoint.nose.value]

        # Calculate eye midpoint
        eye_midpoint = (left_eye + right_eye) / 2
        eye_width = float(np.linalg.norm(right_eye - left_eye))

        roll: float = PoseHeadFactory.calculate_roll(left_eye, right_eye)
        yaw: float = PoseHeadFactory.calculate_yaw(nose, eye_midpoint, eye_width)
        pitch: float = PoseHeadFactory.calculate_pitch(nose, eye_midpoint, eye_width)

        # Calculate scores for each orientation
        # Roll uses only eye points
        roll_score = min(left_eye_score, right_eye_score)
        # Yaw and pitch use nose and eye points
        yaw_score = min(nose_score, left_eye_score, right_eye_score)
        pitch_score = min(nose_score, left_eye_score, right_eye_score)

        return PoseHeadData(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            yaw_score=float(yaw_score),
            pitch_score=float(pitch_score),
            roll_score=float(roll_score)
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
