import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING


from modules.pose.PoseTypes import PoseJoint
from modules.pose.PosePoints import PosePointData

@dataclass(frozen=True)
class PoseHeadData:
    yaw: float = 0.0    # left/right rotation
    pitch: float = 0.0  # up/down tilt
    roll: float = 0.0   # side tilt

class PoseHeadOrientation:
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
            return PoseHeadData()  # Default values if points are missing

        # Calculate eye midpoint
        eye_midpoint = (left_eye + right_eye) / 2

        # Calculate head orientation

        # 1. Roll: angle between horizontal line and line connecting eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = np.arctan2(dy, dx)

        # 2. Yaw: horizontal deviation of nose from eye midpoint
        # Calculate the horizontal position of nose relative to eye midpoint
        eye_width = np.linalg.norm(right_eye - left_eye)
        if eye_width > 0:
            # Normalize by eye width to get a scale-invariant measure
            nose_offset_x = (nose[0] - eye_midpoint[0]) / eye_width
            # Convert to a reasonable angle range (-0.5π to 0.5π)
            yaw = np.arctan(nose_offset_x * 2)
        else:
            yaw = 0.0

        # 3. Pitch: vertical deviation of nose from eye midpoint
        # Calculate vertical position of nose relative to eye level
        if eye_width > 0:  # Use eye width as reference scale
            nose_offset_y = (nose[1] - eye_midpoint[1]) / eye_width
            # Apply offset for neutral position (looking straight)
            # Typical facial proportions: nose is ~0.5-0.6 eye widths below eye midpoint when facing forward
            nose_offset_y -= 0.55  # Empirical neutral position offset
            # Convert to angle range, negative is looking up
            pitch = np.arctan(nose_offset_y * 2)
        else:
            pitch = 0.0

        return PoseHeadData(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll)
        )
