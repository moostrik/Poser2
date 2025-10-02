import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from modules.pose.PoseTypes import PoseJoint, ANATOMICAL_PROPORTIONS
from modules.utils.PointsAndRects import Rect

from modules.pose.PosePoints import PosePointData

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass(frozen=True)
class PoseMeasurementData:
    approximate_length: float = 1.0

class PoseMeasurements:
    hotreload: HotReloadMethods | None = None
    @staticmethod
    def compute(point_data: Optional['PosePointData'], crop_rect: Optional[Rect]) -> PoseMeasurementData:
        if PoseMeasurements.hotreload is None:
            PoseMeasurements.hotreload = HotReloadMethods(PoseMeasurements)

        if point_data is None or crop_rect is None:
            return PoseMeasurementData()


        # TODO REMOVE SCORES AS POINTS ARE NAN WHEN BELOW SCORE THRESHOLD

        points: np.ndarray = point_data.points
        height: float = crop_rect.height if crop_rect is not None else 1.0

        # Anatomical proportions (height multipliers)
        PROPORTION_ARM = 1 / 0.41  # arm length to full height ratio
        PROPORTION_LEG = 1 / 0.48  # leg length to full height ratio
        PROPORTION_SPINE = 1 / 0.52  # spine length to full height ratiocompute
        PROPORTION_HEAD = 1 / 0.13  # head width (ear-to-ear) to full height ratio

        limb_data = {
            "left_arm": {
                "joints": [PoseJoint.left_shoulder, PoseJoint.left_elbow, PoseJoint.left_wrist],
                "proportion": PROPORTION_ARM
            },
            "right_arm": {
                "joints": [PoseJoint.right_shoulder, PoseJoint.right_elbow, PoseJoint.right_wrist],
                "proportion": PROPORTION_ARM
            },
            "left_leg": {
                "joints": [PoseJoint.left_hip, PoseJoint.left_knee, PoseJoint.left_ankle],
                "proportion": PROPORTION_LEG
            },
            "right_leg": {
                "joints": [PoseJoint.right_hip, PoseJoint.right_knee, PoseJoint.right_ankle],
                "proportion": PROPORTION_LEG
            },
            "head": {
                "joints": [PoseJoint.left_ear, PoseJoint.nose, PoseJoint.right_ear],
                "proportion": PROPORTION_HEAD
            },
            "spine": {
                "joints": [PoseJoint.nose, PoseJoint.left_hip, PoseJoint.right_hip],
                "proportion": PROPORTION_SPINE,
                "special": "spine"
            }
        }

        estimates = []

        # Calculate length estimate for each limb
        for limb_name, data in limb_data.items():
            joints = data["joints"]
            proportion = data["proportion"]

            # Check if all joints are present (not NaN)
            if all(not np.isnan(points[joint]).any() for joint in joints):
                length = 0

                if data.get("special") == "spine":
                    mid_hip_x: float = (points[joints[1]][0] + points[joints[2]][0]) / 2
                    mid_hip_y: float = (points[joints[1]][1] + points[joints[2]][1]) / 2
                    mid_hip: np.ndarray = np.array([mid_hip_x, mid_hip_y])
                    length = float(np.linalg.norm(points[joints[0]] - mid_hip))
                else:
                    seg1 = float(np.linalg.norm(points[joints[0]] - points[joints[1]]))
                    seg2 = float(np.linalg.norm(points[joints[1]] - points[joints[2]]))
                    length: float = seg1 + seg2

                height_estimate = length * proportion * height

                estimates.append({
                    "limb": limb_name,
                    "estimate": height_estimate
                })

        if not estimates:
            print ("No valid limb estimates for height")
            return PoseMeasurementData()

        # Take the highest estimate (assume occlusion underestimates)
        best_estimate = max(estimates, key=lambda e: e["estimate"])

        return PoseMeasurementData(approximate_length=best_estimate["estimate"])

# PoseMeasurements.hotreload = HotReloadMethods(PoseMeasurements)