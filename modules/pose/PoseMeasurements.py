import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from modules.pose.PoseTypes import PoseJoint, ANATOMICAL_PROPORTIONS
from modules.utils.PointsAndRects import Rect

from modules.pose.PosePoints import PosePointData

@dataclass(frozen=True)
class PoseMeasurementData:
    approximate_length: float = 1.0

class PoseMeasurements:
    @staticmethod
    def compute(point_data: Optional['PosePointData'], crop_rect: Optional[Rect]) -> PoseMeasurementData:
        if point_data is None or crop_rect is None:
            return PoseMeasurementData()


        points: np.ndarray = point_data.points
        scores: np.ndarray = point_data.scores
        height: float = crop_rect.height if crop_rect is not None else 1.0

        # Anatomical proportions (height multipliers)
        PROPORTION_ARM = 1 / 0.41  # arm length to full height ratio
        PROPORTION_LEG = 1 / 0.48  # leg length to full height ratio
        PROPORTION_SPINE = 1 / 0.52  # spine length to full height ratio

        # Define limb segments to measure
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
            "spine": {
                "joints": [PoseJoint.nose, PoseJoint.left_hip, PoseJoint.right_hip],
                "proportion": PROPORTION_SPINE,
                "special": "spine"  # Special case for spine calculation
            }
        }

        estimates = []

        # Calculate length estimate for each limb
        for limb_name, data in limb_data.items():
            joints = data["joints"]
            proportion = data["proportion"]

            # Check if all joints are visible
            if all(scores[joint] > 0 for joint in joints):
                length = 0

                # Special case for spine (distance from nose to mid-hip)
                if data.get("special") == "spine":
                    # Calculate mid-point between hips
                    mid_hip_x = (points[joints[1]][0] + points[joints[2]][0]) / 2
                    mid_hip_y = (points[joints[1]][1] + points[joints[2]][1]) / 2
                    mid_hip = np.array([mid_hip_x, mid_hip_y])

                    # Distance from nose to mid-hip
                    length = float(np.linalg.norm(points[joints[0]] - mid_hip))
                else:
                    # For arms and legs: sum of segments
                    seg1 = float(np.linalg.norm(points[joints[0]] - points[joints[1]]))
                    seg2 = float(np.linalg.norm(points[joints[1]] - points[joints[2]]))
                    length = seg1 + seg2

                # Calculate confidence as average of joint scores
                confidence = sum(scores[joint] for joint in joints) / len(joints)

                # Convert limb length to height estimate using anatomical proportion
                height_estimate = length * proportion * height

                estimates.append({
                    "limb": limb_name,
                    "estimate": height_estimate,
                    "confidence": confidence
                })

        if not estimates:
            return PoseMeasurementData()

        # Strategy: take the highest reasonable estimate
        # (assumes occlusion is more likely to underestimate than overestimate)
        # Sort by confidence and filter out obviously wrong estimates (too small/large)
        valid_estimates = [e for e in estimates if e["estimate"] > 0.5 * height and e["estimate"] < 3.0 * height]
        if not valid_estimates:
            return PoseMeasurementData()

        # Take estimate with highest confidence, or highest value if confidences are similar
        valid_estimates.sort(key=lambda e: e["confidence"], reverse=True)
        max_confidence = valid_estimates[0]["confidence"]
        high_confidence_estimates = [e for e in valid_estimates if e["confidence"] > max_confidence * 0.8]

        # From the high confidence estimates, take the highest value
        # This helps when parts of the body are occluded (resulting in underestimation)
        best_estimate = max(high_confidence_estimates, key=lambda e: e["estimate"])

        return PoseMeasurementData(approximate_length=best_estimate["estimate"])