import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING
from modules.pose.PoseTypes import PoseJoint
from modules.utils.PointsAndRects import Rect

from modules.pose.features.PosePoints import PosePointData

from modules.utils.HotReloadMethods import HotReloadMethods

# DEFINITIONS
class LimbType(Enum):
    left_arm =  auto()
    right_arm = auto()
    left_leg =  auto()
    right_leg = auto()
    spine =     auto()
    head =      auto()

class LimbCalculationType(Enum):
    SEGMENT_CHAIN = auto()  # Sum of segments (shoulder-elbow-wrist)
    MIDPOINT_TO_POINT = auto()  # Distance from one point to midpoint of others (spine)

ANATOMICAL_PROPORTIONS: dict[str, float] = {
    "arm": 1 / 0.41,    # arm length to full height ratio
    "leg": 1 / 0.48,    # leg length to full height ratio
    "spine": 1 / 0.52,  # spine length to full height ratio
    "head": 1 / 0.13    # head width to full height ratio
}

LIMB_CONFIG: dict[LimbType, dict] = {
    LimbType.left_arm: {
        "joints": (PoseJoint.left_shoulder, PoseJoint.left_elbow, PoseJoint.left_wrist),
        "proportion": ANATOMICAL_PROPORTIONS["arm"],
        "calc_type": LimbCalculationType.SEGMENT_CHAIN,
        "fallback": False
    },
    LimbType.right_arm: {
        "joints": (PoseJoint.right_shoulder, PoseJoint.right_elbow, PoseJoint.right_wrist),
        "proportion": ANATOMICAL_PROPORTIONS["arm"],
        "calc_type": LimbCalculationType.SEGMENT_CHAIN,
        "fallback": False
    },
    LimbType.left_leg: {
        "joints": (PoseJoint.left_hip, PoseJoint.left_knee, PoseJoint.left_ankle),
        "proportion": ANATOMICAL_PROPORTIONS["leg"],
        "calc_type": LimbCalculationType.SEGMENT_CHAIN,
        "fallback": False
    },
    LimbType.right_leg: {
        "joints": (PoseJoint.right_hip, PoseJoint.right_knee, PoseJoint.right_ankle),
        "proportion": ANATOMICAL_PROPORTIONS["leg"],
        "calc_type": LimbCalculationType.SEGMENT_CHAIN,
        "fallback": False
    },
    LimbType.spine: {
        "joints": (PoseJoint.nose, PoseJoint.left_hip, PoseJoint.right_hip),
        "proportion": ANATOMICAL_PROPORTIONS["spine"],
        "calc_type": LimbCalculationType.MIDPOINT_TO_POINT,
        "fallback": False
    },
    LimbType.head: {
        "joints": (PoseJoint.left_ear, PoseJoint.nose, PoseJoint.right_ear),
        "proportion": ANATOMICAL_PROPORTIONS["head"],
        "calc_type": LimbCalculationType.SEGMENT_CHAIN,
        "fallback": True  # Head is a fallback estimate
    }
}

# CLASSES
@dataclass(frozen=True)
class PoseMeasurementData:
    length_estimate: float = np.nan

class PoseMeasurementFactory:
    hotreload: HotReloadMethods | None = None

    @staticmethod
    def calculate_limb_length(points: np.ndarray, limb_type: LimbType) -> Optional[float]:
        """Calculate the length of a limb based on its type"""
        joints: tuple = LIMB_CONFIG[limb_type]["joints"]
        calc_type: LimbCalculationType = LIMB_CONFIG[limb_type]["calc_type"]

        # Check if all required points are valid
        if any(np.isnan(points[joint]).any() for joint in joints):
            return None

        if calc_type == LimbCalculationType.SEGMENT_CHAIN:
            # Sum of segments (e.g., shoulder-elbow + elbow-wrist)
            total_length = 0
            for i in range(len(joints)-1):
                total_length += float(np.linalg.norm(points[joints[i]] - points[joints[i+1]]))
            return total_length  * LIMB_CONFIG[limb_type]["proportion"]

        elif calc_type == LimbCalculationType.MIDPOINT_TO_POINT:
            # Distance from first point to midpoint of other points (spine)
            midpoint = np.mean([points[joints[1]], points[joints[2]]], axis=0)
            return float(np.linalg.norm(points[joints[0]] - midpoint)) * LIMB_CONFIG[limb_type]["proportion"]

        return None

    @staticmethod
    def compute(point_data: Optional['PosePointData'], crop_rect: Optional[Rect]) -> PoseMeasurementData:
        if PoseMeasurementFactory.hotreload is None:
            PoseMeasurementFactory.hotreload = HotReloadMethods(PoseMeasurementFactory)

        if point_data is None or crop_rect is None:
            return PoseMeasurementData()

        points: np.ndarray = point_data.points
        crop_height: float = crop_rect.height

        estimates: dict[LimbType, float] = {}

        # Calculate primary estimates
        for limb_type, config in LIMB_CONFIG.items():
            if config["fallback"]:
                continue
            limb_length: float | None = PoseMeasurementFactory.calculate_limb_length(points, limb_type)
            if limb_length is not None:
                height_estimate: float = limb_length
                estimates[limb_type] = height_estimate

        # If no primary estimates, calculate fallback estimates
        if not estimates:
            for limb_type, config in LIMB_CONFIG.items():
                if not config["fallback"]:
                    continue
                limb_length: float | None = PoseMeasurementFactory.calculate_limb_length(points, limb_type)
                if limb_length is not None:
                    height_estimate: float = limb_length
                    estimates[limb_type] = height_estimate

        if not estimates:
            # print("No valid limb estimates for height")
            return PoseMeasurementData()

        best_limb: LimbType = max(estimates, key=lambda k: estimates[k])
        best_estimate: float = estimates[best_limb]
        # print(best_limb)

        return PoseMeasurementData(length_estimate=best_estimate * crop_height)

# PoseMeasurements.hotreload = HotReloadMethods(PoseMeasurements)