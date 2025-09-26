from dataclasses import dataclass, field
from enum import Enum, IntEnum
import numpy as np
from typing import TypedDict
from typing import Optional, Callable
from modules.tracker.Tracklet import Tracklet
from modules.utils.PointsAndRects import Rect


# MODEL
POSEMODELWIDTH = 192
POSEMODELHEIGHT = 256

class PoseModelType(Enum):
    NONE =   0
    LARGE =  1
    MEDIUM = 2
    SMALL =  3
    TINY =   4
PoseModelTypeNames: list[str] = [e.name for e in PoseModelType]

PoseModelFileNames: list[tuple[str, str]] = [
    ('none', ''),
    ('rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'),
    ('rtmpose-m_8xb256-420e_aic-coco-256x192.py', 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'),
    ('rtmpose-s_8xb256-420e_aic-coco-256x192.py', 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'),
    ('rtmpose-t_8xb256-420e_aic-coco-256x192.py', 'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth')
]

# JOINTS
class PoseJoint(IntEnum):
    nose =          0
    left_eye =      1
    right_eye =     2
    left_ear =      3
    right_ear =     4
    left_shoulder = 5
    right_shoulder= 6
    left_elbow =    7
    right_elbow =   8
    left_wrist =    9
    right_wrist =   10
    left_hip =      11
    right_hip =     12
    left_knee =     13
    right_knee =    14
    left_ankle =    15
    right_ankle =   16
PoseJointNames: list[str] = [e.name for e in PoseJoint]
NUM_POSE_JOINTS: int = len(PoseJoint)

# COLORS
BASE_ALPHA: float = 0.2
CENTER_COLOR: tuple[float, float, float] = (0.8, 0.8, 0.8)   # Light Gray
LEFT_COLOR: tuple[float, float, float] = (1.0, 0.5, 0.0)     # Orange
RIGHT_COLOR: tuple[float, float, float] = (0.0, 1.0, 1.0)    # Cyan
LEFT_POSITIVE: tuple[float, float, float] = (1.0, 1.0, 0.0)  # Yellow
LEFT_NEGATIVE: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
RIGHT_POSITIVE: tuple[float, float, float] = (0.0, 0.5, 1.0) # Light Blue
RIGHT_NEGATIVE: tuple[float, float, float] = (0.0, 1.0, 0.5) # Light Green


# Define color for each joint
PoseJointColors: dict[PoseJoint, tuple[float, float, float]] = {
    # Central point
    PoseJoint.nose: CENTER_COLOR,

    # Left side points - orange
    PoseJoint.left_eye: LEFT_COLOR,
    PoseJoint.left_ear: LEFT_COLOR,
    PoseJoint.left_shoulder: LEFT_COLOR,
    PoseJoint.left_elbow: LEFT_COLOR,
    PoseJoint.left_wrist: LEFT_COLOR,
    PoseJoint.left_hip: LEFT_COLOR,
    PoseJoint.left_knee: LEFT_COLOR,
    PoseJoint.left_ankle: LEFT_COLOR,

    # Right side points - cyan
    PoseJoint.right_eye: RIGHT_COLOR,
    PoseJoint.right_ear: RIGHT_COLOR,
    PoseJoint.right_shoulder: RIGHT_COLOR,
    PoseJoint.right_elbow: RIGHT_COLOR,
    PoseJoint.right_wrist: RIGHT_COLOR,
    PoseJoint.right_hip: RIGHT_COLOR,
    PoseJoint.right_knee: RIGHT_COLOR,
    PoseJoint.right_ankle: RIGHT_COLOR
}

# VERTICES
@dataclass(frozen=True)
class PoseVertexData:
    vertices: np.ndarray
    colors: np.ndarray
PoseVertexList: list[list[PoseJoint]] = [
    [PoseJoint.nose, PoseJoint.left_eye],
    [PoseJoint.nose, PoseJoint.right_eye],
    [PoseJoint.left_eye, PoseJoint.left_ear],
    [PoseJoint.right_eye, PoseJoint.right_ear],
    [PoseJoint.left_shoulder, PoseJoint.right_shoulder],
    [PoseJoint.left_shoulder, PoseJoint.left_elbow],
    [PoseJoint.right_shoulder, PoseJoint.right_elbow],
    [PoseJoint.left_elbow, PoseJoint.left_wrist],
    [PoseJoint.right_elbow, PoseJoint.right_wrist],
    [PoseJoint.left_shoulder, PoseJoint.left_hip],
    [PoseJoint.right_shoulder, PoseJoint.right_hip],
    [PoseJoint.left_hip, PoseJoint.left_knee],
    [PoseJoint.right_hip, PoseJoint.right_knee],
    [PoseJoint.left_knee, PoseJoint.left_ankle],
    [PoseJoint.right_knee, PoseJoint.right_ankle]
]
PoseVertexArray: np.ndarray = np.array([kp.value for pose in PoseVertexList for kp in pose], dtype=np.int32)
PoseVertexIndices: np.ndarray = np.arange(len(PoseVertexArray), dtype=np.int32)
# NUM_POSE_VERTICES: int = len(PoseVertexList)

# POINT DATA
@dataclass (frozen=True)
class PosePointData():
    raw_points: np.ndarray = field(repr=False)     # shape (17, 2)
    raw_scores: np.ndarray = field(repr=False)     # shape (17, 1) - original unmodified scores
    score_threshold: float

    points: np.ndarray = field(init=False)         # filtered points (NaN where score < threshold)
    scores: np.ndarray = field(init=False)         # normalized scores (0 where < threshold, else normalized)

    _vertex_data: Optional[PoseVertexData] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        s_t: float = max(0.0, min(0.99, self.score_threshold))
        object.__setattr__(self, 'score_threshold', s_t)

        # Filter points based on threshold
        filtered = self.raw_scores >= self.score_threshold
        object.__setattr__(self, 'points', np.where(filtered[:, np.newaxis], self.raw_points, np.nan))

        # Normalize scores based on threshold
        normalized: np.ndarray = np.zeros_like(self.raw_scores)
        above_threshold = self.raw_scores >= self.score_threshold
        normalized[above_threshold] = (self.raw_scores[above_threshold] - self.score_threshold) / (1.0 - self.score_threshold)
        object.__setattr__(self, 'scores', normalized)

    @property
    def vertex_data(self) -> PoseVertexData:
        if self._vertex_data is not None:
            return self._vertex_data

        vertices: np.ndarray = np.zeros((len(PoseVertexArray), 2), dtype=np.float32)
        colors: np.ndarray = np.zeros((len(PoseVertexArray), 4), dtype=np.float32)

        for i, (p1, p2) in enumerate(PoseVertexList):
            for j, joint in enumerate((p1, p2)):
                idx: int = i * 2 + j
                vertices[idx] = self.points[joint]
                colors[idx] = [*PoseJointColors[joint], (self.scores[joint] + BASE_ALPHA) / (1.0 + BASE_ALPHA)]

        vertex_data: PoseVertexData = PoseVertexData(vertices, colors)
        object.__setattr__(self, '_vertex_data', vertex_data)
        return vertex_data

# ANGLE DATA
PoseAngleJointTriplets: dict[PoseJoint, tuple[PoseJoint, PoseJoint, PoseJoint]] = {
    PoseJoint.left_shoulder:  ( PoseJoint.left_hip,       PoseJoint.left_shoulder,  PoseJoint.left_elbow  ),
    PoseJoint.right_shoulder: ( PoseJoint.right_hip,      PoseJoint.right_shoulder, PoseJoint.right_elbow ),
    PoseJoint.left_elbow:     ( PoseJoint.left_shoulder,  PoseJoint.left_elbow,     PoseJoint.left_wrist  ),
    PoseJoint.right_elbow:    ( PoseJoint.right_shoulder, PoseJoint.right_elbow,    PoseJoint.right_wrist ),
    PoseJoint.left_hip:       ( PoseJoint.left_shoulder,  PoseJoint.left_hip,       PoseJoint.left_knee   ),
    PoseJoint.right_hip:      ( PoseJoint.right_shoulder, PoseJoint.right_hip,      PoseJoint.right_knee  ),
    PoseJoint.left_knee:      ( PoseJoint.left_hip,       PoseJoint.left_knee,      PoseJoint.left_ankle  ),
    PoseJoint.right_knee:     ( PoseJoint.right_hip,      PoseJoint.right_knee,     PoseJoint.right_ankle )
}
PoseAngleJoints: list[PoseJoint] = list(PoseAngleJointTriplets.keys())
PoseAngleJointNames: list[str] = [e.name for e in PoseAngleJoints]
PoseAngleJointIdx: dict[PoseJoint, int] = {joint: idx for idx, joint in enumerate(PoseAngleJoints)}

NUM_POSE_ANGLES: int = len(PoseAngleJointTriplets)
PoseAngleRotations: dict[PoseJoint, float] = {
    PoseJoint.left_shoulder:  0.0,
    PoseJoint.right_shoulder: 0.0,
    PoseJoint.left_elbow:     np.pi,
    PoseJoint.right_elbow:    np.pi,
    PoseJoint.left_hip:       np.pi,
    PoseJoint.right_hip:      np.pi,
    PoseJoint.left_knee:      np.pi,
    PoseJoint.right_knee:     np.pi
}

@dataclass (frozen=True)
class PoseAngleData():
    angles: np.ndarray = field(default_factory=lambda: np.full(NUM_POSE_ANGLES, np.nan, dtype=np.float32)) # The computed joint angles (in radians [-Pi...Pi], or np.nan if invalid)
    scores: np.ndarray = field(default_factory=lambda: np.zeros(NUM_POSE_ANGLES, dtype=np.float32))        # The minimum confidence score among the three PoseJoints

# HEAD DATA
@dataclass(frozen=True)
class HeadPoseData:
    yaw: float   # left/right rotation
    pitch: float # up/down tilt
    roll: float  # side tilt


# THE POSE
@dataclass (frozen=True)
class Pose:
    tracklet: Tracklet = field(repr=False)

    crop_rect: Optional[Rect] = field(default = None)
    crop_image: Optional[np.ndarray] = field(default = None, repr=False)

    point_data: Optional[PosePointData] = field(default=None, repr=False)
    angle_data: Optional[PoseAngleData] = field(default=None)
    head_data: Optional[HeadPoseData] = field(default=None)

    _approximate_length: float | None = field(init=False, default=None)

    _vertex_data: Optional[PoseVertexData] = field(init=False, default=None, repr=False)

    @property
    def vertex_data(self) -> Optional[PoseVertexData]:
        if self.point_data is None:
            return None

        if self._vertex_data is not None:
            return self._vertex_data

        vertex_data: PoseVertexData = self.point_data.vertex_data

        if self.angle_data is None:
            return vertex_data

        colors: np.ndarray = vertex_data.colors.copy()

        for i, (p1, p2) in enumerate(PoseVertexList):
            for joint_pos, joint in enumerate((p1, p2)):
                idx: int | None = PoseAngleJointIdx.get(joint)
                if idx is not None:
                    angle = self.angle_data.angles[idx]
                    if not np.isnan(angle):
                        if joint.value % 2 == 1:
                            colors[i*2 + joint_pos][0:3] = LEFT_POSITIVE if angle >= 0 else LEFT_NEGATIVE
                        else:
                            colors[i*2 + joint_pos][0:3] = RIGHT_POSITIVE if angle >= 0 else RIGHT_NEGATIVE

        object.__setattr__(self, '_vertex_data', PoseVertexData(vertex_data.vertices, colors))
        return self.vertex_data

    # @property
    # def absolute_points(self) -> Optional[np.ndarray]:
    #     """
    #     Get PoseJoints in the original rectangle coordinates.
    #     Returns a tuple of (PoseJoints, scores) or None if not available.
    #     """
    #     if self.point_data is None or self.crop_rect is None:
    #         return None

    #     PoseJoints: np.ndarray = self.point_data.points  # Normalized coordinates within the model
    #     rect: Rect = self.crop_rect

    #     # Convert from normalized coordinates to actual pixel coordinates in the crop rect
    #     real_PoseJoints: np.ndarray = np.zeros_like(PoseJoints)
    #     real_PoseJoints[:, 0] = PoseJoints[:, 0] * rect.width + rect.x  # x coordinates
    #     real_PoseJoints[:, 1] = PoseJoints[:, 1] * rect.height + rect.y  # y coordinates

    #     return real_PoseJoints

    @property
    def get_approximate_person_length(self) -> float | None:
        if self.point_data is None:
            return None
        if self._approximate_length is not None:
            return self._approximate_length

        points: np.ndarray = self.point_data.points
        scores: np.ndarray = self.point_data.scores
        height: float = self.crop_rect.height if self.crop_rect is not None else 1.0

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
            return None

        # Strategy: take the highest reasonable estimate
        # (assumes occlusion is more likely to underestimate than overestimate)
        # Sort by confidence and filter out obviously wrong estimates (too small/large)
        valid_estimates = [e for e in estimates if e["estimate"] > 0.5 * height and e["estimate"] < 3.0 * height]
        if not valid_estimates:
            return None

        # Take estimate with highest confidence, or highest value if confidences are similar
        valid_estimates.sort(key=lambda e: e["confidence"], reverse=True)
        max_confidence = valid_estimates[0]["confidence"]
        high_confidence_estimates = [e for e in valid_estimates if e["confidence"] > max_confidence * 0.8]

        # From the high confidence estimates, take the highest value
        # This helps when parts of the body are occluded (resulting in underestimation)
        best_estimate = max(high_confidence_estimates, key=lambda e: e["estimate"])

        object.__setattr__(self, '_approximate_length', best_estimate["estimate"])
        return best_estimate["estimate"]

PoseDict = dict[PoseJoint, Pose]
PoseCallback = Callable[[Pose], None]