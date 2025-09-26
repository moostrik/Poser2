from enum import Enum, IntEnum
import numpy as np


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

# ANATOMICAL PROPORTIONS
ANATOMICAL_PROPORTIONS: dict[str, float] = {
    "arm": 1 / 0.41,    # arm length to full height ratio
    "leg": 1 / 0.48,    # leg length to full height ratio
    "spine": 1 / 0.52   # spine length to full height ratio
}
