from enum import Enum, IntEnum
import numpy as np


# MODEL
POSE_MODEL_WIDTH = 192
POSE_MODEL_HEIGHT = 256

class PoseModelType(Enum):
    NONE =   0
    LARGE =  1
    MEDIUM = 2
    SMALL =  3
    TINY =   4
POSE_MODEL_TYPE_NAMES: list[str] = [e.name for e in PoseModelType]

POSE_MODEL_FILE_NAMES: list[tuple[str, str]] = [
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
POSE_JOINT_NAMES: list[str] = [e.name for e in PoseJoint]
POSE_NUM_JOINTS: int = len(PoseJoint)

# COLORS
POSE_COLOR_ALPHA_BASE: float = 0.2
POSE_COLOR_CENTER: tuple[float, float, float] = (0.8, 0.8, 0.8)   # Light Gray
POSE_COLOR_LEFT: tuple[float, float, float] = (1.0, 0.5, 0.0)     # Orange
POSE_COLOR_LEFT_POSITIVE: tuple[float, float, float] = (1.0, 1.0, 0.0)  # Yellow
POSE_COLOR_LEFT_NEGATIVE: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
POSE_COLOR_RIGHT: tuple[float, float, float] = (0.0, 1.0, 1.0)    # Cyan
POSE_COLOR_RIGHT_POSITIVE: tuple[float, float, float] = (0.0, 0.5, 1.0) # Light Blue
POSE_COLOR_RIGHT_NEGATIVE: tuple[float, float, float] = (0.0, 1.0, 0.5) # Light Green


# Define color for each joint
POSE_JOINT_COLORS: dict[PoseJoint, tuple[float, float, float]] = {
    # Central point
    PoseJoint.nose: POSE_COLOR_CENTER,

    # Left side points - orange
    PoseJoint.left_eye: POSE_COLOR_LEFT,
    PoseJoint.left_ear: POSE_COLOR_LEFT,
    PoseJoint.left_shoulder: POSE_COLOR_LEFT,
    PoseJoint.left_elbow: POSE_COLOR_LEFT,
    PoseJoint.left_wrist: POSE_COLOR_LEFT,
    PoseJoint.left_hip: POSE_COLOR_LEFT,
    PoseJoint.left_knee: POSE_COLOR_LEFT,
    PoseJoint.left_ankle: POSE_COLOR_LEFT,

    # Right side points - cyan
    PoseJoint.right_eye: POSE_COLOR_RIGHT,
    PoseJoint.right_ear: POSE_COLOR_RIGHT,
    PoseJoint.right_shoulder: POSE_COLOR_RIGHT,
    PoseJoint.right_elbow: POSE_COLOR_RIGHT,
    PoseJoint.right_wrist: POSE_COLOR_RIGHT,
    PoseJoint.right_hip: POSE_COLOR_RIGHT,
    PoseJoint.right_knee: POSE_COLOR_RIGHT,
    PoseJoint.right_ankle: POSE_COLOR_RIGHT
}

# VERTICES
POSE_VERTEX_LIST: list[list[PoseJoint]] = [
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
POSE_VERTEX_ARRAY: np.ndarray = np.array([kp.value for pose in POSE_VERTEX_LIST for kp in pose], dtype=np.int32)
POSE_VERTEX_INDICES: np.ndarray = np.arange(len(POSE_VERTEX_ARRAY), dtype=np.int32)

# ANGLE DATA
POSE_ANGLE_JOINT_TRIPLETS: dict[PoseJoint, tuple[PoseJoint, PoseJoint, PoseJoint]] = {
    PoseJoint.left_shoulder:  ( PoseJoint.left_hip,       PoseJoint.left_shoulder,  PoseJoint.left_elbow  ),
    PoseJoint.right_shoulder: ( PoseJoint.right_hip,      PoseJoint.right_shoulder, PoseJoint.right_elbow ),
    PoseJoint.left_elbow:     ( PoseJoint.left_shoulder,  PoseJoint.left_elbow,     PoseJoint.left_wrist  ),
    PoseJoint.right_elbow:    ( PoseJoint.right_shoulder, PoseJoint.right_elbow,    PoseJoint.right_wrist ),
    PoseJoint.left_hip:       ( PoseJoint.left_shoulder,  PoseJoint.left_hip,       PoseJoint.left_knee   ),
    PoseJoint.right_hip:      ( PoseJoint.right_shoulder, PoseJoint.right_hip,      PoseJoint.right_knee  ),
    PoseJoint.left_knee:      ( PoseJoint.left_hip,       PoseJoint.left_knee,      PoseJoint.left_ankle  ),
    PoseJoint.right_knee:     ( PoseJoint.right_hip,      PoseJoint.right_knee,     PoseJoint.right_ankle )
}
POSE_ANGLE_JOINTS: list[PoseJoint] = list(POSE_ANGLE_JOINT_TRIPLETS.keys())
POSE_ANGLE_JOINT_NAMES: list[str] = [e.name for e in POSE_ANGLE_JOINTS]
POSE_ANGLE_JOINT_IDXS: dict[PoseJoint, int] = {joint: idx for idx, joint in enumerate(POSE_ANGLE_JOINTS)}

POSE_NUM_ANGLES: int = len(POSE_ANGLE_JOINT_TRIPLETS)
POSE_ANGLE_ROTATIONS: dict[PoseJoint, float] = {
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
