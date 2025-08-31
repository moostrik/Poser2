# based on
# https://www.tensorflow.org/hub/tutorials/movenet
# https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main
# Lightning for low latency, Thunder for high accuracy

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from threading import Lock
import numpy as np
from typing import TypedDict
from typing import Optional, Callable
from modules.tracker.Tracklet import Tracklet, Rect
from pandas import Timestamp

NUM_KEYPOINTS = 17
NUM_KEYPOINT_VALUES = 3 # [y, x, score]
MULTIPOSE_BOX_SIZE = 5  # [ymin, xmin, ymax, xmax, score]
MULTIPOSE_BOX_IDX = NUM_KEYPOINTS * NUM_KEYPOINT_VALUES
MULTIPOSE_BOX_SCORE_IDX = MULTIPOSE_BOX_IDX + 4
MULTIPOSE_INSTANCE_SIZE = NUM_KEYPOINTS * NUM_KEYPOINT_VALUES + MULTIPOSE_BOX_SIZE


class ModelType(Enum):
    NONE =   0
    LARGE =  1
    MEDIUM = 2
    SMALL =  3
    TINY =   4

ModelTypeNames: list[str] = [e.name for e in ModelType]

ModelFileNames: list[tuple[str, str]] = [
    ['none', ''],
    ['rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'],
    ['rtmpose-m_8xb256-420e_aic-coco-256x192.py', 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'],
    ['rtmpose-s_8xb256-420e_aic-coco-256x192.py', 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'],
    ['rtmpose-t_8xb256-420e_aic-coco-256x192.py', 'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth']
]

class Keypoint(IntEnum):
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

KeypointNames: list[str] = [e.name for e in Keypoint]

PoseEdgeList: list[list[Keypoint]] = [
    [Keypoint.nose, Keypoint.left_eye],
    [Keypoint.nose, Keypoint.right_eye],
    [Keypoint.left_eye, Keypoint.left_ear],
    [Keypoint.right_eye, Keypoint.right_ear],
    [Keypoint.left_shoulder, Keypoint.right_shoulder],
    [Keypoint.left_shoulder, Keypoint.left_elbow],
    [Keypoint.right_shoulder, Keypoint.right_elbow],
    [Keypoint.left_elbow, Keypoint.left_wrist],
    [Keypoint.right_elbow, Keypoint.right_wrist],
    [Keypoint.left_shoulder, Keypoint.left_hip],
    [Keypoint.right_shoulder, Keypoint.right_hip],
    [Keypoint.left_hip, Keypoint.left_knee],
    [Keypoint.right_hip, Keypoint.right_knee],
    [Keypoint.left_knee, Keypoint.left_ankle],
    [Keypoint.right_knee, Keypoint.right_ankle]
]

# convert PoseIndices to a 1 dimentional np.ndarray
PoseEdgeFlatList: np.ndarray = np.array([kp.value for pose in PoseEdgeList for kp in pose], dtype=np.int32)

# make an array of increasing indices with the length of KeypointFlatList
PoseEdgeIndices: np.ndarray = np.arange(len(PoseEdgeFlatList), dtype=np.int32)

PoseEdgeColors: list[tuple[float, float, float]] = [
    (1.0, 0.7, 0.4),   # nose-left_eye
    (0.4, 0.7, 1.0),   # nose-right_eye
    (1.0, 0.8, 0.6),   # left_eye-left_ear
    (0.6, 0.8, 1.0),   # right_eye-right_ear
    (1.0, 1.0, 0.0),   # left_shoulder-right_shoulder
    (1.0, 0.7, 0.4),   # left_shoulder-left_elbow
    (0.4, 0.7, 1.0),   # right_shoulder-right_elbow
    (1.0, 0.8, 0.6),   # left_elbow-left_wrist
    (0.6, 0.8, 1.0),   # right_elbow-right_wrist
    (1.0, 0.6, 0.2),   # left_shoulder-left_hip
    (0.2, 0.6, 1.0),   # right_shoulder-right_hip
    (1.0, 0.7, 0.4),   # left_hip-left_knee
    (0.4, 0.7, 1.0),   # right_hip-right_knee
    (1.0, 0.8, 0.6),   # left_knee-left_ankle
    (0.6, 0.8, 1.0),   # right_knee-right_ankle
]

PoseAngleKeypoints: dict[Keypoint, tuple[Keypoint, Keypoint, Keypoint]] = {
    Keypoint.left_shoulder:  ( Keypoint.left_hip,       Keypoint.left_shoulder,  Keypoint.left_elbow  ),
    Keypoint.right_shoulder: ( Keypoint.right_hip,      Keypoint.right_shoulder, Keypoint.right_elbow ),
    Keypoint.left_elbow:     ( Keypoint.left_shoulder,  Keypoint.left_elbow,     Keypoint.left_wrist  ),
    Keypoint.right_elbow:    ( Keypoint.right_shoulder, Keypoint.right_elbow,    Keypoint.right_wrist ),
    # Keypoint.left_hip:       ( Keypoint.left_shoulder,  Keypoint.left_hip,       Keypoint.left_knee   ),
    # Keypoint.right_hip:      ( Keypoint.right_shoulder, Keypoint.right_hip,      Keypoint.right_knee  ),
    # Keypoint.left_knee:      ( Keypoint.left_hip,       Keypoint.left_knee,      Keypoint.left_ankle  ),
    # Keypoint.right_knee:     ( Keypoint.right_hip,      Keypoint.right_knee,     Keypoint.right_ankle ),
}

PoseAngleNames: list[str] = [k.name for k in PoseAngleKeypoints.keys()]

class JointAngle(TypedDict):
    angle: float         # The computed joint angle (degrees, or np.nan if invalid)
    confidence: float    # The minimum confidence score among the three keypoints

JointAngleDict = dict[Keypoint, JointAngle]

class PosePoints():
    def __init__(self, keypoints: np.ndarray, scores: np.ndarray) -> None:
        self.keypoints: np.ndarray = keypoints  # shape (NUM_KEYPOINTS, 2)
        self.scores: np.ndarray = scores        # shape (NUM_KEYPOINTS,)
        # self.mean_score = float(np.mean(scores))

    def getKeypoints(self) -> np.ndarray:
        return self.keypoints

    def getScores(self) -> np.ndarray:
        return self.scores

    def getVertices(self) -> np.ndarray:
        vertices: np.ndarray = np.zeros((len(PoseEdgeFlatList), 2), dtype=np.float32)
        keypoints: np.ndarray = self.getKeypoints()
        for i in range(len(PoseEdgeFlatList)):
            vertices[i] = keypoints[PoseEdgeFlatList[i]]
        return vertices

    def getColors(self, threshold: float = 0.0, r: float = 1.0, g:float = 1.0, b:float = 1.0, a:float = 1.0) -> np.ndarray:
        colors: np.ndarray = np.zeros((len(PoseEdgeFlatList), 4), dtype=np.float32)
        scores: np.ndarray = self.getScores()
        for i in range(len(PoseEdgeList)):
            kp1: int = PoseEdgeList[i][0].value
            kp2: int = PoseEdgeList[i][1].value
            s1: float = scores[kp1]
            s2: float = scores[kp2]
            score: float = min(s1, s2)
            alpha: float = 0.0
            C: tuple[float, float, float] = PoseEdgeColors[i]
            if score > threshold:
                alpha = (score - threshold) / (1 - threshold) * a
            colors[i*2] = [C[0], C[1], C[2], alpha]
            colors[i*2+1] = [C[0], C[1], C[2], alpha]
        return colors

PoseList = list[PosePoints]

@dataclass (frozen=True)
class Pose:
    id: int # Unique identifier for the pose data, typically the tracklet ID
    cam_id: int
    time_stamp: Timestamp
    tracklet: Tracklet = field(repr=False)

    crop_rect: Optional[Rect] = field(default = None)
    smooth_rect: Optional[Rect] = field(default = None)
    image: Optional[np.ndarray] = field(default = None, repr=False)

    points: Optional[PosePoints] = field(default=None, repr=False)
    angles: Optional[JointAngleDict] = field(default=None)

    is_final: bool = field(default=False, repr=False)

    def get_absolute_keypoints(self) -> Optional[np.ndarray]:
        """
        Get keypoints in the original rectangle coordinates.
        Returns a tuple of (keypoints, scores) or None if not available.
        """
        if self.points is None or self.crop_rect is None:
            return None

        keypoints = self.points.keypoints  # Normalized coordinates within the model
        rect = self.crop_rect

        # Convert from normalized coordinates to actual pixel coordinates in the crop rect
        real_keypoints = np.zeros_like(keypoints)
        real_keypoints[:, 0] = keypoints[:, 0] * rect.width + rect.x  # x coordinates
        real_keypoints[:, 1] = keypoints[:, 1] * rect.height + rect.y  # y coordinates

        return real_keypoints


PoseCallback = Callable[[Pose], None]