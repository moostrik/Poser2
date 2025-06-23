# based on
# https://www.tensorflow.org/hub/tutorials/movenet
# https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main
# Lightning for low latency, Thunder for high accuracy

from enum import Enum
import numpy as np
from typing import Callable

NUM_KEYPOINTS = 17
NUM_KEYPOINT_VALUES = 3 # [y, x, score]
MULTIPOSE_BOX_SIZE = 5  # [ymin, xmin, ymax, xmax, score]
MULTIPOSE_BOX_IDX = NUM_KEYPOINTS * NUM_KEYPOINT_VALUES
MULTIPOSE_BOX_SCORE_IDX = MULTIPOSE_BOX_IDX + 4
MULTIPOSE_INSTANCE_SIZE = NUM_KEYPOINTS * NUM_KEYPOINT_VALUES + MULTIPOSE_BOX_SIZE


class ModelType(Enum):
    NONE =      0
    THUNDER =   1
    LIGHTNING = 2
    MULTI =     3

ModelTypeNames: list[str] = [e.name for e in ModelType]

ModelFileNames: list[str] = [
    'none',
    'movenet_singlepose_thunder_4.onnx',
    'movenet_singlepose_lightning_4.onnx',
    'movenet_multipose_lightning_1.onnx'
]

ModelInputSize: list[int] = [
    0,
    256,
    192,
    256,
]

class Keypoints(Enum):
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

KeypointNames: list[str] = [e.name for e in Keypoints]

PoseEdgeList: list[list[Keypoints]] = [
    [Keypoints.nose, Keypoints.left_eye],
    [Keypoints.nose, Keypoints.right_eye],
    [Keypoints.left_eye, Keypoints.left_ear],
    [Keypoints.right_eye, Keypoints.right_ear],
    [Keypoints.left_shoulder, Keypoints.right_shoulder],
    [Keypoints.left_shoulder, Keypoints.left_elbow],
    [Keypoints.right_shoulder, Keypoints.right_elbow],
    [Keypoints.left_elbow, Keypoints.left_wrist],
    [Keypoints.right_elbow, Keypoints.right_wrist],
    [Keypoints.left_shoulder, Keypoints.left_hip],
    [Keypoints.right_shoulder, Keypoints.right_hip],
    [Keypoints.left_hip, Keypoints.left_knee],
    [Keypoints.right_hip, Keypoints.right_knee],
    [Keypoints.left_knee, Keypoints.left_ankle],
    [Keypoints.right_knee, Keypoints.right_ankle]
]

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

# convert PoseIndices to a 1 dimentional np.ndarray
KeypointFlatList: np.ndarray = np.array([kp.value for pose in PoseEdgeList for kp in pose], dtype=np.int32)

# make an array of increasing indices with the length of KeypointFlatList
Indices: np.ndarray = np.arange(len(KeypointFlatList), dtype=np.int32)


class Pose():
    def __init__(self, keypoints: np.ndarray, scores: np.ndarray) -> None:
        self.keypoints: np.ndarray = keypoints  # shape (NUM_KEYPOINTS, 2)
        self.scores: np.ndarray = scores        # shape (NUM_KEYPOINTS,)
        # self.mean_score = float(np.mean(scores))

    def getKeypoints(self) -> np.ndarray:
        return self.keypoints

    def getScores(self) -> np.ndarray:
        return self.scores

    def getVertices(self) -> np.ndarray:
        vertices: np.ndarray = np.zeros((len(KeypointFlatList), 2), dtype=np.float32)
        keypoints: np.ndarray = self.getKeypoints()
        for i in range(len(KeypointFlatList)):
            vertices[i] = keypoints[KeypointFlatList[i]]
        return vertices

    def getColors(self, threshold: float = 0.0, r: float = 1.0, g:float = 1.0, b:float = 1.0, a:float = 1.0) -> np.ndarray:
        colors: np.ndarray = np.zeros((len(KeypointFlatList), 4), dtype=np.float32)
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

PoseList = list[Pose]

