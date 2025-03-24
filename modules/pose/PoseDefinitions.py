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

KeypointList: list[list[Keypoints]] = [
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

# convert PoseIndices to a 1 dimentional np.ndarray
KeypointFlatList: np.ndarray = np.array([kp.value for pose in KeypointList for kp in pose], dtype=np.int32)

# make an array of increasing indices with the length of KeypointFlatList
Indices: np.ndarray = np.arange(len(KeypointFlatList), dtype=np.int32)

class PoseBox():
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, score: float) -> None:
        self._xmin: float =  xmin
        self._ymin: float =  ymin
        self._xmax: float =  xmax
        self._ymax: float =  ymax
        self._score: float = score

    def getTopRight(self) -> tuple[float, float]:
        return self._xmax, self._ymin
    def getTopLeft(self) -> tuple[float, float]:
        return self._xmin, self._ymin
    def getBottomRight(self) -> tuple[float, float]:
        return self._xmax, self._ymax
    def getBottomLeft(self) -> tuple[float, float]:
        return self._xmin, self._ymax
    def getCenter(self) -> tuple[float, float]:
        return (self._xmin + self._xmax) / 2, (self._ymin + self._ymax) / 2
    def getWidth(self) -> float:
        return self._xmax - self._xmin
    def getHeight(self) -> float:
        return self._ymax - self._ymin
    def getSize(self) -> tuple[float, float]:
        return self.getWidth(), self.getHeight()
    def getScore(self) -> float:
        return self._score
    def getXmin(self) -> float:
        return self._xmin
    def getYmin(self) -> float:
        return self._ymin
    def getXmax(self) -> float:
        return self._xmax
    def getYmax(self) -> float:
        return self._ymax


class Pose():
    def __init__(self, keypoints: np.ndarray, scores: np.ndarray) -> None:
        self.keypoints: np.ndarray = keypoints
        self.scores: np.ndarray = scores
        self.mean_score = float(np.mean(scores))

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
        for i in range(len(KeypointList)):
            kp1: int = KeypointList[i][0].value
            kp2: int = KeypointList[i][1].value
            s1: float = scores[kp1]
            s2: float = scores[kp2]
            score: float = min(s1, s2)
            alpha: float = 0.0
            if score > threshold:
                alpha = (score - threshold) / (1 - threshold) * a
            colors[i*2] = [r, g, b, alpha]
            colors[i*2+1] = [r, g, b, alpha]
        return colors


PoseList = list[Pose]

