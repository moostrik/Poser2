from enum import Enum
import numpy as np

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

PoseIndices: list[list[Keypoints]] = [
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
    def __init__(self, keypoints: np.ndarray, box: PoseBox) -> None:
        self.keypoints: np.ndarray = keypoints
        self.box: PoseBox = box
        # keypoints: np.ndarray = np.zeros((NUM_KEYPOINTS, NUM_KEYPOINT_VALUES), dtype=np.float32)
        # box: PoseBox = PoseBox(0.0, 0.0, 1.0, 1.0, 0.0)

PoseList = list[Pose]
