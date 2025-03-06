
import cv2
import cv2 as cv
import numpy as np
import time
import onnxruntime as ort
from enum import Enum
from threading import Thread, Lock
import copy

class ModelType(Enum):
    NONE = 0
    THUNDER = 1
    LIGHTNING = 2
    MULTI = 3

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

class Pose():
    point_list: list = []

PoseList = list[Pose]

class PoseMessage():
    def __init__(self, pose_list: PoseList, pose_image: np.ndarray) -> None:
        self.pose_list: PoseList = pose_list
        self.image: np.ndarray = pose_image

def LoadSession(model_type: ModelType, model_path: str) -> tuple[ort.InferenceSession, int]:
    # check for trailing slash
    if model_path[-1] != '/':
        model_path += '/'
    path: str = model_path + ModelFileNames[model_type.value]
    print('Loading model', ModelTypeNames[model_type.value], 'from path', path)
    onnx_session = ort.InferenceSession(
        path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ],
    )
    input_size: int = ModelInputSize[model_type.value]
    return onnx_session, input_size

def RunSession(onnx_session: ort.InferenceSession, input_size: int, image: np.ndarray):
    image_width, image_height = image.shape[1], image.shape[0]

    input_image: np.ndarray = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = input_image.astype('int32')   # int32へキャスト

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)
        # print(scores)

    return keypoints, scores

    # return keypoints_with_scores

def DrawPose(image, keypoints, scores) -> np.ndarray:
    debug_image = copy.deepcopy(image)
    keypoint_score_th = 0.5
    index01, index02 = 0, 1
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右目
    index01, index02 = 0, 2
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左目 → 左耳
    index01, index02 = 1, 3
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右目 → 右耳
    index01, index02 = 2, 4
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 左肩
    index01, index02 = 0, 5
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右肩
    index01, index02 = 0, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 右肩
    index01, index02 = 5, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左肘
    index01, index02 = 5, 7
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肘 → 左手首
    index01, index02 = 7, 9
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右肘
    index01, index02 = 6, 8
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肘 → 右手首
    index01, index02 = 8, 10
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 右股関節
    index01, index02 = 11, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左股関節
    index01, index02 = 5, 11
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 左ひざ
    index01, index02 = 11, 13
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左ひざ → 左足首
    index01, index02 = 13, 15
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右股関節
    index01, index02 = 6, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右股関節 → 右ひざ
    index01, index02 = 12, 14
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右ひざ → 右足首
    index01, index02 = 14, 16
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)

    # Circle：各点
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    return debug_image

class PoseDetection(Thread):
    def __init__(self, model_type:ModelType) -> None:
        super().__init__()
        self.path = 'C:/Developer/DepthAI/DepthPose/models/'
        self.modelType: ModelType = model_type

        self._input_mutex: Lock = Lock()
        self._image: np.ndarray | None = None
        self._image_consumed: bool = True

        self.running: bool = False
        self.callbacks: set = set()

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:

        session: ort.InferenceSession
        input_size: int
        session, input_size = LoadSession(self.modelType, self.path)

        self.running = True
        while self.running:
            if self.get_image is not None:
                image: np.ndarray | None = self.get_image()
                if image is not None:
                    keypoints_list, scores_list = RunSession(session, input_size, image)
                    # result: list = RunSession(session, input_size, image)
                    pose_image: np.ndarray = DrawPose(image, keypoints_list, scores_list)
                    pose_message = PoseMessage(keypoints_list, pose_image)
                    self.callback(pose_message)

            time.sleep(0.1)

    # IMAGE INPUTS
    def set_image(self, image: np.ndarray | None) -> None:
        with self._input_mutex:
            self._image = image
            self._image_consumed = False
    def get_image(self, get_cumsumed_image = False) -> np.ndarray | None:
        with self._input_mutex:
            if not self._image_consumed:
                self._image_consumed = True
                return self._image

            if self._image_consumed and get_cumsumed_image:
                return self._image

            return None

    # CALLBACKS
    def callback(self, value: PoseMessage) -> None:
        for c in self.callbacks:
            c(value)

    def addMessageCallback(self, callback) -> None:
        self.callbacks.add(callback)

    def clearMessageCallbacks(self) -> None:
        self.callbacks = set()
