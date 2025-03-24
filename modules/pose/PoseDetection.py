# based on
# https://www.tensorflow.org/hub/tutorials/movenet
# https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main
# Lightning for low latency, Thunder for high accuracy


import cv2
import numpy as np
import time
import onnxruntime as ort
from enum import Enum
from threading import Thread, Lock
import copy
import os

from modules.pose.PoseDefinitions import *
from modules.person.Person import Person

def LoadSession(model_type: ModelType, model_path: str) -> tuple[ort.InferenceSession, int]:
    path: str = os.path.join(model_path, ModelFileNames[model_type.value])
    onnx_session = ort.InferenceSession(
        path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ],
    )
    input_size: int = ModelInputSize[model_type.value]
    return onnx_session, input_size

def RunSession(onnx_session: ort.InferenceSession, input_size: int, image: np.ndarray) -> PoseList:
    input_image: np.ndarray = resize_with_pad(image, input_size, input_size)
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = input_image.astype('int32')

    input_name  = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs     = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores: np.ndarray = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # TODO: Modify for use with MultiPose

    box: PoseBox = PoseBox(0, 0, 1.0, 1.0, 1.0)
    pose = Pose(keypoints_with_scores, box)
    return [pose]

def DrawPose(image, poses:PoseList) -> np.ndarray:
    image_width, image_height = image.shape[1], image.shape[0]

    for pose in poses:

        keypoints = []
        scores = []
        for index in range(17):
            keypoint_x = int(image_width * pose.keypoints[index][1])
            keypoint_y = int(image_height * pose.keypoints[index][0])
            score = pose.keypoints[index][2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)
        # print(scores)


        debug_image = copy.deepcopy(image)
        keypoint_score_th = 0.5
        index01, index02 = 0, 1
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 右目
        index01, index02 = 0, 2
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左目 → 左耳
        index01, index02 = 1, 3
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右目 → 右耳
        index01, index02 = 2, 4
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 左肩
        index01, index02 = 0, 5
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 右肩
        index01, index02 = 0, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 右肩
        index01, index02 = 5, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 左肘
        index01, index02 = 5, 7
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肘 → 左手首
        index01, index02 = 7, 9
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右肩 → 右肘
        index01, index02 = 6, 8
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右肘 → 右手首
        index01, index02 = 8, 10
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左股関節 → 右股関節
        index01, index02 = 11, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 左股関節
        index01, index02 = 5, 11
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左股関節 → 左ひざ
        index01, index02 = 11, 13
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左ひざ → 左足首
        index01, index02 = 13, 15
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右肩 → 右股関節
        index01, index02 = 6, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右股関節 → 右ひざ
        index01, index02 = 12, 14
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右ひざ → 右足首
        index01, index02 = 14, 16
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)

        # Circle：各点
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv2.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
                cv2.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    return debug_image

def resize_with_pad(image, target_width, target_height, padding_color=(0, 0, 0)):
    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new dimensions while maintaining the aspect ratio
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the target dimensions and the padding color
    padded_image = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

    # Calculate the position to place the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image on the padded image
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return padded_image


class PoseDetection(Thread):
    onnx_session: ort.InferenceSession | None = None
    onnx_size: int = 256

    def __init__(self, path: str, model_type:ModelType) -> None:
        super().__init__()
        self.path: str = path
        self.modelType: ModelType = model_type
        if model_type is not ModelType.THUNDER and model_type is not ModelType.LIGHTNING:
            print('PoseDetection ModelType must be THUNDER or LIGHTNING, defaulting to THUNDER', model_type)
            self.modelType = ModelType.THUNDER

        self._input_mutex: Lock = Lock()
        self._image: np.ndarray | None = None
        self._image_consumed: bool = True

        self._detection: Person | None = None

        self._running: bool = False
        self._callbacks: set = set()
        self._occupied: bool = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        if PoseDetection.onnx_session is None:
            PoseDetection.onnx_session, PoseDetection.onnx_size = LoadSession(self.modelType, self.path)

        self._running = True
        while self._running:
            detection: Person | None = self.get_detection()
            if detection is not None:
                image: np.ndarray | None = detection.pose_image
                if image is not None:
                    Poses: PoseList = RunSession(PoseDetection.onnx_session, PoseDetection.onnx_size, image)
                    detection.pose = Poses
                    self.callback(detection)
            time.sleep(0.01)

    # # IMAGE INPUTS
    # def set_image(self, ID: int, image: np.ndarray | None) -> None:
    #     with self._input_mutex:
    #         self._image = image
    #         self._occupied = True

    # def get_image(self) -> np.ndarray | None:
    #     with self._input_mutex:
    #         return_image: np.ndarray | None = self._image
    #         self._image = None
    #         return return_image

    # def is_occupied(self) -> bool:
    #     with self._input_mutex:
    #         return not self._occupied

    def get_detection(self) -> Person | None:
        with self._input_mutex:
            return_detection: Person | None = self._detection
            self._detection = None
            return return_detection

    def set_detection(self, detection: Person) -> None:
        with self._input_mutex:
            self._detection = detection

    def get_frame_size(self) -> int:
        return PoseDetection.onnx_size

    # CALLBACKS
    def callback(self, value: Person) -> None:
        for c in self._callbacks:
            c(value)

    def addMessageCallback(self, callback) -> None:
        self._callbacks.add(callback)

    def clearMessageCallbacks(self) -> None:
        self._callbacks = set()
