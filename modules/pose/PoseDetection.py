# Standard library imports
import time
from threading import Thread, Lock
import os
from typing import Optional

# Third-party imports
import cv2
import numpy as np
import onnxruntime as ort

# Local application imports
from modules.pose.PoseDefinitions import *
from modules.person.Person import Person

class Detection(Thread):
    _model_load_lock: Lock = Lock()
    _model_loaded: bool =  False
    _model_type: ModelType = ModelType.NONE
    _model_path: str
    _moodel_size: int
    _model_session: ort.InferenceSession

    def __init__(self, path: str, model_type:ModelType) -> None:
        super().__init__()

        if Detection._model_type is ModelType.NONE:
            Detection._model_type = model_type
            Detection._model_path = path
            Detection._moodel_size = ModelInputSize[model_type.value]
        else:
            if Detection._model_type is not model_type:
                print('Pose Detection WARNING: ModelType is different from the first instance')

        if Detection._model_type is ModelType.NONE:
            print('Pose Detection WARNING: ModelType is NONE')

        self._running: bool = False
        self._input_mutex: Lock = Lock()
        self._input_person: Person | None = None
        self._callbacks: set = set()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self.load_model_once()

        self._running = True
        while self._running:
            person: Person | None = self.get_person()
            if person is not None:
                image: np.ndarray | None = person.pose_image
                if image is not None:
                    poses: PoseList = self.RunSession(Detection._model_session, Detection._moodel_size, image)
                    if len(poses) > 0:
                        person.pose = poses[0]
                self.callback(person)
            time.sleep(0.01)

    def load_model_once(self) -> None:
        with Detection._model_load_lock:
            if not Detection._model_loaded:
                Detection._model_session, Detection._moodel_size = self.LoadSession(self._model_type, Detection._model_path)
                Detection._model_loaded = True

    # GETTERS AND SETTERS
    def get_person(self) -> Person | None:
        with self._input_mutex:
            return_detection: Person | None = self._input_person
            self._input_person = None
            return return_detection

    def person_input(self, detection: Person) -> None:
        with self._input_mutex:
            self._input_person = detection

    def get_frame_size(self) -> int:
        return Detection._moodel_size

    # CALLBACKS
    def callback(self, value: Person) -> None:
        for c in self._callbacks:
            c(value)

    def addMessageCallback(self, callback) -> None:
        self._callbacks.add(callback)

    def clearMessageCallbacks(self) -> None:
        self._callbacks = set()

    # STATIC METHODS
    @staticmethod
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

    @staticmethod
    def RunSession(onnx_session: ort.InferenceSession, input_size: int, image: np.ndarray) -> PoseList:
        height, width = image.shape[:2]
        if height != input_size or width != input_size:
            image = Detection.resize_with_pad(image, input_size, input_size)
        input_image: np.ndarray = image.reshape(-1, input_size, input_size, 3)
        input_image = input_image.astype('int32')

        input_name  = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        outputs     = onnx_session.run([output_name], {input_name: input_image})

        keypoints_with_scores: np.ndarray = outputs[0]
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        if Detection._model_type is ModelType.LIGHTNING or Detection._model_type is ModelType.THUNDER:
            keypoints: np.ndarray = keypoints_with_scores[:, :2]
            keypoints = np.flip(keypoints, axis=1)
            scores: np.ndarray = keypoints_with_scores[:, 2]
            pose = Pose(keypoints, scores)
            return [pose]
        else: # ModelType.MULTI
            poses: PoseList = []
            for kps in keypoints_with_scores:

                mean_score = kps[55]
                if mean_score < 0.1:
                    continue

                # make a nd.array of 17 by 3 for the keypoints
                keypoints: np.ndarray = np.zeros((17, 2), dtype=np.float32)
                scores: np.ndarray = np.zeros((17), dtype=np.float32)
                for index in range(17):
                    x: float = kps[(index * 3) + 1]
                    y: float = kps[(index * 3) + 0]
                    s: float = kps[(index * 3) + 2]

                    keypoints[index] = [x, y]
                    scores[index] = s

                ymin: float = kps[51]
                xmin: float = kps[52]
                ymax: float = kps[53]
                xmax: float = kps[54]

                pose = Pose(keypoints, scores)
                poses.append(pose)
            return poses

    @staticmethod
    def resize_with_pad(image, target_width, target_height, padding_color=(0, 0, 0)) -> np.ndarray:
        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio: float = original_width / original_height

        # Determine the new dimensions while maintaining the aspect ratio
        if target_width / target_height > aspect_ratio:
            new_height: int = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width: int = target_width
            new_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image: np.ndarray = cv2.resize(image, (new_width, new_height))

        # Create a new image with the target dimensions and the padding color
        padded_image: np.ndarray = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

        # Calculate the position to place the resized image
        x_offset: int = (target_width - new_width) // 2
        y_offset: int = (target_height - new_height) // 2

        # Place the resized image on the padded image
        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

        return padded_image