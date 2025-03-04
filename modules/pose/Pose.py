
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import time

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

class Pose():
    def __init__(self) -> None:

        self.frameCallbacks: set = set()

        base_options = python.BaseOptions(
            model_asset_path='C:/Developer/DepthAI/DepthPose/models/pose_landmarker_heavy.task',
            delegate= "GPU"
        )
        options = vision.PoseLandmarkerOptions(
            running_mode= VisionTaskRunningMode.VIDEO,
            # result_callback=self.callback,
            num_poses= 5,
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_in: np.ndarray) -> None:
        # img = mp.Image.f
        # resize image to half size
        resized_image = cv2.resize(image_in, (int(image_in.shape[1] / 2), int(image_in.shape[0] / 2)))

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image)
        # detection_result = self.detector.detect(image)
        # get the time in miliseconds
        t: int = round(time.time() * 1000)
        detection_result = self.detector.detect_for_video(image, t)
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        for c in self.frameCallbacks:
            c(annotated_image)


    def callback(self, results) -> None:
        print(results)
        # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        pass

    # CALLBACKS
    def addFrameCallback(self, callback) -> None:
        self.frameCallbacks.add(callback)

    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks = set()