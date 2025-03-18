import cv2
import numpy as np
from threading import Thread, Lock
from typing import Callable
from time import sleep

from modules.cam.DepthAi.Definitions import Tracklet, Rect
from modules.pose.PoseDetection import PoseDetection, ModelType, PoseMessage, PoseList

from modules.utils.pool import ObjectPool



class Detection():
    _id_counter = 0

    def __init__(self, cam_id: int, tracklet: Tracklet) -> None:
        self.id: str =                  self.create_unique_id(cam_id, tracklet.id)
        self.cam_id: int =              cam_id
        self.tracklet: Tracklet =       tracklet
        self.image: np.ndarray | None = None
        self.pose: PoseList | None =    None

    @staticmethod
    def create_unique_id(cam_id: int, tracklet_id: int) -> str:
        return f"{cam_id}_{tracklet_id}"

DetectionCallback = Callable[[Detection], None]



class Manager(Thread):
    def __init__(self, max_persons: int, model_path:str, model_type: ModelType) -> None:
        super().__init__()
        self.input_mutex: Lock = Lock()
        self.running: bool = False

        self.max_persons: int = max_persons
        self.input_frames: dict[int, np.ndarray] = {}
        self.input_detections: dict[str, Detection] = {}

        self.detector_pool = ObjectPool(PoseDetection, max_persons, model_path, model_type)

        self.callbacks: set[DetectionCallback] = set()


    def run(self) -> None:
        self.running = True

        while self.running:
            detections: dict[str, Detection] = self.get_input_detections()
            for key in detections.keys():
                if detections[key].image is None:
                    roi = detections[key].tracklet.roi
                    cam_id: int = detections[key].cam_id
                    image: np.ndarray = self.get_image(cam_id)
                    detections[key].image = self.get_image_cutout(image, roi, 256)

                    for c in self.callbacks:
                        c(detections[key])




            sleep(0.01)
            pass
            # check if frame with the same camid is not being processed
            # if not, process the frame


    def process_frame(self, id: int) -> bool:
        return True


    def stop(self) -> None:
        self.running = False

    def set_image(self, id: int, image: np.ndarray) -> None :
        with self.input_mutex:
            self.input_frames[id] = image

    def get_image(self, id: int) -> np.ndarray:
        with self.input_mutex:
            return self.input_frames[id]


    def get_input_detections(self) -> dict[str, Detection]:
        with self.input_mutex:
            detections: dict[str, Detection] =  self.input_detections.copy()
            self.input_detections.clear()
            return detections

    def add_tracklet(self, id: int, tracklet: Tracklet) -> None :
        print(tracklet.status)
        if tracklet.status != Tracklet.TrackingStatus.TRACKED:
            return
        unique_id: str = Detection.create_unique_id(id, tracklet.id)
        with self.input_mutex:
            self.input_detections[unique_id] = Detection(id, tracklet)


    def addCallback(self, callback: DetectionCallback) -> None:
        self.callbacks.add(callback)
    def discardCallback(self, callback: DetectionCallback) -> None:
        self.callbacks.discard(callback)
    def clearCallbacks(self) -> None:
        self.callbacks.clear()

    @staticmethod
    def get_image_cutout(image: np.ndarray, roi: Rect, size: int) -> np.ndarray:
        image_height, image_width = image.shape[:2]

        # Calculate the original ROI coordinates
        x = int(roi.x * image_width)
        y = int(roi.y * image_height)
        w = int(roi.width * image_width)
        h = int(roi.height * image_height)

        # Determine the size of the square cutout based on the longest side of the ROI
        side_length = max(w, h)

        # Calculate the new coordinates to center the square cutout around the original ROI
        x_center = x + w // 2
        y_center = y + h // 2
        x_new = x_center - side_length // 2
        y_new = y_center - side_length // 2

        # Calculate padding if the cutout goes outside the image boundaries
        top_padding = max(0, -y_new)
        left_padding = max(0, -x_new)
        bottom_padding = max(0, y_new + side_length - image_height)
        right_padding = max(0, x_new + side_length - image_width)

        # Add padding to the image if necessary
        if top_padding > 0 or left_padding > 0 or bottom_padding > 0 or right_padding > 0:
            image = cv2.copyMakeBorder(
                image,
                top_padding,
                bottom_padding,
                left_padding,
                right_padding,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # You can change the padding color if needed
            )

        # Recalculate the new coordinates after padding
        x_new = max(0, x_new)
        y_new = max(0, y_new)

        # Extract the square cutout
        cutout: np.ndarray = image[y_new:y_new + side_length, x_new:x_new + side_length]

        # Resize the cutout to the desired size
        return cv2.resize(cutout, (size, size), interpolation=cv2.INTER_AREA)