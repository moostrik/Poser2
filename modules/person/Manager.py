import cv2
import numpy as np
from threading import Thread, Lock
from time import sleep

from modules.person.Person import Person, PersonCallback
from modules.cam.DepthAi.Definitions import Tracklet, Rect
from modules.pose.PoseDetection import PoseDetection, ModelType, ModelTypeNames
from modules.utils.pool import ObjectPool

class Manager(Thread):
    def __init__(self, max_persons: int, model_path: str, model_type: ModelType) -> None:
        super().__init__()
        self.input_mutex: Lock = Lock()
        self.running: bool = False

        self.pose_detector_pool: ObjectPool
        self.pose_detector_frame_size: int = 256
        self.pose_detector_active: bool = False
        if model_type == ModelType.NONE:
            print('Running without pose detection')
        else:
            self.pose_detector_pool = ObjectPool(PoseDetection, max_persons, model_path, model_type)
            self.pose_detector_active = True
            print('Running', max_persons, 'pose detections with model', ModelTypeNames[model_type.value])

        self.max_persons: int = max_persons
        self.input_frames: dict[int, np.ndarray] = {}
        self.input_persons: dict[str, Person] = {}

        self.callbacks: set[PersonCallback] = set()
        self.active_detectors: dict[str, PoseDetection] = {}

    def run(self) -> None:
        self.running = True

        if self.pose_detector_active:
            detectors: list[PoseDetection] = self.pose_detector_pool.get_all_objects()
            for detector in detectors:
                detector.addMessageCallback(self.callback)
                detector.start()
                self.pose_detector_frame_size  = detector.get_frame_size()

        while self.running:
            persons: dict[str, Person] = self.get_input_persons()
            for key in persons.keys():
                self.add_player_id(persons[key])
                self.add_cropped_image(persons[key])

                if self.pose_detector_active:
                    self.add_pose(persons[key])
                else:
                    for c in self.callbacks:
                        c(persons[key])

            sleep(0.01)
            pass

    def add_cropped_image(self, person: Person) -> None:
        if person.image is not None:
            return
        roi: Rect = person.tracklet.roi
        image: np.ndarray = self.get_image(person.cam_id)
        person.image = self.get_cropped_image(image, roi, self.pose_detector_frame_size)

    def add_player_id(self, person: Person) -> None:
        person.player_id = Person._id_counter

    def add_pose(self, person: Person) -> None:
        if self.pose_detector_pool is None:
            return
        if person.image is None:
            return
        if person.pose is not None:
            return

        detector: PoseDetection
        key: str = person.id
        if self.active_detectors.get(key) is None:
            detector: PoseDetection = self.pose_detector_pool.acquire()
            self.active_detectors[key] = detector
        else:
            detector = self.active_detectors[key]

        detector.set_detection(person)

    def stop(self) -> None:
        self.running = False

    # INPUTS
    def set_image(self, id: int, image: np.ndarray) -> None :
        with self.input_mutex:
            self.input_frames[id] = image
    def get_image(self, id: int) -> np.ndarray:
        with self.input_mutex:
            if self.input_frames.get(id) is None:
                print('No image with id', id)
                return np.zeros((self.pose_detector_frame_size, self.pose_detector_frame_size, 3), np.uint8)
            return self.input_frames[id]

    def get_input_persons(self) -> dict[str, Person]:
        with self.input_mutex:
            detections: dict[str, Person] =  self.input_persons.copy()
            self.input_persons.clear()
            return detections

    def add_tracklet(self, id: int, tracklet: Tracklet) -> None :
        if tracklet.status != Tracklet.TrackingStatus.TRACKED:
            return
        unique_id: str = Person.create_unique_id(id, tracklet.id)
        with self.input_mutex:
            self.input_persons[unique_id] = Person(id, tracklet)

    # CALLBACKS
    def callback(self, detection: Person) -> None:
        for c in self.callbacks:
            c(detection)

    def addCallback(self, callback: PersonCallback) -> None:
        self.callbacks.add(callback)
    def discardCallback(self, callback: PersonCallback) -> None:
        self.callbacks.discard(callback)
    def clearCallbacks(self) -> None:
        self.callbacks.clear()

    # STATIC METHODS
    @staticmethod
    def get_cropped_image(image: np.ndarray, roi: Rect, size: int) -> np.ndarray:
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