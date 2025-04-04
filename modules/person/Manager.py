import cv2
import numpy as np
from threading import Thread, Lock
from time import time, sleep

from modules.Settings import Settings
from modules.person.Person import Person, PersonDict, PersonCallback, CamTracklet
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.person.pose.PoseDetection import PoseDetection, ModelType, ModelTypeNames
from modules.person.CircularCoordinates import CircularCoordinates

CamTrackletDict = dict[str, CamTracklet]

class IdPool:
    def __init__(self, max_size: int) -> None:
        self._available = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise Exception("No more IDs available")
            return self._available.pop()

    def release(self, obj: int) -> None:
        with self._lock:
            self._available.add(obj)

    def size(self) -> int:
        return len(self._available)

class Manager(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.input_mutex: Lock = Lock()
        self.running: bool = False
        self.max_persons: int = settings.num_players

        self.pose_detectors: dict[int, PoseDetection] = {}
        self.pose_detector_frame_size: int = 256
        if not settings.pose:
            print('Pose Detection: Disabled')
        else:
            for i in range(self.max_persons):
                self.pose_detectors[i] = PoseDetection(settings.model_path, settings.model_type)
            print('Pose Detection:', self.max_persons, 'six instances of model', ModelTypeNames[settings.model_type.value])

        self.input_frames: dict[int, np.ndarray] = {}
        self.input_tracklets: CamTrackletDict = {}

        self.person_id_pool: IdPool = IdPool(self.max_persons)
        self.persons: PersonDict = {}

        self.circular_coordinates: CircularCoordinates = CircularCoordinates(settings.num_cams)

        self.callbacks: set[PersonCallback] = set()

    def stop(self) -> None:
        self.running = False
        self.callbacks.clear()

    def run(self) -> None:
        self.running = True

        for detector in self.pose_detectors.values():
            detector.addMessageCallback(self.callback)
            detector.start()
            self.pose_detector_frame_size = detector.get_frame_size()

        while self.running:
            self.update_persons()
            sleep(0.01)
            pass

    def update_persons(self) -> None:
        tracklets: CamTrackletDict = self.get_tracklets()
        # print number of tracklets

        for key in tracklets.keys():
            cam_id: int = tracklets[key].cam_id
            tracklet: Tracklet = tracklets[key].tracklet
            if tracklet.status == Tracklet.TrackingStatus.NEW or tracklet.status == Tracklet.TrackingStatus.TRACKED:

                person: Person = Person(-1, cam_id, tracklet)
                self.circular_coordinates.add_angle_position(person)
                person_found: Person | None = self.circular_coordinates.find(person, self.persons)

                if person_found is not None:
                    person.id = person_found.id
                    person.start_time = person_found.start_time
                    person.last_time = time()
                    self.persons[person.id] = person
                    continue

                try:
                    person_id: int = self.person_id_pool.acquire()
                except:
                    print('No more person ids available')
                    continue

                person.id = person_id
                # print('New person id:', person.id)
                self.persons[person_id] = person

        for key in self.persons.keys():
            person = self.persons[key]
            if person.last_time < time() - 1.0:
                self.person_id_pool.release(person.id)
                person.active = False

            self.add_cropped_image(person)
            self.add_pose(person) # also handles callback

    def add_cropped_image(self, person: Person) -> None:
        if person.pose_image is not None:
            return
        image: np.ndarray = self.get_image(person.cam_id)
        h, w = image.shape[:2]
        roi: Rect = self.get_crop_rect(w, h, person.tracklet.roi)
        person.pose_rect = roi
        person.pose_image = self.get_cropped_image(image, roi, self.pose_detector_frame_size)

    def add_pose(self, person: Person) -> None:
        if person.pose_image is None:
            print('No image for person', person.id)
            return
        if person.pose is not None:
            return

        detector: PoseDetection | None = self.pose_detectors.get(person.id)
        if detector is not None:
            detector.set_detection(person)
        else:
            self.callback(person)


    # INPUTS
    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None :
        with self.input_mutex:
            self.input_frames[id] = image
    def get_image(self, id: int) -> np.ndarray:
        with self.input_mutex:
            if self.input_frames.get(id) is None:
                # print('No image with id', id)
                return np.zeros((self.pose_detector_frame_size, self.pose_detector_frame_size, 3), np.uint8)
            return self.input_frames[id]

    def get_tracklets(self) -> CamTrackletDict:
        with self.input_mutex:
            tracklets: CamTrackletDict =  self.input_tracklets.copy()
            self.input_tracklets.clear()
            return tracklets
    def add_tracklet(self, id: int, tracklet: Tracklet) -> None :
        unique_id: str = Person.create_cam_id(id, tracklet.id)
        with self.input_mutex:
            self.input_tracklets[unique_id] = CamTracklet(id, tracklet)

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
    def get_crop_rect(image_width: int, image_height: int, roi: Rect, expansion = 0.0) -> Rect:
       # Calculate the original ROI coordinates
        img_x = int(roi.x * image_width)
        img_y = int(roi.y * image_height)
        img_w = int(roi.width * image_width)
        img_h = int(roi.height * image_height)

        # Determine the size of the square cutout based on the longest side of the ROI
        img_wh: int = max(img_w, img_h)
        img_wh += int(img_wh * expansion)

        # Calculate the new coordinates to center the square cutout around the original ROI
        crop_center_x: int = img_x + img_w // 2
        crop_center_y: int = img_y + img_h // 2
        crop_x: int = crop_center_x - img_wh // 2
        crop_y: int = crop_center_y - img_wh // 2
        crop_w: int = img_wh
        crop_h: int = img_wh

        # convert back to normalized coordinates
        norm_x: float = crop_x / image_width
        norm_y: float = crop_y / image_height
        norm_w: float = crop_w / image_width
        norm_h: float = crop_h / image_height

        return Rect(norm_x, norm_y, norm_w, norm_h)

    @staticmethod
    def get_cropped_image(image: np.ndarray, roi: Rect, size: int) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        image_channels = image.shape[2] if len(image.shape) > 2 else 1

        # Calculate the original ROI coordinates
        x: int = int(roi.x * image_width)
        y: int = int(roi.y * image_height)
        w: int = int(roi.width * image_width)
        h: int = int(roi.height * image_height)

        # Extract the roi without padding
        img_x: int = max(0, x)
        img_y: int = max(0, y)
        img_w: int = min(w, image_width - x)
        img_h: int = min(h, image_height - y)

        crop: np.ndarray = image[img_y:img_y + img_h, img_x:img_x + img_w]

        if image_channels == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        # Apply padding if the roi is outside the image bounds
        left_padding: int = -min(0, x)
        top_padding: int = -min(0, y)
        right_padding: int = max(0, x + w - image_width)
        bottom_padding: int = max(0, y + h - image_height)

        if left_padding + right_padding + top_padding + bottom_padding > 0:
            crop = cv2.copyMakeBorder(crop, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize the cutout to the desired size
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

