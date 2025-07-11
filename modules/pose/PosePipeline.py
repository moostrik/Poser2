# Standard library imports
from threading import Lock
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from modules.cam.depthcam.Definitions import FrameType
from modules.person.Person import Person, PersonCallback
from modules.pose.PoseDefinitions import ModelTypeNames
from modules.pose.PoseDetection import Detection
from modules.pose.PoseImageProcessor import PoseImageProcessor
from modules.pose.PoseAngleCalculator import PoseAngleCalculator
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


class PosePipeline:
    def __init__(self, settings: Settings) -> None:

        self.pose_active: bool = settings.pose_active
        self.pose_detector_frame_size: int = 256
        self.pose_crop_expansion: float = settings.pose_crop_expansion
        self.max_detectors: int = settings.max_players
        self.pose_detectors: dict[int, Detection] = {}

        self.image_processor: PoseImageProcessor = PoseImageProcessor(
            crop_expansion=self.pose_crop_expansion,
            output_size=self.pose_detector_frame_size
        )

        if self.pose_active:
            for i in range(self.max_detectors):
                self.pose_detectors[i] = Detection(settings.path_model, settings.pose_model_type)
            print('Pose Detection:', self.max_detectors, 'instances of model', ModelTypeNames[settings.pose_model_type.value])
        else:
            print('Pose Detection: Disabled')

        self.input_mutex: Lock = Lock()
        self.input_frames: dict[int, np.ndarray] = {}

        # Pose angles calculator
        self.joint_angles: PoseAngleCalculator = PoseAngleCalculator()

        # Callbacks
        self.callback_lock = Lock()
        self.person_output_callbacks: set[PersonCallback] = set()
        self.running: bool = False

        hot_reloader = HotReloadMethods(self.__class__)

    def start(self) -> None:
        if self.running:
            return

        self.joint_angles.add_person_callback(self._notify_callback)
        self.joint_angles.start()

        # Start detectors
        for detector in self.pose_detectors.values():
            detector.addMessageCallback(self.joint_angles.person_input)
            detector.start()
            self.pose_detector_frame_size = detector.get_frame_size()

        self.running = True

    def stop(self) -> None:
        self.running = False
        with self.callback_lock:
            self.person_output_callbacks.clear()

        for detector in self.pose_detectors.values():
            detector.stop()

        self.joint_angles.stop()

     # INPUTS
    def add_person(self, person: Person) -> None:

        if person.is_active and person.pose_image is None:
            image: Optional[np.ndarray] = self._get_image(person.cam_id)
            if image is not None:
                pose_image, crop_rect = self.image_processor.process_person_image(person, image)
                person.pose_image = pose_image
                person.pose_crop_rect = crop_rect

        if not self.pose_active:
            self._notify_callback(person)
            return

        detector: Optional[Detection] = self.pose_detectors.get(person.id)
        if detector:
            detector.person_input(person)

    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None :
        if frame_type != FrameType.VIDEO:
            return
        with self.input_mutex:
            self.input_frames[id] = image
    def _get_image(self, id: int) -> Optional[np.ndarray]:
        with self.input_mutex:
            if not self.input_frames.get(id) is None:
                return self.input_frames[id]
            return None

    # External Output Callbacks
    def add_person_callback(self, callback: PersonCallback) -> None:
        """Add callback for processed persons"""
        if self.running:
            print('Pipeline is running, cannot add callback')
            return
        self.person_output_callbacks.add(callback)

    def _notify_callback(self, person: Person) -> None:
        """Handle processed person"""
        with self.callback_lock:
            for callback in self.person_output_callbacks:
                callback(person)