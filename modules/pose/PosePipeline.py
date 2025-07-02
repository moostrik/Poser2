# Standard library imports
from threading import Lock
from typing import Optional

# Third-party imports

# Local application imports
from modules.Settings import Settings
from modules.person.Person import Person, PersonCallback
from modules.pose.PoseDefinitions import ModelTypeNames
from modules.pose.PoseDetection import Detection
from modules.pose.PoseAngleCalculator import PoseAngleCalculator

from modules.utils.HotReloadMethods import HotReloadMethods


class PosePipeline:
    def __init__(self, settings: Settings) -> None:

        # Pose detection components
        self.pose_detectors: dict[int, Detection] = {}
        self.pose_detector_frame_size: int = 256
        self.max_detectors: int = settings.pose_num
        for i in range(self.max_detectors):
            self.pose_detectors[i] = Detection(settings.path_model, settings.pose_model_type)
        print('Pose Detection:', self.max_detectors, 'instances of model', ModelTypeNames[settings.pose_model_type.value])

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

        self.joint_angles.add_person_callback(self._person_output_callback)
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

    # External Input
    def set_person(self, person: Person) -> None:
        detector: Optional[Detection] = self.pose_detectors.get(person.id)

        if detector:
            detector.person_input(person)

    # External Output Callbacks
    def add_person_callback(self, callback: PersonCallback) -> None:
        """Add callback for processed persons"""
        if self.running:
            print('Pipeline is running, cannot add callback')
            return
        self.person_output_callbacks.add(callback)

    def _person_output_callback(self, person: Person) -> None:
        """Handle processed person"""
        with self.callback_lock:
            for callback in self.person_output_callbacks:
                callback(person)