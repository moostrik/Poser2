from __future__ import annotations

import cv2
import numpy as np
import queue
from threading import Thread, Lock
from time import time, sleep
from typing import Any, Callable, Dict, List, Optional, Type

from modules.Settings import Settings
from modules.person.Person import Person, PersonCallback
from modules.pose.Definitions import ModelTypeNames
from modules.pose.Detection import Detection


class Pipeline(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()

        self.running: bool = False

        # Input queue for persons to process
        self.person_queue: queue.Queue[Person] = queue.Queue()
        self.max_persons: int = settings.pose_num

        # Pose detection components
        self.pose_detectors: dict[int, Detection] = {}
        self.pose_detector_frame_size: int = 256

        for i in range(self.max_persons):
            self.pose_detectors[i] = Detection(settings.path_model, settings.pose_model_type)
        print('Pose Detection:', self.max_persons, 'instances of model', ModelTypeNames[settings.pose_model_type.value])

        # Callbacks
        self.callback_lock = Lock()
        self.person_output_callbacks: set[PersonCallback] = set()

    def start(self) -> None:
        if self.running:
            return

        # Start detectors
        for detector in self.pose_detectors.values():
            detector.addMessageCallback(self._pose_detection_callback)
            detector.start()
            self.pose_detector_frame_size = detector.get_frame_size()

        self.running = True
        super().start()

    def stop(self) -> None:
        self.running = False
        with self.callback_lock:
            self.person_output_callbacks.clear()

        # Stop detectors
        for detector in self.pose_detectors.values():
            detector.stop()

    def run(self) -> None:
        while self.running:
            try:
                person: Optional[Person] = self.person_queue.get(block=True, timeout=0.01)
                if person is not None:
                    self._pose_detection(person)
                    self.person_queue.task_done()
            except queue.Empty:
                pass

    def _pose_detection(self, person: Person) -> None:
            detector: Optional[Detection] = self.pose_detectors.get(person.id)
            if detector:
                detector.set_detection(person)

    def _pose_detection_callback(self, person: Person) -> None:
        """Callback for pose detection completion"""
        if person.pose is not None:
            self._pose_filter(person)
        else:
            print(f"Pose detection failed for person {person.id}")

    def _pose_filter(self, person: Person) -> None:
        self._person_output_callback(person)

    # External Input
    def person_input(self, person: Person) -> None:
        """Add a person to the processing queue"""
        self.person_queue.put(person)

    # External Output Calbacks
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