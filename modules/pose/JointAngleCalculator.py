# Standard library imports
import queue
from threading import Event, Lock, Thread
from typing import Optional, Dict, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# Local application imports
from modules.pose.Definitions import *
from modules.person.Person import Person, PersonCallback

from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods


class JointAngleCalculator(Thread):
    def __init__(self) -> None:
        """Initialize the JointAngles calculator."""
        super().__init__()
        self._stop_event = Event()

        # Input
        self.person_input_queue: queue.Queue[Person] = queue.Queue()

        # Parameters
        self.confidence_threshold: float = 0.3

        # Callbacks
        self.callback_lock = Lock()
        self.person_output_callbacks: set[PersonCallback] = set()

        self.hot_reload = HotReloadStaticMethods(self.__class__, True)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()
        with self.callback_lock:
            self.person_output_callbacks.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                person: Optional[Person] = self.person_input_queue.get(block=True, timeout=0.01)
                if person is not None:
                    try:
                        self._process(person, self.confidence_threshold, self._person_output_callback)
                    except Exception as e:
                        print(f"Error processing person {person.id}: {e}")
                    self.person_input_queue.task_done()
            except queue.Empty:
                continue

    def person_input(self, person: Person) -> None:
        """Add a person to the processing queue"""
        self.person_input_queue.put(person)

        # External Output Calbacks
    def add_person_callback(self, callback: PersonCallback) -> None:
        """Add callback for processed persons"""
        with self.callback_lock:
            self.person_output_callbacks.add(callback)

    def _person_output_callback(self, person: Person) -> None:
        """Handle processed person"""
        with self.callback_lock:
            for callback in self.person_output_callbacks:
                callback(person)

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for joint angle calculation."""
        self.confidence_threshold = threshold

    @staticmethod
    def _process(person: Person, confidence_threshold: float, callback:PersonCallback) -> None:
        # callback(person)
        # return
        # Check if person has pose data
        if person.pose is None:
            callback(person)
            return

        # if person.pose_angles is not None:
        #     print(f"Pose angles already calculated for person {person.id}, skipping.")
        #     callback(person)
        #     return

        angles: JointAngleDict = {}
        keypoints: np.ndarray = person.pose.getKeypoints()
        scores: np.ndarray = person.pose.getScores()

        # Calculate angles for each joint in PoseAngleDict
        for joint_name, (kp1, kp2, kp3) in PoseAngleKeypoints.items():
            # Get indices
            # idx1, idx2, idx3 = kp1.value, kp2.value, kp3.value
            idx1: int = kp1.value
            idx2: int = kp2.value
            idx3: int = kp3.value
            # Check confidence scores
            if (scores[kp1.value] > confidence_threshold and
                scores[kp2.value] > confidence_threshold and
                scores[kp2.value] > confidence_threshold):

                # Calculate angle
                p1 = keypoints[idx1]
                p2 = keypoints[idx2]
                p3 = keypoints[idx3]
                angle: float = JointAngleCalculator.calculate_angle(p1, p2, p3)
                confidence: float = min(scores[idx1], scores[idx2], scores[idx3])
                # Store results
                angles[joint_name] = JointAngle(angle = angle, confidence = confidence)

            else:
                # Low confidence, set angle to NaN
                angles[joint_name] = JointAngle(angle = np.nan, confidence = 0.0)

        # Create DataFrame from angles
        person.pose_angles = angles
        callback(person)

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate the signed angle between three points in the 2D plane (0 to 2π radians).

        Args:
            p1: First point coordinates [x, y]
            p2: Second point (vertex) coordinates [x, y]
            p3: Third point coordinates [x, y]

        Returns:
            Angle in radians, in the range [0, 2π)
        """
        v1 = p1 - p2
        v2 = p3 - p2

        angle: float = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle: float = angle % (2 * np.pi)  # Normalize to [0, 2π)

        # angle = angle / (2 * np.pi)     # Normalize to [0, 1]
        # angle = angle - np.pi           # Shift to [-π, π) range

        return angle

