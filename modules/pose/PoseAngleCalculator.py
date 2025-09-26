# Standard library imports
from dataclasses import replace
from queue import Empty, Queue
from threading import Event, Lock, Thread
import traceback
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from modules.pose.PoseDefinitions import *

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseAngleCalculator(Thread):
    def __init__(self) -> None:
        """Initialize the JointAngles calculator."""
        super().__init__()
        self._stop_event = Event()

        # Input
        self.pose_input_queue: Queue[Pose] = Queue()

        # Callbacks
        self.callback_lock = Lock()
        self.pose_output_callbacks: set[PoseCallback] = set()

        self.hot_reload = HotReloadMethods(self.__class__, True)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()
        with self.callback_lock:
            self.pose_output_callbacks.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pose: Optional[Pose] = self.pose_input_queue.get(block=True, timeout=0.01)
                if pose is not None:
                    try:
                        self._process(pose, self._notify_callback)
                    except Exception as e:
                        print(f"Error processing pose {pose.id}: {e}")
                        traceback.print_exc()  # This prints the stack trace
                    self.pose_input_queue.task_done()
            except Empty:
                continue

    def pose_input(self, pose: Pose) -> None:
        """Add a pose to the processing queue"""
        self.pose_input_queue.put(pose)

        # External Output Calbacks
    def add_pose_callback(self, callback: PoseCallback) -> None:
        """Add callback for processed poses"""
        with self.callback_lock:
            self.pose_output_callbacks.add(callback)

    def _notify_callback(self, pose: Pose) -> None:
        """Handle processed pose"""
        with self.callback_lock:
            for callback in self.pose_output_callbacks:
                callback(pose)

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for joint angle calculation."""
        self.confidence_threshold = threshold

    @staticmethod
    def _process(pose: Pose, callback:PoseCallback) -> None:

        if pose.point_data is None:
            # return nan angles and 0 for confidence
            angled_pose: Pose = replace(pose, angle_data=PoseAngleData())
            callback(angled_pose)
            return

        point_values: np.ndarray = pose.point_data.points
        point_scores: np.ndarray = pose.point_data.scores

        angle_values: np.ndarray = np.full(NUM_POSE_ANGLES, np.nan, dtype=np.float32)
        angle_scores: np.ndarray = np.zeros(NUM_POSE_ANGLES, dtype=np.float32)

        for i, (joint, (kp1, kp2, kp3)) in enumerate(PoseAngleJointTriplets.items()):
            idx1, idx2, idx3 = kp1.value, kp2.value, kp3.value
            p1, p2, p3 = point_values[idx1], point_values[idx2], point_values[idx3]
            scores = point_scores[[idx1, idx2, idx3]]
            if not (np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any()):
                # All points are valid (not NaN), calculate the angle
                rotate_by: float = PoseAngleRotations[joint]
                angle: float = PoseAngleCalculator.calculate_angle(p1, p2, p3, rotate_by)
                confidence: float = np.min(scores)
                angle_values[i] = angle
                angle_scores[i] = confidence

        angled_pose: Pose = replace(pose, angle_data=PoseAngleData(angle_values, angle_scores))
        callback(angled_pose)

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, rotate_by: float = 0) -> float:
        """
        Calculate the signed angle between three points in the 2D plane (0 to 2π radians).

        Args:
            p1: First point coordinates [x, y]
            p2: Second point (vertex) coordinates [x, y]
            p3: Third point coordinates [x, y]

        Returns:
            Angle in radians, in the range [-π, π)
        """
        v1 = p1 - p2
        v2 = p3 - p2

        # angle: float = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

        dot = np.dot(v1, v2)
        det = v1[0] * v2[1] - v1[1] * v2[0]  # 2D cross product
        angle = np.arctan2(det, dot)

        # Rotate the angle by a specified amount (in radians)
        angle += rotate_by

        # angle = angle % (2 * np.pi)  # Normalize to [0, 2π)
        # angle -= np.pi  # Shift to [-π, π)

        # Normalize to [-π, π) range
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        # epsilon = 1e-4
        # if np.isclose(angle, -np.pi):
        #     angle = -np.pi + epsilon
        # elif np.isclose(angle, np.pi):
        #     angle = np.pi - epsilon

        return angle

