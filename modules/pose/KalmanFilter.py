# Standard library imports

# Third-party imports
import numpy as np
from cv2 import KalmanFilter

# Local application imports
from modules.pose.Definitions import Pose, NUM_KEYPOINTS

class Window:
    def __init__(self) -> None:
        self.kalman_filters: list[KalmanFilter] = [self._init_kalman() for _ in range(NUM_KEYPOINTS)]

    def _init_kalman(self) -> KalmanFilter:
        """Initialize a Kalman filter for a single keypoint (2D)."""
        kf = KalmanFilter(4, 2)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4,1), dtype=np.float32)
        return kf

    def filter(self, pose: Pose) -> Pose:
        filtered_keypoints: np.ndarray = np.zeros_like(pose.keypoints)
        for i, kf in enumerate(self.kalman_filters):
            filtered_keypoints[i] = kf.correct(np.array(pose.keypoints[i], dtype=np.float32))
        filtered_pose = Pose(filtered_keypoints, pose.scores)
        return filtered_pose
