# Standard library imports

# Third-party imports
import numpy as np
from cv2 import KalmanFilter as Filter

# Local application imports
from modules.pose.Definitions import Pose, NUM_KEYPOINTS
from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods

class KalmanFilter:
    def __init__(self) -> None:
        self.HotReload = HotReloadStaticMethods(self.__class__, True)
        self.HotReload.add_reload_callback(self.on_hot_reload)

        self.kalman_filters: list[Filter] = [self._init_kalman() for _ in range(NUM_KEYPOINTS)]

    def apply(self, pose: Pose) -> Pose:
        return self.apply_static(pose, self.kalman_filters)

    def on_hot_reload(self) -> None:
        print('KalmanFilter: Hot reload detected, reinitializing Kalman filters')
        self.kalman_filters = [self._init_kalman() for _ in range(NUM_KEYPOINTS)]

    @staticmethod
    def apply_static(pose: Pose, filters: list[Filter]) -> Pose:
        return pose
        filtered_keypoints: np.ndarray = np.zeros_like(pose.keypoints)
        for i, kf in enumerate(filters):
            kf.predict()
            measurement = np.array(pose.keypoints[i], dtype=np.float32).reshape(2, 1)
            state = kf.correct(measurement)
            filtered_keypoints[i] = state[:2, 0]
        return Pose(filtered_keypoints, pose.scores)

    @staticmethod
    def _init_kalman() -> Filter:
        kf = Filter(4, 2)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4,1), dtype=np.float32)
        return kf


