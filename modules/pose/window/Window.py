# Standard library imports
from collections import deque
import time
from typing import Any, Callable, Dict, List, Optional, Type

# Third-party imports
import cv2
import numpy as np
import pandas as pd
from cv2 import KalmanFilter

# Local application imports
from modules.pose.Definitions import *

class Window:
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.kalman_filters = [self._init_kalman() for _ in range(NUM_KEYPOINTS)]

    def update_pose(self, pose: Pose, timestamp: float) -> pd.DataFrame:
        """Add a new pose to the window, apply Kalman and rolling filter, and return the window as DataFrame."""
        # Apply Kalman filter to each keypoint
        filtered_keypoints = np.zeros_like(pose.keypoints)
        for i, kf in enumerate(self.kalman_filters):
            filtered_keypoints[i] = kf.correct(np.array(pose.keypoints[i], dtype=np.float32))
        filtered_pose = Pose(filtered_keypoints, pose.scores)
        # Add to window
        self.window.append({
            'timestamp': timestamp,
            'keypoints': filtered_pose.keypoints.flatten(),
            'scores': filtered_pose.scores
        })
        # Create DataFrame
        df = pd.DataFrame(list(self.window))
        # Apply rolling mean to keypoints
        if len(df) >= 2:
            keypoints_cols = [f'kp_{i}' for i in range(NUM_KEYPOINTS * 2)]
            keypoints_data = np.stack(list(df['keypoints'].values))
            keypoints_df = pd.DataFrame(keypoints_data, columns=keypoints_cols)
            smoothed = keypoints_df.rolling(window=min(self.window_size, len(df)), min_periods=1).mean()
            for i, col in enumerate(keypoints_cols):
                df[col] = smoothed[col]
        return df

    def _init_kalman(self):
        """Initialize a Kalman filter for a single keypoint (2D)."""
        kf = KalmanFilter(4, 2)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4,1), dtype=np.float32)
        return kf

    def show_latest_pose(self, df: pd.DataFrame, window_name: str = "PoseWindow", img_size: int = 512) -> None:
        """Visualize the latest pose in the DataFrame using OpenCV."""
        if df.empty:
            print("No data to display.")
            return
        # Get the latest row
        latest = df.iloc[-1]
        # Extract keypoints (assumes columns named kp_0, kp_1, ..., kp_33)
        keypoints = np.array([latest[f'kp_{i}'] for i in range(NUM_KEYPOINTS * 2)], dtype=np.float32)
        keypoints = keypoints.reshape(NUM_KEYPOINTS, 2)
        # Create blank image
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        # Draw keypoints
        for x, y in keypoints:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)