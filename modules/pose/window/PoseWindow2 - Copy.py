# Standard library imports
from collections import deque
import time
from typing import Any, Callable, Dict, List, Optional, Type

# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Local application imports
from modules.pose.detection.Definitions import *

class PoseWindow:
    def __init__(self, window_size: int = 30) -> None:
        """Initialize the pose window with specified size."""
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self._init_kalman()

    def update_pose(self, pose: Pose, timestamp: float) -> pd.DataFrame:
        """ Add a new pose to the Window, apply filters and return smoothed data."""
        # Apply Kalman filter to the pose
        filtered_pose = self._apply_kalman(pose)

        # Create a record with timestamp and keypoint data
        pose_data = {
            'timestamp': timestamp,
            'mean_score': filtered_pose.mean_score
        }

        # Add keypoint positions and scores
        keypoints = filtered_pose.getKeypoints()
        scores = filtered_pose.getScores()

        for i, kp_name in enumerate(KeypointNames):
            pose_data[f'{kp_name}_x'] = keypoints[i, 0]
            pose_data[f'{kp_name}_y'] = keypoints[i, 1]
            pose_data[f'{kp_name}_score'] = scores[i]

        # Add to window
        self.window.append(pose_data)

        # Convert to DataFrame and apply smoothing
        df = pd.DataFrame(list(self.window))

        if len(df) > 2:  # Need at least 3 points for meaningful smoothing
            # Apply rolling window smoothing to keypoint positions
            window_size = min(5, len(df))

            for kp_name in KeypointNames:
                # Only smooth positions with reasonable confidence
                mask = df[f'{kp_name}_score'] > 0.3

                if mask.sum() > window_size // 2:  # Enough valid points
                    for coord in ['x', 'y']:
                        col = f'{kp_name}_{coord}'
                        df.loc[mask, f'{col}_smooth'] = df.loc[mask, col].rolling(
                            window=window_size, min_periods=1, center=True
                        ).mean()

                        # Fill non-smoothed values with original
                        df[f'{col}_smooth'] = df[f'{col}_smooth'].fillna(df[col])
                else:
                    # Not enough valid points, use original values
                    for coord in ['x', 'y']:
                        col = f'{kp_name}_{coord}'
                        df[f'{col}_smooth'] = df[col]
        else:
            # Not enough frames yet, use original values
            for kp_name in KeypointNames:
                for coord in ['x', 'y']:
                    col = f'{kp_name}_{coord}'
                    df[f'{col}_smooth'] = df[col]

        return df

    def _init_kalman(self) -> None:
        """ Initialize the Kalman filter parameters."""
        # Initialize one Kalman filter per keypoint
        self.kalman_filters = []

        for _ in range(NUM_KEYPOINTS):
            # State: [x, y, dx, dy] - position and velocity
            kf = cv2.KalmanFilter(4, 2)

            # State transition matrix
            kf.transitionMatrix = np.array([
                [1, 0, 1, 0],  # x = x + dx
                [0, 1, 0, 1],  # y = y + dy
                [0, 0, 1, 0],  # dx = dx
                [0, 0, 0, 1]   # dy = dy
            ], np.float32)

            # Measurement matrix (we only measure position)
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], np.float32)

            # Process noise covariance
            kf.processNoiseCov = np.array([
                [0.01, 0, 0, 0],
                [0, 0.01, 0, 0],
                [0, 0, 0.1, 0],
                [0, 0, 0, 0.1]
            ], np.float32)

            # Measurement noise covariance
            kf.measurementNoiseCov = np.array([
                [0.1, 0],
                [0, 0.1]
            ], np.float32)

            # Initial state
            kf.errorCovPost = np.eye(4, dtype=np.float32)

            self.kalman_filters.append(kf)

        # Track if each filter has been initialized
        self.filter_initialized = [False] * NUM_KEYPOINTS

    def _apply_kalman(self, pose: Pose) -> Pose:
        """ Apply Kalman filter to the pose."""
        keypoints = pose.getKeypoints().copy()
        scores = pose.getScores().copy()

        for i in range(NUM_KEYPOINTS):
            # Skip filtering low confidence keypoints
            if scores[i] < 0.2:
                continue

            point = keypoints[i].reshape(2, 1)

            if not self.filter_initialized[i]:
                # Initialize filter with first good measurement
                self.kalman_filters[i].statePost = np.array([
                    [point[0, 0]],
                    [point[1, 0]],
                    [0],
                    [0]
                ], dtype=np.float32)
                self.filter_initialized[i] = True
                continue

            # Prediction
            predicted = self.kalman_filters[i].predict()

            # Adaptive measurement noise based on confidence
            self.kalman_filters[i].measurementNoiseCov = np.array([
                [1.0/scores[i], 0],
                [0, 1.0/scores[i]]
            ], np.float32)

            # Correction
            corrected = self.kalman_filters[i].correct(point)

            # Update keypoint with filtered position
            keypoints[i, 0] = corrected[0, 0]
            keypoints[i, 1] = corrected[1, 0]

        # Create new pose with filtered keypoints
        filtered_pose = Pose(keypoints, scores)
        return filtered_pose

    def get_latest_smoothed_pose(self) -> Optional[Pose]:
        """Return the latest smoothed pose from the window."""
        if not self.window:
            return None

        df = pd.DataFrame(list(self.window))
        if len(df) == 0:
            return None

        # Get the last row (most recent)
        latest = df.iloc[-1]

        # Extract keypoints and scores
        keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        scores = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        for i, kp_name in enumerate(KeypointNames):
            # Use smoothed values if available, otherwise original
            if f'{kp_name}_x_smooth' in latest:
                keypoints[i, 0] = latest[f'{kp_name}_x_smooth']
                keypoints[i, 1] = latest[f'{kp_name}_y_smooth']
            else:
                keypoints[i, 0] = latest[f'{kp_name}_x']
                keypoints[i, 1] = latest[f'{kp_name}_y']

            scores[i] = latest[f'{kp_name}_score']

        return Pose(keypoints, scores)