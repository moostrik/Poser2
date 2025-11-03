
from dataclasses import replace

from threading import Lock
from traceback import print_exc

import numpy as np

from modules.pose.Pose import Pose, PoseDict, PoseDictCallback, PosePointData, PoseAngleFactory
from modules.Settings import Settings
from modules.pose.features.PoseAngles import PoseAngleData


class PoseConfidenceFilter():
    """Filters poses based on confidence thresholds."""
    def __init__(self, settings: Settings) -> None:
        self.confidence_threshold: float = max(0.0, min(0.99, settings.pose_conf_threshold))

        # Callbacks
        self.callback_lock: Lock = Lock()
        self.pose_output_callbacks: set[PoseDictCallback] = set()

    def add_poses(self, poses: PoseDict) -> None:
        filtered_poses: PoseDict = {}
        for pose_id, pose in poses.items():
            if pose.point_data.valid_count == 0:
                filtered_poses[pose_id] = pose
                continue

            values: np.ndarray = pose.point_data.values
            scores: np.ndarray = pose.point_data.scores
            filtered = scores >= self.confidence_threshold
            filtered_values: np.ndarray = np.where(filtered[:, np.newaxis], values, np.nan)
            rescaled_scores: np.ndarray = np.where(filtered, (scores - self.confidence_threshold) / (1 - self.confidence_threshold), 0.0)
            filtered_points = PosePointData(values=filtered_values, scores=rescaled_scores)
            filtered_angles: PoseAngleData = PoseAngleFactory.from_points(filtered_points)
            filtered_pose: Pose = replace(pose, point_data=filtered_points, angle_data=filtered_angles)
            filtered_poses[pose_id] = filtered_pose

        # Notify callbacks with filtered poses
        self._notify_pose_callbacks(filtered_poses)

    # CALLBACK METHODS
    def add_pose_callback(self, callback: PoseDictCallback) -> None:
        """Register a callback to be invoked when pose detection completes"""

        with self.callback_lock:
            self.pose_output_callbacks.add(callback)

    def _notify_pose_callbacks(self, poses: PoseDict) -> None:
        """Invoke all registered pose output callbacks"""
        with self.callback_lock:
            for callback in self.pose_output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"PosePipeline: Error in pose output callback: {str(e)}")
                    print_exc()