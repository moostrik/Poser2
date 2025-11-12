from dataclasses import replace
from queue import Queue, Empty
from threading import Lock
from typing import Optional
import time

from modules.pose.Pose import Pose, PoseDict
from modules.pose.nodes.Nodes import BatchExtractorNode
from modules.pose.detection.Detection import Detection, DetectionInput, DetectionOutput
from modules.pose.features import Point2DFeature
from modules.utils.ImageUtils import resize_image
from modules.pose.detection.Detection import POSE_MODEL_WIDTH, POSE_MODEL_HEIGHT
import numpy as np


class PointsExtractorNode(BatchExtractorNode):
    """GPU-based batch extractor for 2D pose points using RTMPose detection."""

    def __init__(self, detection: Detection):
        """Initialize with a Detection instance.

        Args:
            detection: Pre-configured Detection instance (must be started)
        """
        self._detection = detection
        self._lock = Lock()

        # Batch accumulation
        self._pending_poses: PoseDict = {}
        self._batch_counter: int = 0

        # Result tracking
        self._result_queue: Queue[DetectionOutput] = Queue(maxsize=4)
        self._waiting_batches: dict[int, PoseDict] = {}  # batch_id -> poses

        # Register callback with detection
        self._detection.register_callback(self._on_detection_result)

    def add(self, poses: PoseDict) -> None:
        """Add poses to the batch queue.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
        """
        with self._lock:
            self._pending_poses.update(poses)

    def process(self) -> PoseDict:
        """Process all queued poses and return enriched poses.

        Returns:
            Dictionary of poses with updated Point2DFeature, keyed by tracklet ID
        """
        if not self._detection.is_ready:
            print("PointsExtractor: Detection not ready")
            return {}

        # Get pending poses
        with self._lock:
            if not self._pending_poses:
                return {}

            poses_to_process = self._pending_poses
            self._pending_poses = {}
            self._batch_counter += 1
            batch_id = self._batch_counter

        # Prepare images for detection
        images = []
        tracklet_ids = []

        for tracklet_id, pose in poses_to_process.items():
            # Get crop image from pose (assumes crop_image is set by upstream)
            if not hasattr(pose, 'crop_image') or pose.crop_image is None:
                continue

            # Resize to model input size
            resized = resize_image(pose.crop_image, POSE_MODEL_WIDTH, POSE_MODEL_HEIGHT)
            images.append(resized)
            tracklet_ids.append(tracklet_id)

        if not images:
            return {}

        # Submit batch for detection
        detection_input = DetectionInput(batch_id=batch_id, images=images)

        with self._lock:
            self._waiting_batches[batch_id] = poses_to_process

        self._detection.submit_batch(detection_input)

        # Wait for results (with timeout)
        timeout = 2.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = self._result_queue.get(timeout=0.1)

                if result.batch_id == batch_id:
                    return self._process_result(result, tracklet_ids)
                else:
                    # Got result for different batch, put it back
                    self._result_queue.put(result)

            except Empty:
                continue

        print(f"PointsExtractor: Timeout waiting for batch {batch_id}")

        with self._lock:
            self._waiting_batches.pop(batch_id, None)

        return {}

    def _on_detection_result(self, output: DetectionOutput) -> None:
        """Callback invoked by Detection thread with results."""
        try:
            self._result_queue.put_nowait(output)
        except:
            print("PointsExtractor: Result queue full, dropping results")

    def _process_result(self, output: DetectionOutput, tracklet_ids: list[int]) -> PoseDict:
        """Process detection output and create updated poses.

        Args:
            output: Detection output with points and scores
            tracklet_ids: List of tracklet IDs corresponding to batch order

        Returns:
            Dictionary of updated poses
        """
        with self._lock:
            original_poses = self._waiting_batches.pop(output.batch_id, None)

        if original_poses is None:
            print(f"PointsExtractor: No waiting batch for ID {output.batch_id}")
            return {}

        updated_poses: PoseDict = {}

        for idx, tracklet_id in enumerate(tracklet_ids):
            if idx >= len(output.point_batch):
                continue

            original_pose = original_poses.get(tracklet_id)
            if original_pose is None:
                continue

            # Create Point2DFeature from detection results
            points = output.point_batch[idx]  # (17, 2) normalized
            scores = output.score_batch[idx]  # (17,)

            point_feature = Point2DFeature(
                values=points.astype(np.float32),
                scores=scores.astype(np.float32)
            )

            # Create updated pose with new points
            updated_pose = replace(original_pose, points=point_feature)
            updated_poses[tracklet_id] = updated_pose

        return updated_poses

    def reset(self) -> None:
        """Clear all pending data."""
        with self._lock:
            self._pending_poses.clear()
            self._waiting_batches.clear()

        # Clear result queue
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Empty:
                break