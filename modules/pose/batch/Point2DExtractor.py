from dataclasses import replace
from threading import Lock

from modules.pose.Pose import PoseDict
from modules.pose.detection.Detection import Detection, DetectionInput, DetectionOutput
from modules.pose.features import Point2DFeature
from modules.pose.callback.mixins import PoseDictCallbackMixin
import numpy as np


class Point2DExtractor(PoseDictCallbackMixin):
    """GPU-based batch extractor for 2D pose points using RTMPose detection.

    Processes entire PoseDict batches through GPU inference asynchronously.
    Results are broadcast via callbacks when ready.

    Inherits callback system from PoseDictCallbackMixin for broadcasting results.
    """

    def __init__(self, detection: Detection, emit_dropped: bool = False):
        """Initialize Point2D extractor.
        
        Args:
            detection: Detection instance for GPU inference
            emit_dropped: If True, broadcast dropped batches with empty Point2DFeatures
        """
        super().__init__()
        self._detection = detection
        self._emit_dropped = emit_dropped
        self._lock = Lock()
        self._batch_counter: int = 0

        # Track which poses correspond to which batch
        self._waiting_batches: dict[int, tuple[PoseDict, list[int]]] = {}

        self._detection.register_callback(self._on_detection_result)

    def process(self, poses: PoseDict, images: dict[int, np.ndarray]) -> None:
        """Submit poses for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            images: Pre-cropped/resized images (256x192) keyed by tracklet ID
        """
        if not self._detection.is_ready or not poses:
            return

        # Prepare batch
        with self._lock:
            self._batch_counter += 1
            batch_id = self._batch_counter

        tracklet_ids = []
        image_list = []

        for tracklet_id in poses.keys():
            if tracklet_id in images:
                tracklet_ids.append(tracklet_id)
                image_list.append(images[tracklet_id])

        if not image_list:
            return

        # Store original poses for later
        with self._lock:
            self._waiting_batches[batch_id] = (poses, tracklet_ids)

        # Submit and return immediately
        self._detection.submit_batch(DetectionInput(batch_id=batch_id, images=image_list))

    def _on_detection_result(self, output: DetectionOutput) -> None:
        """Callback from Detection thread when results are ready or dropped."""
        with self._lock:
            batch_data = self._waiting_batches.pop(output.batch_id, None)

        if not batch_data:
            return

        original_poses, tracklet_ids = batch_data

        # If batch was dropped, optionally emit with empty points
        if not output.processed:
            if self._emit_dropped:
                dropped_poses: PoseDict = {}
                for tracklet_id in tracklet_ids:
                    if tracklet_id in original_poses:
                        # Create pose with empty Point2DFeature
                        dropped_poses[tracklet_id] = replace(
                            original_poses[tracklet_id],
                            points=Point2DFeature.create_empty()
                        )
                
                if dropped_poses:
                    self._notify_callbacks(dropped_poses)
            
            return

        # Process successful results
        updated_poses: PoseDict = {}

        for idx, tracklet_id in enumerate(tracklet_ids):
            if idx < len(output.point_batch) and tracklet_id in original_poses:
                point_feature = Point2DFeature(
                    values=output.point_batch[idx].astype(np.float32),
                    scores=output.score_batch[idx].astype(np.float32)
                )
                updated_poses[tracklet_id] = replace(original_poses[tracklet_id], points=point_feature)

        # Broadcast results
        if updated_poses:
            self._notify_callbacks(updated_poses)

    def reset(self) -> None:
        with self._lock:
            self._waiting_batches.clear()