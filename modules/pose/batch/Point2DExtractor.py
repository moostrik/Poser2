from dataclasses import replace
from threading import Lock

from modules.pose.Pose import PoseDict
from modules.pose.detection.Detection import Detection, DetectionInput, DetectionOutput
from modules.pose.features import Point2DFeature
from modules.pose.callback.mixins import PoseDictCallbackMixin
import numpy as np


class Point2DExtractor(PoseDictCallbackMixin):
    """GPU-based batch extractor for 2D pose points using RTMPose detection.

    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the Detection queue to maintain real-time performance. Dropped
    batches do not emit callbacks - only successfully processed batches are broadcast.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time visualization where recent data is more valuable than old data.
    """

    def __init__(self, detection: Detection):
        super().__init__()
        self._detection = detection
        self._lock = Lock()
        self._batch_counter: int = 0
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

        tracklet_ids: list[int] = []
        image_list: list[np.ndarray] = []

        for tracklet_id in poses.keys():
            if tracklet_id in images:
                tracklet_ids.append(tracklet_id)
                image_list.append(images[tracklet_id])

        if not image_list:
            return

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter
            self._waiting_batches[batch_id] = (poses, tracklet_ids)

        self._detection.submit_batch(DetectionInput(batch_id=batch_id, images=image_list))

    def _on_detection_result(self, output: DetectionOutput) -> None:
        """Callback from Detection thread when results are ready or dropped.

        Only successful batches (processed=True) emit callbacks. Dropped batches
        are silently ignored to prioritize real-time performance over completeness.
        """
        with self._lock:
            batch_data: tuple[PoseDict, list[int]] | None = self._waiting_batches.pop(output.batch_id, None)

        if not batch_data:
            print(f"Point2DExtractor Warning: No waiting batch for batch_id {output.batch_id} (possible reset during processing)")
            return

        original_poses, tracklet_ids = batch_data
        result_poses: PoseDict = {}

        if output.processed:
            for idx, tracklet_id in enumerate(tracklet_ids):
                if idx < len(output.point_batch) and tracklet_id in original_poses:
                    point_feature = Point2DFeature(
                        values=output.point_batch[idx],
                        scores=output.score_batch[idx]
                    )
                    result_poses[tracklet_id] = replace(original_poses[tracklet_id], points=point_feature)

            self._notify_pose_dict_callbacks(result_poses)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._waiting_batches.clear()
            self._batch_counter = 0