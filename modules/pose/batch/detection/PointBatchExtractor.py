from dataclasses import replace
from threading import Lock
from typing import Union

from modules.pose.batch.detection.InOut import DetectionInput, DetectionOutput
from modules.pose.batch.detection.ONNXDetection import ONNXDetection
from modules.pose.batch.detection.TRTDetection import TRTDetection
from modules.pose.batch.GPUFrame import GPUFrameDict
from modules.pose.features import Points2D
from modules.pose.callback.mixins import PoseDictCallbackMixin
from modules.pose.Frame import FrameDict
from modules.pose.Settings import Settings, ModelType
from modules.utils.PerformanceTimer import PerformanceTimer

Detection = Union[ONNXDetection, TRTDetection]

class PointBatchExtractor(PoseDictCallbackMixin):
    """GPU-based batch extractor for 2D pose points using RTMPose detection.

    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the Detection queue to maintain real-time performance. Dropped
    batches do not emit callbacks - only successfully processed batches are broadcast.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time visualization where recent data is more valuable than old data.
    """

    def __init__(self, settings: Settings):
        super().__init__()
        self._detection: Detection = TRTDetection(settings)
        if settings.model_type is ModelType.ONNX:
            self._detection = ONNXDetection(settings)
        elif settings.model_type is ModelType.TRT:
            self._detection = TRTDetection(settings)

        self._lock = Lock()
        self._batch_counter: int = 0
        self._waiting_batches: dict[int, tuple[FrameDict, list[int]]] = {}

        self._process_timer =   PerformanceTimer(name="RTM Pose Detection", sample_count=100, report_interval=100, color='yellow', omit_init=2)
        self._wait_timer =      PerformanceTimer(name="RTM Pose Wait     ", sample_count=100, report_interval=100, color='yellow', omit_init=2)

        self._verbose: bool = settings.verbose

        self._detection.register_callback(self._on_detection_result)

    def start(self) -> None:
        """Start the detection processing thread."""
        self._detection.start()

    def stop(self) -> None:
        """Stop the detection processing thread."""
        self._detection.stop()

    def process(self, poses: FrameDict, gpu_frames: GPUFrameDict) -> None:
        """Submit poses with GPU images for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            gpu_frames: GPU frames with crops already on GPU, keyed by tracklet ID
        """
        if not self._detection.is_ready:
            return

        tracklet_ids: list[int] = []
        gpu_image_list: list = []  # list[cp.ndarray]

        for tracklet_id in poses.keys():
            if tracklet_id in gpu_frames:
                tracklet_ids.append(tracklet_id)
                gpu_image_list.append(gpu_frames[tracklet_id].crop)

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter
            self._waiting_batches[batch_id] = (poses, tracklet_ids)

        self._detection.submit_batch(DetectionInput(batch_id=batch_id, gpu_images=gpu_image_list))

    def _on_detection_result(self, output: DetectionOutput) -> None:
        """Callback from Detection thread when results are ready or dropped.

        Only successful batches (processed=True) emit callbacks. Dropped batches
        are silently ignored to prioritize real-time performance over completeness.
        """
        with self._lock:
            batch_data: tuple[FrameDict, list[int]] | None = self._waiting_batches.pop(output.batch_id, None)

        if not batch_data:
            print(f"Point2DExtractor Warning: No waiting batch for batch_id {output.batch_id} (possible reset during processing)")
            return

        original_poses, tracklet_ids = batch_data
        result_poses: FrameDict = {}

        # print(output.inference_time_ms)
        self._process_timer.add_time(output.inference_time_ms, report=self._verbose)
        self._wait_timer.add_time(output.lock_time_ms, report=self._verbose)

        if output.processed:
            for idx, tracklet_id in enumerate(tracklet_ids):
                if idx < len(output.point_batch) and tracklet_id in original_poses:
                    point_feature = Points2D(
                        values=output.point_batch[idx],
                        scores=output.score_batch[idx]
                    )
                    result_poses[tracklet_id] = replace(original_poses[tracklet_id], points=point_feature)

            self._notify_poses_callbacks(result_poses)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._waiting_batches.clear()
            self._batch_counter = 0