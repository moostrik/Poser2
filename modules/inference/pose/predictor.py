from threading import Lock
from typing import TYPE_CHECKING, Union
from .. import crop, ModelType
from modules.pose import Points2D, FrameDictCallbackMixin, replace, FrameDict
from .settings import Settings
from modules.utils import PerformanceTimer

from .io import DetectionInput, DetectionOutput
from .runner_onnx import RunnerONNX

if TYPE_CHECKING:
    from .runner_trt import RunnerTRT

import logging
logger = logging.getLogger(__name__)

Detection = Union[RunnerONNX, "RunnerTRT"]

class Predictor(FrameDictCallbackMixin):
    """GPU-based batch extractor for 2D pose points using RTMPose detection.

    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the Detection queue to maintain real-time performance. Dropped
    batches do not emit callbacks - only successfully processed batches are broadcast.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time visualization where recent data is more valuable than old data.
    """

    def __init__(self, settings: Settings):
        super().__init__()
        # One runner per model type; TensorRT is imported only when it is used.
        self._detection: Detection
        if settings.model_type is ModelType.ONNX:
            self._detection = RunnerONNX(settings)
        else:
            from .runner_trt import RunnerTRT
            self._detection = RunnerTRT(settings)

        self._lock = Lock()
        self._batch_counter: int = 0
        self._waiting_batches: dict[int, tuple[FrameDict, list[int]]] = {}

        self._process_timer =   PerformanceTimer(name="RTM Pose Detection", sample_count=1000, report_interval=100, color='yellow', omit_init=25)
        self._wait_timer =      PerformanceTimer(name="RTM Pose Wait     ", sample_count=1000, report_interval=100, color='yellow', omit_init=25)

        self._verbose: bool = settings.verbose
        self._settings: Settings = settings

        self._detection.add_pose_callback(self._on_detection_result)

    def start(self) -> None:
        """Start the detection processing thread."""
        self._detection.start()

    def stop(self) -> None:
        """Stop the detection processing thread."""
        self._detection.stop()

    def submit(self, poses: FrameDict, crop_frames: crop.ImageDict) -> None:
        """Submit poses with GPU images for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by track id
            crop_frames: Crop frames with crops already on GPU, keyed by track id
        """
        if not self._detection.is_ready:
            return
        if not self._settings.enabled:
            return

        track_ids: list[int] = []
        gpu_image_list: list = []  # list[cp.ndarray]

        for track_id in poses.keys():
            if track_id in crop_frames:
                track_ids.append(track_id)
                gpu_image_list.append(crop_frames[track_id].crop)

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter
            self._waiting_batches[batch_id] = (poses, track_ids)

        self._detection.submit(DetectionInput(batch_id=batch_id, gpu_images=gpu_image_list))

    def _on_detection_result(self, output: DetectionOutput) -> None:
        """Callback from Detection thread when results are ready or dropped.

        Only successful batches (processed=True) emit callbacks. Dropped batches
        are silently ignored to prioritize real-time performance over completeness.
        """
        with self._lock:
            batch_data: tuple[FrameDict, list[int]] | None = self._waiting_batches.pop(output.batch_id, None)

        if not batch_data:
            logger.warning(f"Point2DExtractor Warning: No waiting batch for batch_id {output.batch_id} (possible reset during processing)")
            return

        original_poses, track_ids = batch_data
        result_poses: FrameDict = {}

        # print(output.inference_time_ms)
        self._process_timer.add_time(output.inference_time_ms, report=self._verbose)
        # self._wait_timer.add_time(output.lock_time_ms, report=self._verbose)

        if output.processed:
            for idx, track_id in enumerate(track_ids):
                if idx < len(output.point_batch) and track_id in original_poses:
                    point_feature = Points2D(
                        values=output.point_batch[idx],
                        scores=output.score_batch[idx]
                    )
                    result_poses[track_id] = replace(original_poses[track_id], {Points2D: point_feature})

            self._notify_frames_callbacks(result_poses)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._waiting_batches.clear()
            self._batch_counter = 0