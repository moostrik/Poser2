from threading import Lock
from typing import Union

import torch

from .. import crop, ModelType
from .image import Image, ImageDict, ImageCallback
from modules.pose.frame import FrameDict
from .settings import Settings
from modules.utils import PerformanceTimer


from .io import SegmentationInput, SegmentationOutput
from .runner_onnx import RunnerONNX
from .runner_trt import RunnerTRT

import logging
logger = logging.getLogger(__name__)


Segmentation = Union[RunnerONNX, RunnerTRT]


class Predictor:
    """GPU-based batch extractor for person segmentation masks using RVM.

    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the Segmentation queue to maintain real-time performance.

    Broadcasts (FrameDict, GPUFrameDict) with masks attached via callbacks. If a batch
    is dropped, the original frames are forwarded without masks to maintain data flow.
    Masks stay on GPU to avoid unnecessary CPU transfers.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time visualization where recent data is more valuable than old data.
    """

    def __init__(self, settings: Settings):
        self._segmentation: Segmentation = RunnerONNX(settings)
        if settings.model_type is ModelType.ONNX:
            self._segmentation = RunnerONNX(settings)
        elif settings.model_type is ModelType.TRT:
            self._segmentation = RunnerTRT(settings)
        self._lock: Lock = Lock()
        self._batch_counter: int = 0
        self._verbose: bool = settings.verbose

        # Callbacks
        self._callbacks: set[ImageCallback] = set()
        self._callback_lock: Lock = Lock()

        # Pending frames awaiting segmentation results, keyed by batch_id
        self._pending_frames: dict[int, tuple[FrameDict, crop.ImageDict]] = {}

        # Track active tracklet IDs for detecting removed tracklets
        self._previous_tracklet_ids: set[int] = set()

        # Track inference times
        self._process_timer =   PerformanceTimer(name="RVM Segmentation  ", sample_count=1000, report_interval=100, color='cyan', omit_init=25)
        self._wait_timer =      PerformanceTimer(name="RVM Wait        ", sample_count=1000, report_interval=100, color='cyan', omit_init=25)
        self._settings: Settings = settings

        self._segmentation.register_callback(self._on_segmentation_result)

    def start(self) -> None:
        """Start the segmentation processing thread."""
        self._segmentation.start()

    def stop(self) -> None:
        """Stop the segmentation processing thread."""
        self._segmentation.stop()

    def process(self, poses: FrameDict, crop_frames: crop.ImageDict) -> None:
        """Submit poses with GPU images for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            crop_frames: Crop frames with crops already on GPU, keyed by tracklet ID
        """
        if not self._segmentation.is_ready:
            return
        if not self._settings.enabled:
            return

        tracklet_ids: list[int] = []
        gpu_image_list: list = []  # list[cp.ndarray]

        for tracklet_id in poses.keys():
            if tracklet_id in crop_frames:
                tracklet_ids.append(tracklet_id)
                gpu_image_list.append(crop_frames[tracklet_id].crop)

        # If no crops available, forward frames without segmentation to maintain data flow
        if not gpu_image_list:
            self._notify_callbacks(poses, {})
            return

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter
            self._pending_frames[batch_id] = (poses, crop_frames)

        self._segmentation.submit(SegmentationInput(
            batch_id=batch_id,
            gpu_images=gpu_image_list,
            tracklet_ids=tracklet_ids
        ))

    def _on_segmentation_result(self, output: SegmentationOutput) -> None:
        """Callback from Segmentation thread when results are ready or dropped.

        Forwards (FrameDict, GPUFrameDict) with masks attached. If batch was dropped,
        forwards original frames without masks to maintain data flow.
        """
        # Retrieve and remove pending frames for this batch
        with self._lock:
            pending = self._pending_frames.pop(output.batch_id, None)

        if pending is None:
            return

        poses, crop_frames = pending

        # If batch was dropped or no masks, forward empty dict to maintain data flow
        if not output.processed or output.mask_tensor is None or output.fgr_tensor is None:
            self._notify_callbacks(poses, {})
            return

        # Track inference time
        self._process_timer.add_time(output.inference_time_ms, report=self._verbose)

        # Build ImageDict from segmentation results
        result_frames: ImageDict = {}
        for idx, tracklet_id in enumerate(output.tracklet_ids):
            if idx < output.mask_tensor.shape[0]:
                mask = output.mask_tensor[idx]  # (H, W) tensor on GPU
                foreground = output.fgr_tensor[idx]
                result_frames[tracklet_id] = Image(
                    mask=mask,
                    foreground=foreground,
                )

        # Broadcast to callbacks
        self._notify_callbacks(poses, result_frames)

    def _notify_callbacks(self, poses: FrameDict, mask_frames: ImageDict) -> None:
        """Emit callbacks with poses and segmentation frames."""
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(poses, mask_frames)
                except Exception as e:
                    logger.exception("Error in callback")
    def add_segmentation_image_callback(self, callback: ImageCallback) -> None:
        """Register callback to receive poses and segmentation images."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def remove_segmentation_image_callback(self, callback: ImageCallback) -> None:
        """Unregister callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._batch_counter = 0
            self._pending_frames.clear()
