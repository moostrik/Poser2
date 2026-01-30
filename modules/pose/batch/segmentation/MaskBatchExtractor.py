from dataclasses import replace
from threading import Lock
from traceback import print_exc
from typing import Union

import torch

from modules.pose.batch.GPUFrame import GPUFrame, GPUFrameDict, GPUFrameCallback
from modules.pose.Frame import FrameDict
from modules.pose.Settings import Settings, ModelType
from modules.utils.PerformanceTimer import PerformanceTimer


from .InOut import SegmentationInput, SegmentationOutput
from .ONNXSegmentation import ONNXSegmentation
from .TRTSegmentation import TRTSegmentation


Segmentation = Union[ONNXSegmentation, TRTSegmentation]


class MaskBatchExtractor:
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
        self._segmentation: Segmentation = ONNXSegmentation(settings)
        if settings.model_type is ModelType.ONNX:
            self._segmentation = ONNXSegmentation(settings)
        elif settings.model_type is ModelType.TRT:
            self._segmentation = TRTSegmentation(settings)
        self._lock: Lock = Lock()
        self._batch_counter: int = 0
        self._verbose: bool = settings.verbose

        # Callbacks
        self._callbacks: set[GPUFrameCallback] = set()
        self._callback_lock: Lock = Lock()

        # Pending frames awaiting segmentation results, keyed by batch_id
        self._pending_frames: dict[int, tuple[FrameDict, GPUFrameDict]] = {}

        # Track active tracklet IDs for detecting removed tracklets
        self._previous_tracklet_ids: set[int] = set()

        # Track inference times
        self._process_timer =   PerformanceTimer(name="RVM Segmentation  ", sample_count=1000, report_interval=100, color='cyan', omit_init=25)
        self._wait_timer =      PerformanceTimer(name="RVM Wait        ", sample_count=1000, report_interval=100, color='cyan', omit_init=25)

        self._segmentation.register_callback(self._on_segmentation_result)

    def start(self) -> None:
        """Start the segmentation processing thread."""
        self._segmentation.start()

    def stop(self) -> None:
        """Stop the segmentation processing thread."""
        self._segmentation.stop()

    def process(self, poses: FrameDict, gpu_frames: GPUFrameDict) -> None:
        """Submit poses with GPU images for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            gpu_frames: GPU frames with crops already on GPU, keyed by tracklet ID
        """
        if not self._segmentation.is_ready:
            return

        tracklet_ids: list[int] = []
        gpu_image_list: list = []  # list[cp.ndarray]

        for tracklet_id in poses.keys():
            if tracklet_id in gpu_frames:
                tracklet_ids.append(tracklet_id)
                gpu_image_list.append(gpu_frames[tracklet_id].crop)

        if not gpu_image_list:
            return

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter
            self._pending_frames[batch_id] = (poses, gpu_frames)

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

        poses, gpu_frames = pending

        # If batch was dropped or no masks, forward original frames without masks
        if not output.processed or output.mask_tensor is None or output.fgr_tensor is None:
            self._notify_callbacks(poses, gpu_frames)
            return

        # Track inference time
        self._process_timer.add_time(output.inference_time_ms, report=self._verbose)

        # Create updated GPUFrameDict with masks attached
        result_frames: GPUFrameDict = {}
        for idx, tracklet_id in enumerate(output.tracklet_ids):
            if tracklet_id in gpu_frames and idx < output.mask_tensor.shape[0]:
                mask = output.mask_tensor[idx]  # (H, W) tensor on GPU
                foreground = output.fgr_tensor[idx]
                # foreground = foreground * mask.unsqueeze(0) # Do in visualisation if needed
                result_frames[tracklet_id] = replace(gpu_frames[tracklet_id], mask=mask, foreground=foreground)
            elif tracklet_id in gpu_frames:
                # No mask for this tracklet, forward original
                result_frames[tracklet_id] = gpu_frames[tracklet_id]

        # Include any frames that weren't in the output (shouldn't happen, but safety)
        for tracklet_id, frame in gpu_frames.items():
            if tracklet_id not in result_frames:
                result_frames[tracklet_id] = frame

        # Broadcast to callbacks
        self._notify_callbacks(poses, result_frames)

    def _notify_callbacks(self, poses: FrameDict, gpu_frames: GPUFrameDict) -> None:
        """Emit callbacks with poses and GPU frames."""
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(poses, gpu_frames)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_callback(self, callback: GPUFrameCallback) -> None:
        """Register callback to receive poses and GPU frames with masks."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def remove_callback(self, callback: GPUFrameCallback) -> None:
        """Unregister callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._batch_counter = 0
            self._pending_frames.clear()
