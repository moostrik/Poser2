from threading import Lock
from typing import Union

import torch

from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.batch.GPUFrame import GPUFrameDict
from modules.pose.Frame import FrameDict
from modules.pose.Settings import Settings, ModelType
from modules.utils.PerformanceTimer import PerformanceTimer


from .InOut import SegmentationInput, SegmentationOutput
from .ONNXSegmentation import ONNXSegmentation
from .TRTSegmentation_T import TRTSegmentation


Segmentation = Union[ONNXSegmentation, TRTSegmentation]


class MaskBatchExtractor(TypedCallbackMixin[dict[int, torch.Tensor]]):
    """GPU-based batch extractor for person segmentation masks using RVM.

    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the Segmentation queue to maintain real-time performance.

    Broadcasts GPU tensors (dict[tracklet_id, torch.Tensor]) via callbacks for efficient
    OpenGL texture conversion. Masks stay on GPU to avoid unnecessary CPU transfers.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time visualization where recent data is more valuable than old data.
    """

    def __init__(self, settings: Settings):
        super().__init__()
        self._segmentation: Segmentation = ONNXSegmentation(settings)
        if settings.model_type is ModelType.ONNX:
            self._segmentation = ONNXSegmentation(settings)
        elif settings.model_type is ModelType.TRT:
            self._segmentation = TRTSegmentation(settings)
        self._lock: Lock = Lock()
        self._batch_counter: int = 0
        self._verbose: bool = settings.verbose

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

        self._segmentation.submit_batch(SegmentationInput(
            batch_id=batch_id,
            gpu_images=gpu_image_list,
            tracklet_ids=tracklet_ids
        ))

    def _on_segmentation_result(self, output: SegmentationOutput) -> None:
        """Callback from Segmentation thread when results are ready or dropped.

        Broadcasts GPU tensor dict via callbacks. Dropped batches are silently ignored.
        """
        if not output.processed or output.mask_tensor is None:
            return

        # Track inference time
        self._process_timer.add_time(output.inference_time_ms, report=self._verbose)
        # self._wait_timer.add_time(output.lock_time_ms, report=self._verbose)

        # Create dict mapping tracklet_id -> GPU tensor (H, W)
        mask_dict: dict[int, torch.Tensor] = {}
        for idx, tracklet_id in enumerate(output.tracklet_ids):
            if idx < output.mask_tensor.shape[0]:
                mask_dict[tracklet_id] = output.mask_tensor[idx]  # (H, W) tensor on GPU

        # Broadcast to callbacks
        self._notify_callbacks(mask_dict)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._batch_counter = 0
