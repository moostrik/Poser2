from collections import deque
from threading import Lock

import numpy as np
import torch

from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.Frame import FrameDict
from modules.pose.segmentation.RVMSegmentation import RVMSegmentation, SegmentationInput, SegmentationOutput
from modules.pose.Settings import Settings
from modules.cam.depthcam.Definitions import FrameType
from modules.utils.PerformanceTimer import PerformanceTimer


class MaskBatchExtractor(TypedCallbackMixin[dict[int, torch.Tensor]]):
    """GPU-based batch extractor for person segmentation masks using MODNet.

    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the Segmentation queue to maintain real-time performance.

    Broadcasts GPU tensors (dict[tracklet_id, torch.Tensor]) via callbacks for efficient
    OpenGL texture conversion. Masks stay on GPU to avoid unnecessary CPU transfers.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time visualization where recent data is more valuable than old data.
    """

    def __init__(self, settings: Settings):
        super().__init__()
        self._segmentation = RVMSegmentation(settings)
        self._lock = Lock()
        self._batch_counter: int = 0
        self._images: dict[int, np.ndarray] = {}

        # Track inference times
        self._timer = PerformanceTimer(name="RVM Segmentation", sample_count=100, report_interval=100)

        self._segmentation.register_callback(self._on_segmentation_result)

    def start(self) -> None:
        """Start the segmentation processing thread."""
        self._segmentation.start()

    def stop(self) -> None:
        """Stop the segmentation processing thread."""
        self._segmentation.stop()

    def set_crop_images(self, images: dict[int, np.ndarray]) -> None:
        """Set images for processing.

        Args:
            images: Dictionary of pre-cropped/resized images (256x192) keyed by tracklet ID
        """
        with self._lock:
            self._images = images

    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None:
        """Update the camera image for a specific camera ID"""
        if frame_type != FrameType.VIDEO:
            return
        with self._lock:
            self._images[id] = image

    def process(self, poses: FrameDict) -> None:
        """Submit batch for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
        """
        if not self._segmentation.is_ready:
            return

        tracklet_id_list: list[int] = []
        image_list: list[np.ndarray] = []

        for tracklet_id in poses.keys():
            if tracklet_id in self._images:
                tracklet_id_list.append(tracklet_id)
                image_list.append(self._images[tracklet_id])

        if not image_list:
            return

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter

        self._segmentation.submit_batch(SegmentationInput(
            batch_id=batch_id,
            images=image_list,
            tracklet_ids=tracklet_id_list
        ))

    def _on_segmentation_result(self, output: SegmentationOutput) -> None:
        """Callback from Segmentation thread when results are ready or dropped.

        Broadcasts GPU tensor dict via callbacks. Dropped batches are silently ignored.
        """
        if not output.processed or output.mask_tensor is None:
            return

        # Track inference time
        self._timer.add_time(output.inference_time_ms)

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
            self._images.clear()
