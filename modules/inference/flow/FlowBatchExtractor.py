from collections import deque
from threading import Lock
from typing import Union

import torch

from ..crop_extractor import CropImageDict
from .flow_image import FlowImage, FlowImageDict, FlowImageCallback
from modules.pose.frame import FrameDict

import logging
logger = logging.getLogger(__name__)
from ..model_types import ModelType
from .FlowSettings import FlowSettings
from modules.utils import PerformanceTimer

from .InOut import OpticalFlowInput, OpticalFlowOutput
from .ONNXOpticalFlow import ONNXOpticalFlow, OpticalFlowInput, OpticalFlowOutput
from .TRTOpticalFlow import TRTOpticalFlow

OpticalFlow = Union[ONNXOpticalFlow, TRTOpticalFlow]


FlowCallback = FlowImageCallback


class FlowBatchExtractor:
    """GPU-based batch extractor for optical flow using RAFT.

    Computes dense optical flow between consecutive frames for each tracked person.
    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the optical flow queue to maintain real-time performance.

    Broadcasts GPU tensors (dict[tracklet_id, torch.Tensor]) via callbacks for efficient
    OpenGL texture conversion. Flow tensors stay on GPU to avoid unnecessary CPU transfers.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time fluid simulation where recent data is more valuable than old data.
    """

    def __init__(self, settings: FlowSettings):
        self._optical_flow: OpticalFlow = ONNXOpticalFlow(settings)
        if settings.model_type is ModelType.ONNX:
            self._optical_flow = ONNXOpticalFlow(settings)
        elif settings.model_type is ModelType.TRT:
            self._optical_flow = TRTOpticalFlow(settings)
        self._lock = Lock()
        self._batch_counter: int = 0
        self._verbose: bool = settings.verbose

        # Track inference times
        self._process_timer =   PerformanceTimer(name="RAFT Optical Flow", sample_count=1000, report_interval=100, color='magenta', omit_init=25)
        self._wait_timer =      PerformanceTimer(name="RAFT Wait        ", sample_count=1000, report_interval=100, color='magenta', omit_init=25)

        self._callbacks: set[FlowCallback] = set()
        self._callback_lock = Lock()

        self._optical_flow.register_callback(self._on_optical_flow_result)

    def add_flow_callback(self, callback: FlowCallback) -> None:
        with self._callback_lock:
            self._callbacks.add(callback)

    def remove_flow_callback(self, callback: FlowCallback) -> None:
        with self._callback_lock:
            self._callbacks.discard(callback)

    def _notify_callbacks(self, flow_dict: FlowImageDict) -> None:
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(flow_dict)
                except Exception:
                    logger.exception("Error in callback")

    def start(self) -> None:
        """Start the optical flow processing thread."""
        self._optical_flow.start()

    def stop(self) -> None:
        """Stop the optical flow processing thread."""
        self._optical_flow.stop()

    def process(self, poses: FrameDict, crop_frames: CropImageDict) -> None:
        """Submit batch for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            crop_frames: Crop frames with crop and prev_crop tensors, keyed by tracklet ID
        """
        if not self._optical_flow.is_ready:
            return

        tracklet_id_list: list[int] = []
        pair_list: list[tuple[torch.Tensor, torch.Tensor]] = []

        for tracklet_id in poses.keys():
            if tracklet_id in crop_frames:
                crop_frame = crop_frames[tracklet_id]
                # Only add if prev_crop and crop exist (need both frames for optical flow)
                if crop_frame.prev_crop is not None:
                    tracklet_id_list.append(tracklet_id)
                    pair_list.append((crop_frame.prev_crop, crop_frame.crop))

        # If no pairs available, emit empty flow dict to maintain data flow
        if not pair_list:
            self._notify_callbacks({})
            return

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter

        self._optical_flow.submit(OpticalFlowInput(
            batch_id=batch_id,
            gpu_image_pairs=pair_list,
            tracklet_ids=tracklet_id_list
        ))

    def _on_optical_flow_result(self, output: OpticalFlowOutput) -> None:
        """Callback from optical flow thread when results are ready or dropped.

        Broadcasts GPU tensor dict via callbacks. Dropped batches are silently ignored.
        """
        if not output.processed or output.flow_tensor is None:
            return

        # Track inference time
        self._process_timer.add_time(output.inference_time_ms, report=self._verbose)
        self._wait_timer.add_time(output.lock_time_ms, report=self._verbose)

        # Create FlowImageDict mapping tracklet_id -> FlowImage
        # Flow tensor format: [0] = x-displacement, [1] = y-displacement
        flow_dict: FlowImageDict = {}
        for idx, tracklet_id in enumerate(output.tracklet_ids):
            if idx < output.flow_tensor.shape[0]:
                flow_dict[tracklet_id] = FlowImage(
                    track_id=tracklet_id,
                    flow=output.flow_tensor[idx],  # (2, H, W) tensor on GPU
                )

        # Broadcast to callbacks
        self._notify_callbacks(flow_dict)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._batch_counter = 0
