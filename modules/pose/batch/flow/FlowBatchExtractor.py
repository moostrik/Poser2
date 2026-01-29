from collections import deque
from threading import Lock
from typing import Union

import torch

from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.batch.GPUFrame import GPUFrameDict
from modules.pose.Frame import FrameDict
from modules.pose.Settings import Settings, ModelType
from modules.utils.PerformanceTimer import PerformanceTimer

from .InOut import OpticalFlowInput, OpticalFlowOutput
from .ONNXOpticalFlow_T import ONNXOpticalFlow, OpticalFlowInput, OpticalFlowOutput
from .TRTOpticalFlow_T import TRTOpticalFlow

OpticalFlow = Union[ONNXOpticalFlow, TRTOpticalFlow]


class FlowBatchExtractor(TypedCallbackMixin[dict[int, torch.Tensor]]):
    """GPU-based batch extractor for optical flow using RAFT.

    Computes dense optical flow between consecutive frames for each tracked person.
    Batches are processed asynchronously on GPU. Under load, pending batches may
    be dropped by the optical flow queue to maintain real-time performance.

    Broadcasts GPU tensors (dict[tracklet_id, torch.Tensor]) via callbacks for efficient
    OpenGL texture conversion. Flow tensors stay on GPU to avoid unnecessary CPU transfers.

    This design prioritizes low latency over data completeness, making it suitable
    for real-time fluid simulation where recent data is more valuable than old data.
    """

    def __init__(self, settings: Settings):
        super().__init__()
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


        self._optical_flow.register_callback(self._on_optical_flow_result)

    def start(self) -> None:
        """Start the optical flow processing thread."""
        self._optical_flow.start()

    def stop(self) -> None:
        """Stop the optical flow processing thread."""
        self._optical_flow.stop()

    def process(self, poses: FrameDict, gpu_frames: GPUFrameDict) -> None:
        """Submit batch for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            gpu_frames: GPU frames with crop and prev_crop tensors, keyed by tracklet ID
        """
        if not self._optical_flow.is_ready:
            return

        tracklet_id_list: list[int] = []
        pair_list: list[tuple[torch.Tensor, torch.Tensor]] = []

        for tracklet_id in poses.keys():
            if tracklet_id in gpu_frames:
                gpu_frame = gpu_frames[tracklet_id]
                # Only add if prev_crop exists (need both frames for optical flow)
                if gpu_frame.prev_crop is not None:
                    tracklet_id_list.append(tracklet_id)
                    pair_list.append((gpu_frame.prev_crop, gpu_frame.crop))

        if not pair_list:
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

        # Create dict mapping tracklet_id -> GPU tensor (2, H, W)
        # Flow tensor format: [0] = x-displacement, [1] = y-displacement
        flow_dict: dict[int, torch.Tensor] = {}
        for idx, tracklet_id in enumerate(output.tracklet_ids):
            if idx < output.flow_tensor.shape[0]:
                flow_dict[tracklet_id] = output.flow_tensor[idx]  # (2, H, W) tensor on GPU

        # Broadcast to callbacks
        self._notify_callbacks(flow_dict)

    def reset(self) -> None:
        """Clear all pending and buffered data."""
        with self._lock:
            self._batch_counter = 0
