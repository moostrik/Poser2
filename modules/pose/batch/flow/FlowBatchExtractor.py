from collections import deque
from threading import Lock
from typing import Union

import numpy as np
import torch

from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.Frame import FrameDict
from modules.pose.batch.flow.ONNXOpticalFlow import ONNXOpticalFlow, OpticalFlowInput, OpticalFlowOutput
from modules.pose.batch.flow.TensorRTOpticalFlow import TensorRTOpticalFlow
from modules.pose.Settings import Settings, ModelType
from modules.cam.depthcam.Definitions import FrameType
from modules.utils.PerformanceTimer import PerformanceTimer

OpticalFlow = Union[ONNXOpticalFlow, TensorRTOpticalFlow]


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
        elif settings.model_type is ModelType.TENSORRT:
            self._optical_flow = TensorRTOpticalFlow(settings)
        self._lock = Lock()
        self._batch_counter: int = 0
        self._verbose: bool = settings.verbose

        # Track inference times
        self._timer = PerformanceTimer(name="RAFT Optical Flow", sample_count=100, report_interval=100)

        self._optical_flow.register_callback(self._on_optical_flow_result)

    def start(self) -> None:
        """Start the optical flow processing thread."""
        self._optical_flow.start()

    def stop(self) -> None:
        """Stop the optical flow processing thread."""
        self._optical_flow.stop()

    def process(self, poses: FrameDict, frame_pairs: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        """Submit batch for async processing. Results broadcast via callbacks.

        Args:
            poses: Dictionary of poses keyed by tracklet ID
            frame_pairs: Dictionary of (previous_frame, current_frame) tuples keyed by tracklet ID
        """
        if not self._optical_flow.is_ready:
            return

        tracklet_id_list: list[int] = []
        pair_list: list[tuple[np.ndarray, np.ndarray]] = []

        for tracklet_id in poses.keys():
            if tracklet_id in frame_pairs:
                tracklet_id_list.append(tracklet_id)
                pair_list.append(frame_pairs[tracklet_id])

        if not pair_list:
            if self._verbose and len(poses) > 0:
                print(f"FlowBatchExtractor: No frame pairs available")
            return

        with self._lock:
            self._batch_counter += 1
            batch_id: int = self._batch_counter

        self._optical_flow.submit_batch(OpticalFlowInput(
            batch_id=batch_id,
            frame_pairs=pair_list,
            tracklet_ids=tracklet_id_list
        ))

    def _on_optical_flow_result(self, output: OpticalFlowOutput) -> None:
        """Callback from optical flow thread when results are ready or dropped.

        Broadcasts GPU tensor dict via callbacks. Dropped batches are silently ignored.
        """
        if not output.processed or output.flow_tensor is None:
            return

        # Track inference time
        self._timer.add_time(output.inference_time_ms, report=True)

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
