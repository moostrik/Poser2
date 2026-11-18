"""RAFT through TensorRT: the previous and current crops into preallocated batches on its stream, the flow out on the
GPU."""

import logging

import tensorrt as trt
import torch

from ..trt_runtime import get_tensorrt_runtime
from .settings import OpticalFlowPredictorSettings

logger = logging.getLogger(__name__)

PRECISIONS = {trt.DataType.FLOAT: "FP32", trt.DataType.HALF: "FP16", trt.DataType.INT8: "INT8"}
TORCH_DTYPES = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}


class RunnerTRT:
    def __init__(self, settings: OpticalFlowPredictorSettings) -> None:
        self.model_file: str = f"{settings.model_path}/{settings.model}"
        self.model_width: int = settings.width
        self.model_height: int = settings.height
        self.max_batch: int = min(settings.max_players, 4)
        self.model_precision: str = "UNKNOWN"

        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._stream: torch.cuda.Stream
        self._input_names: list[str] = []
        self._output_name: str = ''
        self._torch_dtype: torch.dtype = torch.float32
        self._img1_buffer: torch.Tensor     # (max_batch, 3, H, W) the previous crops
        self._img2_buffer: torch.Tensor     # (max_batch, 3, H, W) the current crops

    def setup(self) -> None:
        """Load the engine, on the thread that will run it."""
        with open(self.model_file, 'rb') as f:
            engine_data = f.read()
        self._engine = get_tensorrt_runtime().deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine {self.model_file}")
        self._context = self._engine.create_execution_context()

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_name = name
        if len(self._input_names) != 2:
            raise RuntimeError(f"Expected 2 inputs, got {len(self._input_names)}")

        input_dtype = self._engine.get_tensor_dtype(self._input_names[0])
        self.model_precision = PRECISIONS.get(input_dtype, "UNKNOWN")
        self._torch_dtype = TORCH_DTYPES.get(input_dtype, torch.float32)

        self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            shape = (self.max_batch, 3, self.model_height, self.model_width)
            self._img1_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
            self._img2_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
        self._stream.synchronize()

        # The inputs' addresses stay (a slice keeps the base pointer); the output's is set per call
        self._context.set_tensor_address(self._input_names[0], self._img1_buffer.data_ptr())
        self._context.set_tensor_address(self._input_names[1], self._img2_buffer.data_ptr())

    def infer(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], ready: torch.cuda.Event | None) -> torch.Tensor:
        """The flow (N, 2, H, W), x and y in the model's pixels, from each previous to current crop (3, h, w) RGB in
        [0, 1], read once `ready` is recorded."""
        if ready is not None:
            self._stream.wait_event(ready)      # the crops are written on the crop stream
        batch_size = len(pairs)
        input_h, input_w = pairs[0][0].shape[1], pairs[0][0].shape[2]
        h, w = self.model_height, self.model_width
        img1_buffer = self._img1_buffer[:batch_size]
        img2_buffer = self._img2_buffer[:batch_size]
        with torch.cuda.stream(self._stream):       # the output made on the stream that writes it
            flow_out = torch.empty((batch_size, 2, h, w), dtype=self._torch_dtype, device='cuda')
            prev_chw = torch.stack([p[0] for p in pairs], dim=0).mul(255.0).to(self._torch_dtype)     # RAFT reads 0..255
            curr_chw = torch.stack([p[1] for p in pairs], dim=0).mul(255.0).to(self._torch_dtype)
            if input_h != h or input_w != w:
                prev_chw = torch.nn.functional.interpolate(prev_chw, size=(h, w), mode='bilinear', align_corners=False)
                curr_chw = torch.nn.functional.interpolate(curr_chw, size=(h, w), mode='bilinear', align_corners=False)
            img1_buffer.copy_(prev_chw)
            img2_buffer.copy_(curr_chw)

        self._context.set_input_shape(self._input_names[0], tuple(img1_buffer.shape))
        self._context.set_input_shape(self._input_names[1], tuple(img2_buffer.shape))
        self._context.set_tensor_address(self._output_name, flow_out.data_ptr())
        self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return flow_out
