"""RTMPose through TensorRT: the crops normalized into a preallocated batch on its stream, the SimCC outputs back
to the CPU. It takes turns with the other TensorRT models on the shared execution lock."""

import logging
import time

import numpy as np
import tensorrt as trt
import torch

from ..trt_runtime import get_tensorrt_runtime, get_init_lock, get_exec_lock
from .decode import NUM_KEYPOINTS
from .settings import Settings

logger = logging.getLogger(__name__)

# ImageNet normalization (RGB), for inputs in [0, 1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

PRECISIONS = {trt.DataType.FLOAT: "FP32", trt.DataType.HALF: "FP16", trt.DataType.INT8: "INT8"}
TORCH_DTYPES = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}


class RunnerTRT:
    def __init__(self, settings: Settings) -> None:
        self.model_file: str = settings.model_path + '/' + settings.model
        self.model_width: int = settings.width
        self.model_height: int = settings.height
        self.max_batch: int = 8
        self.model_precision: str = "UNKNOWN"

        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._stream: torch.cuda.Stream
        self._input_name: str = ''
        self._output_names: list[str] = []
        self._torch_dtype: torch.dtype = torch.float32
        self._mean_gpu: torch.Tensor
        self._std_gpu: torch.Tensor
        self._input_buffer: torch.Tensor     # (max_batch, 3, H, W) normalized input
        self._resize_buffer: torch.Tensor    # (max_batch, 3, H, W) before normalization

    def setup(self) -> None:
        """Load the engine, on the thread that will run it."""
        with get_init_lock():       # one Myelin graph loads at a time
            runtime = get_tensorrt_runtime()
            with open(self.model_file, 'rb') as f:
                engine_data = f.read()
            self._engine = runtime.deserialize_cuda_engine(engine_data)
            if self._engine is None:
                raise RuntimeError(f"Failed to load engine {self.model_file}")
            self._context = self._engine.create_execution_context()

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_names.append(name)
        if len(self._output_names) != 2:
            raise RuntimeError(f"Expected 2 outputs, got {len(self._output_names)}")

        input_dtype = self._engine.get_tensor_dtype(self._input_name)
        self.model_precision = PRECISIONS.get(input_dtype, "UNKNOWN")
        self._torch_dtype = TORCH_DTYPES.get(input_dtype, torch.float32)

        self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            self._mean_gpu = IMAGENET_MEAN.to(device='cuda', dtype=self._torch_dtype)
            self._std_gpu = IMAGENET_STD.to(device='cuda', dtype=self._torch_dtype)
            shape = (self.max_batch, 3, self.model_height, self.model_width)
            self._input_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
            self._resize_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
        self._stream.synchronize()

        # The input's address stays (a slice keeps the base pointer); the outputs' are set per call
        self._context.set_tensor_address(self._input_name, self._input_buffer.data_ptr())

    def infer(self, gpu_images: list[torch.Tensor], ready: torch.cuda.Event | None) -> tuple[np.ndarray, np.ndarray, float]:
        """The SimCC outputs (N, K, W) and (N, K, H) for the crops (3, H, W) RGB in [0, 1], read once `ready` is
        recorded; and the time waited for the shared execution lock (ms)."""
        if ready is not None:
            self._stream.wait_event(ready)      # the crops are written on the crop stream
        batch_size = len(gpu_images)
        input_h, input_w = gpu_images[0].shape[1], gpu_images[0].shape[2]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)
        input_buffer = self._input_buffer[:batch_size]

        simcc_x_out = torch.empty((batch_size, NUM_KEYPOINTS, self.model_width * 2), dtype=self._torch_dtype, device='cuda')
        simcc_y_out = torch.empty((batch_size, NUM_KEYPOINTS, self.model_height * 2), dtype=self._torch_dtype, device='cuda')

        with torch.cuda.stream(self._stream):
            batch_chw = torch.stack(gpu_images, dim=0).to(self._torch_dtype)
            if needs_resize:
                batch_chw = torch.nn.functional.interpolate(
                    batch_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False)
            torch.sub(batch_chw, self._mean_gpu, out=self._resize_buffer[:batch_size])
            torch.div(self._resize_buffer[:batch_size], self._std_gpu, out=input_buffer)

        lock_wait_start: float = time.perf_counter()
        with get_exec_lock():
            lock_acquired: float = time.perf_counter()
            self._context.set_input_shape(self._input_name, tuple(input_buffer.shape))
            self._context.set_tensor_address(self._output_names[0], simcc_x_out.data_ptr())
            self._context.set_tensor_address(self._output_names[1], simcc_y_out.data_ptr())
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            self._stream.synchronize()

        return simcc_x_out.cpu().numpy(), simcc_y_out.cpu().numpy(), (lock_acquired - lock_wait_start) * 1000.0
