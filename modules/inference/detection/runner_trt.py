"""The person detector's model through TensorRT, one engine per input size. It takes turns with the
other TensorRT models on the shared execution lock."""

import numpy as np
import tensorrt as trt
import torch

from ..trt_runtime import get_exec_lock, get_init_lock, get_tensorrt_runtime


class RunnerTRT:
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._stream: torch.cuda.Stream
        self._input_name: str = ''
        self._output_name: str = ''
        self._output_shape: tuple[int, int] = (0, 0)    # per image: 4 + classes, candidates

    def setup(self) -> None:
        """Load the engine, on the thread that will run it."""
        with open(self.model_file, 'rb') as f:
            engine_data: bytes = f.read()
        with get_init_lock():
            self._engine = get_tensorrt_runtime().deserialize_cuda_engine(engine_data)
            self._context = self._engine.create_execution_context()
        for i in range(self._engine.num_io_tensors):
            name: str = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        shape = self._engine.get_tensor_shape(self._output_name)
        self._output_shape = (shape[1], shape[2])
        self._stream = torch.cuda.Stream()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """The output (N, 4 + classes, candidates) of a float32 batch (N, 3, height, width) in C order."""
        with torch.cuda.stream(self._stream):
            gpu_batch: torch.Tensor = torch.from_numpy(batch).to('cuda')
            output: torch.Tensor = torch.empty((batch.shape[0], *self._output_shape), dtype=gpu_batch.dtype, device='cuda')
        with get_exec_lock():
            self._context.set_input_shape(self._input_name, tuple(gpu_batch.shape))
            self._context.set_tensor_address(self._input_name, gpu_batch.data_ptr())
            self._context.set_tensor_address(self._output_name, output.data_ptr())
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            self._stream.synchronize()
        return output.cpu().numpy()
