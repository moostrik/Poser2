"""The person detector's model through TensorRT, one engine per input size. It takes turns with the
other TensorRT models on the shared execution lock."""

import numpy as np
import tensorrt as trt
import torch

from ..trt_runtime import get_exec_lock, get_init_lock, get_tensorrt_runtime
from .decode import PERSON_ROWS


class RunnerTRT:
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._input_name: str = ''
        self._output_name: str = ''
        self._input_dtype: torch.dtype = torch.float32
        self._output_dtype: torch.dtype = torch.float32
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
        self._input_dtype = torch_dtype(self._engine.get_tensor_dtype(self._input_name))
        self._output_dtype = torch_dtype(self._engine.get_tensor_dtype(self._output_name))
        shape = self._engine.get_tensor_shape(self._output_name)
        self._output_shape = (shape[1], shape[2])

    def infer(self, batch: torch.Tensor, stream: torch.cuda.Stream) -> np.ndarray:
        """The person rows (N, `PERSON_ROWS`, candidates) of an RGB batch (N, 3, height, width) in [0, 1] on the
        GPU, written on `stream`; only those rows come back to the CPU."""
        with torch.cuda.stream(stream):
            model_input: torch.Tensor = batch.to(self._input_dtype).contiguous()
            output: torch.Tensor = torch.empty((batch.shape[0], *self._output_shape), dtype=self._output_dtype, device='cuda')
        with get_exec_lock():
            self._context.set_input_shape(self._input_name, tuple(model_input.shape))
            self._context.set_tensor_address(self._input_name, model_input.data_ptr())
            self._context.set_tensor_address(self._output_name, output.data_ptr())
            self._context.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()
        with torch.cuda.stream(stream):
            return output[:, :PERSON_ROWS].float().cpu().numpy()


def torch_dtype(dtype: trt.DataType) -> torch.dtype:
    """The torch type of a TensorRT tensor type."""
    return torch.from_numpy(np.empty(0, dtype=trt.nptype(dtype))).dtype
