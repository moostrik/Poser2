"""A face model through TensorRT: all its outputs back to the CPU in the engine's order, the ONNX model's. It takes
turns with the other TensorRT models on the shared execution lock."""

import numpy as np
import tensorrt as trt
import torch

from ..trt_runtime import get_exec_lock, get_init_lock, get_tensorrt_runtime


class RunnerTRT:
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._input_name: str = ''
        self._output_names: list[str] = []

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
                self._output_names.append(name)

    def infer(self, batch: torch.Tensor, stream: torch.cuda.Stream) -> list[np.ndarray]:
        """The model's outputs for a float32 batch on the GPU, written on `stream`."""
        with torch.cuda.stream(stream):
            model_input: torch.Tensor = batch.float().contiguous()
        with get_exec_lock():
            self._context.set_input_shape(self._input_name, tuple(model_input.shape))
            self._context.set_tensor_address(self._input_name, model_input.data_ptr())
            outputs: list[torch.Tensor] = []
            with torch.cuda.stream(stream):
                for name in self._output_names:
                    shape: tuple[int, ...] = tuple(self._context.get_tensor_shape(name))
                    dtype: torch.dtype = torch.from_numpy(np.empty(0, dtype=trt.nptype(self._engine.get_tensor_dtype(name)))).dtype
                    output: torch.Tensor = torch.empty(shape, dtype=dtype, device='cuda')
                    self._context.set_tensor_address(name, output.data_ptr())
                    outputs.append(output)
            self._context.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()
        return [output.float().cpu().numpy() for output in outputs]
