"""A model through TensorRT: its engine file deserialized by the shared runtime, its own execution context run on the
worker's stream (`execute_async_v3`), the outputs made on that stream at the shapes the inputs give."""

import logging

import numpy as np
import tensorrt as trt
import torch

from .engine import Engine
from .trt_runtime import get_tensorrt_runtime

logger = logging.getLogger(__name__)

PRECISIONS = {trt.DataType.FLOAT: "FP32", trt.DataType.HALF: "FP16", trt.DataType.INT8: "INT8"}


class TrtEngine(Engine):
    def __init__(self, model_file: str) -> None:
        super().__init__(model_file)
        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._stream: torch.cuda.Stream

    def setup(self, stream: torch.cuda.Stream) -> None:
        with open(self.model_file, 'rb') as f:
            engine_data = f.read()
        self._engine = get_tensorrt_runtime().deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine {self.model_file}")
        self._context = self._engine.create_execution_context()
        self._stream = stream
        for i in range(self._engine.num_io_tensors):
            name: str = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self.precision = PRECISIONS.get(self._engine.get_tensor_dtype(self.input_names[0]), "UNKNOWN")

    def dtype(self, name: str) -> torch.dtype:
        return torch.from_numpy(np.empty(0, dtype=trt.nptype(self._engine.get_tensor_dtype(name)))).dtype

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(self._stream):
            contiguous: dict[str, torch.Tensor] = {name: tensor.contiguous() for name, tensor in inputs.items()}
            for name, tensor in contiguous.items():
                self._context.set_input_shape(name, tuple(tensor.shape))
                self._context.set_tensor_address(name, tensor.data_ptr())
            for name in self.output_names:      # made on the stream that writes them
                shape: tuple[int, ...] = tuple(self._context.get_tensor_shape(name))
                outputs[name] = torch.empty(shape, dtype=self.dtype(name), device='cuda')
                self._context.set_tensor_address(name, outputs[name].data_ptr())
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        return outputs
