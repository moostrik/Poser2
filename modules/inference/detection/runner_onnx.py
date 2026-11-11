"""The person detector's model through ONNX Runtime on CUDA, its input and output bound on the GPU."""

import logging

import numpy as np
import onnxruntime as ort
import torch

from ..ort_runtime import preload_cuda
from .decode import PERSON_ROWS

logger = logging.getLogger(__name__)

ELEMENT_TYPES: dict[str, type] = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}


class RunnerONNX:
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ''
        self._output_name: str = ''
        self._element_type: type = np.float32
        self._dtype: torch.dtype = torch.float32
        self._output_shape: tuple[int, int] = (0, 0)    # per image: 4 + classes, candidates

    def setup(self) -> None:
        """Load the model, on the thread that will run it. It runs on CUDA only: the batch is on the GPU."""
        preload_cuda()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(self.model_file, options, providers=['CUDAExecutionProvider'])
        if self._session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA is not available to ONNX Runtime')
        model_input = self._session.get_inputs()[0]
        model_output = self._session.get_outputs()[0]
        self._input_name = model_input.name
        self._output_name = model_output.name
        self._element_type = ELEMENT_TYPES[model_input.type]
        self._dtype = torch.from_numpy(np.empty(0, dtype=self._element_type)).dtype
        rows, candidates = model_output.shape[1], model_output.shape[2]
        assert isinstance(rows, int) and isinstance(candidates, int), f'the output needs a fixed size per image, not {model_output.shape}'
        self._output_shape = (rows, candidates)

    def infer(self, batch: torch.Tensor, stream: torch.cuda.Stream) -> np.ndarray:
        """The person rows (N, `PERSON_ROWS`, candidates) of an RGB batch (N, 3, height, width) in [0, 1] on the
        GPU, written on `stream`; only those rows come back to the CPU."""
        assert self._session is not None
        with torch.cuda.stream(stream):
            model_input: torch.Tensor = batch.to(self._dtype).contiguous()
            output: torch.Tensor = torch.empty((batch.shape[0], *self._output_shape), dtype=self._dtype, device='cuda')
        stream.synchronize()        # ONNX Runtime runs on its own stream: the input must be written first
        binding = self._session.io_binding()
        binding.bind_input(name=self._input_name, device_type='cuda', device_id=0, element_type=self._element_type,
                           shape=tuple(model_input.shape), buffer_ptr=model_input.data_ptr())
        binding.bind_output(name=self._output_name, device_type='cuda', device_id=0, element_type=self._element_type,
                            shape=tuple(output.shape), buffer_ptr=output.data_ptr())
        self._session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        with torch.cuda.stream(stream):
            return output[:, :PERSON_ROWS].float().cpu().numpy()
