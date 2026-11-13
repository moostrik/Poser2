"""A face model through ONNX Runtime on CUDA: its input bound on the GPU, all its outputs back to the CPU in the
model's order."""

import logging

import numpy as np
import onnxruntime as ort
import torch

from ..ort_runtime import preload_cuda

logger = logging.getLogger(__name__)


class RunnerONNX:
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ''
        self._output_names: list[str] = []

    def setup(self) -> None:
        """Load the model, on the thread that will run it. It runs on CUDA only: the input is on the GPU."""
        preload_cuda()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(self.model_file, options, providers=['CUDAExecutionProvider'])
        if self._session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA is not available to ONNX Runtime')
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

    def infer(self, batch: torch.Tensor, stream: torch.cuda.Stream) -> list[np.ndarray]:
        """The model's outputs for a float32 batch on the GPU, written on `stream`."""
        assert self._session is not None
        with torch.cuda.stream(stream):
            model_input: torch.Tensor = batch.float().contiguous()
        stream.synchronize()        # ONNX Runtime runs on its own stream: the input must be written first
        binding = self._session.io_binding()
        binding.bind_input(name=self._input_name, device_type='cuda', device_id=0, element_type=np.float32,
                           shape=tuple(model_input.shape), buffer_ptr=model_input.data_ptr())
        for name in self._output_names:
            binding.bind_output(name, 'cuda')
        self._session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()
