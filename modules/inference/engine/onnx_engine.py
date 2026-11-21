"""A model through ONNX Runtime's CUDA provider, run on the worker's stream (`user_compute_stream`), its inputs and
outputs bound on the GPU and the outputs handed on as torch tensors without a copy (DLPack). Given `max_batch`, every
input's batch is padded to it, the shape the model is warmed up at, and the outputs cut back."""

import logging

import numpy as np
import onnxruntime as ort
import torch

from .engine import Engine
from .ort_runtime import preload_cuda

logger = logging.getLogger(__name__)

TYPES: dict[str, tuple[type, torch.dtype]] = {
    'tensor(float)': (np.float32, torch.float32), 'tensor(float16)': (np.float16, torch.float16),
    'tensor(double)': (np.float64, torch.float64), 'tensor(int8)': (np.int8, torch.int8)}


class OnnxEngine(Engine):
    def __init__(self, model_file: str, max_batch: int | None = None) -> None:
        super().__init__(model_file)
        self.max_batch: int | None = max_batch
        self._session: ort.InferenceSession | None = None
        self._types: dict[str, str] = {}
        self._stream: torch.cuda.Stream

    def setup(self, stream: torch.cuda.Stream) -> None:
        self._stream = stream
        preload_cuda()
        providers = [('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': False,
            'has_user_compute_stream': True,
            'user_compute_stream': str(stream.cuda_stream),
        })]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(self.model_file, options, providers=providers)
        if self._session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError("CUDA is not available to ONNX Runtime: the engine binds CUDA tensors")
        self.input_names = [i.name for i in self._session.get_inputs()]
        self.output_names = [o.name for o in self._session.get_outputs()]
        self._types = {i.name: i.type for i in self._session.get_inputs()} | {o.name: o.type for o in self._session.get_outputs()}

    def dtype(self, name: str) -> torch.dtype:
        return TYPES[self._types[name]][1]

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self._session is not None
        batch: int = next(iter(inputs.values())).shape[0]
        padding: int = (self.max_batch - batch) if self.max_batch is not None and batch < self.max_batch else 0
        binding = self._session.io_binding()
        bound: list[torch.Tensor] = []         # kept alive until the run is queued
        with torch.cuda.stream(self._stream):
            for name, tensor in inputs.items():
                if padding:
                    tensor = torch.cat([tensor, tensor.new_zeros((padding, *tensor.shape[1:]))])
                tensor = tensor.contiguous()
                bound.append(tensor)
                binding.bind_input(name=name, device_type='cuda', device_id=0, element_type=TYPES[self._types[name]][0],
                                   shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())
        for name in self.output_names:
            binding.bind_output(name, 'cuda')
        self._session.run_with_iobinding(binding)
        outputs: dict[str, torch.Tensor] = {name: torch.from_dlpack(value)
                                            for name, value in zip(self.output_names, binding.get_outputs())}
        return {name: tensor[:batch] for name, tensor in outputs.items()} if padding else outputs
