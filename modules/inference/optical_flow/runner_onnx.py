"""RAFT through ONNX Runtime on CUDA: the previous and current crops into preallocated batches on its stream, the flow
out on the GPU. The batch is padded to `max_batch`, the size it is warmed up at."""

import logging

import numpy as np
import onnxruntime as ort
import torch

from ..ort_runtime import preload_cuda
from .settings import OpticalFlowPredictorSettings

logger = logging.getLogger(__name__)

PRECISIONS: dict[str, str] = {'tensor(float)': 'FP32', 'tensor(float16)': 'FP16', 'tensor(double)': 'FP64', 'tensor(int8)': 'INT8'}
DTYPES: dict[str, type] = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(double)': np.float64, 'tensor(int8)': np.int8}


class RunnerONNX:
    def __init__(self, settings: OpticalFlowPredictorSettings) -> None:
        self.model_file: str = f"{settings.model_path}/{settings.model}"
        self.model_width: int = settings.width
        self.model_height: int = settings.height
        self.max_batch: int = min(settings.max_players, 8)
        self.model_precision: str = "UNKNOWN"

        self._session: ort.InferenceSession | None = None
        self._model_dtype: type = np.float32
        self._torch_dtype: torch.dtype = torch.float32
        self._stream: torch.cuda.Stream
        self._img1_buffer: torch.Tensor     # (max_batch, 3, H, W) the previous crops
        self._img2_buffer: torch.Tensor     # (max_batch, 3, H, W) the current crops

    def setup(self) -> None:
        """Load the model and warm it up, on the thread that will run it. It runs on CUDA only: the input is on the GPU."""
        preload_cuda()
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1
        sess_options.log_severity_level = 3
        self._session = ort.InferenceSession(self.model_file, sess_options=sess_options, providers=providers)
        if 'CUDAExecutionProvider' not in self._session.get_providers():
            raise RuntimeError("CUDA provider not available: the runner binds CUDA tensors, the CPU provider cannot take them")

        input_type: str = self._session.get_inputs()[0].type
        self.model_precision = PRECISIONS.get(input_type, f"UNKNOWN({input_type})")
        self._model_dtype = DTYPES.get(input_type, np.float32)
        self._torch_dtype = {np.float32: torch.float32, np.float16: torch.float16}.get(self._model_dtype, torch.float32)

        self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            shape = (self.max_batch, 3, self.model_height, self.model_width)
            self._img1_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
            self._img2_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
        self._stream.synchronize()
        self._warmup()

    def infer(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], ready: torch.cuda.Event | None) -> torch.Tensor:
        """The flow (N, 2, H, W), x and y in the model's pixels, from each previous to current crop (3, h, w) RGB in
        [0, 1], read once `ready` is recorded."""
        if ready is not None:
            self._stream.wait_event(ready)      # the crops are written on the crop stream
        batch_size = len(pairs)
        input_h, input_w = pairs[0][0].shape[1], pairs[0][0].shape[2]
        h, w = self.model_height, self.model_width
        with torch.cuda.stream(self._stream):       # the output made on the stream that waits for it
            flow_out = torch.empty((self.max_batch, 2, h, w), dtype=self._torch_dtype, device='cuda')
            prev_chw = torch.stack([p[0] for p in pairs], dim=0).mul(255.0).to(self._torch_dtype)     # RAFT reads 0..255
            curr_chw = torch.stack([p[1] for p in pairs], dim=0).mul(255.0).to(self._torch_dtype)
            if input_h != h or input_w != w:
                prev_chw = torch.nn.functional.interpolate(prev_chw, size=(h, w), mode='bilinear', align_corners=False)
                curr_chw = torch.nn.functional.interpolate(curr_chw, size=(h, w), mode='bilinear', align_corners=False)
            self._img1_buffer[:batch_size].copy_(prev_chw)
            self._img2_buffer[:batch_size].copy_(curr_chw)
            if batch_size < self.max_batch:     # padded to max_batch, the shape it is warmed up at
                self._img1_buffer[batch_size:].zero_()
                self._img2_buffer[batch_size:].zero_()
        self._stream.synchronize()      # ONNX Runtime runs on its own stream: the input must be written first

        assert self._session is not None
        io_binding = self._session.io_binding()
        input_names = [i.name for i in self._session.get_inputs()]
        for name, tensor in zip(input_names, (self._img1_buffer, self._img2_buffer)):
            io_binding.bind_input(name=name, device_type='cuda', device_id=0, element_type=self._model_dtype,
                                  shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())
        io_binding.bind_output(name=self._session.get_outputs()[0].name, device_type='cuda', device_id=0,
                               element_type=self._model_dtype, shape=tuple(flow_out.shape), buffer_ptr=flow_out.data_ptr())
        self._session.run_with_iobinding(io_binding)
        self._stream.synchronize()
        return flow_out[:batch_size]

    def _warmup(self) -> None:
        """The CUDA kernels made for the fixed batch size, so none is compiled while running."""
        try:
            dummy = torch.zeros((3, self.model_height, self.model_width), dtype=torch.float32, device='cuda')
            self.infer([(dummy, dummy)] * self.max_batch, None)
        except Exception as e:
            logger.info(f"Optical Flow: Warmup failed (non-critical) - {str(e)}")
