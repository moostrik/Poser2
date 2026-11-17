"""RTMPose through ONNX Runtime on CUDA: the crops normalized into a preallocated batch on its stream, the SimCC
outputs back to the CPU. The batch is padded to `max_batch`, the size it is warmed up at."""

import logging

import numpy as np
import onnxruntime as ort
import torch

from ..ort_runtime import preload_cuda
from .decode import NUM_KEYPOINTS
from .settings import Settings

logger = logging.getLogger(__name__)

# ImageNet normalization (RGB), for inputs in [0, 1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

PRECISIONS: dict[str, str] = {'tensor(float)': 'FP32', 'tensor(float16)': 'FP16', 'tensor(double)': 'FP64', 'tensor(int8)': 'INT8'}
DTYPES: dict[str, type] = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(double)': np.float64, 'tensor(int8)': np.int8}


class RunnerONNX:
    def __init__(self, settings: Settings) -> None:
        self.model_file: str = settings.model_path + '/' + settings.model
        self.model_width: int = settings.width
        self.model_height: int = settings.height
        self.max_batch: int = min(settings.max_players, 8)
        self.model_precision: str = "UNKNOWN"

        self._session: ort.InferenceSession | None = None
        self._model_dtype: type = np.float32
        self._torch_dtype: torch.dtype = torch.float32
        self._stream: torch.cuda.Stream
        self._mean_gpu: torch.Tensor
        self._std_gpu: torch.Tensor
        self._input_buffer: torch.Tensor     # (max_batch, 3, H, W) normalized input
        self._resize_buffer: torch.Tensor    # (max_batch, 3, H, W) before normalization

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
        self._session = ort.InferenceSession(self.model_file, sess_options, providers=providers)
        if 'CUDAExecutionProvider' not in self._session.get_providers():
            raise RuntimeError("CUDA provider not available: the runner binds CUDA tensors, the CPU provider cannot take them")

        input_type: str = self._session.get_inputs()[0].type
        self.model_precision = PRECISIONS.get(input_type, f"UNKNOWN({input_type})")
        self._model_dtype = DTYPES.get(input_type, np.float32)
        self._torch_dtype = {np.float32: torch.float32, np.float16: torch.float16}.get(self._model_dtype, torch.float32)

        self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            self._mean_gpu = IMAGENET_MEAN.to(device='cuda', dtype=self._torch_dtype)
            self._std_gpu = IMAGENET_STD.to(device='cuda', dtype=self._torch_dtype)
            shape = (self.max_batch, 3, self.model_height, self.model_width)
            self._input_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
            self._resize_buffer = torch.empty(shape, dtype=self._torch_dtype, device='cuda')
        self._stream.synchronize()
        self._warmup()

    def infer(self, gpu_images: list[torch.Tensor], ready: torch.cuda.Event | None) -> tuple[np.ndarray, np.ndarray, float]:
        """The SimCC outputs (N, K, W) and (N, K, H) for the crops (3, H, W) RGB in [0, 1], read once `ready` is
        recorded; and the time waited for the GPU's lock, 0: ONNX Runtime takes none."""
        if ready is not None:
            self._stream.wait_event(ready)      # the crops are written on the crop stream
        simcc_x, simcc_y = self._infer(gpu_images)
        return simcc_x, simcc_y, 0.0

    def _infer(self, gpu_images: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        batch_size = len(gpu_images)
        if batch_size == 0 or self._session is None:
            return np.empty((0, NUM_KEYPOINTS, 0)), np.empty((0, NUM_KEYPOINTS, 0))

        input_h, input_w = gpu_images[0].shape[1], gpu_images[0].shape[2]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)
        input_buffer = self._input_buffer[:self.max_batch]

        with torch.cuda.stream(self._stream):
            batch_chw = torch.stack(gpu_images, dim=0).to(self._torch_dtype)
            if needs_resize:
                batch_chw = torch.nn.functional.interpolate(
                    batch_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False)
            # Padded to max_batch: the shape it is warmed up at
            if batch_size < self.max_batch:
                padding = torch.zeros((self.max_batch - batch_size, 3, self.model_height, self.model_width),
                                      dtype=self._torch_dtype, device='cuda')
                batch_chw = torch.cat([batch_chw, padding], dim=0)
            torch.sub(batch_chw, self._mean_gpu, out=self._resize_buffer)
            torch.div(self._resize_buffer, self._std_gpu, out=input_buffer)

        self._stream.synchronize()      # ONNX Runtime runs on its own stream: the input must be written first

        input_name = self._session.get_inputs()[0].name
        output_names = [o.name for o in self._session.get_outputs()]
        io_binding = self._session.io_binding()
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=self._model_dtype,
                              shape=tuple(input_buffer.shape), buffer_ptr=input_buffer.data_ptr())
        io_binding.bind_output(output_names[0], device_type='cuda', device_id=0)
        io_binding.bind_output(output_names[1], device_type='cuda', device_id=0)
        self._session.run_with_iobinding(io_binding)
        self._stream.synchronize()

        outputs = io_binding.copy_outputs_to_cpu()
        return outputs[0][:batch_size], outputs[1][:batch_size]

    def _warmup(self) -> None:
        """The CUDA kernels made for the fixed batch size, so none is compiled while running."""
        if self._session is None:
            return
        try:
            dummy_img = torch.zeros((self.model_height, self.model_width, 3), dtype=torch.float32, device='cuda')
            self._infer([dummy_img] * self.max_batch)
        except Exception as e:
            logger.info(f"Pose: Warmup failed (non-critical) - {str(e)}")
