"""Robust Video Matting through ONNX Runtime on CUDA: the crops and each player's recurrent states into a
preallocated batch on its stream, the alpha matte and the model's foreground out on the GPU. The batch is padded to
`max_batch`, the size it is warmed up at."""

import logging

import numpy as np
import onnxruntime as ort
import torch

from ..ort_runtime import preload_cuda
from .recurrent_state import RecurrentState
from .settings import SegmentationPredictorSettings

logger = logging.getLogger(__name__)

PRECISIONS: dict[str, str] = {'tensor(float)': 'FP32', 'tensor(float16)': 'FP16', 'tensor(double)': 'FP64', 'tensor(int8)': 'INT8'}
DTYPES: dict[str, type] = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(double)': np.float64, 'tensor(int8)': np.int8}
RECURRENT_CHANNELS: tuple[tuple[int, int], ...] = ((16, 2), (20, 4), (40, 8), (64, 16))     # (channels, downscale) of r1..r4


class RunnerONNX:
    def __init__(self, settings: SegmentationPredictorSettings) -> None:
        self.model_file: str = f"{settings.model_path}/{settings.model}"
        self.model_width: int = settings.width
        self.model_height: int = settings.height
        self.max_batch: int = min(settings.max_players, 4)
        self.model_precision: str = "UNKNOWN"

        self._session: ort.InferenceSession | None = None
        self._model_dtype: type = np.float16
        self._torch_dtype: torch.dtype = torch.float16
        self._stream: torch.cuda.Stream
        self._states: dict[int, RecurrentState] = {}    # per track id
        self._zero_states: list[torch.Tensor] = []      # (1, C, h, w) per recurrent input, for a new player
        self._src_buffer: torch.Tensor                  # (max_batch, 3, H, W)
        self._state_buffers: list[torch.Tensor] = []    # (max_batch, C, h, w) per recurrent input

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

        input_type: str = self._session.get_inputs()[0].type        # 'src'
        self.model_precision = PRECISIONS.get(input_type, f"UNKNOWN({input_type})")
        self._model_dtype = DTYPES.get(input_type, np.float16)
        self._torch_dtype = {np.float32: torch.float32, np.float16: torch.float16}.get(self._model_dtype, torch.float16)

        self._stream = torch.cuda.Stream()
        h, w = self.model_height, self.model_width
        with torch.cuda.stream(self._stream):
            self._zero_states = [torch.zeros((1, c, h // s, w // s), dtype=self._torch_dtype, device='cuda') for c, s in RECURRENT_CHANNELS]
            self._src_buffer = torch.empty((self.max_batch, 3, h, w), dtype=self._torch_dtype, device='cuda')
            self._state_buffers = [torch.empty((self.max_batch, c, h // s, w // s), dtype=self._torch_dtype, device='cuda') for c, s in RECURRENT_CHANNELS]
        self._stream.synchronize()
        self._warmup()

    def reset_states(self) -> None:
        """Every player starts again without memory."""
        self._states.clear()

    def infer(self, gpu_images: list[torch.Tensor], track_ids: list[int],
              ready: torch.cuda.Event | None) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        """The alpha matte (N, H, W) and the foreground (N, 3, H, W) at the model's size for the crops (3, h, w) RGB in
        [0, 1] of the players `track_ids`, read once `ready` is recorded; and the event recorded after they are written."""
        if ready is not None:
            self._stream.wait_event(ready)      # the crops are written on the crop stream
        batch_size = len(gpu_images)
        input_h, input_w = gpu_images[0].shape[1], gpu_images[0].shape[2]
        h, w = self.model_height, self.model_width

        # Padded to max_batch, the shape it is warmed up at, with ids no player has
        images: list[torch.Tensor] = list(gpu_images)
        states: list[RecurrentState | None] = [self._states.get(track_id) for track_id in track_ids]
        if batch_size < self.max_batch:
            images += [torch.zeros((3, input_h, input_w), dtype=torch.float32, device='cuda')] * (self.max_batch - batch_size)
            states += [None] * (self.max_batch - batch_size)

        with torch.cuda.stream(self._stream):       # the outputs made on the stream that waits for them
            fgr_out = torch.empty((self.max_batch, 3, h, w), dtype=self._torch_dtype, device='cuda')
            pha_out = torch.empty((self.max_batch, 1, h, w), dtype=self._torch_dtype, device='cuda')
            state_outs = [torch.empty((self.max_batch, c, h // s, w // s), dtype=self._torch_dtype, device='cuda') for c, s in RECURRENT_CHANNELS]
            batch_chw = torch.stack(images, dim=0).to(self._torch_dtype)
            if input_h != h or input_w != w:
                batch_chw = torch.nn.functional.interpolate(batch_chw, size=(h, w), mode='bilinear', align_corners=False)
            self._src_buffer.copy_(batch_chw)
            for i, state in enumerate(states):
                sources = (state.r1, state.r2, state.r3, state.r4) if state is not None else self._zero_states
                for buffer, source in zip(self._state_buffers, sources):
                    buffer[i:i + 1].copy_(source)
        self._stream.synchronize()      # ONNX Runtime runs on its own stream: the input must be written first

        assert self._session is not None
        io_binding = self._session.io_binding()
        for name, tensor in zip(('src', 'r1i', 'r2i', 'r3i', 'r4i'), [self._src_buffer, *self._state_buffers]):
            io_binding.bind_input(name, device_type='cuda', device_id=0, element_type=self._model_dtype,
                                  shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())
        for name, tensor in zip(('fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'), [fgr_out, pha_out, *state_outs]):
            io_binding.bind_output(name, device_type='cuda', device_id=0, element_type=self._model_dtype,
                                   shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())
        self._session.run_with_iobinding(io_binding)
        self._stream.synchronize()

        # The new states are views into this call's outputs, for the players only
        r1o, r2o, r3o, r4o = state_outs
        self._states = {track_id: RecurrentState(r1o[i:i + 1], r2o[i:i + 1], r3o[i:i + 1], r4o[i:i + 1])
                        for i, track_id in enumerate(track_ids)}
        written: torch.cuda.Event = torch.cuda.Event()
        written.record(self._stream)
        return pha_out[:batch_size].squeeze(1), fgr_out[:batch_size], written

    def _warmup(self) -> None:
        """The CUDA kernels made for the fixed batch size, so none is compiled while running."""
        try:
            dummy = torch.zeros((3, self.model_height, self.model_width), dtype=torch.float32, device='cuda')
            self.infer([dummy] * self.max_batch, list(range(self.max_batch)), None)
        except Exception as e:
            logger.info(f"Segmentation: Warmup failed (non-critical) - {str(e)}")
        self._states.clear()
