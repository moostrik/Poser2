"""Robust Video Matting through TensorRT: the crops and each player's recurrent states into a preallocated batch on
its stream; out on the GPU the alpha matte at the crops' size and the crops multiplied by it."""

import logging

import tensorrt as trt
import torch

from ..trt_runtime import get_tensorrt_runtime
from .recurrent_state import RecurrentState
from .settings import SegmentationPredictorSettings

logger = logging.getLogger(__name__)

PRECISIONS = {trt.DataType.FLOAT: "FP32", trt.DataType.HALF: "FP16", trt.DataType.INT8: "INT8"}
TORCH_DTYPES = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
RECURRENT_CHANNELS: tuple[tuple[int, int], ...] = ((16, 2), (20, 4), (40, 8), (64, 16))     # (channels, downscale) of r1..r4
INPUT_NAMES: tuple[str, ...] = ('src', 'r1i', 'r2i', 'r3i', 'r4i')
OUTPUT_NAMES: tuple[str, ...] = ('fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o')


class RunnerTRT:
    def __init__(self, settings: SegmentationPredictorSettings) -> None:
        self.model_file: str = f"{settings.model_path}/{settings.model}"
        self.model_width: int = settings.width
        self.model_height: int = settings.height
        self.max_batch: int = min(settings.max_players, 4)
        self.model_precision: str = "UNKNOWN"

        self._engine: trt.ICudaEngine
        self._context: trt.IExecutionContext
        self._stream: torch.cuda.Stream
        self._torch_dtype: torch.dtype = torch.float16
        self._states: dict[int, RecurrentState] = {}    # per track id
        self._zero_states: list[torch.Tensor] = []      # (1, C, h, w) per recurrent input, for a new player
        self._src_buffer: torch.Tensor                  # (max_batch, 3, H, W)
        self._state_buffers: list[torch.Tensor] = []    # (max_batch, C, h, w) per recurrent input

    def setup(self) -> None:
        """Load the engine, on the thread that will run it."""
        with open(self.model_file, 'rb') as f:
            engine_data = f.read()
        self._engine = get_tensorrt_runtime().deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine {self.model_file}")
        self._context = self._engine.create_execution_context()

        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        inputs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        if len(inputs) != len(INPUT_NAMES) or len(names) - len(inputs) != len(OUTPUT_NAMES):
            raise RuntimeError(f"Expected 5 inputs and 6 outputs, got {len(inputs)} and {len(names) - len(inputs)}")

        src_dtype = self._engine.get_tensor_dtype('src')
        self.model_precision = PRECISIONS.get(src_dtype, "UNKNOWN")
        self._torch_dtype = TORCH_DTYPES.get(src_dtype, torch.float16)

        self._stream = torch.cuda.Stream()
        h, w = self.model_height, self.model_width
        with torch.cuda.stream(self._stream):
            self._zero_states = [torch.zeros((1, c, h // s, w // s), dtype=self._torch_dtype, device='cuda') for c, s in RECURRENT_CHANNELS]
            self._src_buffer = torch.empty((self.max_batch, 3, h, w), dtype=self._torch_dtype, device='cuda')
            self._state_buffers = [torch.empty((self.max_batch, c, h // s, w // s), dtype=self._torch_dtype, device='cuda') for c, s in RECURRENT_CHANNELS]
        self._stream.synchronize()

        # The inputs' addresses stay (a slice keeps the base pointer); the outputs' are set per call
        for name, tensor in zip(INPUT_NAMES, [self._src_buffer, *self._state_buffers]):
            self._context.set_tensor_address(name, tensor.data_ptr())

    def reset_states(self) -> None:
        """Every player starts again without memory."""
        self._states.clear()

    def infer(self, gpu_images: list[torch.Tensor], track_ids: list[int],
              ready: torch.cuda.Event | None) -> tuple[torch.Tensor, torch.Tensor]:
        """The alpha matte (N, h, w) and the foreground (N, 3, h, w), the crops multiplied by it, for the crops (3, h, w)
        RGB in [0, 1] of the players `track_ids`, read once `ready` is recorded."""
        if ready is not None:
            self._stream.wait_event(ready)      # the crops are written on the crop stream
        batch_size = len(gpu_images)
        input_h, input_w = gpu_images[0].shape[1], gpu_images[0].shape[2]
        h, w = self.model_height, self.model_width
        needs_resize = (input_h != h or input_w != w)
        src_buffer = self._src_buffer[:batch_size]
        state_buffers = [buffer[:batch_size] for buffer in self._state_buffers]

        outputs = [torch.empty((batch_size, 3, h, w), dtype=self._torch_dtype, device='cuda'),
                   torch.empty((batch_size, 1, h, w), dtype=self._torch_dtype, device='cuda')]
        outputs += [torch.empty((batch_size, c, h // s, w // s), dtype=self._torch_dtype, device='cuda') for c, s in RECURRENT_CHANNELS]

        with torch.cuda.stream(self._stream):
            batch_chw = torch.stack(gpu_images, dim=0).to(self._torch_dtype)
            original = batch_chw
            if needs_resize:
                batch_chw = torch.nn.functional.interpolate(batch_chw, size=(h, w), mode='bilinear', align_corners=False)
            src_buffer.copy_(batch_chw)
            for i, track_id in enumerate(track_ids):
                state = self._states.get(track_id)
                sources = (state.r1, state.r2, state.r3, state.r4) if state is not None else self._zero_states
                for buffer, source in zip(state_buffers, sources):
                    buffer[i:i + 1].copy_(source)

        for name, tensor in zip(INPUT_NAMES, [src_buffer, *state_buffers]):
            self._context.set_input_shape(name, tuple(tensor.shape))
        for name, tensor in zip(OUTPUT_NAMES, outputs):
            self._context.set_tensor_address(name, tensor.data_ptr())
        self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()

        # The new states are views into this call's outputs
        _, pha_out, r1o, r2o, r3o, r4o = outputs
        self._states = {track_id: RecurrentState(r1o[i:i + 1], r2o[i:i + 1], r3o[i:i + 1], r4o[i:i + 1])
                        for i, track_id in enumerate(track_ids)}

        # The matte at the crops' size, and the crops multiplied by it
        with torch.cuda.stream(self._stream):
            mask = pha_out.squeeze(1)
            if needs_resize:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(input_h, input_w), mode='bilinear', align_corners=False).squeeze(1)
            foreground = original * mask.unsqueeze(1)
        return mask, foreground
