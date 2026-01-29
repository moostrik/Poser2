# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import torch
import tensorrt as trt

from ..tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock
from .InOut import SegmentationInput, SegmentationOutput, SegmentationOutputCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

class RecurrentState:
    """Container for RVM recurrent states (r1, r2, r3, r4)."""
    def __init__(self, r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, r4: torch.Tensor):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4

class TRTSegmentation(Thread):
    """Asynchronous GPU person segmentation using Robust Video Matting (RVM) with TensorRT.

    TensorRT-optimized inference for maximum performance. Drop-in replacement for ONNXSegmentation.

    Uses preallocated buffers and dedicated CUDA stream for ultra-low latency.
    All preprocessing and inference runs on the same dedicated stream to avoid sync overhead.

    Uses recurrent states for temporal coherence, eliminating flickering artifacts.
    Single-slot queue: only the most recent batch waits processing; older pending batches dropped.
    Batches already processing cannot be cancelled.

    Maintains per-tracklet recurrent states (r1-r4) for temporal consistency.
    All results delivered via callbacks in notification order.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.enabled: bool = settings.segmentation_enabled
        if not self.enabled:
            print('TRT RVM Segmentation WARNING: Segmentation is disabled')

        self.model_path: str = settings.model_path
        self.model_name: str = settings.segmentation_model
        self.model_file: str = f"{self.model_path}/{self.model_name}"
        self.model_width: int = settings.segmentation_width
        self.model_height: int = settings.segmentation_height
        self.resolution_name: str = settings.segmentation_resolution.name
        self.verbose: bool = settings.verbose

        # Thread coordination
        self._shutdown_event: Event = Event()
        self._notify_update_event: Event = Event()
        self._model_ready: Event = Event()

        # Input queue
        self._input_lock: Lock = Lock()
        self._pending_batch: SegmentationInput | None = None
        self._input_timestamp: float = time.time()
        self._last_dropped_batch_id: int = 0

        # Callbacks
        self._callback_lock: Lock = Lock()
        self._callbacks: set[SegmentationOutputCallback] = set()
        self._callback_queue: Queue[SegmentationOutput | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._dispatch_callbacks, daemon=True)

        # Per-tracklet recurrent states for temporal coherence
        self._recurrent_states: dict[int, RecurrentState] = {}
        self._frame_counter: int = 0  # For periodic state reset
        self._state_reset_interval: int = settings.segmentation_reset_interval  # Reset all states every N frames (0=disabled)

        # TensorRT engine and context (initialized in run())
        self.engine: trt.ICudaEngine  # type: ignore
        self.context: trt.IExecutionContext  # type: ignore
        self.stream: torch.cuda.Stream  # type: ignore
        self.input_names: list[str]
        self.output_names: list[str]

        # Model configuration (initialized in _setup())
        self._max_inputs: int = min(settings.max_poses, 4)
        self._torch_dtype: torch.dtype

        # Preallocated zero recurrent states for new tracklets (read-only, shared)
        self._zero_r1: torch.Tensor
        self._zero_r2: torch.Tensor
        self._zero_r3: torch.Tensor
        self._zero_r4: torch.Tensor

        # Preallocated buffers (initialized in _setup())
        self._src_buffer: torch.Tensor    # (max_batch, 3, H, W) normalized input
        self._r1i_buffer: torch.Tensor    # (max_batch, 16, H/2, W/2)
        self._r2i_buffer: torch.Tensor    # (max_batch, 20, H/4, W/4)
        self._r3i_buffer: torch.Tensor    # (max_batch, 40, H/8, W/8)
        self._r4i_buffer: torch.Tensor    # (max_batch, 64, H/16, W/16)
        # Note: Output buffers allocated fresh each call (recurrent states are views into them)

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and system is ready to process segmentation"""
        return self._model_ready.is_set() and not self._shutdown_event.is_set() and self.is_alive()

    def start(self) -> None:
        if not self.enabled:
            return
        self._callback_thread.start()
        super().start()

    def stop(self) -> None:
        """Stop both inference and callback threads gracefully."""
        if not self.enabled:
            return
        self._shutdown_event.set()

        # Wake up inference thread
        self._notify_update_event.set()
        self.join(timeout=2.0)

        if self.is_alive():
            print("Warning: TRT RVM Segmentation inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: TRT RVM Segmentation callback thread did not stop cleanly")

    def run(self) -> None:
        self._setup()

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process()

            except Exception as e:
                print(f"TRT RVM Segmentation Error: {str(e)}")
                traceback.print_exc()

    def submit(self, input_batch: SegmentationInput) -> None:
        """Submit batch for processing. Replaces pending batch if not yet started.

        Dropped batches receive callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        # Validate batch size
        if len(input_batch.gpu_images) > self._max_batch:
            print(f"TRT RVM Segmentation Warning: Batch size {len(input_batch.gpu_images)} exceeds max {self._max_batch}, will process only first {self._max_batch} images")

        dropped_batch: SegmentationInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"TRT RVM Segmentation: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
                self._last_dropped_batch_id = dropped_batch.batch_id

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        # Notify about dropped batch
        if dropped_batch is not None:
            dropped_output = SegmentationOutput(
                batch_id=dropped_batch.batch_id,
                tracklet_ids=dropped_batch.tracklet_ids,
                processed=False
            )
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                pass

        self._notify_update_event.set()

    def _setup(self) -> None:
        """Initialize TensorRT engine, context, and buffers. Called from run()."""
        try:
            # Acquire global lock to prevent concurrent Myelin graph loading
            with get_init_lock():
                runtime = get_tensorrt_runtime()

                # print(f"TensorRT Segmentation: Loading engine from {self.model_file}")
                with open(self.model_file, 'rb') as f:
                    engine_data = f.read()

                self.engine = runtime.deserialize_cuda_engine(engine_data)
                if self.engine is None:
                    print("TRT RVM Segmentation ERROR: Failed to load engine")
                    return

                self.context = self.engine.create_execution_context()

            # Lock released - continue with setup

            # Get input/output names
            self.input_names = []
            self.output_names = []

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)

                if mode == trt.TensorIOMode.INPUT:  # type: ignore
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

            # Expected: 5 inputs (src, r1i-r4i), 6 outputs (fgr, pha, r1o-r4o)
            if len(self.input_names) != 5:
                print(f"TRT RVM Segmentation ERROR: Expected 5 inputs, got {len(self.input_names)}")
                return

            if len(self.output_names) != 6:
                print(f"TRT RVM Segmentation ERROR: Expected 6 outputs, got {len(self.output_names)}")
                return

            # Determine model precision from 'src' input tensor
            src_dtype = self.engine.get_tensor_dtype('src')
            precision_map = {
                trt.DataType.FLOAT: "FP32",   # type: ignore
                trt.DataType.HALF: "FP16",    # type: ignore
                trt.DataType.INT8: "INT8",    # type: ignore
            }
            self.model_precision = precision_map.get(src_dtype, "UNKNOWN")

            # Map TensorRT dtype to PyTorch dtype
            dtype_to_torch = {
                trt.DataType.FLOAT: torch.float32, # type: ignore
                trt.DataType.HALF: torch.float16,  # type: ignore
            }
            self._torch_dtype = dtype_to_torch.get(src_dtype, torch.float16)

            # Clear any existing states
            self._recurrent_states.clear()

            # Preallocate zero recurrent states (read-only, shared across all new tracklets)
            self._zero_r1 = torch.zeros((1, 16, self.model_height // 2, self.model_width // 2), dtype=self._torch_dtype, device='cuda')
            self._zero_r2 = torch.zeros((1, 20, self.model_height // 4, self.model_width // 4), dtype=self._torch_dtype, device='cuda')
            self._zero_r3 = torch.zeros((1, 40, self.model_height // 8, self.model_width // 8), dtype=self._torch_dtype, device='cuda')
            self._zero_r4 = torch.zeros((1, 64, self.model_height // 16, self.model_width // 16), dtype=self._torch_dtype, device='cuda')

            self._max_batch = self._max_inputs

            # Dedicated CUDA stream for all GPU operations (preprocessing + inference)
            self.stream = torch.cuda.Stream()

            # Preallocate all INPUT buffers on the dedicated stream for optimal memory placement
            with torch.cuda.stream(self.stream):
                # Input buffer: normalized CHW format ready for TRT
                self._src_buffer = torch.empty(
                    (self._max_batch, 3, self.model_height, self.model_width),
                    dtype=self._torch_dtype, device='cuda'
                )

                # Recurrent input buffers
                self._r1i_buffer = torch.empty(
                    (self._max_batch, 16, self.model_height // 2, self.model_width // 2),
                    dtype=self._torch_dtype, device='cuda'
                )
                self._r2i_buffer = torch.empty(
                    (self._max_batch, 20, self.model_height // 4, self.model_width // 4),
                    dtype=self._torch_dtype, device='cuda'
                )
                self._r3i_buffer = torch.empty(
                    (self._max_batch, 40, self.model_height // 8, self.model_width // 8),
                    dtype=self._torch_dtype, device='cuda'
                )
                self._r4i_buffer = torch.empty(
                    (self._max_batch, 64, self.model_height // 16, self.model_width // 16),
                    dtype=self._torch_dtype, device='cuda'
                )

            self.stream.synchronize()

            # Set persistent tensor addresses for INPUT buffers (base pointers don't change when slicing)
            # Output addresses must be set per-call since they're allocated fresh each inference
            self.context.set_tensor_address('src', self._src_buffer.data_ptr())
            self.context.set_tensor_address('r1i', self._r1i_buffer.data_ptr())
            self.context.set_tensor_address('r2i', self._r2i_buffer.data_ptr())
            self.context.set_tensor_address('r3i', self._r3i_buffer.data_ptr())
            self.context.set_tensor_address('r4i', self._r4i_buffer.data_ptr())

            self._model_ready.set()
            print(f"TRT RVM Segmentation: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"TRT RVM Segmentation Error: Failed to load model - {str(e)}")
            traceback.print_exc()
            return

    def _claim(self) -> SegmentationInput | None:
        """Atomically get and clear pending batch. Once retrieved, cannot be cancelled."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process(self) -> None:
        """Process pending batch using batched TensorRT inference with recurrent states."""
        batch: SegmentationInput | None = self._claim()

        if batch is None:
            return

        # Increment frame counter for periodic state reset
        self._frame_counter += 1

        # Periodic state reset as failsafe (0=disabled)
        if self._state_reset_interval > 0 and self._frame_counter % self._state_reset_interval == 0:
            if self.verbose:
                print(f"RVM TRT Segmentation: Periodic state reset at frame {self._frame_counter}")
            self._recurrent_states.clear()

        output = SegmentationOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=False)

        # Run inference using selected mode
        gpu_images = batch.gpu_images[:self._max_batch]
        tracklets_to_process = batch.tracklet_ids[:self._max_batch]

        if gpu_images:
            try:
                mask_tensor, fgr_tensor, inference_time_ms, lock_wait_ms = self._infer(gpu_images, tracklets_to_process)
                output = SegmentationOutput(
                    batch_id=batch.batch_id,
                    mask_tensor=mask_tensor,
                    fgr_tensor=fgr_tensor,
                    tracklet_ids=batch.tracklet_ids,
                    processed=True,
                    inference_time_ms=inference_time_ms,
                    lock_time_ms=lock_wait_ms
                )
            except Exception as e:
                print(f"TRT RVM Segmentation Error: Batched inference failed: {str(e)}")
                traceback.print_exc()

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TRT RVM Segmentation Warning: Callback queue full, dropping inference results")

    def _infer(self, gpu_imgs: list[torch.Tensor], tracklet_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Run batched TensorRT inference with per-tracklet recurrent states.

        All preprocessing and inference runs on the dedicated stream for zero sync overhead.
        Uses preallocated INPUT buffers for zero allocation latency.
        OUTPUT buffers allocated fresh each call (recurrent states are views into them).

        Args:
            gpu_imgs: List of RGB uint8 tensors on GPU (H, W, 3)
            tracklet_ids: Tracklet IDs for each image

        Returns:
            (alpha_matte (B,H,W), foreground (B,3,H,W), inference_ms, lock_wait_ms)
        """
        batch_size = len(gpu_imgs)
        if batch_size == 0:
            return torch.empty(0, dtype=self._torch_dtype, device='cuda'), torch.empty(0, 3, 0, 0, dtype=self._torch_dtype, device='cuda'), 0.0, 0.0

        method_start: float = time.perf_counter()

        # Get input dimensions from first image
        input_h, input_w = gpu_imgs[0].shape[0], gpu_imgs[0].shape[1]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)

        # Get preallocated INPUT buffer slices for current batch
        src_buffer = self._src_buffer[:batch_size]
        r1i_buffer = self._r1i_buffer[:batch_size]
        r2i_buffer = self._r2i_buffer[:batch_size]
        r3i_buffer = self._r3i_buffer[:batch_size]
        r4i_buffer = self._r4i_buffer[:batch_size]

        # Allocate OUTPUT buffers fresh each call (recurrent states will be views into these)
        fgr_out = torch.empty((batch_size, 3, self.model_height, self.model_width), dtype=self._torch_dtype, device='cuda')
        pha_out = torch.empty((batch_size, 1, self.model_height, self.model_width), dtype=self._torch_dtype, device='cuda')
        r1o_out = torch.empty((batch_size, 16, self.model_height // 2, self.model_width // 2), dtype=self._torch_dtype, device='cuda')
        r2o_out = torch.empty((batch_size, 20, self.model_height // 4, self.model_width // 4), dtype=self._torch_dtype, device='cuda')
        r3o_out = torch.empty((batch_size, 40, self.model_height // 8, self.model_width // 8), dtype=self._torch_dtype, device='cuda')
        r4o_out = torch.empty((batch_size, 64, self.model_height // 16, self.model_width // 16), dtype=self._torch_dtype, device='cuda')

        # All preprocessing on dedicated stream (no cross-stream sync needed)
        with torch.cuda.stream(self.stream):
            # Stack GPU tensors: (B, H, W, 3) float32 RGB [0,1]
            batch_hwc = torch.stack(gpu_imgs, dim=0)

            # HWC -> CHW: (B, 3, H, W), convert to model dtype if needed
            batch_chw = batch_hwc.permute(0, 3, 1, 2).to(self._torch_dtype)

            # Resize if needed (crop size != model size)
            if needs_resize:
                batch_chw = torch.nn.functional.interpolate(
                    batch_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False
                )

            # Already normalized to [0, 1], just copy into preallocated buffer
            src_buffer.copy_(batch_chw)

            # Gather recurrent states for all tracklets into preallocated buffers
            for i, tid in enumerate(tracklet_ids):
                state = self._recurrent_states.get(tid)
                if state is not None:
                    r1i_buffer[i:i+1].copy_(state.r1)
                    r2i_buffer[i:i+1].copy_(state.r2)
                    r3i_buffer[i:i+1].copy_(state.r3)
                    r4i_buffer[i:i+1].copy_(state.r4)
                else:
                    r1i_buffer[i:i+1].copy_(self._zero_r1)
                    r2i_buffer[i:i+1].copy_(self._zero_r2)
                    r3i_buffer[i:i+1].copy_(self._zero_r3)
                    r4i_buffer[i:i+1].copy_(self._zero_r4)

        # TensorRT inference - acquire global lock (still on same stream)
        lock_wait_start: float = time.perf_counter()
        with get_exec_lock():
            lock_acquired: float = time.perf_counter()

            # Only set_input_shape per-call (input addresses are persistent from _setup)
            self.context.set_input_shape('src', tuple(src_buffer.shape))
            self.context.set_input_shape('r1i', tuple(r1i_buffer.shape))
            self.context.set_input_shape('r2i', tuple(r2i_buffer.shape))
            self.context.set_input_shape('r3i', tuple(r3i_buffer.shape))
            self.context.set_input_shape('r4i', tuple(r4i_buffer.shape))

            # Output addresses must be set per-call (fresh allocation each inference)
            self.context.set_tensor_address('fgr', fgr_out.data_ptr())
            self.context.set_tensor_address('pha', pha_out.data_ptr())
            self.context.set_tensor_address('r1o', r1o_out.data_ptr())
            self.context.set_tensor_address('r2o', r2o_out.data_ptr())
            self.context.set_tensor_address('r3o', r3o_out.data_ptr())
            self.context.set_tensor_address('r4o', r4o_out.data_ptr())

            # Execute on dedicated CUDA stream (preprocessing already complete on this stream)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

        # Build new recurrent states from outputs (views into fresh buffers, no clone needed)
        new_states_dict: dict[int, RecurrentState] = {}
        for i, tid in enumerate(tracklet_ids):
            new_states_dict[tid] = RecurrentState(
                r1=r1o_out[i:i+1],
                r2=r2o_out[i:i+1],
                r3=r3o_out[i:i+1],
                r4=r4o_out[i:i+1],
            )
        self._recurrent_states = new_states_dict

        # Squeeze alpha channel: (B, 1, H, W) -> (B, H, W)
        # No clone needed - output buffers are fresh each call
        pha_tensor = pha_out.squeeze(1)

        method_end = time.perf_counter()

        lock_wait_ms = (lock_acquired - lock_wait_start) * 1000.0
        total_time_ms = (method_end - method_start) * 1000.0
        process_time_ms = total_time_ms - lock_wait_ms

        return pha_tensor, fgr_out, process_time_ms, lock_wait_ms

    # CALLBACK METHODS
    def register_callback(self, callback: SegmentationOutputCallback) -> None:
        """Register callback to receive segmentation results (success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: SegmentationOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

    def _dispatch_callbacks(self) -> None:
        """Dispatch queued results to registered callbacks on dedicated thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1:
                    print("TRT RVM Segmentation Warning: Callback queue size > 1, consumers may be falling behind")

                output: SegmentationOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"TensorRT Segmentation Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue
