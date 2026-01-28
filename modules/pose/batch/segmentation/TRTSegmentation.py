# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import torch
import tensorrt as trt
import cupy as cp

from ..tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock
from ..cuda_image_ops import batched_bilinear_resize_inplace, normalize_hwc_to_chw_inplace
from .InOut import SegmentationInput, SegmentationOutput, SegmentationOutputCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

class RecurrentState:
    """Container for RVM recurrent states (r1, r2, r3, r4)."""
    def __init__(self, r1: cp.ndarray, r2: cp.ndarray, r3: cp.ndarray, r4: cp.ndarray):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4

class TRTSegmentation(Thread):
    """Asynchronous GPU person segmentation using Robust Video Matting (RVM) with TensorRT.

    TensorRT-optimized inference for maximum performance. Drop-in replacement for ONNXSegmentation.

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
        self._callback_thread: Thread = Thread(target=self._callback_worker_loop, daemon=True)

        # Per-tracklet recurrent states for temporal coherence
        self._recurrent_states: dict[int, RecurrentState] = {}
        self._frame_counter: int = 0  # For periodic state reset
        self._state_reset_interval: int = settings.segmentation_reset_interval  # Reset all states every N frames (0=disabled)

        # TensorRT engine and context (initialized in run())
        self.engine: trt.ICudaEngine  # type: ignore
        self.context: trt.IExecutionContext  # type: ignore
        self.input_names: list[str]
        self.output_names: list[str]
        self.stream: cp.cuda.Stream

        # Thread pool for parallel inference
        self._max_inputs: int = min(settings.max_poses, 4)

        # Preallocated zero recurrent states for new tracklets (read-only, shared)
        self._zero_r1: cp.ndarray
        self._zero_r2: cp.ndarray
        self._zero_r3: cp.ndarray
        self._zero_r4: cp.ndarray

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
                self._process_pending_batch()

            except Exception as e:
                print(f"TRT RVM Segmentation Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: SegmentationInput) -> None:
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

                # Create dedicated CUDA stream
                self.stream = cp.cuda.Stream(non_blocking=True)

                # Tune CuPy memory pool for better performance
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=2 * 1024**3)  # 2GB limit (up from 512MB default)

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

            # Map TensorRT dtype to CuPy dtype for buffer allocations
            dtype_to_cupy = {
                trt.DataType.FLOAT: cp.float32, # type: ignore
                trt.DataType.HALF: cp.float16,  # type: ignore
                trt.DataType.INT8: cp.int8,     # type: ignore
            }
            self._model_dtype = dtype_to_cupy.get(src_dtype, cp.float16)  # Default to FP16

            # Clear any existing states
            self._recurrent_states.clear()

            # Preallocate zero recurrent states (read-only, shared across all new tracklets)
            self._zero_r1 = cp.zeros((1, 16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype)
            self._zero_r2 = cp.zeros((1, 20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype)
            self._zero_r3 = cp.zeros((1, 40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype)
            self._zero_r4 = cp.zeros((1, 64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype)

            # Preallocate BATCHED buffers for true batch inference (inputs only)
            self._max_batch = self._max_inputs  # Match max batch to max workers
            self._batch_buffers = {
                'img_uint8': cp.empty((self._max_batch, self.model_height, self.model_width, 3), dtype=cp.uint8),
                'src': cp.empty((self._max_batch, 3, self.model_height, self.model_width), dtype=self._model_dtype),
                # Batched recurrent state INPUT buffers
                'r1i': cp.empty((self._max_batch, 16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype),
                'r2i': cp.empty((self._max_batch, 20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype),
                'r3i': cp.empty((self._max_batch, 40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype),
                'r4i': cp.empty((self._max_batch, 64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype),
            }

            # Set persistent tensor addresses for INPUT buffers (base pointers don't change when slicing)
            # Output addresses must be set per-call since they're allocated fresh each inference
            self.context.set_tensor_address('src', self._batch_buffers['src'].data.ptr)
            self.context.set_tensor_address('r1i', self._batch_buffers['r1i'].data.ptr)
            self.context.set_tensor_address('r2i', self._batch_buffers['r2i'].data.ptr)
            self.context.set_tensor_address('r3i', self._batch_buffers['r3i'].data.ptr)
            self.context.set_tensor_address('r4i', self._batch_buffers['r4i'].data.ptr)

            self._model_ready.set()
            print(f"TRT RVM Segmentation: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"TRT RVM Segmentation Error: Failed to load model - {str(e)}")
            traceback.print_exc()
            return

    def _retrieve_pending_batch(self) -> SegmentationInput | None:
        """Atomically get and clear pending batch. Once retrieved, cannot be cancelled."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process_pending_batch(self) -> None:
        """Process pending batch using batched TensorRT inference with recurrent states."""
        batch: SegmentationInput | None = self._retrieve_pending_batch()

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
                mask_tensor, fgr_tensor, inference_time_ms, lock_wait_ms = self._infer_batch(gpu_images, tracklets_to_process)
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

    def _infer_batch(self, gpu_imgs: list[cp.ndarray], tracklet_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Run batched TensorRT inference with per-tracklet recurrent states.

        Args:
            gpu_imgs: List of RGB uint8 images on GPU (H, W, 3) - already correct size
            tracklet_ids: Tracklet IDs for each image

        Returns:
            (alpha_matte (B,H,W), foreground (B,3,H,W), inference_ms, lock_wait_ms)
        """
        batch_size = len(gpu_imgs)
        if batch_size == 0:
            torch_dtype = torch.float16 if self._model_dtype == cp.float16 else torch.float32
            return torch.empty(0, dtype=torch_dtype, device='cuda'), torch.empty(0, 3, 0, 0, dtype=torch_dtype, device='cuda'), 0.0, 0.0

        method_start: float = time.perf_counter()

        # Gather recurrent states for all tracklets
        states: list[RecurrentState | None] = []
        for tid in tracklet_ids:
            states.append(self._recurrent_states.get(tid))

        # Allocate output buffers
        fgr_gpu = cp.empty((batch_size, 3, self.model_height, self.model_width), dtype=self._model_dtype)
        pha_gpu = cp.empty((batch_size, 1, self.model_height, self.model_width), dtype=self._model_dtype)
        r1o_gpu = cp.empty((batch_size, 16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype)
        r2o_gpu = cp.empty((batch_size, 20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype)
        r3o_gpu = cp.empty((batch_size, 40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype)
        r4o_gpu = cp.empty((batch_size, 64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype)

        # Get batched INPUT buffers (sliced to actual batch size)
        buf = self._batch_buffers
        src_gpu = buf['src'][:batch_size]
        r1i_gpu = buf['r1i'][:batch_size]
        r2i_gpu = buf['r2i'][:batch_size]
        r3i_gpu = buf['r3i'][:batch_size]
        r4i_gpu = buf['r4i'][:batch_size]

        # CuPy preprocessing on GPU (no CPU upload needed)
        with self.stream:
            # Stack GPU images into batch (B, H, W, 3)
            batch_hwc = cp.stack(gpu_imgs, axis=0)

            # Check if resize is needed (crop size != model size)
            src_h, src_w = batch_hwc.shape[1:3]
            if src_h != self.model_height or src_w != self.model_width:
                # Batched resize directly into preallocated buffer
                batched_bilinear_resize_inplace(batch_hwc, buf['img_uint8'][:batch_size], self.stream)
                batch_hwc = buf['img_uint8'][:batch_size]

            # Fused normalize [0,1] + HWC->CHW directly into src buffer
            # Note: Images are already RGB from GPUCropProcessor, no BGR->RGB needed
            normalize_hwc_to_chw_inplace(batch_hwc, src_gpu, self.stream)

            # Gather recurrent states into batched buffers
            for i, state in enumerate(states):
                if state is not None:
                    r1i_gpu[i] = state.r1[0]
                    r2i_gpu[i] = state.r2[0]
                    r3i_gpu[i] = state.r3[0]
                    r4i_gpu[i] = state.r4[0]
                else:
                    r1i_gpu[i] = self._zero_r1[0]
                    r2i_gpu[i] = self._zero_r2[0]
                    r3i_gpu[i] = self._zero_r3[0]
                    r4i_gpu[i] = self._zero_r4[0]
        # self.stream.synchronize()  # Ensure preprocessing complete before TRT reads

        # TensorRT operations only - acquire global lock
        lock_wait_start: float = time.perf_counter()
        with get_exec_lock():
            lock_acquired: float = time.perf_counter()

            # Only set_input_shape per-call (input addresses are persistent from _setup)
            self.context.set_input_shape('src', (batch_size, 3, self.model_height, self.model_width))
            self.context.set_input_shape('r1i', (batch_size, 16, self.model_height // 2, self.model_width // 2))
            self.context.set_input_shape('r2i', (batch_size, 20, self.model_height // 4, self.model_width // 4))
            self.context.set_input_shape('r3i', (batch_size, 40, self.model_height // 8, self.model_width // 8))
            self.context.set_input_shape('r4i', (batch_size, 64, self.model_height // 16, self.model_width // 16))

            # Output addresses must be set per-call (fresh allocation each inference)
            self.context.set_tensor_address('fgr', fgr_gpu.data.ptr)
            self.context.set_tensor_address('pha', pha_gpu.data.ptr)
            self.context.set_tensor_address('r1o', r1o_gpu.data.ptr)
            self.context.set_tensor_address('r2o', r2o_gpu.data.ptr)
            self.context.set_tensor_address('r3o', r3o_gpu.data.ptr)
            self.context.set_tensor_address('r4o', r4o_gpu.data.ptr)

            self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()

        # Output handling (no lock needed - fresh buffers)
        new_states: list[RecurrentState] = []
        for i in range(batch_size):
            new_states.append(RecurrentState(
                r1=r1o_gpu[i:i+1],
                r2=r2o_gpu[i:i+1],
                r3=r3o_gpu[i:i+1],
                r4=r4o_gpu[i:i+1],
            ))

        pha_tensor = torch.as_tensor(pha_gpu, device='cuda').squeeze(1)  # (B, H, W)
        fgr_tensor = torch.as_tensor(fgr_gpu, device='cuda')  # (B, 3, H, W)

        # Update recurrent states - replace entire dict to remove stale IDs
        # Note: This intentionally purges all non-current tracklets every frame to prevent
        # memory growth. Tracklets that temporarily disappear will restart with zero states.
        new_states_dict: dict[int, RecurrentState] = {}
        for tid, new_state in zip(tracklet_ids, new_states):
            new_states_dict[tid] = new_state
        self._recurrent_states = new_states_dict

        method_end = time.perf_counter()

        lock_wait_ms = (lock_acquired - lock_wait_start) * 1000.0
        total_time_ms = (method_end - method_start) * 1000.0
        process_time_ms = total_time_ms - lock_wait_ms

        return pha_tensor, fgr_tensor, process_time_ms, lock_wait_ms

    # CALLBACK METHODS
    def register_callback(self, callback: SegmentationOutputCallback) -> None:
        """Register callback to receive segmentation results (success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: SegmentationOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

    def _callback_worker_loop(self) -> None:
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
