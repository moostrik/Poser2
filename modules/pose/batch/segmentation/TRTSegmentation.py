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
from concurrent.futures import ThreadPoolExecutor


from ..tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock
from .InOut import SegmentationInput, SegmentationOutput, SegmentationOutputCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings


from modules.utils.HotReloadMethods import HotReloadMethods

class RecurrentState:
    """Container for RVM recurrent states (r1, r2, r3, r4)."""
    def __init__(self, r1: cp.ndarray, r2: cp.ndarray, r3: cp.ndarray, r4: cp.ndarray):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4

class TRTSegmentation(Thread):
    """Asynchronous GPU person segmentation using Robust Video Matting (RVM) with TensorRT.

    Optimized inference using TensorRT engines for maximum performance.
    Architecture identical to RVMSegmentation for drop-in replacement.

    RVM uses recurrent states to maintain temporal coherence across video frames,
    eliminating flickering artifacts common in frame-independent methods like MODNet.

    Uses a single-slot queue: only the most recent submitted batch waits to be processed.
    Older pending batches are dropped. Batches already processing on GPU cannot be cancelled.

    Maintains per-tracklet recurrent states (r1, r2, r3, r4) for temporal consistency.
    All results (success and dropped) are delivered via callbacks in notification order.
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
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = min(settings.max_poses, 4)

        # Preallocated zero recurrent states for new tracklets (read-only, shared)
        self._zero_r1: cp.ndarray
        self._zero_r2: cp.ndarray
        self._zero_r3: cp.ndarray
        self._zero_r4: cp.ndarray

        # Preallocated input buffers per worker (for preprocessing)
        self._input_buffers: list[dict[str, cp.ndarray]] = []

        # Profiling accumulators (periodic averaging)
        self._profile_count: int = 0
        self._profile_interval: int = 100  # Print every N frames (reduced overhead)
        self._profile_lock_wait: float = 0.0
        self._profile_preprocess: float = 0.0
        self._profile_inference: float = 0.0
        self._profile_postprocess: float = 0.0

        # Toggle between batched and single-image inference (for performance comparison)
        self._use_batched_inference: bool = True  # Set False to use single-image mode

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
        if not self.enabled:
            return
        """Stop both inference and callback threads gracefully"""
        self._shutdown_event.set()

        # Shutdown thread pool
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

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
        """Main inference thread loop. Initializes TensorRT engine and processes batches."""
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
                trt.float32: "FP32",  # type: ignore
                trt.float16: "FP16",  # type: ignore
                trt.int8: "INT8",     # type: ignore
            }
            self.model_precision = precision_map.get(src_dtype, "UNKNOWN")

            # Map TensorRT dtype to CuPy dtype for buffer allocations
            dtype_to_cupy = {
                trt.float32: cp.float32,  # type: ignore
                trt.float16: cp.float16,  # type: ignore
                trt.int8: cp.int8,        # type: ignore
            }
            self._model_dtype = dtype_to_cupy.get(src_dtype, cp.float16)  # Default to FP16

            # Clear any existing states
            self._recurrent_states.clear()

            # Preallocate zero recurrent states (read-only, shared across all new tracklets)
            self._zero_r1 = cp.zeros((1, 16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype)
            self._zero_r2 = cp.zeros((1, 20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype)
            self._zero_r3 = cp.zeros((1, 40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype)
            self._zero_r4 = cp.zeros((1, 64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype)

            # Preallocate input buffers per worker (for single-image fallback)
            self._input_buffers = []
            for _ in range(self._max_workers):
                self._input_buffers.append({
                    'img_uint8': cp.empty((self.model_height, self.model_width, 3), dtype=cp.uint8),
                    'img_float': cp.empty((self.model_height, self.model_width, 3), dtype=self._model_dtype),
                    'src': cp.empty((1, 3, self.model_height, self.model_width), dtype=self._model_dtype),
                })

            # Preallocate BATCHED buffers for true batch inference (inputs only)
            # Output buffers are allocated fresh each batch to avoid .copy() overhead
            self._max_batch = self._max_workers  # Match max batch to max workers
            self._batch_buffers = {
                'img_uint8': cp.empty((self._max_batch, self.model_height, self.model_width, 3), dtype=cp.uint8),
                'src': cp.empty((self._max_batch, 3, self.model_height, self.model_width), dtype=self._model_dtype),
                # Batched recurrent state INPUT buffers
                'r1i': cp.empty((self._max_batch, 16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype),
                'r2i': cp.empty((self._max_batch, 20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype),
                'r3i': cp.empty((self._max_batch, 40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype),
                'r4i': cp.empty((self._max_batch, 64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype),
            }

            # Create thread pool for parallel inference
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="TRT-RVM-Worker")

            self._model_ready.set()
            print(f"TRT RVM Segmentation: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"TRT RVM Segmentation Error: Failed to load model - {str(e)}")
            traceback.print_exc()
            return

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
        """Submit batch for processing. Replaces any pending (not yet started) batch.

        Dropped batches trigger callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        dropped_batch: SegmentationInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
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

    def _retrieve_pending_batch(self) -> SegmentationInput | None:
        """Atomically get and clear pending batch. Once retrieved, batch cannot be cancelled.

        Returns:
            The pending batch if one exists, None otherwise
        """
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process_pending_batch(self) -> None:
        """Process the pending batch using RVM TensorRT inference with recurrent states."""
        batch: SegmentationInput | None = self._retrieve_pending_batch()

        if batch is None:
            return

        # Increment frame counter for periodic state reset
        self._frame_counter += 1

        # Periodic state reset as failsafe (0=disabled)
        if self._state_reset_interval > 0 and self._frame_counter % self._state_reset_interval == 0:
            if self.verbose:
                print(f"TRT RVM Segmentation: Periodic state reset at frame {self._frame_counter}")
            self._recurrent_states.clear()

        output = SegmentationOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Run inference using selected mode
        if batch.images:
            # Limit processing to max batch size
            images_to_process = batch.images[:self._max_batch]
            tracklets_to_process = batch.tracklet_ids[:self._max_batch]


            try:
                mask_tensor, inference_time_ms, lock_wait_ms = self._infer_batch(images_to_process, tracklets_to_process)
                processed = True
            except Exception as e:
                print(f"TRT RVM Segmentation Error: Batched inference failed: {str(e)}")
                traceback.print_exc()
                # Fallback to empty result
                mask_tensor = torch.empty(0, dtype=torch.float32, device='cuda')
                inference_time_ms = 0.0
                lock_wait_ms = 0.0
                processed = False

            output = SegmentationOutput(
                batch_id=batch.batch_id,
                mask_tensor=mask_tensor,
                tracklet_ids=batch.tracklet_ids,
                processed=processed,
                inference_time_ms=inference_time_ms,
                lock_time_ms=lock_wait_ms
            )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TRT RVM Segmentation Warning: Callback queue full, dropping inference results")

    def _infer_batch(self, images: list[np.ndarray], tracklet_ids: list[int]) -> tuple[torch.Tensor, float, float]:
        """Run RVM TensorRT inference on BATCH of images with per-tracklet recurrent states.

        Args:
            images: List of input images (H, W, 3) BGR uint8
            tracklet_ids: List of tracklet IDs (same length as images)

        Returns:
            Tuple of (alpha matte tensor (B, H, W) FP32 on CUDA [0, 1], process_time_ms, lock_wait_ms)
        """
        batch_size = len(images)
        if batch_size == 0:
            return torch.empty(0, dtype=torch.float32, device='cuda'), 0.0, 0.0

        method_start: float = time.perf_counter()

        # Gather recurrent states for all tracklets
        states: list[RecurrentState | None] = []
        for tid in tracklet_ids:
            states.append(self._recurrent_states.get(tid))

        # CPU preprocessing
        stacked_imgs = np.stack(images, axis=0)  # (B, H, W, 3) uint8

        # Allocate output buffers
        fgr_gpu = cp.empty((batch_size, 3, self.model_height, self.model_width), dtype=self._model_dtype)
        pha_gpu = cp.empty((batch_size, 1, self.model_height, self.model_width), dtype=self._model_dtype)
        r1o_gpu = cp.empty((batch_size, 16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype)
        r2o_gpu = cp.empty((batch_size, 20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype)
        r3o_gpu = cp.empty((batch_size, 40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype)
        r4o_gpu = cp.empty((batch_size, 64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype)

        # Run batched inference with global lock
        lock_wait_start: float = time.perf_counter()
        with get_exec_lock():
            lock_acquired: float = time.perf_counter()

            # Get batched INPUT buffers (sliced to actual batch size)
            buf = self._batch_buffers
            src_gpu = buf['src'][:batch_size]
            r1i_gpu = buf['r1i'][:batch_size]
            r2i_gpu = buf['r2i'][:batch_size]
            r3i_gpu = buf['r3i'][:batch_size]
            r4i_gpu = buf['r4i'][:batch_size]

            with self.stream:
                # OPTIMIZED: Stack images on CPU first, single transfer to GPU
                buf['img_uint8'][:batch_size].set(stacked_imgs)

                # Vectorized BGR->RGB, uint8->float, normalize (operates on full batch)
                img_float_batch = (buf['img_uint8'][:batch_size, :, :, ::-1] / 255.0).astype(self._model_dtype)

                # Vectorized transpose HWC -> CHW for full batch
                src_gpu[:] = cp.ascontiguousarray(cp.transpose(img_float_batch, (0, 3, 1, 2)))

                # Gather recurrent states into batched buffers
                for i, state in enumerate(states):
                    if state is not None:
                        r1i_gpu[i] = state.r1[0]  # Remove batch dim from stored state
                        r2i_gpu[i] = state.r2[0]
                        r3i_gpu[i] = state.r3[0]
                        r4i_gpu[i] = state.r4[0]
                    else:
                        # Use zeros for new tracklets
                        r1i_gpu[i] = self._zero_r1[0]
                        r2i_gpu[i] = self._zero_r2[0]
                        r3i_gpu[i] = self._zero_r3[0]
                        r4i_gpu[i] = self._zero_r4[0]

            # Set batched input shapes
            self.context.set_input_shape('src', (batch_size, 3, self.model_height, self.model_width))
            self.context.set_input_shape('r1i', (batch_size, 16, self.model_height // 2, self.model_width // 2))
            self.context.set_input_shape('r2i', (batch_size, 20, self.model_height // 4, self.model_width // 4))
            self.context.set_input_shape('r3i', (batch_size, 40, self.model_height // 8, self.model_width // 8))
            self.context.set_input_shape('r4i', (batch_size, 64, self.model_height // 16, self.model_width // 16))

            # Set tensor addresses
            self.context.set_tensor_address('src', buf['src'].data.ptr)
            self.context.set_tensor_address('r1i', buf['r1i'].data.ptr)
            self.context.set_tensor_address('r2i', buf['r2i'].data.ptr)
            self.context.set_tensor_address('r3i', buf['r3i'].data.ptr)
            self.context.set_tensor_address('r4i', buf['r4i'].data.ptr)
            self.context.set_tensor_address('fgr', fgr_gpu.data.ptr)
            self.context.set_tensor_address('pha', pha_gpu.data.ptr)
            self.context.set_tensor_address('r1o', r1o_gpu.data.ptr)
            self.context.set_tensor_address('r2o', r2o_gpu.data.ptr)
            self.context.set_tensor_address('r3o', r3o_gpu.data.ptr)
            self.context.set_tensor_address('r4o', r4o_gpu.data.ptr)

            # Execute batched inference (no sync needed before - TensorRT waits for stream)
            with self.stream:
                self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()


            # Split output states - NO COPY needed, buffers are fresh allocations
            new_states: list[RecurrentState] = []
            for i in range(batch_size):
                new_states.append(RecurrentState(
                    r1=r1o_gpu[i:i+1],  # View into fresh buffer - safe to keep
                    r2=r2o_gpu[i:i+1],
                    r3=r3o_gpu[i:i+1],
                    r4=r4o_gpu[i:i+1],
                ))

            # Convert batched pha to PyTorch tensor
            pha_tensor = torch.as_tensor(pha_gpu, device='cuda').squeeze(1).float()  # (B, H, W)

        # Update recurrent states - replace entire dict to remove stale IDs
        new_states_dict: dict[int, RecurrentState] = {}
        for tid, new_state in zip(tracklet_ids, new_states):
            new_states_dict[tid] = new_state

        self._recurrent_states = new_states_dict

        method_end = time.perf_counter()

        lock_wait_ms = (lock_acquired - lock_wait_start) * 1000.0
        total_time_ms = (method_end - method_start) * 1000.0
        process_time_ms = total_time_ms - lock_wait_ms

        return pha_tensor, process_time_ms, lock_wait_ms

    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks. Runs on separate thread."""
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

    def register_callback(self, callback: SegmentationOutputCallback) -> None:
        """Register callback to receive segmentation results (both success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)
