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

from .InOut import OpticalFlowInput, OpticalFlowOutput, OpticalFlowOutputCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

from modules.pose.batch.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock


class TRTOpticalFlow(Thread):
    """Asynchronous GPU optical flow computation using RAFT with TensorRT.

    Optimized inference using TensorRT engines for maximum performance.
    Architecture identical to RAFTOpticalFlow for drop-in replacement.

    RAFT (Recurrent All-Pairs Field Transforms) computes dense optical flow between
    consecutive video frames using iterative refinement.

    Uses a single-slot queue: only the most recent submitted batch waits to be processed.
    Older pending batches are dropped. Batches already processing on GPU cannot be cancelled.

    All results (success and dropped) are delivered via callbacks in notification order.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.enabled: bool = settings.flow_enabled if hasattr(settings, 'flow_enabled') else False
        if not self.enabled:
            print('TensorRT Optical Flow WARNING: Optical flow is disabled')

        self.model_path: str = settings.model_path
        self.model_name: str = settings.flow_model
        self.model_file: str = f"{self.model_path}/{self.model_name}"
        self.model_width: int = settings.flow_width
        self.model_height: int = settings.flow_height
        self.resolution_name: str = settings.flow_resolution.name
        self.verbose: bool = settings.verbose

        # Thread coordination
        self._shutdown_event: Event = Event()
        self._notify_update_event: Event = Event()
        self._model_ready: Event = Event()

        # Input queue
        self._input_lock: Lock = Lock()
        self._pending_batch: OpticalFlowInput | None = None
        self._input_timestamp: float = time.time()
        self._last_dropped_batch_id: int = 0

        # Callbacks
        self._callback_lock: Lock = Lock()
        self._callbacks: set[OpticalFlowOutputCallback] = set()
        self._callback_queue: Queue[OpticalFlowOutput | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._callback_worker_loop, daemon=True)

        # TensorRT engine and context (initialized in run())
        self.engine: trt.ICudaEngine  # type: ignore
        self.context: trt.IExecutionContext  # type: ignore
        self.input_names: list[str]
        self.output_name: str
        self.stream: cp.cuda.Stream

        # Model configuration (initialized in _setup())
        self._max_batch: int = min(getattr(settings, 'max_poses', 3), 4)
        self._model_dtype: cp.dtype
        self.model_precision: str
        self._batch_buffers: dict[str, cp.ndarray]

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and system is ready to process optical flow"""
        return self._model_ready.is_set() and not self._shutdown_event.is_set() and self.is_alive()

    def _setup(self) -> None:
        """Initialize TensorRT engine, context, and buffers. Called from run()."""
        # Acquire global lock to prevent concurrent Myelin graph loading
        with get_init_lock():
            runtime = get_tensorrt_runtime()

            print(f"TensorRT Optical Flow: Loading engine from {self.model_file}")
            with open(self.model_file, 'rb') as f:
                engine_data = f.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to load engine")

            self.context = self.engine.create_execution_context()

            # Create dedicated CUDA stream
            self.stream = cp.cuda.Stream(non_blocking=True)

            # Tune CuPy memory pool for better performance
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=2 * 1024**3)  # 2GB limit

        # Lock released - continue with setup

        # Get input/output names
        self.input_names = []
        self.output_name = ""

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:  # type: ignore
                self.input_names.append(name)
            else:
                self.output_name = name

        if len(self.input_names) != 2:
            raise ValueError(f"Expected 2 inputs, got {len(self.input_names)}")

        # Determine model precision from input tensor
        input_dtype = self.engine.get_tensor_dtype(self.input_names[0])
        precision_map = {
            trt.DataType.FLOAT: "FP32", # type: ignore
            trt.DataType.HALF: "FP16",  # type: ignore
            trt.DataType.INT8: "INT8",  # type: ignore
        }
        self.model_precision = precision_map.get(input_dtype, "UNKNOWN")

        # Map TensorRT dtype to CuPy dtype
        dtype_to_cupy = {
            trt.DataType.FLOAT: cp.float32, # type: ignore
            trt.DataType.HALF: cp.float16,  # type: ignore
            trt.DataType.INT8: cp.int8,     # type: ignore
        }
        self._model_dtype = dtype_to_cupy.get(input_dtype, cp.float32)

        # Preallocate BATCHED buffers for batch inference
        self._batch_buffers = {
            'img1_uint8': cp.empty((self._max_batch, self.model_height, self.model_width, 3), dtype=cp.uint8),
            'img2_uint8': cp.empty((self._max_batch, self.model_height, self.model_width, 3), dtype=cp.uint8),
            'img1_input': cp.empty((self._max_batch, 3, self.model_height, self.model_width), dtype=self._model_dtype),
            'img2_input': cp.empty((self._max_batch, 3, self.model_height, self.model_width), dtype=self._model_dtype),
            'flow_output': cp.empty((self._max_batch, 2, self.model_height, self.model_width), dtype=self._model_dtype),
        }

        self._model_ready.set()
        print(f"TensorRT Optical Flow: {self.resolution_name} model loaded ({self.model_width}x{self.model_height}) "
              f"precision={self.model_precision} max_batch={self._max_batch}")

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

        # Wake up inference thread
        self._notify_update_event.set()
        self.join(timeout=2.0)

        if self.is_alive():
            print("Warning: TensorRT Optical Flow inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: TensorRT Optical Flow callback thread did not stop cleanly")

    def run(self) -> None:
        """Main inference thread loop. Initializes TensorRT engine and processes batches."""
        try:
            self._setup()
        except Exception as e:
            print(f"TensorRT Optical Flow Error: Failed to load model - {str(e)}")
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
                print(f"TensorRT Optical Flow Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: OpticalFlowInput) -> None:
        """Submit batch for processing. Replaces any pending (not yet started) batch."""
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        dropped_batch: OpticalFlowInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                lag = int((time.time() - self._input_timestamp) * 1000)
                # if self.verbose:
                print(f"TensorRT Optical Flow: Dropped batch {dropped_batch.batch_id} with lag {lag} ms")
                self._last_dropped_batch_id = dropped_batch.batch_id

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        # Notify about dropped batch
        if dropped_batch is not None:
            dropped_output = OpticalFlowOutput(
                batch_id=dropped_batch.batch_id,
                tracklet_ids=dropped_batch.tracklet_ids,
                processed=False
            )
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                pass

        self._notify_update_event.set()

    def _retrieve_pending_batch(self) -> OpticalFlowInput | None:
        """Atomically get and clear pending batch."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process_pending_batch(self) -> None:
        """Process the pending batch using TensorRT inference."""
        batch: OpticalFlowInput | None = self._retrieve_pending_batch()

        if batch is None:
            return

        output = OpticalFlowOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Run inference
        if batch.frame_pairs:
            # Validate batch size
            if len(batch.frame_pairs) > self._max_batch:
                print(f"TensorRT Optical Flow Warning: Batch size {len(batch.frame_pairs)} "
                      f"exceeds max {self._max_batch}, will process only first {self._max_batch}")
                batch.frame_pairs = batch.frame_pairs[:self._max_batch]
                batch.tracklet_ids = batch.tracklet_ids[:self._max_batch]

            try:
                flow_tensor, inference_time_ms, lock_wait_ms = self._infer_batch(batch.frame_pairs)

                output = OpticalFlowOutput(
                    batch_id=batch.batch_id,
                    flow_tensor=flow_tensor,
                    tracklet_ids=batch.tracklet_ids,
                    processed=True,
                    inference_time_ms=inference_time_ms,
                    lock_time_ms=lock_wait_ms
                )
            except Exception as e:
                print(f"TensorRT Optical Flow Error: Inference failed: {str(e)}")
                traceback.print_exc()
                # Create zero flow on error
                h, w = batch.frame_pairs[0][0].shape[:2]
                flow_tensor = torch.zeros((len(batch.frame_pairs), 2, h, w), dtype=torch.float32, device='cuda')
                output = OpticalFlowOutput(
                    batch_id=batch.batch_id,
                    flow_tensor=flow_tensor,
                    tracklet_ids=batch.tracklet_ids,
                    processed=True,
                    inference_time_ms=0.0,
                    lock_time_ms=0.0
                )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TensorRT Optical Flow Warning: Callback queue full, dropping results")

    def _infer_batch(self, frame_pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, float, float]:
        """Run TensorRT RAFT inference on batch of frame pairs.

        Args:
            frame_pairs: List of (prev_frame, curr_frame) tuples, each (H, W, 3) BGR uint8

        Returns:
            Tuple of (flow_tensor, inference_time_ms, lock_wait_ms)
            - flow_tensor: (B, 2, H, W) FP32 on CUDA, where [:, 0] is x-flow, [:, 1] is y-flow
            - inference_time_ms: Time spent processing (excluding lock wait)
            - lock_wait_ms: Time spent waiting for lock
        """
        method_start = time.perf_counter()
        batch_size = len(frame_pairs)

        # CPU preprocessing: Stack and BGR -> RGB (outside lock to minimize contention)
        prev_frames = np.stack([pair[0] for pair in frame_pairs], axis=0)  # (B, H, W, 3)
        curr_frames = np.stack([pair[1] for pair in frame_pairs], axis=0)  # (B, H, W, 3)
        prev_frames_rgb = prev_frames[:, :, :, ::-1]  # BGR to RGB
        curr_frames_rgb = curr_frames[:, :, :, ::-1]

        buf = self._batch_buffers

        lock_wait_start = time.perf_counter()
        with get_exec_lock():
            lock_acquired = time.perf_counter()

            # Get buffer slices for current batch
            image1_gpu = buf['img1_input'][:batch_size]
            image2_gpu = buf['img2_input'][:batch_size]
            flow_gpu = buf['flow_output'][:batch_size]

            with self.stream:
                # Single GPU transfer for entire batch
                buf['img1_uint8'][:batch_size].set(prev_frames_rgb)
                buf['img2_uint8'][:batch_size].set(curr_frames_rgb)

                # Vectorized GPU operations on entire batch
                img1_float = buf['img1_uint8'][:batch_size].astype(self._model_dtype)
                img2_float = buf['img2_uint8'][:batch_size].astype(self._model_dtype)

                # Transpose HWC -> CHW: (B, H, W, 3) -> (B, 3, H, W)
                img1_chw = cp.ascontiguousarray(cp.transpose(img1_float, (0, 3, 1, 2)))
                img2_chw = cp.ascontiguousarray(cp.transpose(img2_float, (0, 3, 1, 2)))

                # Store in preallocated input buffers
                image1_gpu[:] = img1_chw
                image2_gpu[:] = img2_chw

            # Set input shapes for current batch
            self.context.set_input_shape(self.input_names[0], image1_gpu.shape)
            self.context.set_input_shape(self.input_names[1], image2_gpu.shape)

            # Set tensor addresses
            self.context.set_tensor_address(self.input_names[0], image1_gpu.data.ptr)
            self.context.set_tensor_address(self.input_names[1], image2_gpu.data.ptr)
            self.context.set_tensor_address(self.output_name, flow_gpu.data.ptr)

            # Execute inference
            with self.stream:
                self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()

        method_end = time.perf_counter()

        # Calculate timing metrics
        lock_wait_ms = (lock_acquired - lock_wait_start) * 1000.0
        total_time_ms = (method_end - method_start) * 1000.0
        process_time_ms = total_time_ms - lock_wait_ms

        # Convert CuPy array to PyTorch tensor (zero-copy, both on CUDA)
        flow_tensor = torch.as_tensor(flow_gpu, device='cuda')  # (B, 2, H, W)

        return flow_tensor, process_time_ms, lock_wait_ms

    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks."""
        while not self._shutdown_event.is_set():
            try:
                output: OpticalFlowOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"TensorRT Optical Flow Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def register_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Register callback to receive optical flow results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
