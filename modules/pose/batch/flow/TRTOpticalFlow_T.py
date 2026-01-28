# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import torch
import tensorrt as trt

from .InOut import OpticalFlowInput, OpticalFlowOutput, OpticalFlowOutputCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

from modules.pose.batch.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock


class TRTOpticalFlow(Thread):
    """Asynchronous GPU optical flow computation using RAFT with TensorRT.

    TensorRT-optimized inference for maximum performance. Drop-in replacement for ONNXOpticalFlow.

    Uses preallocated buffers and dedicated CUDA stream for ultra-low latency.
    All preprocessing and inference runs on the same dedicated stream to avoid sync overhead.

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
        self.stream: torch.cuda.Stream  # type: ignore
        self.input_names: list[str]
        self.output_name: str

        # Model configuration (initialized in _setup())
        self._max_batch: int = min(getattr(settings, 'max_poses', 3), 4)
        self._torch_dtype: torch.dtype
        self.model_precision: str

        # Preallocated INPUT buffers (initialized in _setup())
        # Note: Output buffer allocated fresh each call
        self._img1_buffer: torch.Tensor  # (max_batch, 3, H, W) input 1
        self._img2_buffer: torch.Tensor  # (max_batch, 3, H, W) input 2

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and system is ready to process optical flow"""
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

        # Validate batch size
        if len(input_batch.gpu_image_pairs) > self._max_batch:
            print(f"TensorRT Optical Flow Warning: Batch size {len(input_batch.gpu_image_pairs)} exceeds max {self._max_batch}, will process only first {self._max_batch} pairs")

        dropped_batch: OpticalFlowInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"TensorRT Optical Flow: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
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

        # Map TensorRT dtype to PyTorch dtype
        dtype_to_torch = {
            trt.DataType.FLOAT: torch.float32, # type: ignore
            trt.DataType.HALF: torch.float16,  # type: ignore
        }
        self._torch_dtype = dtype_to_torch.get(input_dtype, torch.float32)

        # Dedicated CUDA stream for all GPU operations (preprocessing + inference)
        self.stream = torch.cuda.Stream()

        # Preallocate INPUT buffers on the dedicated stream for optimal memory placement
        with torch.cuda.stream(self.stream):
            # Input buffers: CHW format ready for TRT
            self._img1_buffer = torch.empty(
                (self._max_batch, 3, self.model_height, self.model_width),
                dtype=self._torch_dtype, device='cuda'
            )
            self._img2_buffer = torch.empty(
                (self._max_batch, 3, self.model_height, self.model_width),
                dtype=self._torch_dtype, device='cuda'
            )

        self.stream.synchronize()

        # Set persistent tensor addresses for INPUT buffers (base pointers don't change when slicing)
        # Output address must be set per-call since it's allocated fresh each inference
        self.context.set_tensor_address(self.input_names[0], self._img1_buffer.data_ptr())
        self.context.set_tensor_address(self.input_names[1], self._img2_buffer.data_ptr())

        self._model_ready.set()
        print(f"TensorRT Optical Flow: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

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

        output = OpticalFlowOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=False)

        # Run inference
        if batch.gpu_image_pairs:
            # Limit processing to max batch size
            pairs_to_process = batch.gpu_image_pairs[:self._max_batch]
            tracklets_to_process = batch.tracklet_ids[:self._max_batch]

            try:
                flow_tensor, inference_time_ms, lock_wait_ms = self._infer_batch(pairs_to_process)

                output = OpticalFlowOutput(
                    batch_id=batch.batch_id,
                    flow_tensor=flow_tensor,
                    tracklet_ids=tracklets_to_process,
                    processed=True,
                    inference_time_ms=inference_time_ms,
                    lock_time_ms=lock_wait_ms
                )
            except Exception as e:
                print(f"TensorRT Optical Flow Error: Inference failed: {str(e)}")
                traceback.print_exc()
        else:
            output = OpticalFlowOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TensorRT Optical Flow Warning: Callback queue full, dropping results")

    def _infer_batch(self, gpu_pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, float, float]:
        """Run TensorRT RAFT inference on batch of GPU frame pairs.

        All preprocessing and inference runs on the dedicated stream for zero sync overhead.
        Uses preallocated buffers for zero allocation latency.

        Args:
            gpu_pairs: List of (prev_crop, curr_crop) tuples, each (H, W, 3) RGB uint8 on GPU

        Returns:
            Tuple of (flow_tensor, inference_time_ms, lock_wait_ms)
            - flow_tensor: (B, 2, H, W) on CUDA, where [:, 0] is x-flow, [:, 1] is y-flow
            - inference_time_ms: Time spent processing (excluding lock wait)
            - lock_wait_ms: Time spent waiting for lock
        """
        method_start = time.perf_counter()
        batch_size = len(gpu_pairs)

        # Get input dimensions from first pair
        input_h, input_w = gpu_pairs[0][0].shape[0], gpu_pairs[0][0].shape[1]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)

        # Get preallocated INPUT buffer slices for current batch
        img1_buffer = self._img1_buffer[:batch_size]
        img2_buffer = self._img2_buffer[:batch_size]

        # Allocate output buffer fresh each call (no clone needed)
        flow_out = torch.empty((batch_size, 2, self.model_height, self.model_width), dtype=self._torch_dtype, device='cuda')

        # All preprocessing on dedicated stream (no cross-stream sync needed)
        with torch.cuda.stream(self.stream):
            # Stack GPU tensors: (B, H, W, 3)
            prev_batch = torch.stack([p[0] for p in gpu_pairs], dim=0)
            curr_batch = torch.stack([p[1] for p in gpu_pairs], dim=0)

            # Convert to model dtype and HWC -> CHW: (B, 3, H, W)
            prev_chw = prev_batch.to(self._torch_dtype).permute(0, 3, 1, 2)
            curr_chw = curr_batch.to(self._torch_dtype).permute(0, 3, 1, 2)

            # Resize if needed (crop size != model size)
            if needs_resize:
                prev_chw = torch.nn.functional.interpolate(
                    prev_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False
                )
                curr_chw = torch.nn.functional.interpolate(
                    curr_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False
                )

            # Copy into preallocated buffers
            img1_buffer.copy_(prev_chw)
            img2_buffer.copy_(curr_chw)

        # TensorRT inference - acquire global lock (still on same stream)
        lock_wait_start = time.perf_counter()
        with get_exec_lock():
            lock_acquired = time.perf_counter()

            # Only set_input_shape per-call (input addresses are persistent from _setup)
            self.context.set_input_shape(self.input_names[0], tuple(img1_buffer.shape))
            self.context.set_input_shape(self.input_names[1], tuple(img2_buffer.shape))

            # Output address must be set per-call (fresh allocation each inference)
            self.context.set_tensor_address(self.output_name, flow_out.data_ptr())

            # Execute on dedicated CUDA stream (preprocessing already complete on this stream)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

        method_end = time.perf_counter()

        # Calculate timing metrics
        lock_wait_ms = (lock_acquired - lock_wait_start) * 1000.0
        total_time_ms = (method_end - method_start) * 1000.0
        process_time_ms = total_time_ms - lock_wait_ms

        # No clone needed - output buffer is fresh each call
        return flow_out, process_time_ms, lock_wait_ms

    # CALLBACK METHODS
    def register_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Register callback to receive optical flow results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

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
