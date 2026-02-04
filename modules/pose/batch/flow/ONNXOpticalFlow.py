# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import torch
import onnxruntime as ort

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

from .InOut import OpticalFlowInput, OpticalFlowOutput, OpticalFlowOutputCallback


class ONNXOpticalFlow(Thread):
    """Asynchronous GPU optical flow computation using RAFT with ONNX Runtime.

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
            print('Optical Flow WARNING: Optical flow is disabled')

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
        self._callback_thread: Thread = Thread(target=self._dispatch_callbacks, daemon=True)

        # ONNX session (initialized in run thread)
        self._session: ort.InferenceSession | None = None
        self._model_dtype = np.float32  # Default, determined in setup
        self._torch_dtype: torch.dtype  # PyTorch dtype for preprocessing
        self.model_precision: str = "UNKNOWN"

        # Dedicated CUDA stream (initialized in _setup())
        self.stream: torch.cuda.Stream

        # Preallocated input buffers (initialized in _setup())
        self._img1_buffer: torch.Tensor  # (max_batch, 3, H, W) input 1
        self._img2_buffer: torch.Tensor  # (max_batch, 3, H, W) input 2

        # Batch inference settings
        self._max_batch: int = min(settings.max_poses, 8)

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
            print("Warning: ONNX Optical Flow inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: ONNX Optical Flow callback thread did not stop cleanly")

    def run(self) -> None:
        """Main inference thread loop."""
        self._setup()

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process()
            except Exception as e:
                print(f"ONNX Optical Flow Error: {str(e)}")
                traceback.print_exc()

    def submit(self, input_batch: OpticalFlowInput) -> None:
        """Submit batch for processing. Replaces any pending (not yet started) batch."""
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        # Validate batch size
        if len(input_batch.gpu_image_pairs) > self._max_batch:
            print(f"ONNX Optical Flow Warning: Batch size {len(input_batch.gpu_image_pairs)} exceeds max {self._max_batch}, will process only first {self._max_batch} pairs")

        dropped_batch: OpticalFlowInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"ONNX Optical Flow: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
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
        """Initialize ONNX session and warmup. Called from run()."""
        try:
            # Initialize ONNX Runtime session with CUDA provider
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = 1
            sess_options.log_severity_level = 3  # 0=verbose, 1=info, 2=warning, 3=error

            self._session = ort.InferenceSession(
                self.model_file,
                sess_options=sess_options,
                providers=providers
            )

            # Verify CUDA provider is active
            providers_used = self._session.get_providers()
            if 'CUDAExecutionProvider' not in providers_used:
                print("ONNX Optical Flow WARNING: CUDA provider not available, using CPU")

            # Determine model precision from input tensor
            input_tensor = self._session.get_inputs()[0]
            input_type = input_tensor.type

            # Parse ONNX type string (e.g., 'tensor(float)', 'tensor(float16)')
            precision_map = {
                'tensor(float)': 'FP32',
                'tensor(float16)': 'FP16',
                'tensor(double)': 'FP64',
                'tensor(int8)': 'INT8',
            }
            self.model_precision = precision_map.get(input_type, f"UNKNOWN({input_type})")

            # Map ONNX type to numpy dtype for preprocessing
            dtype_map = {
                'tensor(float)': np.float32,
                'tensor(float16)': np.float16,
                'tensor(double)': np.float64,
                'tensor(int8)': np.int8,
            }
            self._model_dtype = dtype_map.get(input_type, np.float32)

            # Map numpy dtype to torch dtype for preprocessing
            torch_dtype_map = {
                np.float32: torch.float32,
                np.float16: torch.float16,
            }
            self._torch_dtype = torch_dtype_map.get(self._model_dtype, torch.float32)

            # Dedicated CUDA stream for all GPU operations (preprocessing + inference)
            self.stream = torch.cuda.Stream()

            # Preallocate INPUT buffers on the dedicated stream for optimal memory placement
            with torch.cuda.stream(self.stream):
                self._img1_buffer = torch.empty(
                    (self._max_batch, 3, self.model_height, self.model_width),
                    dtype=self._torch_dtype, device='cuda'
                )
                self._img2_buffer = torch.empty(
                    (self._max_batch, 3, self.model_height, self.model_width),
                    dtype=self._torch_dtype, device='cuda'
                )

            self.stream.synchronize()

            # Warmup model
            self._warmup()

            self._model_ready.set()
            print(f"ONNX Optical Flow: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"ONNX Optical Flow Error: Failed to load model - {str(e)}")
            traceback.print_exc()

    def _claim(self) -> OpticalFlowInput | None:
        """Atomically get and clear pending batch."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process(self) -> None:
        """Process the pending batch using batched ONNX inference."""
        batch: OpticalFlowInput | None = self._claim()

        if batch is None:
            return

        output = OpticalFlowOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=False)

        # Run inference
        if batch.gpu_image_pairs:
            # Limit processing to max batch size
            pairs_to_process = batch.gpu_image_pairs[:self._max_batch]
            tracklets_to_process = batch.tracklet_ids[:self._max_batch]

            try:
                flow_tensor, inference_time_ms = self._infer(pairs_to_process)

                output = OpticalFlowOutput(
                    batch_id=batch.batch_id,
                    flow_tensor=flow_tensor,
                    tracklet_ids=tracklets_to_process,
                    processed=True,
                    inference_time_ms=inference_time_ms
                )
            except Exception as e:
                print(f"ONNX Optical Flow Error: Batched inference failed: {str(e)}")
                traceback.print_exc()
        else:
            output = OpticalFlowOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("ONNX Optical Flow Warning: Callback queue full, dropping results")

    def _infer(self, gpu_pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, float]:
        """Run batched ONNX inference on GPU frame pairs.

        All preprocessing runs on the dedicated stream for zero sync overhead.
        Uses preallocated buffers for zero allocation latency.
        Uses fixed batch size padding to avoid ONNX Runtime dynamic shape recompilation.

        Args:
            gpu_pairs: List of (prev_crop, curr_crop) tuples, each (H, W, 3) RGB float32 [0,1] on GPU

        Returns:
            (flow_tensor (actual_batch, 2, H, W) on CUDA, inference_time_ms)
        """
        actual_batch_size = len(gpu_pairs)
        if actual_batch_size == 0 or self._session is None:
            return torch.empty(0, 2, self.model_height, self.model_width, dtype=self._torch_dtype, device='cuda'), 0.0

        batch_start = time.perf_counter()

        # Get input dimensions from first pair (CHW format: 3, H, W)
        input_h, input_w = gpu_pairs[0][0].shape[1], gpu_pairs[0][0].shape[2]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)

        # Get preallocated INPUT buffer slices for fixed batch
        img1_buffer = self._img1_buffer[:self._max_batch]
        img2_buffer = self._img2_buffer[:self._max_batch]

        # Allocate OUTPUT buffer fresh each call
        flow_out = torch.empty((self._max_batch, 2, self.model_height, self.model_width), dtype=self._torch_dtype, device='cuda')

        # All preprocessing on dedicated stream (no cross-stream sync needed)
        with torch.cuda.stream(self.stream):
            # Stack GPU tensors: (B, 3, H, W) float32 RGB CHW [0,1]
            prev_chw = torch.stack([p[0] for p in gpu_pairs], dim=0)
            curr_chw = torch.stack([p[1] for p in gpu_pairs], dim=0)

            # Scale to [0, 255] for RAFT, convert to model dtype
            prev_chw = prev_chw.mul(255.0).to(self._torch_dtype)
            curr_chw = curr_chw.mul(255.0).to(self._torch_dtype)

            # Resize if needed (crop size != model size)
            if needs_resize:
                prev_chw = torch.nn.functional.interpolate(
                    prev_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False
                )
                curr_chw = torch.nn.functional.interpolate(
                    curr_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False
                )

            # Copy real batch into preallocated buffers first
            img1_buffer[:actual_batch_size].copy_(prev_chw)
            img2_buffer[:actual_batch_size].copy_(curr_chw)

            # Zero out padding region to avoid ONNX Runtime recompilation
            num_padding = self._max_batch - actual_batch_size
            if num_padding > 0:
                img1_buffer[actual_batch_size:].zero_()
                img2_buffer[actual_batch_size:].zero_()

        # CRITICAL: Synchronize stream before IOBinding to ensure all preprocessing is complete
        self.stream.synchronize()

        # Use preprocessed tensors directly for ONNX Runtime
        torch_img1 = img1_buffer
        torch_img2 = img2_buffer

        # Use IOBinding for GPU input/output
        io_binding = self._session.io_binding()
        input_names = [inp.name for inp in self._session.get_inputs()]

        # Bind inputs from PyTorch tensors
        io_binding.bind_input(
            name=input_names[0],
            device_type='cuda',
            device_id=0,
            element_type=self._model_dtype,
            shape=tuple(torch_img1.shape),
            buffer_ptr=torch_img1.data_ptr()
        )
        io_binding.bind_input(
            name=input_names[1],
            device_type='cuda',
            device_id=0,
            element_type=self._model_dtype,
            shape=tuple(torch_img2.shape),
            buffer_ptr=torch_img2.data_ptr()
        )

        # Bind output from preallocated PyTorch tensor
        output_name = self._session.get_outputs()[0].name
        io_binding.bind_output(
            name=output_name,
            device_type='cuda',
            device_id=0,
            element_type=self._model_dtype,
            shape=tuple(flow_out.shape),
            buffer_ptr=flow_out.data_ptr()
        )

        # Run inference on the same stream
        self._session.run_with_iobinding(io_binding)

        # Synchronize stream to ensure outputs are ready
        self.stream.synchronize()

        # Slice to actual batch size
        flow_tensor = flow_out[:actual_batch_size]

        inference_time_ms = (time.perf_counter() - batch_start) * 1000.0

        return flow_tensor, inference_time_ms

    def _warmup(self) -> None:
        """Initialize CUDA kernels for fixed batch size to prevent runtime recompilation."""
        if self._session is None:
            return

        try:
            # Create dummy GPU images (float32 RGB [0,1])
            dummy_img = torch.zeros((self.model_height, self.model_width, 3), dtype=torch.float32, device='cuda')

            # Warmup with max_batch size (all inference runs with this size)
            dummy_pairs = [(dummy_img, dummy_img)] * self._max_batch

            flow_tensor, ms = self._infer(dummy_pairs)

        except Exception as e:
            print(f"ONNX Optical Flow: Warmup failed (non-critical) - {str(e)}")
            traceback.print_exc()

    # CALLBACK METHODS
    def register_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Register callback to receive optical flow results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

    def _dispatch_callbacks(self) -> None:
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
                        print(f"ONNX Optical Flow Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue
