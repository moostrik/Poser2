# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import cupy as cp
import torch
import onnxruntime as ort

from ..cuda_image_ops import batched_bilinear_resize_inplace, normalize_hwc_to_chw_inplace
from .InOut import SegmentationInput, SegmentationOutput, SegmentationOutputCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings


class RecurrentState:
    """Container for RVM recurrent states (r1, r2, r3, r4)."""
    def __init__(self, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, r4: np.ndarray):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4


class ONNXSegmentation(Thread):
    """Asynchronous GPU person segmentation using Robust Video Matting (RVM) with ONNX Runtime.

    ONNX Runtime implementation for testing/validation. Use TRTSegmentation for production.

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
            print('Segmentation WARNING: Segmentation is disabled')

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

        # ONNX session (initialized in run thread)
        self._session: ort.InferenceSession | None = None
        self._model_dtype = np.float16  # Default, determined in setup
        self._cupy_dtype = cp.float16  # CuPy dtype for GPU preprocessing

        # Pre-allocated zero recurrent states (reused for all new tracklets to avoid recompilation)
        self._zero_r1: np.ndarray | None = None
        self._zero_r2: np.ndarray | None = None
        self._zero_r3: np.ndarray | None = None
        self._zero_r4: np.ndarray | None = None

        # GPU zero states for IOBinding path
        self._zero_r1_gpu: cp.ndarray
        self._zero_r2_gpu: cp.ndarray
        self._zero_r3_gpu: cp.ndarray
        self._zero_r4_gpu: cp.ndarray

        # Batch inference settings
        self._max_batch: int = min(settings.max_poses, 4)

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
            print("Warning: ONNX Segmentation inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: ONNX Segmentation callback thread did not stop cleanly")

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
                print(f"ONNX Segmentation Error: {str(e)}")
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
            print(f"ONNX Segmentation Warning: Batch size {len(input_batch.gpu_images)} exceeds max {self._max_batch}, will process only first {self._max_batch} images")

        dropped_batch: SegmentationInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"ONNX Segmentation: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
                self._last_dropped_batch_id = dropped_batch.batch_id

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        # Notify about dropped batch
        if dropped_batch is not None:
            dropped_output = SegmentationOutput(
                batch_id=dropped_batch.batch_id,
                tracklet_ids=dropped_batch.tracklet_ids,
                processed=False  # Mark as not processed
            )
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                if self.verbose:
                    print("ONNX Segmentation Warning: Callback queue full, not critical for dropped notifications")
                pass  # Queue full, not critical for dropped notifications

        self._notify_update_event.set()

    def _setup(self) -> None:
        """Initialize ONNX session, executor, and warmup. Called from run()."""
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
                print("ONNX Segmentation WARNING: CUDA provider not available, using CPU")

            # Determine model precision from 'src' input tensor
            src_input = self._session.get_inputs()[0]  # First input is 'src'
            input_type = src_input.type

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
            self._model_dtype = dtype_map.get(input_type, np.float16)

            # CuPy dtype for GPU preprocessing
            self._cupy_dtype = cp.float16 if self._model_dtype == np.float16 else cp.float32

            # Pre-allocate zero recurrent states on CPU (for state storage)
            self._zero_r1 = np.zeros((16, self.model_height // 2, self.model_width // 2), dtype=self._model_dtype)
            self._zero_r2 = np.zeros((20, self.model_height // 4, self.model_width // 4), dtype=self._model_dtype)
            self._zero_r3 = np.zeros((40, self.model_height // 8, self.model_width // 8), dtype=self._model_dtype)
            self._zero_r4 = np.zeros((64, self.model_height // 16, self.model_width // 16), dtype=self._model_dtype)

            # Pre-allocate zero recurrent states on GPU (for IOBinding input)
            self._zero_r1_gpu = cp.zeros((16, self.model_height // 2, self.model_width // 2), dtype=self._cupy_dtype)
            self._zero_r2_gpu = cp.zeros((20, self.model_height // 4, self.model_width // 4), dtype=self._cupy_dtype)
            self._zero_r3_gpu = cp.zeros((40, self.model_height // 8, self.model_width // 8), dtype=self._cupy_dtype)
            self._zero_r4_gpu = cp.zeros((64, self.model_height // 16, self.model_width // 16), dtype=self._cupy_dtype)

            # Clear any existing states
            self._recurrent_states.clear()

            # Warmup: initialize CUDA kernels with dummy inference
            self._model_warmup(self._session)

            self._model_ready.set()
            print(f"ONNX Segmentation: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"ONNX Segmentation Error: Failed to load model - {str(e)}")
            traceback.print_exc()

    def _retrieve_pending_batch(self) -> SegmentationInput | None:
        """Atomically get and clear pending batch. Once retrieved, cannot be cancelled."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None  # Clear slot - batch is now committed
            return batch

    def _process_pending_batch(self) -> None:
        """Process pending batch using batched ONNX inference with recurrent states."""
        batch: SegmentationInput | None = self._retrieve_pending_batch()

        if batch is None:
            return

        # Increment frame counter for periodic state reset
        self._frame_counter += 1

        # Periodic state reset as failsafe (0=disabled)
        if self._state_reset_interval > 0 and self._frame_counter % self._state_reset_interval == 0:
            if self.verbose:
                print(f"ONNX Segmentation: Periodic state reset at frame {self._frame_counter}")
            self._recurrent_states.clear()

        output = SegmentationOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=False)

        # Run inference
        gpu_images = batch.gpu_images[:self._max_batch]
        tracklets_to_process = batch.tracklet_ids[:self._max_batch]

        if gpu_images:
            try:
                mask_tensor, fgr_tensor, inference_time_ms = self._infer_batch_gpu(gpu_images, tracklets_to_process)
                output = SegmentationOutput(
                    batch_id=batch.batch_id,
                    mask_tensor=mask_tensor,
                    fgr_tensor=fgr_tensor,
                    tracklet_ids=batch.tracklet_ids,
                    processed=True,
                    inference_time_ms=inference_time_ms
                )
            except Exception as e:
                print(f"ONNX Segmentation Error: Batched inference failed: {str(e)}")
                traceback.print_exc()

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("ONNX Segmentation Warning: Callback queue full, dropping inference results")

    def _infer_batch_gpu(self, gpu_imgs: list[torch.Tensor], tracklet_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run batched ONNX inference with per-tracklet recurrent states using GPU images.

        Uses IOBinding for GPU input/output to minimize data transfers.

        Args:
            gpu_imgs: List of RGB uint8 tensors on GPU (H, W, 3)
            tracklet_ids: Tracklet IDs for each image

        Returns:
            (alpha_matte (actual_batch,H,W), foreground (actual_batch,3,H,W), inference_ms)
        """
        actual_batch_size = len(gpu_imgs)
        if actual_batch_size == 0 or self._session is None:
            torch_dtype = torch.float16 if self._model_dtype == np.float16 else torch.float32
            return torch.empty(0, dtype=torch_dtype, device='cuda'), torch.empty(0, 3, 0, 0, dtype=torch_dtype, device='cuda'), 0.0

        batch_start = time.perf_counter()

        # Get input image dimensions from first real image
        input_h, input_w = gpu_imgs[0].shape[0], gpu_imgs[0].shape[1]

        # Pad to fixed batch size using missing IDs to avoid ONNX Runtime recompilation
        num_padding = self._max_batch - actual_batch_size
        padding_ids: list[int] = []

        # Gather recurrent states for real tracklets first
        states: list[RecurrentState | None] = []
        for tid in tracklet_ids:
            states.append(self._recurrent_states.get(tid))

        if num_padding > 0:
            # Find which IDs from 0 to max_batch-1 are missing
            all_ids = set(range(self._max_batch))
            used_ids = set(tracklet_ids)
            available_ids = list(all_ids - used_ids)

            # Add padding with zero images at SAME SIZE as input images (not model size)
            for i in range(num_padding):
                padding_id = available_ids[i]
                gpu_imgs.append(torch.zeros((input_h, input_w, 3), dtype=torch.uint8, device='cuda'))
                tracklet_ids.append(padding_id)
                padding_ids.append(padding_id)
                states.append(None)  # Padding gets zero states

        # Stack torch tensors and convert to CuPy for preprocessing kernels
        batch_torch = torch.stack(gpu_imgs, dim=0)  # (B, H, W, 3)
        batch_hwc = cp.asarray(batch_torch)  # Zero-copy view

        # Check if resize is needed (crop size != model size)
        src_h, src_w = batch_hwc.shape[1:3]
        if src_h != self.model_height or src_w != self.model_width:
            # Batched resize into new buffer
            resized = cp.empty((batch_hwc.shape[0], self.model_height, self.model_width, 3), dtype=cp.uint8)
            batched_bilinear_resize_inplace(batch_hwc, resized)
            batch_hwc = resized

        # Fused normalize [0,1] + HWC->CHW into preallocated buffer
        # Note: Images are already RGB from GPUCropProcessor, no BGR->RGB needed
        src_batch = cp.empty((batch_hwc.shape[0], 3, self.model_height, self.model_width), dtype=self._cupy_dtype)
        normalize_hwc_to_chw_inplace(batch_hwc, src_batch)

        # Prepare batched recurrent state inputs on GPU
        r1i_list = []
        r2i_list = []
        r3i_list = []
        r4i_list = []

        for state in states:
            if state is not None:
                r1i_list.append(cp.asarray(state.r1[0]))
                r2i_list.append(cp.asarray(state.r2[0]))
                r3i_list.append(cp.asarray(state.r3[0]))
                r4i_list.append(cp.asarray(state.r4[0]))
            else:
                r1i_list.append(self._zero_r1_gpu)
                r2i_list.append(self._zero_r2_gpu)
                r3i_list.append(self._zero_r3_gpu)
                r4i_list.append(self._zero_r4_gpu)

        # Stack recurrent states into batches
        r1i = cp.stack(r1i_list, axis=0)
        r2i = cp.stack(r2i_list, axis=0)
        r3i = cp.stack(r3i_list, axis=0)
        r4i = cp.stack(r4i_list, axis=0)

        # Synchronize CuPy operations before ONNX Runtime reads the data
        cp.cuda.Device().synchronize()

        # Convert CuPy arrays to PyTorch tensors (zero-copy via DLPack)
        torch_src = torch.as_tensor(src_batch, device='cuda:0')
        torch_r1i = torch.as_tensor(r1i, device='cuda:0')
        torch_r2i = torch.as_tensor(r2i, device='cuda:0')
        torch_r3i = torch.as_tensor(r3i, device='cuda:0')
        torch_r4i = torch.as_tensor(r4i, device='cuda:0')

        # Use IOBinding for GPU input/output
        io_binding = self._session.io_binding()
        element_type = np.float16 if self._model_dtype == np.float16 else np.float32

        # Bind inputs from PyTorch tensors
        io_binding.bind_input('src', device_type='cuda', device_id=0,
                              element_type=element_type, shape=tuple(torch_src.shape),
                              buffer_ptr=torch_src.data_ptr())
        io_binding.bind_input('r1i', device_type='cuda', device_id=0,
                              element_type=element_type, shape=tuple(torch_r1i.shape),
                              buffer_ptr=torch_r1i.data_ptr())
        io_binding.bind_input('r2i', device_type='cuda', device_id=0,
                              element_type=element_type, shape=tuple(torch_r2i.shape),
                              buffer_ptr=torch_r2i.data_ptr())
        io_binding.bind_input('r3i', device_type='cuda', device_id=0,
                              element_type=element_type, shape=tuple(torch_r3i.shape),
                              buffer_ptr=torch_r3i.data_ptr())
        io_binding.bind_input('r4i', device_type='cuda', device_id=0,
                              element_type=element_type, shape=tuple(torch_r4i.shape),
                              buffer_ptr=torch_r4i.data_ptr())

        # Bind outputs to GPU
        io_binding.bind_output('fgr', device_type='cuda', device_id=0)
        io_binding.bind_output('pha', device_type='cuda', device_id=0)
        io_binding.bind_output('r1o', device_type='cuda', device_id=0)
        io_binding.bind_output('r2o', device_type='cuda', device_id=0)
        io_binding.bind_output('r3o', device_type='cuda', device_id=0)
        io_binding.bind_output('r4o', device_type='cuda', device_id=0)

        # Run inference
        self._session.run_with_iobinding(io_binding)

        # Synchronize to ensure outputs are ready before accessing
        torch.cuda.synchronize()

        # Get outputs as OrtValues on GPU
        outputs = io_binding.get_outputs()

        # Convert to numpy for recurrent state storage (copies from GPU)
        fgr_np = outputs[0].numpy()[:actual_batch_size]  # (actual_batch, 3, H, W)
        pha_np = outputs[1].numpy()[:actual_batch_size]  # (actual_batch, 1, H, W)
        r1o_np = outputs[2].numpy()
        r2o_np = outputs[3].numpy()
        r3o_np = outputs[4].numpy()
        r4o_np = outputs[5].numpy()

        # Create new states ONLY for real tracklets (skip padding IDs)
        new_states_dict: dict[int, RecurrentState] = {}
        for i, tid in enumerate(tracklet_ids):
            if tid not in padding_ids:
                new_states_dict[tid] = RecurrentState(
                    r1=r1o_np[i:i+1],
                    r2=r2o_np[i:i+1],
                    r3=r3o_np[i:i+1],
                    r4=r4o_np[i:i+1]
                )

        # Update recurrent states
        self._recurrent_states = new_states_dict

        # Convert to PyTorch CUDA tensors
        fgr_tensor = torch.from_numpy(fgr_np).cuda()  # (actual_batch, 3, H, W)
        pha_tensor = torch.from_numpy(pha_np).cuda().squeeze(1)  # (actual_batch, H, W)

        inference_time_ms = (time.perf_counter() - batch_start) * 1000.0

        return pha_tensor, fgr_tensor, inference_time_ms

    def _model_warmup(self, session: ort.InferenceSession) -> None:
        """Initialize CUDA kernels for fixed batch size to prevent runtime recompilation."""
        try:
            # Create dummy GPU images
            dummy_img = torch.zeros((self.model_height, self.model_width, 3), dtype=torch.uint8, device='cuda')
            dummy_images = [dummy_img] * self._max_batch
            dummy_ids = list(range(self._max_batch))

            pha, fgr, ms = self._infer_batch_gpu(dummy_images, dummy_ids)

            # Clear warmup states
            self._recurrent_states.clear()

        except Exception as e:
            print(f"ONNX Segmentation: Warmup failed (non-critical) - {str(e)}")
            traceback.print_exc()

    # CALLBACK METHODS
    def register_callback(self, callback: SegmentationOutputCallback) -> None:
        """Register callback to receive segmentation results (success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: SegmentationOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks on dedicated thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1:
                    print("ONNX Segmentation Warning: Callback queue size > 1, consumers may be falling behind")

                output: SegmentationOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"ONNX Segmentation Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue
