# Standard library imports
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback
from typing import Callable

# Third-party imports
import numpy as np
import torch
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings


@dataclass
class SegmentationInput:
    """Batch of images for segmentation. Images should be (H, W, 3) BGR uint8."""
    batch_id: int
    images: list[np.ndarray]
    tracklet_ids: list[int] = field(default_factory=list)  # Corresponding tracklet IDs for each image


@dataclass
class SegmentationOutput:
    """Results from segmentation. processed=False indicates batch was dropped."""
    batch_id: int
    mask_tensor: torch.Tensor | None = None    # GPU tensor (B, H, W) FP16, alpha matte [0, 1]
    tracklet_ids: list[int] = field(default_factory=list)  # Corresponding tracklet IDs
    processed: bool = True          # False if batch was dropped before processing
    inference_time_ms: float = 0.0  # For monitoring


SegmentationOutputCallback = Callable[[SegmentationOutput], None]


class RecurrentState:
    """Container for RVM recurrent states (r1, r2, r3, r4) - stored as CuPy arrays on GPU."""
    def __init__(self, r1, r2, r3, r4):
        # Accept either CuPy or NumPy arrays, always store as CuPy
        import cupy as cp
        self.r1 = r1 if isinstance(r1, cp.ndarray) else cp.asarray(r1)
        self.r2 = r2 if isinstance(r2, cp.ndarray) else cp.asarray(r2)
        self.r3 = r3 if isinstance(r3, cp.ndarray) else cp.asarray(r3)
        self.r4 = r4 if isinstance(r4, cp.ndarray) else cp.asarray(r4)


class ONNXSegmentation(Thread):
    """Asynchronous GPU person segmentation using Robust Video Matting (RVM) with ONNX Runtime.

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
            print('Segmentation WARNING: Segmentation is disabled')

        self.model_path: str = settings.model_path
        self.model_name: str = settings.segmentation_model
        self.model_file: str = f"{self.model_path}/{self.model_name}"
        self.model_width: int = settings.segmentation_width
        self.model_height: int = settings.segmentation_height
        self.resolution_name: str = settings.segmentation_resolution.name
        self.verbose: bool = settings.verbose

        # RVM-specific settings
        self.downsample_ratio: float = getattr(settings, 'segmentation_downsample_ratio', 1.0)

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
        self._state_lock: Lock = Lock()
        self._recurrent_states: dict[int, RecurrentState] = {}
        self._frame_counter: int = 0  # For periodic state reset
        self._state_reset_interval: int = settings.segmentation_reset_interval  # Reset all states every N frames (0=disabled)

        # ONNX session (initialized in run thread)
        self._session: ort.InferenceSession | None = None

        # Thread pool for parallel inference
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = min(settings.max_poses, 4)  # Limit concurrent inferences

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
            print("Warning: RVM Segmentation inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: RVM Segmentation callback thread did not stop cleanly")

    def run(self) -> None:
        """Main inference thread loop. Initializes ONNX session and processes batches."""
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
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.intra_op_num_threads = 1
            # sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.log_severity_level = 3  # 0=verbose, 1=info, 2=warning, 3=error

            self._session = ort.InferenceSession(
                self.model_file,
                sess_options=sess_options,
                providers=providers
            )

            # Verify CUDA provider is active
            providers_used = self._session.get_providers()
            if 'CUDAExecutionProvider' not in providers_used:
                print("RVM Segmentation WARNING: CUDA provider not available, using CPU")

            # Clear any existing states (in case of code changes or restarts)
            with self._state_lock:
                self._recurrent_states.clear()

            # Warmup: initialize CUDA kernels with dummy inference
            self._model_warmup(self._session)

            # Create thread pool for parallel inference
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="RVM-Worker")

            self._model_ready.set()  # Signal model is ready
            print(f"RVM Segmentation: {self.resolution_name} model loaded ({self.model_width}x{self.model_height}, {providers_used[0]}) with {self._max_workers} workers")

        except Exception as e:
            print(f"RVM Segmentation Error: Failed to load model - {str(e)}")
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
                print(f"RVM Segmentation Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: SegmentationInput) -> None:
        """Submit batch for processing. Replaces any pending (not yet started) batch.

        Dropped batches trigger callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        # Don't accept inputs until model is loaded
        if not self._model_ready.is_set():
            return

        dropped_batch: SegmentationInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                # if self.verbose:
                lag = int((time.time() - self._input_timestamp) * 1000)
                print(f"RVM Segmentation: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
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
                    print("RVM Segmentation Warning: Callback queue full, not critical for dropped notifications")
                pass  # Queue full, not critical for dropped notifications

        self._notify_update_event.set()

    def _retrieve_pending_batch(self) -> SegmentationInput | None:
        """Atomically get and clear pending batch. Once retrieved, batch cannot be cancelled.

        Returns:
            The pending batch if one exists, None otherwise
        """
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None  # Clear slot - batch is now committed
            return batch

    def _process_pending_batch(self) -> None:
        """Process the pending batch using RVM ONNX inference with recurrent states."""
        # Get pending input
        batch: SegmentationInput | None = self._retrieve_pending_batch()

        if batch is None:
            if self.verbose:
                print("RVM Segmentation Warning: No pending batch to process, this should not happen")
            return

        # Increment frame counter for periodic state reset
        self._frame_counter += 1

        # Periodic state reset as failsafe (0=disabled)
        if self._state_reset_interval > 0 and self._frame_counter % self._state_reset_interval == 0:
            if self.verbose:
                print(f"RVM Segmentation: Periodic state reset at frame {self._frame_counter}")
            self.clear_all_states()

        output = SegmentationOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Run inference
        if batch.images and self._session is not None and self._executor is not None:
            batch_start = time.perf_counter()

            # Process tracklets in parallel using thread pool
            futures = []
            for img, tracklet_id in zip(batch.images, batch.tracklet_ids):
                future = self._executor.submit(self._infer_single_image, img, tracklet_id)
                futures.append((future, tracklet_id))

            # Collect results in original order
            mask_list: list[torch.Tensor] = []
            for future, tracklet_id in futures:
                try:
                    mask = future.result()
                    mask_list.append(mask)
                except Exception as e:
                    print(f"RVM Segmentation Error: Inference failed for tracklet {tracklet_id}: {str(e)}")
                    # Create empty mask on error
                    h, w = batch.images[0].shape[:2]
                    mask_list.append(torch.zeros((h, w), dtype=torch.float16, device='cuda'))

            # Stack into batch tensor
            if mask_list:
                mask_tensor = torch.stack(mask_list)  # (B, H, W) FP16 on CUDA
            else:
                mask_tensor = torch.empty(0, dtype=torch.float16, device='cuda')

            inference_time_ms: float = (time.perf_counter() - batch_start) * 1000.0

            # Create output (processed=True by default) - keep tensor on GPU
            output = SegmentationOutput(
                batch_id=batch.batch_id,
                mask_tensor=mask_tensor,
                tracklet_ids=batch.tracklet_ids,
                processed=True,
                inference_time_ms=inference_time_ms
            )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("RVM Segmentation Warning: Callback queue full, dropping inference results")

    def _infer_single_image(self, img: np.ndarray, tracklet_id: int) -> torch.Tensor:
        """Run RVM inference on single image with per-tracklet recurrent state.

        Args:
            img: Input image (H, W, 3) BGR uint8
            tracklet_id: Unique identifier for tracking temporal state

        Returns:
            Alpha matte tensor (H, W) FP16 on CUDA, range [0, 1]
        """
        if self._session is None:
            return torch.zeros((img.shape[0], img.shape[1]), dtype=torch.float16, device='cuda')

        # Preprocess: BGR -> RGB, normalize to [0, 1], HWC -> NCHW
        img_rgb = img[:, :, ::-1].astype(np.float32)  # BGR to RGB, FP32
        img_norm = img_rgb / 255.0  # Normalize to [0, 1]
        img_chw = np.transpose(img_norm, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
        img_nchw = np.expand_dims(img_chw, axis=0).astype(np.float32)  # (1, 3, H, W) FP32

        # Get or initialize recurrent state for this tracklet
        with self._state_lock:
            state = self._recurrent_states.get(tracklet_id)

        # New TensorRT-compatible ONNX model has fixed recurrent state shapes for 256x192:
        # r1: [1, 16, 128, 96], r2: [1, 20, 64, 48], r3: [1, 40, 32, 24], r4: [1, 64, 16, 12]
        # No downsample_ratio input (hardcoded to 1.0 during export)
        # Model uses FP32 inputs (ONNX Runtime, not FP16 like old model)

        # Prepare ONNX inputs
        if state is not None:
            # Use existing states from previous frame
            onnx_inputs = {
                'src': img_nchw,
                'r1i': state.r1,
                'r2i': state.r2,
                'r3i': state.r3,
                'r4i': state.r4,
            }
        else:
            # Initialize with dynamic shapes based on model resolution (first frame)
            # r1: h/2, r2: h/4, r3: h/8, r4: h/16
            onnx_inputs = {
                'src': img_nchw,
                'r1i': np.zeros((1, 16, self.model_height // 2, self.model_width // 2), dtype=np.float32),
                'r2i': np.zeros((1, 20, self.model_height // 4, self.model_width // 4), dtype=np.float32),
                'r3i': np.zeros((1, 40, self.model_height // 8, self.model_width // 8), dtype=np.float32),
                'r4i': np.zeros((1, 64, self.model_height // 16, self.model_width // 16), dtype=np.float32),
            }

        # Run ONNX inference
        outputs = self._session.run(None, onnx_inputs)

        # ONNX outputs: [fgr, pha, r1o, r2o, r3o, r4o]
        # We need: pha (alpha matte) and update recurrent states
        pha_np = outputs[1]  # (1, 1, H, W) FP16

        # Update recurrent states for next frame
        if len(outputs) >= 6:
            new_state = RecurrentState(r1=outputs[2], r2=outputs[3], r3=outputs[4], r4=outputs[5]) # type: ignore
            with self._state_lock:
                self._recurrent_states[tracklet_id] = new_state

        # Convert numpy to PyTorch CUDA tensor for OpenGL compatibility
        pha_tensor = torch.from_numpy(pha_np).cuda()  # (1, 1, H, W) FP16
        pha_tensor = pha_tensor.squeeze(0).squeeze(0)  # (H, W)

        return pha_tensor

    def clear_tracklet_state(self, tracklet_id: int) -> None:
        """Clear recurrent state for a specific tracklet (e.g., when person leaves frame)."""
        with self._state_lock:
            if tracklet_id in self._recurrent_states:
                del self._recurrent_states[tracklet_id]
                if self.verbose:
                    print(f"RVM Segmentation: Cleared state for tracklet {tracklet_id}")

    def clear_all_states(self) -> None:
        """Clear all recurrent states (e.g., on scene change or reset)."""
        with self._state_lock:
            self._recurrent_states.clear()
            if self.verbose:
                print("RVM Segmentation: Cleared all recurrent states")

    # CALLBACK
    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks. Runs on separate thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1:
                    print("RVM Segmentation Warning: Callback queue size > 1, consumers may be falling behind")

                output: SegmentationOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"RVM Segmentation Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def register_callback(self, callback: SegmentationOutputCallback) -> None:
        """Register callback to receive segmentation results (both success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def _model_warmup(self, session: ort.InferenceSession) -> None:
        """Initialize CUDA kernels with dummy inference to avoid cold-start errors."""
        try:
            # Create dummy input matching model dimensions
            dummy_img = np.zeros((self.model_height, self.model_width, 3), dtype=np.uint8)

            # Run a few warmup inferences with tracklet ID 999 (will be cleared)
            for i in range(3):
                _ = self._infer_single_image(dummy_img, tracklet_id=999)

            # Clear warmup state
            with self._state_lock:
                if 999 in self._recurrent_states:
                    del self._recurrent_states[999]

            print("RVM Segmentation: Warmup complete")
        except Exception as e:
            print(f"RVM Segmentation: Warmup failed (non-critical) - {str(e)}")
