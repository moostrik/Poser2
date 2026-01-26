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

# Reuse dataclasses from RVMSegmentation
from modules.pose.batch.segmentation.ONNXSegmentation import (
    SegmentationInput,
    SegmentationOutput,
    SegmentationOutputCallback,
    RecurrentState
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

from modules.pose.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock

from modules.utils.HotReloadMethods import HotReloadMethods


class TensorRTSegmentation(Thread):
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
            print('TensorRT Segmentation WARNING: Segmentation is disabled')

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
        self._state_lock: Lock = Lock()
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

        # Preallocated GPU output buffers (reused across frames for efficiency)
        self._output_buffers: dict[str, cp.ndarray] = {}

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
            print("Warning: TensorRT Segmentation inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: TensorRT Segmentation callback thread did not stop cleanly")

    def run(self) -> None:
        """Main inference thread loop. Initializes TensorRT engine and processes batches."""
        try:
            # Acquire global lock to prevent concurrent Myelin graph loading
            with get_init_lock():
                runtime = get_tensorrt_runtime()

                print(f"TensorRT Segmentation: Loading engine from {self.model_file}")
                with open(self.model_file, 'rb') as f:
                    engine_data = f.read()

                self.engine = runtime.deserialize_cuda_engine(engine_data)
                if self.engine is None:
                    print("TensorRT Segmentation ERROR: Failed to load engine")
                    return

                self.context = self.engine.create_execution_context()

                # Create dedicated CUDA stream
                self.stream = cp.cuda.Stream(non_blocking=True)

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
                print(f"TensorRT Segmentation ERROR: Expected 5 inputs, got {len(self.input_names)}")
                return

            if len(self.output_names) != 6:
                print(f"TensorRT Segmentation ERROR: Expected 6 outputs, got {len(self.output_names)}")
                return

            # Clear any existing states
            with self._state_lock:
                self._recurrent_states.clear()

            # Preallocate output buffers (reused across all inferences)
            self._output_buffers = {
                'fgr': cp.empty((1, 3, self.model_height, self.model_width), dtype=cp.float32),
                'pha': cp.empty((1, 1, self.model_height, self.model_width), dtype=cp.float32),
                'r1o': cp.empty((1, 16, self.model_height // 2, self.model_width // 2), dtype=cp.float32),
                'r2o': cp.empty((1, 20, self.model_height // 4, self.model_width // 4), dtype=cp.float32),
                'r3o': cp.empty((1, 40, self.model_height // 8, self.model_width // 8), dtype=cp.float32),
                'r4o': cp.empty((1, 64, self.model_height // 16, self.model_width // 16), dtype=cp.float32),
            }

            # Create thread pool for parallel inference
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="TRT-RVM-Worker")

            self._model_ready.set()
            print(f"TensorRT Segmentation: {self.resolution_name} model loaded ({self.model_width}x{self.model_height}) with {self._max_workers} workers")

        except Exception as e:
            print(f"TensorRT Segmentation Error: Failed to load model - {str(e)}")
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
                print(f"TensorRT Segmentation Error: {str(e)}")
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
                print(f"TensorRT Segmentation: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
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
                print(f"TensorRT Segmentation: Periodic state reset at frame {self._frame_counter}")
            self.clear_all_states()

        output = SegmentationOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Run inference
        if batch.images and self._executor is not None:
            # Process tracklets in parallel using thread pool
            futures = []
            inference_times = []
            for img, tracklet_id in zip(batch.images, batch.tracklet_ids):
                future = self._executor.submit(self._infer_single_image, img, tracklet_id)
                futures.append((future, tracklet_id))

            # Collect results in original order
            mask_list: list[torch.Tensor] = []
            for future, tracklet_id in futures:
                try:
                    mask, inf_time = future.result()
                    mask_list.append(mask)
                    inference_times.append(inf_time)
                except Exception as e:
                    print(f"TensorRT Segmentation Error: Inference failed for tracklet {tracklet_id}: {str(e)}")
                    # Create empty mask on error
                    h, w = batch.images[0].shape[:2]
                    mask_list.append(torch.zeros((h, w), dtype=torch.float32, device='cuda'))
                    inference_times.append(0.0)

            # Stack into batch tensor
            if mask_list:
                mask_tensor = torch.stack(mask_list)  # (B, H, W) FP32 on CUDA
            else:
                mask_tensor = torch.empty(0, dtype=torch.float32, device='cuda')

            # Use max inference time from parallel workers (actual lock-held time)
            inference_time_ms: float = max(inference_times) if inference_times else 0.0

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
            print("TensorRT Segmentation Warning: Callback queue full, dropping inference results")

    def _infer_single_image(self, img: np.ndarray, tracklet_id: int) -> tuple[torch.Tensor, float]:
        """Run RVM TensorRT inference on single image with per-tracklet recurrent state.

        Args:
            img: Input image (H, W, 3) BGR uint8
            tracklet_id: Unique identifier for tracking temporal state

        Returns:
            Tuple of (alpha matte tensor (H, W) FP32 on CUDA [0, 1], inference_time_ms)
        """
        h, w = img.shape[:2]

        # Move raw image to GPU first, then do all preprocessing on GPU
        img_gpu = cp.asarray(img)  # (H, W, 3) uint8 BGR -> GPU
        img_rgb_gpu = img_gpu[:, :, ::-1]  # BGR to RGB on GPU (zero-copy view)
        img_float_gpu = img_rgb_gpu.astype(cp.float32)  # uint8 -> float32 on GPU
        img_norm_gpu = img_float_gpu / 255.0  # Normalize to [0, 1] on GPU
        img_chw_gpu = cp.transpose(img_norm_gpu, (2, 0, 1))  # (3, H, W) on GPU
        img_chw_gpu = cp.ascontiguousarray(img_chw_gpu)  # Ensure contiguous memory for TensorRT
        src_gpu = cp.expand_dims(img_chw_gpu, axis=0)  # (1, 3, H, W) on GPU

        # Get or initialize recurrent state for this tracklet
        with self._state_lock:
            state = self._recurrent_states.get(tracklet_id)

        # Prepare recurrent state inputs (dynamic shapes based on model resolution)
        if state is not None:
            # States are already CuPy arrays on GPU - use directly
            r1_gpu = state.r1
            r2_gpu = state.r2
            r3_gpu = state.r3
            r4_gpu = state.r4
        else:
            # Initialize with dynamic shapes for first frame (r1: h/2, r2: h/4, r3: h/8, r4: h/16)
            r1_gpu = cp.zeros((1, 16, self.model_height // 2, self.model_width // 2), dtype=cp.float32)
            r2_gpu = cp.zeros((1, 20, self.model_height // 4, self.model_width // 4), dtype=cp.float32)
            r3_gpu = cp.zeros((1, 40, self.model_height // 8, self.model_width // 8), dtype=cp.float32)
            r4_gpu = cp.zeros((1, 64, self.model_height // 16, self.model_width // 16), dtype=cp.float32)

        # Run inference with global lock to prevent race conditions
        with get_exec_lock():
            # Start timing after acquiring lock (excludes wait time)
            lock_acquired = time.perf_counter()

            # Set input shapes (must be inside lock!)
            self.context.set_input_shape('src', src_gpu.shape)
            self.context.set_input_shape('r1i', r1_gpu.shape)
            self.context.set_input_shape('r2i', r2_gpu.shape)
            self.context.set_input_shape('r3i', r3_gpu.shape)
            self.context.set_input_shape('r4i', r4_gpu.shape)

            # Use preallocated GPU output buffers (reused across frames)
            fgr_gpu = self._output_buffers['fgr']
            pha_gpu = self._output_buffers['pha']
            r1o_gpu = self._output_buffers['r1o']
            r2o_gpu = self._output_buffers['r2o']
            r3o_gpu = self._output_buffers['r3o']
            r4o_gpu = self._output_buffers['r4o']

            # Set tensor addresses
            self.context.set_tensor_address('src', src_gpu.data.ptr)
            self.context.set_tensor_address('r1i', r1_gpu.data.ptr)
            self.context.set_tensor_address('r2i', r2_gpu.data.ptr)
            self.context.set_tensor_address('r3i', r3_gpu.data.ptr)
            self.context.set_tensor_address('r4i', r4_gpu.data.ptr)
            self.context.set_tensor_address('fgr', fgr_gpu.data.ptr)
            self.context.set_tensor_address('pha', pha_gpu.data.ptr)
            self.context.set_tensor_address('r1o', r1o_gpu.data.ptr)
            self.context.set_tensor_address('r2o', r2o_gpu.data.ptr)
            self.context.set_tensor_address('r3o', r3o_gpu.data.ptr)
            self.context.set_tensor_address('r4o', r4o_gpu.data.ptr)

            # Execute inference
            with self.stream:
                self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()

            # Calculate time spent in lock (actual inference time)
            inference_time = (time.perf_counter() - lock_acquired) * 1000.0

        # Update recurrent states for next frame (keep on GPU as CuPy arrays)
        new_state = RecurrentState(
            r1=r1o_gpu,
            r2=r2o_gpu,
            r3=r3o_gpu,
            r4=r4o_gpu
        )
        with self._state_lock:
            self._recurrent_states[tracklet_id] = new_state

        # Convert CuPy array to PyTorch tensor (zero-copy, both on CUDA)
        pha_tensor = torch.as_tensor(pha_gpu, device='cuda')  # (1, 1, H, W)
        pha_tensor = pha_tensor.squeeze(0).squeeze(0)  # (H, W)

        return pha_tensor, inference_time

    def clear_tracklet_state(self, tracklet_id: int) -> None:
        """Clear recurrent state for a specific tracklet (e.g., when person leaves frame)."""
        with self._state_lock:
            if tracklet_id in self._recurrent_states:
                del self._recurrent_states[tracklet_id]
                if self.verbose:
                    print(f"TensorRT Segmentation: Cleared state for tracklet {tracklet_id}")

    def clear_all_states(self) -> None:
        """Clear all recurrent states (e.g., on scene change or reset)."""
        with self._state_lock:
            self._recurrent_states.clear()
            if self.verbose:
                print("TensorRT Segmentation: Cleared all recurrent states")

    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks. Runs on separate thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1:
                    print("TensorRT Segmentation Warning: Callback queue size > 1, consumers may be falling behind")

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
