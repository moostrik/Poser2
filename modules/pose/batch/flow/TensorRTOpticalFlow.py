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

# Reuse dataclasses from RAFTOpticalFlow
from modules.pose.batch.flow.ONNXOpticalFlow import (
    OpticalFlowInput,
    OpticalFlowOutput,
    OpticalFlowOutputCallback
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings

from modules.pose.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock


class TensorRTOpticalFlow(Thread):
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
        self.model_name: str = 'raft-sintel_256x192_iter12_batch3.trt'  # TensorRT engine
        self.model_file: str = f"{self.model_path}/{self.model_name}"
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

        # Thread pool for parallel inference
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = min(getattr(settings, 'max_poses', 3), 4)

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
            # Acquire global lock to prevent concurrent Myelin graph loading
            with get_init_lock():
                runtime = get_tensorrt_runtime()

                print(f"TensorRT Optical Flow: Loading engine from {self.model_file}")
                with open(self.model_file, 'rb') as f:
                    engine_data = f.read()

                self.engine = runtime.deserialize_cuda_engine(engine_data)
                if self.engine is None:
                    print("TensorRT Optical Flow ERROR: Failed to load engine")
                    return

                self.context = self.engine.create_execution_context()

                # Create dedicated CUDA stream (matching Detection config)
                self.stream = cp.cuda.Stream(non_blocking=True)

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
                print(f"TensorRT Optical Flow ERROR: Expected 2 inputs, got {len(self.input_names)}")
                return

            # Create thread pool for parallel inference
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="TRT-RAFT-Worker")

            self._model_ready.set()
            print(f"TensorRT Optical Flow: Model '{self.model_name}' loaded successfully with {self._max_workers} workers")

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
        if batch.frame_pairs and self._executor is not None:
            # Process frame pairs in parallel using thread pool
            futures = []
            inference_times = []
            for (prev_frame, curr_frame), tracklet_id in zip(batch.frame_pairs, batch.tracklet_ids):
                future = self._executor.submit(self._infer_optical_flow, prev_frame, curr_frame)
                futures.append((future, tracklet_id))

            # Collect results in original order
            flow_list: list[torch.Tensor] = []
            for future, tracklet_id in futures:
                try:
                    flow, inf_time = future.result()
                    flow_list.append(flow)
                    inference_times.append(inf_time)
                except Exception as e:
                    print(f"TensorRT Optical Flow Error: Inference failed for tracklet {tracklet_id}: {str(e)}")
                    # Create zero flow on error
                    h, w = batch.frame_pairs[0][0].shape[:2]
                    flow_list.append(torch.zeros((2, h, w), dtype=torch.float32, device='cuda'))
                    inference_times.append(0.0)

            # Stack into batch tensor
            if flow_list:
                flow_tensor = torch.stack(flow_list)  # (B, 2, H, W) FP32 on CUDA
            else:
                flow_tensor = torch.empty(0, dtype=torch.float32, device='cuda')

            # Use max inference time from parallel workers (actual lock-held time)
            inference_time_ms: float = max(inference_times) if inference_times else 0.0

            output = OpticalFlowOutput(
                batch_id=batch.batch_id,
                flow_tensor=flow_tensor,
                tracklet_ids=batch.tracklet_ids,
                processed=True,
                inference_time_ms=inference_time_ms
            )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TensorRT Optical Flow Warning: Callback queue full, dropping results")

    def _infer_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> tuple[torch.Tensor, float]:
        """Run TensorRT RAFT inference on frame pair. Returns (flow_tensor, inference_time_ms).

        Args:
            prev_frame: Previous frame (H, W, 3) BGR uint8
            curr_frame: Current frame (H, W, 3) BGR uint8

        Returns:
            Flow tensor (2, H, W) FP32 on CUDA, where [0] is x-flow, [1] is y-flow
        """
        h, w = prev_frame.shape[:2]

        # Preprocess: BGR -> RGB, HWC -> NCHW, keep in [0, 255] range (RAFT expects this)
        def preprocess(img: np.ndarray) -> cp.ndarray:
            img_rgb = img[:, :, ::-1].astype(np.float32)  # BGR to RGB, FP32
            img_chw = np.transpose(img_rgb, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
            img_nchw = np.expand_dims(img_chw, axis=0)  # (1, 3, H, W)
            return cp.asarray(img_nchw)  # Move to GPU

        image1_gpu = preprocess(prev_frame)
        image2_gpu = preprocess(curr_frame)

        # Run inference with global lock to prevent race conditions
        with get_exec_lock():
            # Start timing after acquiring lock (excludes wait time)
            lock_acquired = time.perf_counter()

            # Set input shapes for current batch (must be inside lock!)
            self.context.set_input_shape(self.input_names[0], image1_gpu.shape)
            self.context.set_input_shape(self.input_names[1], image2_gpu.shape)

            # Get output shape
            output_shape = self.context.get_tensor_shape(self.output_name)

            # Allocate GPU output buffer
            flow_gpu = cp.empty(output_shape, dtype=cp.float32)

            # Set tensor addresses
            self.context.set_tensor_address(self.input_names[0], image1_gpu.data.ptr)
            self.context.set_tensor_address(self.input_names[1], image2_gpu.data.ptr)
            self.context.set_tensor_address(self.output_name, flow_gpu.data.ptr)

            # Execute inference
            with self.stream:
                self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()

            # Calculate time spent in lock (actual inference time)
            inference_time = (time.perf_counter() - lock_acquired) * 1000.0

        # Convert CuPy array to PyTorch tensor (zero-copy, both on CUDA)
        flow_tensor = torch.as_tensor(flow_gpu, device='cuda')  # (1, 2, H, W)
        flow_tensor = flow_tensor.squeeze(0)  # (2, H, W)

        return flow_tensor, inference_time

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
