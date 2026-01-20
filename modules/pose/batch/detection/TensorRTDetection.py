# Standard library imports
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback
from typing import Callable

# Third-party imports
import numpy as np
import tensorrt as trt
import cupy as cp

# Reuse dataclasses from MMDetection
from modules.pose.batch.detection.InOut import DetectionInput, DetectionOutput, PoseDetectionOutputCallback

from modules.pose.Settings import Settings
from modules.pose.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock

# ImageNet normalization (RGB order)
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)


class TensorRTDetection(Thread):
    """Asynchronous GPU pose detection using TensorRT.

    Optimized inference using TensorRT engines for maximum performance.
    Architecture identical to ONNXDetection for drop-in replacement.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.model_file: str = settings.model_path + '/' + settings.pose_model
        self.model_width: int = settings.pose_width
        self.model_height: int = settings.pose_height
        self.num_keypoints: int = 17  # RTMPose COCO format
        self.simcc_split_ratio: float = 2.0  # From RTMPose config

        self.verbose: bool = settings.verbose
        self.resolution_name: str = settings.pose_resolution.name

        # Thread coordination
        self._shutdown_event: Event = Event()
        self._notify_update_event: Event = Event()
        self._model_ready: Event = Event()

        # Input queue (single slot)
        self._input_lock: Lock = Lock()
        self._pending_batch: DetectionInput | None = None
        self._input_timestamp: float = time.time()
        self._last_dropped_batch_id: int = 0

        # Callbacks
        self._callback_lock: Lock = Lock()
        self._callbacks: set[PoseDetectionOutputCallback] = set()
        self._callback_queue: Queue[DetectionOutput | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._callback_worker_loop, daemon=True)

        # TensorRT engine and context (initialized in run())
        self.engine: trt.ICudaEngine # type: ignore
        self.context: trt.IExecutionContext # type: ignore
        self.input_name: str
        self.output_names: list[str]
        self.stream: cp.cuda.Stream

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model_ready.is_set() and not self._shutdown_event.is_set() and self.is_alive()

    def start(self) -> None:
        self._callback_thread.start()
        super().start()

    def stop(self) -> None:
        self._shutdown_event.set()
        self._notify_update_event.set()
        self.join(timeout=2.0)

        if self.is_alive():
            print("Warning: TensorRT inference thread did not stop cleanly")

        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: TensorRT callback thread did not stop cleanly")

    def run(self) -> None:
        # Acquire global lock to prevent concurrent Myelin graph loading
        with get_init_lock():
            runtime = get_tensorrt_runtime()

            print(f"TensorRT Detection: Loading engine from {self.model_file}")
            with open(self.model_file, 'rb') as f:
                engine_data = f.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                print("TensorRT Detection ERROR: Failed to load engine")
                return

            self.context = self.engine.create_execution_context()

            # Create dedicated CUDA stream for better performance
            self.stream = cp.cuda.Stream(non_blocking=True)

        # Lock released - continue with setup
        # Initialize output names list
        self.output_names = []

        # Print tensor information
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            if mode == trt.TensorIOMode.INPUT: # type: ignore
                self.input_name = name
            else:
                self.output_names.append(name)

        self._model_ready.set()
        print(f"TensorRT Detection: {self.resolution_name} model ready ({self.model_width}x{self.model_height})")

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process_pending_batch()
            except Exception as e:
                print(f"TensorRT Detection Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: DetectionInput) -> None:
        """Submit batch for processing."""
        if self._shutdown_event.is_set():
            return

        dropped_batch: DetectionInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                # if self.verbose:
                lag = int((time.time() - self._input_timestamp) * 1000)
                print(f"TensorRT Detection: Dropped batch {dropped_batch.batch_id} with lag {lag} ms")
                self._last_dropped_batch_id = dropped_batch.batch_id

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        if dropped_batch is not None:
            dropped_output = DetectionOutput(batch_id=dropped_batch.batch_id, processed=False)
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                pass

        self._notify_update_event.set()

    def _retrieve_pending_batch(self) -> DetectionInput | None:
        """Atomically get and clear pending batch."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process_pending_batch(self) -> None:
        batch: DetectionInput | None = self._retrieve_pending_batch()

        if batch is None:
            return

        output = DetectionOutput(batch_id=batch.batch_id, processed=True)

        # Run inference
        if batch.images:
            # Preprocess and run inference (timing done inside _infer_batch)
            keypoints, scores, inference_time_ms = self._infer_batch(batch.images, self.model_width, self.model_height)

            # Normalize coordinates to [0, 1]
            keypoints[:, :, 0] /= self.model_width
            keypoints[:, :, 1] /= self.model_height

            # Convert to list format
            point_list = [keypoints[i] for i in range(len(keypoints))]
            score_list = [scores[i] for i in range(len(scores))]

            output = DetectionOutput(
                batch_id=batch.batch_id,
                point_batch=point_list,
                score_batch=score_list,
                processed=True,
                inference_time_ms=inference_time_ms
            )
        else:
            output = DetectionOutput(batch_id=batch.batch_id, processed=True)

        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TensorRT Detection Warning: Callback queue full")

    def _callback_worker_loop(self) -> None:
        """Dispatch results to callbacks."""
        while not self._shutdown_event.is_set():
            try:
                output: DetectionOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"TensorRT Detection Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def register_callback(self, callback: PoseDetectionOutputCallback) -> None:
        """Register callback to receive results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    # STATIC METHODS

    @staticmethod
    def _preprocess_batch(imgs: list[np.ndarray], width: int, height: int) -> np.ndarray:
        """Convert BGR uint8 images to normalized RGB FP32 batch (B, 3, H, W)."""
        if not imgs:
            return np.empty((0, 3, height, width), dtype=np.float32)

        batch_hwc = np.stack(imgs, axis=0).astype(np.float32)
        batch_rgb = batch_hwc[:, :, :, ::-1]
        batch_chw = batch_rgb.transpose(0, 3, 1, 2)
        batch_norm = (batch_chw - IMAGENET_MEAN) / IMAGENET_STD

        return batch_norm

    @staticmethod
    def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float, apply_softmax: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Decode SimCC using MMPose's exact method from get_simcc_maximum.

        Reference: mmpose/codecs/utils/post_processing.py::get_simcc_maximum
        """
        N, K, _ = simcc_x.shape

        # Reshape for processing: (N, K, W) -> (N*K, W)
        simcc_x_flat = simcc_x.reshape(N * K, -1)
        simcc_y_flat = simcc_y.reshape(N * K, -1)

        if apply_softmax:
            # Exact MMPose softmax implementation
            simcc_x_flat = simcc_x_flat - np.max(simcc_x_flat, axis=1, keepdims=True)
            simcc_y_flat = simcc_y_flat - np.max(simcc_y_flat, axis=1, keepdims=True)
            ex, ey = np.exp(simcc_x_flat), np.exp(simcc_y_flat)
            simcc_x_flat = ex / np.sum(ex, axis=1, keepdims=True)
            simcc_y_flat = ey / np.sum(ey, axis=1, keepdims=True)

        # Get coordinates from argmax
        x_locs = np.argmax(simcc_x_flat, axis=1)
        y_locs = np.argmax(simcc_y_flat, axis=1)

        # Get max values (scores) at predicted locations
        max_val_x = np.amax(simcc_x_flat, axis=1)
        max_val_y = np.amax(simcc_y_flat, axis=1)

        # MMPose takes MINIMUM of x and y scores
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        scores = max_val_x

        # Convert to coordinates
        x_coords = x_locs.astype(np.float32) / split_ratio
        y_coords = y_locs.astype(np.float32) / split_ratio

        # Reshape back to (N, K, 2) and (N, K)
        keypoints = np.stack([x_coords, y_coords], axis=-1).reshape(N, K, 2)
        scores = scores.reshape(N, K)

        # Mark invalid keypoints
        keypoints[scores <= 0.] = -1


        scores = np.clip(scores, 0.0, 1.0)

        return keypoints, scores

    def _infer_batch(self, imgs: list[np.ndarray], width: int, height: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Run TensorRT inference on batch of images. Returns (keypoints, scores, inference_time_ms)."""
        if not imgs:
            return np.empty((0, 17, 2)), np.empty((0, 17)), 0.0

        # Preprocess
        batch = self._preprocess_batch(imgs, width, height)


        # Use CuPy for GPU arrays
        batch_gpu = cp.asarray(batch)

        # Run inference with global lock to prevent race conditions
        with get_exec_lock():
            # Start timing after acquiring lock (excludes wait time)
            lock_acquired = time.perf_counter()

            # Set input shape for current batch (must be inside lock!)
            self.context.set_input_shape(self.input_name, batch_gpu.shape)

            # Get output shapes from engine
            output0_shape = self.context.get_tensor_shape(self.output_names[0])
            output1_shape = self.context.get_tensor_shape(self.output_names[1])

            # Allocate GPU output buffers
            output0_gpu = cp.empty(output0_shape, dtype=cp.float32)
            output1_gpu = cp.empty(output1_shape, dtype=cp.float32)

            # Set tensor addresses
            self.context.set_tensor_address(self.input_name, batch_gpu.data.ptr)
            self.context.set_tensor_address(self.output_names[0], output0_gpu.data.ptr)
            self.context.set_tensor_address(self.output_names[1], output1_gpu.data.ptr)

            # Execute inference
            with self.stream:
                self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()

        # Copy back to CPU
        simcc_x = cp.asnumpy(output0_gpu)
        simcc_y = cp.asnumpy(output1_gpu)

        # Return time spent in lock (actual inference time)
        inference_time = (time.perf_counter() - lock_acquired) * 1000.0

        # Decode SimCC outputs
        keypoints, scores = TensorRTDetection._decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)

        return keypoints, scores, inference_time