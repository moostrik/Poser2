# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import tensorrt as trt
import cupy as cp

# Reuse dataclasses from MMDetection
from modules.pose.batch.detection.InOut import DetectionInput, DetectionOutput, PoseDetectionOutputCallback

from modules.pose.Settings import Settings
from modules.pose.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock

# ImageNet normalization (RGB order) - CPU constants for reference
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)


class TRTDetection(Thread):
    """Asynchronous GPU pose detection using RTMPose with TensorRT.

    TensorRT-optimized inference for maximum performance. Drop-in replacement for ONNXDetection.

    Single-slot queue: only the most recent batch waits processing; older pending batches dropped.
    Batches already processing cannot be cancelled.

    All results delivered via callbacks in notification order.
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

        # Preallocated GPU buffers (allocated after engine loads)
        self._max_batch: int = 8
        self._batch_buffers: dict[str, cp.ndarray]
        self._mean_gpu: cp.ndarray  # ImageNet mean on GPU
        self._std_gpu: cp.ndarray  # ImageNet std on GPU

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
        self._setup()

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
        """Submit batch for processing. Replaces pending batch if not yet started.

        Dropped batches receive callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        # Validate batch size
        if len(input_batch.images) > self._max_batch:
            print(f"TensorRT Detection Warning: Batch size {len(input_batch.images)} exceeds max {self._max_batch}, will process only first {self._max_batch} images")

        dropped_batch: DetectionInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
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

    def _setup(self) -> None:
        """Initialize TensorRT engine, context, and buffers. Called from run()."""
        try:
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

                # Create dedicated CUDA stream
                self.stream = cp.cuda.Stream(non_blocking=True)

                # Tune CuPy memory pool for better performance
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=2 * 1024**3)  # 2GB limit (up from 512MB default)

            # Lock released - continue with setup

            # Get input/output names
            self.output_names = []

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)

                if mode == trt.TensorIOMode.INPUT: # type: ignore
                    self.input_name = name
                else:
                    self.output_names.append(name)

            # Validate expected tensor counts
            if len(self.output_names) != 2:
                print(f"TensorRT Detection ERROR: Expected 2 outputs, got {len(self.output_names)}")
                return

            # Determine model precision from input tensor
            input_dtype = self.engine.get_tensor_dtype(self.input_name)
            precision_map = {
                trt.DataType.FLOAT: "FP32",   # type: ignore
                trt.DataType.HALF: "FP16",    # type: ignore
                trt.DataType.INT8: "INT8",    # type: ignore
            }
            self.model_precision = precision_map.get(input_dtype, "UNKNOWN")

            # Map TensorRT dtype to CuPy dtype for buffer allocations
            dtype_to_cupy = {
                trt.DataType.FLOAT: cp.float32, # type: ignore
                trt.DataType.HALF: cp.float16,  # type: ignore
                trt.DataType.INT8: cp.int8,     # type: ignore
            }
            self._model_dtype = dtype_to_cupy.get(input_dtype, cp.float32)  # Default to FP32

            # Preallocate GPU buffers for max batch size
            # SimCC output dimensions: simcc_x = (B, 17, W*2), simcc_y = (B, 17, H*2)
            simcc_x_width = self.model_width * 2
            simcc_y_height = self.model_height * 2

            # Preallocate BATCHED buffers for batch inference
            self._batch_buffers = {
                'img_uint8': cp.empty((self._max_batch, self.model_height, self.model_width, 3), dtype=cp.uint8),
                'input': cp.empty((self._max_batch, 3, self.model_height, self.model_width), dtype=self._model_dtype),
                'simcc_x': cp.empty((self._max_batch, self.num_keypoints, simcc_x_width), dtype=self._model_dtype),
                'simcc_y': cp.empty((self._max_batch, self.num_keypoints, simcc_y_height), dtype=self._model_dtype),
            }

            # GPU constants for normalization (CHW format for broadcasting)
            self._mean_gpu = cp.asarray(IMAGENET_MEAN, dtype=self._model_dtype)
            self._std_gpu = cp.asarray(IMAGENET_STD, dtype=self._model_dtype)

            self._model_ready.set()
            print(f"TensorRT Detection: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"TensorRT Detection Error: Failed to load model - {str(e)}")
            traceback.print_exc()
            return

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
            # Limit processing to max batch size
            images_to_process = batch.images[:self._max_batch]

            # Run inference with timing breakdown
            keypoints, scores, process_time_ms, lock_wait_ms = self._infer_batch(images_to_process, self.model_width, self.model_height)

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
                inference_time_ms=process_time_ms,
                lock_time_ms=lock_wait_ms
            )
        else:
            output = DetectionOutput(batch_id=batch.batch_id, processed=True)

        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TensorRT Detection Warning: Callback queue full")

    def _infer_batch(self, imgs: list[np.ndarray], width: int, height: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Run TensorRT inference on batch of images.

        Args:
            imgs: List of BGR uint8 images (H, W, 3)
            width: Model input width
            height: Model input height

        Returns:
            (keypoints, scores, process_time_ms, lock_wait_ms)
        """
        batch_size = len(imgs)

        method_start: float = time.perf_counter()

        # CPU preprocessing
        stacked_imgs = np.stack(imgs, axis=0)  # (B, H, W, 3) uint8

        # Get batched buffers (sliced to actual batch size)
        buf = self._batch_buffers
        input_gpu = buf['input'][:batch_size]
        simcc_x_gpu = buf['simcc_x'][:batch_size]
        simcc_y_gpu = buf['simcc_y'][:batch_size]

        # CuPy preprocessing on our own buffers/stream (no lock needed)
        with self.stream:
            buf['img_uint8'][:batch_size].set(stacked_imgs)
            img_float_batch = buf['img_uint8'][:batch_size].astype(self._model_dtype)
            img_rgb_batch = img_float_batch[:, :, :, ::-1]  # BGR→RGB
            img_chw_batch = cp.ascontiguousarray(cp.transpose(img_rgb_batch, (0, 3, 1, 2)))  # HWC→CHW
            input_gpu[:] = (img_chw_batch - self._mean_gpu) / self._std_gpu
        # self.stream.synchronize()  # Ensure preprocessing complete before TRT reads

        # TensorRT operations only - acquire global lock
        lock_wait_start: float = time.perf_counter()
        with get_exec_lock():
            lock_acquired: float = time.perf_counter()

            self.context.set_input_shape(self.input_name, input_gpu.shape)
            self.context.set_tensor_address(self.input_name, input_gpu.data.ptr)
            self.context.set_tensor_address(self.output_names[0], simcc_x_gpu.data.ptr)
            self.context.set_tensor_address(self.output_names[1], simcc_y_gpu.data.ptr)

            self.context.execute_async_v3(stream_handle=self.stream.ptr)
            self.stream.synchronize()

        # Transfer outputs to CPU (no lock needed - our own buffers)
        simcc_x_cpu = cp.asnumpy(simcc_x_gpu)
        simcc_y_cpu = cp.asnumpy(simcc_y_gpu)

        # Decode on CPU (outside lock - no GPU resource needed)
        keypoints, scores = TRTDetection._decode_simcc_cpu(simcc_x_cpu, simcc_y_cpu, self.simcc_split_ratio)

        method_end = time.perf_counter()

        lock_wait_ms = (lock_acquired - lock_wait_start) * 1000.0
        total_time_ms = (method_end - method_start) * 1000.0
        process_time_ms = total_time_ms - lock_wait_ms

        return keypoints, scores, process_time_ms, lock_wait_ms

    # CALLBACK METHODS
    def register_callback(self, callback: PoseDetectionOutputCallback) -> None:
        """Register callback to receive results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def unregister_callback(self, callback: PoseDetectionOutputCallback) -> None:
        """Unregister previously registered callback."""
        with self._callback_lock:
            self._callbacks.discard(callback)

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

    # STATIC METHODS
    @staticmethod
    def _decode_simcc_gpu(simcc_x: cp.ndarray, simcc_y: cp.ndarray, split_ratio: float) -> tuple[cp.ndarray, cp.ndarray]:
        """Decode SimCC on GPU using CuPy. Returns (keypoints, scores) as CuPy arrays.

        Reference: mmpose/codecs/utils/post_processing.py::get_simcc_maximum
        """
        N, K, _ = simcc_x.shape

        # Reshape for processing: (N, K, W) -> (N*K, W)
        simcc_x_flat = simcc_x.reshape(N * K, -1)
        simcc_y_flat = simcc_y.reshape(N * K, -1)

        # Get coordinates from argmax
        x_locs = cp.argmax(simcc_x_flat, axis=1)
        y_locs = cp.argmax(simcc_y_flat, axis=1)

        # Get max values (scores) at predicted locations
        max_val_x = cp.amax(simcc_x_flat, axis=1)
        max_val_y = cp.amax(simcc_y_flat, axis=1)

        # MMPose takes MINIMUM of x and y scores
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        scores = max_val_x

        # Convert to coordinates
        x_coords = x_locs.astype(cp.float32) / split_ratio
        y_coords = y_locs.astype(cp.float32) / split_ratio

        # Reshape back to (N, K, 2) and (N, K)
        keypoints = cp.stack([x_coords, y_coords], axis=-1).reshape(N, K, 2)
        scores = scores.reshape(N, K)

        # Mark invalid keypoints
        keypoints[scores <= 0.] = -1

        scores = cp.clip(scores, 0.0, 1.0)

        return keypoints, scores

    @staticmethod
    def _decode_simcc_cpu(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float, apply_softmax: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Decode SimCC on CPU using NumPy. Kept as fallback.

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