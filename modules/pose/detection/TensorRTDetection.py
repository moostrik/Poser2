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
from modules.pose.detection.MMDetection import (
    DetectionInput,
    DetectionOutput,
    PoseDetectionOutputCallback,
    ModelType,
    POSE_MODEL_WIDTH,
    POSE_MODEL_HEIGHT
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.pose.Settings import Settings

# TensorRT model file names and keypoint counts
TENSORRT_MODEL_CONFIG: dict[ModelType, tuple[str, int]] = {
    ModelType.NONE: ('', 17),
    ModelType.LARGE: ('rtmpose-l_256x192.trt', 17),
    ModelType.MEDIUM: ('rtmpose-m_256x192.trt', 17),
    ModelType.SMALL: ('rtmpose-s_256x192.trt', 17),
    ModelType.TINY: ('rtmpose-t_256x192.trt', 17),
}

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

        self.enabled: bool = settings.model_type is not ModelType.NONE
        if settings.model_type is ModelType.NONE:
            print('TensorRT Detection WARNING: ModelType is NONE')

        self.model_type: ModelType = settings.model_type
        model_filename, self.num_keypoints = TENSORRT_MODEL_CONFIG[settings.model_type]
        self.model_file: str = settings.model_path + '/' + model_filename
        self.model_width: int = POSE_MODEL_WIDTH
        self.model_height: int = POSE_MODEL_HEIGHT
        self.simcc_split_ratio: float = 2.0  # From RTMPose config

        self.verbose: bool = settings.verbose

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
        if not self.enabled:
            return
        self._callback_thread.start()
        super().start()

    def stop(self) -> None:
        if not self.enabled:
            return
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
        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING) # type: ignore
        runtime = trt.Runtime(logger) # type: ignore

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

        # Initialize output names list
        self.output_names = []

        # Print tensor information
        print("TensorRT Engine Tensors:")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            print(f"  {name}: mode={mode}, shape={shape}, dtype={dtype}")

            if mode == trt.TensorIOMode.INPUT: # type: ignore
                self.input_name = name
            else:
                self.output_names.append(name)

        self._model_ready.set()
        print(f"TensorRT Detection: {self.model_type.name} model ready")

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

        if batch.images:
            batch_start = time.perf_counter()

            # Preprocess and run inference
            keypoints, scores = self._infer_batch(batch.images)

            # Normalize coordinates to [0, 1]
            keypoints[:, :, 0] /= self.model_width
            keypoints[:, :, 1] /= self.model_height

            inference_time_ms = (time.perf_counter() - batch_start) * 1000.0

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
    def _preprocess_batch(imgs: list[np.ndarray]) -> np.ndarray:
        """Convert BGR uint8 images to normalized RGB FP32 batch (B, 3, H, W)."""
        if not imgs:
            return np.empty((0, 3, POSE_MODEL_HEIGHT, POSE_MODEL_WIDTH), dtype=np.float32)

        batch_hwc = np.stack(imgs, axis=0).astype(np.float32)
        batch_rgb = batch_hwc[:, :, :, ::-1]
        batch_chw = batch_rgb.transpose(0, 3, 1, 2)
        batch_norm = (batch_chw - IMAGENET_MEAN) / IMAGENET_STD

        return batch_norm

    @staticmethod
    def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """Decode SimCC outputs to keypoints and scores."""
        x_locs = np.argmax(simcc_x, axis=-1)
        y_locs = np.argmax(simcc_y, axis=-1)

        x_coords = x_locs.astype(np.float32) / split_ratio
        y_coords = y_locs.astype(np.float32) / split_ratio

        batch_size, num_keypoints = simcc_x.shape[:2]
        x_scores = np.zeros((batch_size, num_keypoints), dtype=np.float32)
        y_scores = np.zeros((batch_size, num_keypoints), dtype=np.float32)

        for b in range(batch_size):
            for k in range(num_keypoints):
                x_scores[b, k] = simcc_x[b, k, x_locs[b, k]]
                y_scores[b, k] = simcc_y[b, k, y_locs[b, k]]

        scores = np.maximum(x_scores, y_scores)
        scores = np.clip(scores, 0.0, 1.0)

        keypoints = np.stack([x_coords, y_coords], axis=-1)

        return keypoints, scores

    def _infer_batch(self, imgs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run TensorRT inference on batch of images."""
        if not imgs:
            return np.empty((0, 17, 2)), np.empty((0, 17))

        # Preprocess
        batch = self._preprocess_batch(imgs)

        # Use CuPy for GPU arrays
        batch_gpu = cp.asarray(batch)

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

        # Run inference on dedicated stream
        with self.stream:
            self.context.execute_async_v3(stream_handle=self.stream.ptr)

        # Wait for completion and copy back to CPU
        self.stream.synchronize()
        simcc_x = cp.asnumpy(output0_gpu)
        simcc_y = cp.asnumpy(output1_gpu)

        # Decode SimCC outputs
        keypoints, scores = TensorRTDetection._decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)

        return keypoints, scores

        # Decode SimCC outputs
        keypoints, scores = TensorRTDetection._decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)

        return keypoints, scores