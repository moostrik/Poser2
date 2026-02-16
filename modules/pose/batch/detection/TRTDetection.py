# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import torch
import tensorrt as trt

# Reuse dataclasses from MMDetection
from modules.pose.batch.detection.InOut import DetectionInput, DetectionOutput, PoseDetectionOutputCallback

from modules.pose.Settings import Settings
from modules.pose.tensorrt_shared import get_tensorrt_runtime, get_init_lock, get_exec_lock

# ImageNet normalization constants (RGB order) - scaled to [0,1] range
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class TRTDetection(Thread):
    """Asynchronous GPU pose detection using RTMPose with TensorRT.

    TensorRT-optimized inference for maximum performance. Drop-in replacement for ONNXDetection.

    Uses preallocated buffers and dedicated CUDA stream for ultra-low latency.
    All preprocessing and inference runs on the same dedicated stream to avoid sync overhead.

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
        self._callback_thread: Thread = Thread(target=self._dispatch_callbacks, daemon=True)

        # TensorRT engine and context (initialized in run())
        self.engine: trt.ICudaEngine  # type: ignore
        self.context: trt.IExecutionContext  # type: ignore
        self.stream: torch.cuda.Stream
        self.input_name: str
        self.output_names: list[str]

        # Model configuration (initialized in _setup())
        self._max_batch: int = 8
        self._torch_dtype: torch.dtype
        self._mean_gpu: torch.Tensor  # ImageNet mean on GPU
        self._std_gpu: torch.Tensor  # ImageNet std on GPU

        # Preallocated INPUT buffers (initialized in _setup())
        # Note: Output buffers allocated fresh each call
        self._input_buffer: torch.Tensor  # (max_batch, 3, H, W) normalized input
        self._resize_buffer: torch.Tensor  # (max_batch, 3, H, W) for resize output

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
                self._process()
            except Exception as e:
                print(f"TensorRT Detection Error: {str(e)}")
                traceback.print_exc()

    def submit(self, input_batch: DetectionInput) -> None:
        """Submit batch for processing. Replaces pending batch if not yet started.

        Dropped batches receive callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        # Validate batch size
        if len(input_batch.gpu_images) > self._max_batch:
            print(f"TensorRT Detection Warning: Batch size {len(input_batch.gpu_images)} exceeds max {self._max_batch}, will process only first {self._max_batch} images")

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
        """Initialize TensorRT engine, context, and preallocated buffers. Called from run()."""
        try:
            # Acquire global lock to prevent concurrent Myelin graph loading
            with get_init_lock():
                runtime = get_tensorrt_runtime()

                with open(self.model_file, 'rb') as f:
                    engine_data = f.read()

                self.engine = runtime.deserialize_cuda_engine(engine_data)
                if self.engine is None:
                    print("TensorRT Detection ERROR: Failed to load engine")
                    return

                self.context = self.engine.create_execution_context()

            # Lock released - continue with setup

            # Get input/output names
            self.output_names = []

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)

                if mode == trt.TensorIOMode.INPUT:  # type: ignore
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
                trt.DataType.FLOAT: "FP32",  # type: ignore
                trt.DataType.HALF: "FP16",   # type: ignore
                trt.DataType.INT8: "INT8",   # type: ignore
            }
            self.model_precision = precision_map.get(input_dtype, "UNKNOWN")

            # Map TensorRT dtype to PyTorch dtype
            dtype_to_torch = {
                trt.DataType.FLOAT: torch.float32,  # type: ignore
                trt.DataType.HALF: torch.float16,   # type: ignore
            }
            self._torch_dtype = dtype_to_torch.get(input_dtype, torch.float32)

            # SimCC output dimensions
            self._simcc_x_width = self.model_width * 2
            self._simcc_y_height = self.model_height * 2

            # Dedicated CUDA stream for all GPU operations (preprocessing + inference)
            self.stream = torch.cuda.Stream()

            # Preallocate INPUT buffers on the dedicated stream for optimal memory placement
            # Note: Output buffers allocated fresh each call (no clone needed)
            with torch.cuda.stream(self.stream):
                # GPU constants for ImageNet normalization
                self._mean_gpu = IMAGENET_MEAN.to(device='cuda', dtype=self._torch_dtype)
                self._std_gpu = IMAGENET_STD.to(device='cuda', dtype=self._torch_dtype)

                # Input buffer: normalized CHW format ready for TRT
                self._input_buffer = torch.empty(
                    (self._max_batch, 3, self.model_height, self.model_width),
                    dtype=self._torch_dtype, device='cuda'
                )

                # Resize buffer: intermediate for bilinear resize
                self._resize_buffer = torch.empty(
                    (self._max_batch, 3, self.model_height, self.model_width),
                    dtype=self._torch_dtype, device='cuda'
                )

            self.stream.synchronize()

            # Set persistent tensor address for INPUT buffer (base pointer doesn't change when slicing)
            # Output addresses must be set per-call since they're allocated fresh each inference
            self.context.set_tensor_address(self.input_name, self._input_buffer.data_ptr())

            self._model_ready.set()
            print(f"TensorRT Detection: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"TensorRT Detection Error: Failed to load model - {str(e)}")
            traceback.print_exc()
            return

    def _claim(self) -> DetectionInput | None:
        """Atomically get and clear pending batch."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process(self) -> None:
        batch: DetectionInput | None = self._claim()

        if batch is None:
            return

        gpu_images = batch.gpu_images[:self._max_batch]

        if not gpu_images:
            output = DetectionOutput(batch_id=batch.batch_id, processed=True)
        else:
            keypoints, scores, process_time_ms, lock_wait_ms = self._infer(gpu_images)

            # Normalize coordinates to [0, 1]
            keypoints[:, :, 0] /= self.model_width
            keypoints[:, :, 1] /= self.model_height

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

        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("TensorRT Detection Warning: Callback queue full")

    def _infer(self, gpu_imgs: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Run TensorRT inference on batch of GPU images.

        All preprocessing and inference runs on the dedicated stream for zero sync overhead.
        Uses preallocated buffers for zero allocation latency.

        Args:
            gpu_imgs: List of RGB float32 tensors on GPU (3, H, W) CHW [0,1]

        Returns:
            (keypoints, scores, process_time_ms, lock_wait_ms)
        """
        batch_size = len(gpu_imgs)
        method_start: float = time.perf_counter()

        # Get input dimensions from first image (CHW format: 3, H, W)
        input_h, input_w = gpu_imgs[0].shape[1], gpu_imgs[0].shape[2]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)

        # Get preallocated INPUT buffer slice for current batch
        input_buffer = self._input_buffer[:batch_size]

        # Allocate OUTPUT buffers fresh each call (no clone needed)
        simcc_x_out = torch.empty(
            (batch_size, self.num_keypoints, self._simcc_x_width),
            dtype=self._torch_dtype, device='cuda'
        )
        simcc_y_out = torch.empty(
            (batch_size, self.num_keypoints, self._simcc_y_height),
            dtype=self._torch_dtype, device='cuda'
        )

        # All preprocessing on dedicated stream (no cross-stream sync needed)
        with torch.cuda.stream(self.stream):
            # Stack GPU tensors: (B, 3, H, W) float32 RGB CHW [0,1]
            batch_chw = torch.stack(gpu_imgs, dim=0).to(self._torch_dtype)

            # Resize if needed (crop size != model size)
            if needs_resize:
                batch_chw = torch.nn.functional.interpolate(
                    batch_chw, size=(self.model_height, self.model_width), mode='bilinear', align_corners=False
                )

            # ImageNet normalization directly into preallocated input buffer
            torch.sub(batch_chw, self._mean_gpu, out=self._resize_buffer[:batch_size])
            torch.div(self._resize_buffer[:batch_size], self._std_gpu, out=input_buffer)

        # TensorRT inference - acquire global lock (still on same stream)
        lock_wait_start: float = time.perf_counter()
        with get_exec_lock():
            lock_acquired: float = time.perf_counter()

            # Only set_input_shape per-call (input address is persistent from _setup)
            self.context.set_input_shape(self.input_name, tuple(input_buffer.shape))

            # Output addresses must be set per-call (fresh allocation each inference)
            self.context.set_tensor_address(self.output_names[0], simcc_x_out.data_ptr())
            self.context.set_tensor_address(self.output_names[1], simcc_y_out.data_ptr())

            # Execute on dedicated CUDA stream (preprocessing already complete on this stream)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize() # .cpu() below syncs anyway

        # Transfer to CPU (no clone needed - output buffers are fresh each call)
        simcc_x_cpu = simcc_x_out.cpu().numpy()
        simcc_y_cpu = simcc_y_out.cpu().numpy()

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

    def _dispatch_callbacks(self) -> None:
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