# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import cupy as cp
import onnxruntime as ort
import torch

from modules.pose.batch.detection.InOut import DetectionInput, DetectionOutput, PoseDetectionOutputCallback
from modules.pose.batch.cuda_image_ops import batched_bilinear_resize

from modules.pose.Settings import Settings


class ONNXDetection(Thread):
    """Asynchronous GPU pose detection using ONNX Runtime.

    Faster alternative to MMDetection using exported ONNX models.
    Architecture identical to MMDetection for drop-in replacement.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.model_file: str = settings.model_path + '/' + settings.pose_model
        self.model_width: int = settings.pose_width
        self.model_height: int = settings.pose_height
        self.model_num_warmups: int = settings.max_poses
        self.num_keypoints: int = 17  # RTMPose COCO format
        self.simcc_split_ratio: float = 2.0  # From RTMPose config

        self.verbose: bool = settings.verbose
        self.resolution_name: str = settings.pose_resolution.name

        # Thread coordination (identical to MMDetection)
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

        # ONNX session (initialized in run thread)
        self._session: ort.InferenceSession | None = None
        self._model_dtype = np.float32  # Default, determined in setup
        self.model_precision: str = "UNKNOWN"

        # Batch inference settings
        self._max_batch: int = min(settings.max_poses, 8)

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
            print("Warning: ONNX inference thread did not stop cleanly")

        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: ONNX callback thread did not stop cleanly")

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
                print(f"ONNX Detection Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: DetectionInput) -> None:
        """Submit batch for processing. Identical to MMDetection."""
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        # Validate batch size
        if len(input_batch.gpu_images) > self._max_batch:
            print(f"ONNX Detection Warning: Batch size {len(input_batch.gpu_images)} exceeds max {self._max_batch}, will process only first {self._max_batch} images")

        dropped_batch: DetectionInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"ONNX Detection: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
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
        """Initialize ONNX session and warmup. Called from run()."""
        try:
            # Create ONNX Runtime session with CUDA
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
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

            self._session = ort.InferenceSession(self.model_file, sess_options, providers=providers)

            # Verify CUDA provider is active
            providers_used = self._session.get_providers()
            if 'CUDAExecutionProvider' not in providers_used:
                print("ONNX Detection WARNING: CUDA provider not available, using CPU")

            # Determine model precision from 'input' tensor
            input_tensor = self._session.get_inputs()[0]  # First input is 'input'
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

            # Initialize ImageNet normalization constants with correct dtype
            self._imagenet_mean = np.array([123.675, 116.28, 103.53], dtype=self._model_dtype).reshape(1, 3, 1, 1)
            self._imagenet_std = np.array([58.395, 57.12, 57.375], dtype=self._model_dtype).reshape(1, 3, 1, 1)

            # GPU normalization constants (CuPy)
            cupy_dtype = cp.float16 if self._model_dtype == np.float16 else cp.float32
            self._mean_gpu = cp.asarray(self._imagenet_mean, dtype=cupy_dtype)
            self._std_gpu = cp.asarray(self._imagenet_std, dtype=cupy_dtype)
            self._cupy_dtype = cupy_dtype

            # Warmup model
            self._model_warmup()

            self._model_ready.set()
            print(f"ONNX Detection: {self.resolution_name} model ready: {self.model_width}x{self.model_height} {self.model_precision}")

        except Exception as e:
            print(f"ONNX Detection Error: Failed to load model - {str(e)}")
            traceback.print_exc()

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

        gpu_images = batch.gpu_images[:self._max_batch]

        if not gpu_images:
            output = DetectionOutput(batch_id=batch.batch_id, processed=True)
        else:
            batch_start = time.perf_counter()
            keypoints, scores = self._infer_batch_gpu(gpu_images)

            keypoints[:, :, 0] /= self.model_width
            keypoints[:, :, 1] /= self.model_height

            inference_time_ms = (time.perf_counter() - batch_start) * 1000.0
            point_list = [keypoints[i] for i in range(len(keypoints))]
            score_list = [scores[i] for i in range(len(scores))]

            output = DetectionOutput(
                batch_id=batch.batch_id,
                point_batch=point_list,
                score_batch=score_list,
                processed=True,
                inference_time_ms=inference_time_ms
            )

        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("ONNX Detection Warning: Callback queue full")

    def _infer_batch_gpu(self, gpu_imgs: list[cp.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on GPU images using IOBinding for zero-copy."""
        batch_size = len(gpu_imgs)

        if batch_size == 0 or self._session is None:
            return np.empty((0, 17, 2)), np.empty((0, 17))

        # Stack all GPU images into batch
        batch_src = cp.stack(gpu_imgs, axis=0)  # (B, H, W, 3)

        # Resize if needed
        src_h, src_w = batch_src.shape[1:3]
        if src_h != self.model_height or src_w != self.model_width:
            resized_batch = batched_bilinear_resize(batch_src, self.model_height, self.model_width)
        else:
            resized_batch = batch_src

        # Pad batch to max_batch size for consistent tensor shapes
        if batch_size < self._max_batch:
            pad_size = self._max_batch - batch_size
            padding = cp.zeros((pad_size, self.model_height, self.model_width, 3), dtype=cp.uint8)
            resized_batch = cp.concatenate([resized_batch, padding], axis=0)

        # Convert HWC -> CHW and normalize on GPU
        batch_float = resized_batch.astype(self._cupy_dtype)  # (B, H, W, 3)
        batch_chw = cp.transpose(batch_float, (0, 3, 1, 2))  # (B, 3, H, W)
        batch_normalized = (batch_chw - self._mean_gpu) / self._std_gpu
        batch_normalized = cp.ascontiguousarray(batch_normalized)

        # Convert CuPy array to PyTorch tensor (zero-copy via DLPack)
        torch_input = torch.from_dlpack(batch_normalized) # type: ignore

        # Get input/output names
        input_name = self._session.get_inputs()[0].name
        output_names = [o.name for o in self._session.get_outputs()]

        # Use IOBinding for GPU input/output
        io_binding = self._session.io_binding()

        # Bind input from PyTorch tensor
        io_binding.bind_input(
            name=input_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float16 if self._model_dtype == np.float16 else np.float32,
            shape=tuple(torch_input.shape),
            buffer_ptr=torch_input.data_ptr()
        )

        # Bind outputs to GPU
        io_binding.bind_output(output_names[0], device_type='cuda', device_id=0)
        io_binding.bind_output(output_names[1], device_type='cuda', device_id=0)

        # Run inference
        self._session.run_with_iobinding(io_binding)

        # Get outputs as numpy (copies from GPU)
        outputs = io_binding.copy_outputs_to_cpu()
        simcc_x = outputs[0][:batch_size]
        simcc_y = outputs[1][:batch_size]

        # Decode SimCC outputs
        keypoints, scores = ONNXDetection._decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)

        return keypoints, scores

    def _model_warmup(self) -> None:
        """Initialize CUDA kernels for fixed batch size to prevent runtime recompilation."""
        if self._session is None:
            return

        try:
            # Create realistic dummy input on GPU
            dummy_img = cp.zeros((self.model_height, self.model_width, 3), dtype=cp.uint8)

            # Warmup with max_batch size (all inference runs with this size)
            dummy_images = [dummy_img] * self._max_batch

            keypoints, scores = self._infer_batch_gpu(dummy_images)

        except Exception as e:
            print(f"ONNX Detection: Warmup failed (non-critical) - {str(e)}")
            traceback.print_exc()

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
        """Dispatch results to callbacks. Identical to MMDetection."""
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
                        print(f"ONNX Detection Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    # STATIC METHODS
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
