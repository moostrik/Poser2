# Standard library imports
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback

# Third-party imports
import numpy as np
import onnxruntime as ort
import torch

from modules.pose.batch.detection.InOut import DetectionInput, DetectionOutput, PoseDetectionOutputCallback

from modules.pose.Settings import Settings

# ImageNet normalization constants (RGB order) - scaled to [0,1] range
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


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

            # Map numpy dtype to torch dtype for preprocessing
            torch_dtype_map = {
                np.float32: torch.float32,
                np.float16: torch.float16,
            }
            self._torch_dtype = torch_dtype_map.get(self._model_dtype, torch.float32)

            # Dedicated CUDA stream for all GPU operations (preprocessing + inference)
            self.stream = torch.cuda.Stream()

            # Preallocate INPUT buffers on the dedicated stream
            with torch.cuda.stream(self.stream):
                # GPU constants for ImageNet normalization
                self._mean_gpu = IMAGENET_MEAN.to(device='cuda', dtype=self._torch_dtype)
                self._std_gpu = IMAGENET_STD.to(device='cuda', dtype=self._torch_dtype)

                # Input buffer: normalized CHW format ready for ONNX Runtime
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

    def _infer_batch_gpu(self, gpu_imgs: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on GPU images using IOBinding for zero-copy.

        All preprocessing runs on the dedicated stream for zero sync overhead.
        Uses preallocated buffers for zero allocation latency.

        Args:
            gpu_imgs: List of RGB float32 tensors on GPU (H, W, 3) in [0,1] range

        Returns:
            (keypoints, scores)
        """
        batch_size = len(gpu_imgs)

        if batch_size == 0 or self._session is None:
            return np.empty((0, 17, 2)), np.empty((0, 17))

        # Get input dimensions from first image
        input_h, input_w = gpu_imgs[0].shape[0], gpu_imgs[0].shape[1]
        needs_resize = (input_h != self.model_height or input_w != self.model_width)

        # Get preallocated INPUT buffer slice for current batch
        input_buffer = self._input_buffer[:self._max_batch]

        # All preprocessing on dedicated stream (no cross-stream sync needed)
        with torch.cuda.stream(self.stream):
            # Stack GPU tensors: (B, H, W, 3) float32 RGB [0,1]
            batch_hwc = torch.stack(gpu_imgs, dim=0)

            # HWC -> CHW: (B, 3, H, W), convert to model dtype
            batch_chw = batch_hwc.permute(0, 3, 1, 2).to(self._torch_dtype)

            # Resize if needed (crop size != model size)
            if needs_resize:
                batch_chw = torch.nn.functional.interpolate(
                    batch_chw, size=(self.model_height, self.model_width),
                    mode='bilinear', align_corners=False
                )

            # Pad batch to max_batch size for consistent tensor shapes
            if batch_size < self._max_batch:
                pad_size = self._max_batch - batch_size
                padding = torch.zeros(
                    (pad_size, 3, self.model_height, self.model_width),
                    dtype=self._torch_dtype, device='cuda'
                )
                batch_chw = torch.cat([batch_chw, padding], dim=0)

            # ImageNet normalization directly into preallocated input buffer
            torch.sub(batch_chw, self._mean_gpu, out=self._resize_buffer)
            torch.div(self._resize_buffer, self._std_gpu, out=input_buffer)

        # CRITICAL: Synchronize stream before IOBinding to ensure all preprocessing is complete
        self.stream.synchronize()

        # Use preprocessed tensor directly for ONNX Runtime
        torch_input = input_buffer

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
            element_type=self._model_dtype,
            shape=tuple(torch_input.shape),
            buffer_ptr=torch_input.data_ptr()
        )

        # Bind outputs to GPU
        io_binding.bind_output(output_names[0], device_type='cuda', device_id=0)
        io_binding.bind_output(output_names[1], device_type='cuda', device_id=0)

        # Run inference on the same stream
        self._session.run_with_iobinding(io_binding)

        # Synchronize stream to ensure outputs are ready
        self.stream.synchronize()

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
            # Create realistic dummy input on GPU (float32 RGB [0,1])
            dummy_img = torch.zeros((self.model_height, self.model_width, 3), dtype=torch.float32, device='cuda')

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
