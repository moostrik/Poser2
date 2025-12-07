# Create: modules/pose/detection/ONNXDetection.py

# Standard library imports
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback
from typing import Callable

# Third-party imports
import numpy as np
import onnxruntime as ort

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

# ONNX model file names and keypoint counts
# Format: (filename, num_keypoints)
ONNX_MODEL_CONFIG: dict[ModelType, tuple[str, int]] = {
    ModelType.NONE: ('', 17),
    ModelType.LARGE: ('rtmpose-l_256x192.onnx', 17),
    ModelType.MEDIUM: ('rtmpose-m_256x192.onnx', 17),
    ModelType.SMALL: ('rtmpose-s_256x192.onnx', 17),
    ModelType.TINY: ('rtmpose-t_256x192.onnx', 17),
    ModelType.W_L: ('wb_rtmpose-l_256x192.onnx', 133),  # Wholebody: 133 keypoints
    ModelType.W_M: ('wb_rtmpose-m_256x192.onnx', 133),
    ModelType.W_S: ('wb_rtmpose-s_256x192.onnx', 133),
}

# For Body8 models, add your specific model type or override:
# ONNX_MODEL_CONFIG[ModelType.CUSTOM] = ('body8_model.onnx', 8)

# ImageNet normalization (RGB order)
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)


class ONNXDetection(Thread):
    """Asynchronous GPU pose detection using ONNX Runtime.

    Faster alternative to MMDetection using exported ONNX models.
    Architecture identical to MMDetection for drop-in replacement.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.enabled: bool = settings.model_type is not ModelType.NONE
        if settings.model_type is ModelType.NONE:
            print('ONNX Detection WARNING: ModelType is NONE')

        self.model_type: ModelType = settings.model_type
        model_filename, self.num_keypoints = ONNX_MODEL_CONFIG[settings.model_type]
        self.model_file: str = settings.model_path + '/' + model_filename
        self.model_width: int = POSE_MODEL_WIDTH
        self.model_height: int = POSE_MODEL_HEIGHT
        self.model_num_warmups: int = settings.max_poses
        self.confidence_threshold: float = settings.confidence_threshold
        self.simcc_split_ratio: float = 2.0  # From RTMPose config

        self.verbose: bool = settings.verbose

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
            print("Warning: ONNX inference thread did not stop cleanly")

        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: ONNX callback thread did not stop cleanly")

    def run(self) -> None:
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

        session = ort.InferenceSession(self.model_file, sess_options, providers=providers)

        if 'CUDAExecutionProvider' not in session.get_providers():
            print("ONNX Detection WARNING: CUDA not available, using CPU")

        self._model_warmup(session, self.model_num_warmups)
        self._model_ready.set()
        print(f"ONNX Detection: {self.model_type.name} model ready")

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process_pending_batch(session)
            except Exception as e:
                print(f"ONNX Detection Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: DetectionInput) -> None:
        """Submit batch for processing. Identical to MMDetection."""
        if self._shutdown_event.is_set():
            return

        dropped_batch: DetectionInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"ONNX Detection: Dropped batch {dropped_batch.batch_id} with lag {lag} ms")
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

    def _process_pending_batch(self, session: ort.InferenceSession) -> None:
        batch: DetectionInput | None = self._retrieve_pending_batch()

        if batch is None:
            return

        if batch.images:
            batch_start = time.perf_counter()

            # Preprocess and run inference
            keypoints, scores = self._infer_batch(session, batch.images)

            # Normalize coordinates to [0, 1]
            keypoints[:, :, 0] /= self.model_width
            keypoints[:, :, 1] /= self.model_height

            inference_time_ms = (time.perf_counter() - batch_start) * 1000.0

            # Convert to list format matching MMDetection
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
            print("ONNX Detection Warning: Callback queue full")

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

    def register_callback(self, callback: PoseDetectionOutputCallback) -> None:
        """Register callback to receive results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    # STATIC METHODS

    @staticmethod
    def _preprocess_batch(imgs: list[np.ndarray], debug: bool = False) -> np.ndarray:
        """Convert BGR uint8 images to normalized RGB FP32 batch (B, 3, H, W)."""
        if not imgs:
            return np.empty((0, 3, POSE_MODEL_HEIGHT, POSE_MODEL_WIDTH), dtype=np.float32)

        # Stack and convert to float32
        batch_hwc = np.stack(imgs, axis=0).astype(np.float32)  # (B, H, W, 3)

        if debug:
            print(f"  Input images: shape={batch_hwc.shape}, dtype={batch_hwc.dtype}, range=[{batch_hwc.min():.1f}, {batch_hwc.max():.1f}]")

        # BGR to RGB
        batch_rgb = batch_hwc[:, :, :, ::-1]

        # HWC to CHW
        batch_chw = batch_rgb.transpose(0, 3, 1, 2)  # (B, 3, H, W)

        # Normalize with ImageNet stats
        batch_norm = (batch_chw - IMAGENET_MEAN) / IMAGENET_STD

        if debug:
            print(f"  Normalized: shape={batch_norm.shape}, range=[{batch_norm.min():.2f}, {batch_norm.max():.2f}]")

        return batch_norm

    @staticmethod
    def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """Decode SimCC coordinate classifications to keypoints.

        Args:
            simcc_x: (B, num_keypoints, W_bins) X-coordinate logits
            simcc_y: (B, num_keypoints, H_bins) Y-coordinate logits
            split_ratio: Coordinate upsampling factor

        Returns:
            keypoints: (B, num_keypoints, 2) pixel coordinates
            scores: (B, num_keypoints) confidence scores
        """
        # Apply softmax along the bins dimension
        exp_x = np.exp(simcc_x - np.max(simcc_x, axis=-1, keepdims=True))
        exp_y = np.exp(simcc_y - np.max(simcc_y, axis=-1, keepdims=True))

        prob_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        prob_y = exp_y / np.sum(exp_y, axis=-1, keepdims=True)

        # Get coordinates from argmax
        x_locs = np.argmax(prob_x, axis=-1)  # (B, num_keypoints)
        y_locs = np.argmax(prob_y, axis=-1)  # (B, num_keypoints)

        # Convert bin indices to pixel coordinates
        x_coords = x_locs / split_ratio
        y_coords = y_locs / split_ratio

        # Confidence = max probability (not product, that's too strict)
        x_scores = np.max(prob_x, axis=-1)
        y_scores = np.max(prob_y, axis=-1)
        scores = (x_scores + y_scores) / 2.0  # Average instead of product

        # Stack coordinates
        keypoints = np.stack([x_coords, y_coords], axis=-1)

        return keypoints, scores

    def _infer_batch(self, session: ort.InferenceSession, imgs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run ONNX inference on batch of images."""
        if not imgs:
            return np.empty((0, 17, 2)), np.empty((0, 17))

        # Preprocess
        debug = self.verbose and not hasattr(self, '_onnx_debug')
        batch = self._preprocess_batch(imgs, debug=debug)

        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch})

        # Decode SimCC outputs
        simcc_x, simcc_y = outputs[0], outputs[1]  # (B, 17, 384), (B, 17, 512)
        keypoints, scores = ONNXDetection._decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)

        # Debug first batch
        if self.verbose and not hasattr(self, '_onnx_debug'):
            self._onnx_debug = True
            print(f"\nONNX Detection Debug:")
            print(f"  SimCC outputs: x={simcc_x.shape} y={simcc_y.shape}")
            print(f"  Decoded: keypoints={keypoints.shape} scores={scores.shape}")
            if keypoints.shape[0] > 0:
                print(f"  Sample keypoint 0 (nose): x={keypoints[0,0,0]:.1f} y={keypoints[0,0,1]:.1f} score={scores[0,0]:.3f}")
                print(f"  All scores: {scores[0]}")

        return keypoints, scores

    @staticmethod
    def _model_warmup(session: ort.InferenceSession, num_imgs: int) -> None:
        """Initialize CUDA kernels with dummy batches."""
        if num_imgs <= 0:
            return
        if num_imgs > 8:
            num_imgs = 8

        input_name = session.get_inputs()[0].name

        for batch_size in range(1, num_imgs + 1):
            dummy = np.zeros((batch_size, 3, POSE_MODEL_HEIGHT, POSE_MODEL_WIDTH), dtype=np.float32)
            _ = session.run(None, {input_name: dummy})

        print("ONNX Detection: Warmup complete")