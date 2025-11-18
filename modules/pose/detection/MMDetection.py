# Standard library imports
from dataclasses import dataclass, field
from enum import IntEnum
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback
from typing import Callable

# Third-party imports
from mmengine.structures.instance_data import InstanceData
from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
import numpy as np
import torch

# Ensure numpy functions can be safely used in torch serialization
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.UInt8DType]) # pyright: ignore

# DEFINITIONS
POSE_MODEL_WIDTH = 192
POSE_MODEL_HEIGHT = 256

class ModelType(IntEnum):
    NONE =   0
    LARGE =  1
    MEDIUM = 2
    SMALL =  3
    TINY =   4
POSE_MODEL_TYPE_NAMES: list[str] = [e.name for e in ModelType]

POSE_MODEL_FILE_NAMES: list[tuple[str, str]] = [
    ('none', ''),
    ('rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'),
    ('rtmpose-m_8xb256-420e_aic-coco-256x192.py', 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'),
    ('rtmpose-s_8xb256-420e_aic-coco-256x192.py', 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'),
    ('rtmpose-t_8xb256-420e_aic-coco-256x192.py', 'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth')
]

@dataclass
class DetectionInput:
    """Batch of images for pose detection. Images must be 192x256 (HxW)."""
    batch_id: int
    images: list[np.ndarray]

    def __post_init__(self) -> None:
        """Validate image dimensions (debug builds only)."""
        for i, img in enumerate(self.images):
            assert img.shape[:2] == (POSE_MODEL_HEIGHT, POSE_MODEL_WIDTH), \
                f"Image {i} has incorrect shape {img.shape}, expected (256, 192)"

@dataclass
class DetectionOutput:
    """Results from pose detection. processed=False indicates batch was dropped."""
    batch_id: int
    point_batch: list[np.ndarray] = field(default_factory=list)   # List of (num_keypoints, 2) arrays, normalized [0, 1]
    score_batch: list[np.ndarray] = field(default_factory=list)   # List of (num_keypoints,) arrays, confidence scores [0, 1]
    processed: bool = True          # False if batch was dropped before processing
    inference_time_ms: float = 0.0  # For monitoring

PoseDetectionOutputCallback = Callable[[DetectionOutput], None]

class MMDetection(Thread):
    """Asynchronous GPU pose detection using RTMPose.

    Uses a single-slot queue: only the most recent submitted batch waits to be processed.
    Older pending batches are dropped. Batches already processing on GPU cannot be cancelled.

    All results (success and dropped) are delivered via callbacks in notification order,
    which may differ from batch_id order due to async processing.
    """

    def __init__(self, path: str, model_type: ModelType, num_warmups: int, confidence_threshold: float = 0.3, verbose: bool = False) -> None:
        super().__init__()

        if model_type is ModelType.NONE:
            print('Pose Detection WARNING: ModelType is NONE')

        self.model_config_file: str = path + '/' + POSE_MODEL_FILE_NAMES[model_type.value][0]
        self.model_checkpoint_file: str = path + '/' + POSE_MODEL_FILE_NAMES[model_type.value][1]
        self.model_width: int = POSE_MODEL_WIDTH
        self.model_height: int = POSE_MODEL_HEIGHT
        self.model_num_warmups: int = num_warmups
        self.confidence_threshold: float = confidence_threshold

        self.verbose: bool = verbose

        # Thread coordination
        self._shutdown_event: Event = Event()
        self._notify_update_event: Event = Event()
        self._model_ready: Event = Event()

        # Input queue
        self._input_lock: Lock = Lock()
        self._pending_batch: DetectionInput | None = None
        self._input_timestamp: float = time.time()

        # Callbacks
        self._callback_lock: Lock = Lock()
        self._callbacks: set[PoseDetectionOutputCallback] = set()
        self._callback_queue: Queue[DetectionOutput | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._callback_worker_loop, daemon=True)

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and system is ready to process poses"""
        return self._model_ready.is_set() and not self._shutdown_event.is_set() and self.is_alive()

    def start(self) -> None:
        self._callback_thread.start()
        super().start()

    def stop(self) -> None:
        """Stop both inference and callback threads gracefully"""
        self._shutdown_event.set()

        # Wake up inference thread
        self._notify_update_event.set()
        self.join(timeout=2.0)

        if self.is_alive():
            print("Warning: Inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: Callback thread did not stop cleanly")

    def run(self) -> None:
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream(device=0, priority=-1)
        torch.cuda.set_stream(stream)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        model: torch.nn.Module = init_model(self.model_config_file, self.model_checkpoint_file, device='cuda:0')
        model.half()
        scope = model.cfg.get('default_scope', 'mmpose') # pyright: ignore
        if scope is not None:
            init_default_scope(scope)
        pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline) # pyright: ignore
        self._model_warmup(model, pipeline, self.model_num_warmups)
        self._model_ready.set()  # Signal model is ready
        print("PoseDetection: Model warmup complete")

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process_pending_batch(model, pipeline, stream)

            except Exception as e:
                print(f"Pose Detection Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: DetectionInput) -> None:
        """Submit batch for processing. Replaces any pending (not yet started) batch.

        Dropped batches trigger callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        dropped_batch: DetectionInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"Pose Detection: Dropped batch {dropped_batch.batch_id} with lag {lag} ms")

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        # Notify about dropped batch
        if dropped_batch is not None:
            dropped_output = DetectionOutput(
                batch_id=dropped_batch.batch_id,
                processed=False  # Mark as not processed
            )
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                if self.verbose:
                    print("Pose Detection Warning: Callback queue full, not critical for dropped notifications")
                pass  # Queue full, not critical for dropped notifications

        self._notify_update_event.set()

    def _retrieve_pending_batch(self) -> DetectionInput | None:
        """Atomically get and clear pending batch. Once retrieved, batch cannot be cancelled.

        Returns:
            The pending batch if one exists, None otherwise
        """
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None  # Clear slot - batch is now committed
            return batch

    def _process_pending_batch(self, model: torch.nn.Module, pipeline: Compose, stream: torch.cuda.Stream) -> None:
        # Get pending input
        batch: DetectionInput | None = self._retrieve_pending_batch()

        if batch is None:
            if self.verbose:
                print("Pose Detection Warning: No pending batch to process, this should not happen")
            return
        output = DetectionOutput(batch_id=batch.batch_id, processed=True)

        # Run inference
        if batch.images:
            batch_start = time.perf_counter()

            with torch.cuda.stream(stream):
                data_samples: list[list[PoseDataSample]] = MMDetection._infer_batch(model, pipeline, batch.images)
                point_list, score_list = MMDetection._extract_pose_points(data_samples, self.model_width, self.model_height, self.confidence_threshold)
                stream.synchronize()

            inference_time_ms: float = (time.perf_counter() - batch_start) * 1000.0

            # print(f"Pose Detection: Processed batch {input_data.batch_id} with {len(input_data.images)} images in   {inference_time_ms:.0f}   ms")

            # Create output (processed=True by default)
            output = DetectionOutput(
                batch_id=batch.batch_id,
                point_batch=point_list,
                score_batch=score_list,
                processed=True,
                inference_time_ms=inference_time_ms
            )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("Pose Detection Warning: Callback queue full, dropping inference results")

    # CALLBACK
    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks. Runs on separate thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1:
                    print("Pose Detection Warning: Callback queue size > 1, consumers may be falling behind")

                output: DetectionOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"Pose Detection Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def register_callback(self, callback: PoseDetectionOutputCallback) -> None:
        """Register callback to receive detection results (both success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)

    # STATIC METHODS
    @staticmethod
    def _model_warmup(model: torch.nn.Module, pipeline: Compose, num_imgs: int) -> None:
        """Initialize CUDA kernels with dummy batches of increasing sizes."""
        if num_imgs <= 0:
            return
        if num_imgs > 8:
            num_imgs = 8  # Limit to 8 for warmup

        dummy_sizes: list[int] = [i for i in range(1, num_imgs + 1)]
        for batch_size in dummy_sizes:
            # Create batch of dummy images
            dummy_imgs: list[np.ndarray] = [np.zeros((POSE_MODEL_HEIGHT, POSE_MODEL_WIDTH, 3), dtype=np.uint8) for _ in range(batch_size)]
            _: list[list[PoseDataSample]] = MMDetection._infer_batch(model, pipeline, dummy_imgs)

            # Ensure GPU operations are complete
            torch.cuda.synchronize()

    @staticmethod
    def _infer_batch(model: torch.nn.Module, pipeline: Compose, imgs: list[np.ndarray]) -> list[list[PoseDataSample]]:
        """Run pose detection inference on batch of images."""
        if not imgs:
            return []

        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Cache frequently used values
            model_meta = model.dataset_meta
            batch_size = len(imgs)

            # Pre-allocate and reuse arrays
            data_list = [None] * batch_size
            bbox_score = np.ones(1, dtype=np.float32)

            # Optimized loop
            for idx, img in enumerate(imgs):
                h, w = img.shape[:2]

                # Single dict creation with unpacking
                data_info = {
                    'img': img,
                    'bbox': np.array([[0, 0, w, h]], dtype=np.float32),
                    'bbox_score': bbox_score,
                    **model_meta # pyright: ignore
                }
                data_list[idx] = pipeline(data_info) # pyright: ignore

            batch = pseudo_collate(data_list)

            with torch.inference_mode():
                all_results = model.test_step(batch) # pyright: ignore

            # Optimized result extraction
            results_by_image = [[sample] for sample in all_results]

            return results_by_image

    @staticmethod
    def _extract_pose_points(data_samples: list[list[PoseDataSample]], model_width: int, model_height: int, confidence_threshold: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Extract normalized keypoints and scores from inference results. Returns first person per image."""
        keypoints_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []

        for data_samples_for_image in data_samples:
            pose_found = False

            for data_sample in data_samples_for_image:
                pred_instances: InstanceData = data_sample.pred_instances
                keypoints: np.ndarray = pred_instances.get('keypoints', np.full((0, 17, 2), np.nan, dtype=np.float32))
                scores: np.ndarray = pred_instances.get('keypoint_scores', np.zeros((0, 17), dtype=np.float32))
                scores = np.clip(scores, 0.0, 1.0)

                if keypoints.shape[0] == 0:
                    continue  # No pose detected

                # Take only first person's pose in keypoints array
                norm_keypoints: np.ndarray = keypoints[0].copy() / np.array([model_width, model_height])
                person_scores: np.ndarray = scores[0].copy()

                # Append to lists
                keypoints_list.append(norm_keypoints)
                scores_list.append(person_scores)
                pose_found = True
                break  # Stop after finding first pose

            # If no pose found for this image, add empty placeholders
            if not pose_found:
                keypoints_list.append(np.full((17, 2), np.nan, dtype=np.float32))
                scores_list.append(np.zeros(17, dtype=np.float32))

        return keypoints_list, scores_list
