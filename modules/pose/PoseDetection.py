# Standard library imports
from queue import Queue, Empty
import time
import traceback
from dataclasses import replace
from threading import Thread, Lock, Event

# Third-party imports
import cv2
from mmengine.structures.instance_data import InstanceData
import numpy as np
import torch
from enum import IntEnum
from pandas import Timestamp

from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy

# Local application imports
from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.features.PosePoints import PosePointData

# Ensure numpy functions can be safely used in torch serialization
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.UInt8DType]) # pyright: ignore

from modules.utils.HotReloadMethods import HotReloadMethods



# DEFINITIONS
POSE_MODEL_WIDTH = 192
POSE_MODEL_HEIGHT = 256

class PoseModelType(IntEnum):
    NONE =   0
    LARGE =  1
    MEDIUM = 2
    SMALL =  3
    TINY =   4
POSE_MODEL_TYPE_NAMES: list[str] = [e.name for e in PoseModelType]

POSE_MODEL_FILE_NAMES: list[tuple[str, str]] = [
    ('none', ''),
    ('rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'),
    ('rtmpose-m_8xb256-420e_aic-coco-256x192.py', 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'),
    ('rtmpose-s_8xb256-420e_aic-coco-256x192.py', 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'),
    ('rtmpose-t_8xb256-420e_aic-coco-256x192.py', 'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth')
]

class PoseDetection(Thread):
    def __init__(self, path: str, model_type: PoseModelType, num_warmups: int, confidence_threshold: float = 0.3, verbose: bool = False) -> None:
        super().__init__()

        if model_type is PoseModelType.NONE:
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

        # Pose data
        self._poses_lock: Lock = Lock()
        self._poses_dict: dict[int, Pose] = {}
        self._pose_timestamp: Timestamp = Timestamp.now()

        # Callbacks
        self._callbacks: set[PoseDictCallback] = set()
        self._callback_queue: Queue[PoseDict | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._callback_worker, daemon=True)

        self._hot_reloader = HotReloadMethods(self.__class__)

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

        if self.is_alive() and self.verbose:
            print("Warning: Inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive() and self.verbose:
            print("Warning: Callback thread did not stop cleanly")

    def run(self) -> None:
        model: torch.nn.Module = init_model(self.model_config_file, self.model_checkpoint_file, device='cuda:0')
        model.half()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        scope = model.cfg.get('default_scope', 'mmpose') # pyright: ignore
        if scope is not None:
            init_default_scope(scope)
        pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline) # pyright: ignore

        self._model_warmup(model, pipeline, self.model_num_warmups, self.verbose)

        self._model_ready.set()  # Signal model is ready

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait(timeout=1.0)

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                with self._poses_lock:
                    poses: PoseDict = self._poses_dict
                    self._poses_dict = {}

                # Separate poses with and without crop_image
                poses_with_images: list[Pose] = [pose for pose in poses.values() if pose.crop_image is not None]
                images: list[np.ndarray] = [pose.crop_image for pose in poses_with_images if pose.crop_image is not None]

                # Run inference only on poses with images
                if images:
                    # Pass pipeline to inference
                    data_samples: list[list[PoseDataSample]] = PoseDetection._run_inference(model, pipeline, images, False)
                    point_data_list: list[PosePointData | None] = PoseDetection._extract_point_data_from_samples(data_samples, self.model_width, self.model_height, self.confidence_threshold)

                    # time.sleep(0.1)  # Yield to check if submit_poses works properly

                    # Create updated poses dict with all poses
                    updated_poses: PoseDict = {}
                    image_idx = 0
                    for pose in poses.values():
                        if pose.crop_image is not None:
                            updated_poses[pose.tracklet.id] = replace(pose, point_data=point_data_list[image_idx])
                            image_idx += 1
                        else:
                            updated_poses[pose.tracklet.id] = pose
                else:
                    updated_poses = poses

                try:
                    self._callback_queue.put_nowait(updated_poses)
                except:
                    if self.verbose:
                        print("Pose Detection Warning: Callback queue full, dropping inference results")

            except Exception as e:
                if self.verbose:
                    print(f"Pose Detection Error: {str(e)}")
                    traceback.print_exc()

        if self.verbose:
            print("PoseDetection: Inference thread stopped")

        if self.verbose:
            print("PoseDetection: Callback worker thread stopped")

    def submit_poses(self, poses: PoseDict) -> None:
        """Submit new poses for detection processing."""
        if self._shutdown_event.is_set():
            return

        old_poses: PoseDict | None = None
        lag: float = 0.0

        with self._poses_lock:
            if self._poses_dict:
                old_poses = self._poses_dict
                lag = (Timestamp.now() - self._pose_timestamp).total_seconds()
            self._poses_dict = poses
            self._pose_timestamp = Timestamp.now()

        if old_poses is not None:
            print(f"Pose Detection Warning: Still processing, dropped batch (lag: {lag:.3f}s)")

        self._notify_update_event.set()

    # CALLBACK
    def _callback_worker(self) -> None:
        """Worker thread that processes callbacks without blocking the main thread"""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1 and self.verbose:
                    print("Pose Detection Warning: Callback queue size > 1, consumers may be falling behind")

                poses: PoseDict | None = self._callback_queue.get(timeout=0.5)

                if poses is None:
                    break

                for c in self._callbacks:
                    try:
                        c(poses)
                    except Exception as e:
                        if self.verbose:
                            print(f"Pose Detection Callback Error: {str(e)}")
                            traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def add_poses_callback(self, callback: PoseDictCallback) -> None:
        self._callbacks.add(callback)

    # STATIC METHODS
    @staticmethod
    def _model_warmup(model: torch.nn.Module, pipeline: Compose, num_imgs: int, verbose: bool) -> None:
        """Pre-warm the model with dummy inputs to initialize CUDA kernels and memory"""
        if num_imgs <= 0:
            return
        if num_imgs > 8:
            num_imgs = 8  # Limit to 8 for warmup

        dummy_sizes: list[int] = [i for i in range(1, num_imgs + 1)]
        for batch_size in dummy_sizes:
            # Create batch of dummy images
            dummy_imgs: list[np.ndarray] = [np.zeros((POSE_MODEL_HEIGHT, POSE_MODEL_WIDTH, 3), dtype=np.uint8) for _ in range(batch_size)]
            _: list[list[PoseDataSample]] = PoseDetection._run_inference(model, pipeline, dummy_imgs, False)

            # Ensure GPU operations are complete
            torch.cuda.synchronize()

        if verbose:
            print("PoseDetection: Model warmup complete")

    @staticmethod
    def _run_inference(model: torch.nn.Module, pipeline: Compose, imgs: list[np.ndarray], verbose: bool) -> list[list[PoseDataSample]]:
        if not imgs:
            return []

        with torch.cuda.amp.autocast(dtype=torch.float16):
            total_start = time.perf_counter()

            # Cache frequently used values
            model_meta = model.dataset_meta
            batch_size = len(imgs)

            # Pre-allocate and reuse arrays
            data_list = [None] * batch_size
            bbox_score = np.ones(1, dtype=np.float32)

            prep_start = time.perf_counter()

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
            prep_time = time.perf_counter() - prep_start

            inference_start = time.perf_counter()
            with torch.inference_mode():
                all_results = model.test_step(batch) # pyright: ignore

            torch.cuda.synchronize()
            inference_time = time.perf_counter() - inference_start

            # Optimized result extraction
            results_by_image = [[sample] for sample in all_results]

            if verbose:
                print(f"  Preparation:   {prep_time*1000:.1f}ms,   Inference:   {inference_time*1000:.1f}ms,   batch size:   {batch_size}")

            return results_by_image

    @staticmethod
    def _extract_point_data_from_samples(data_samples: list[list[PoseDataSample]], model_width: int, model_height: int, confidence_threshold: float) -> list[PosePointData | None]:
        """Process pose data samples and return only the first detected pose for each image."""
        first_poses: list[PosePointData | None] = []

        for data_samples_for_image in data_samples:
            pose_found = False

            for data_sample in data_samples_for_image:
                pred_instances: InstanceData = data_sample.pred_instances
                keypoints: np.ndarray = pred_instances.get('keypoints', np.full((0, 17, 2), np.nan, dtype=np.float32))
                scores: np.ndarray = pred_instances.get('keypoint_scores', np.zeros((0, 17), dtype=np.float32))

                if keypoints.shape[0] == 0:
                    continue  # No pose detected

                # Take only first person's pose in keypoints array
                norm_keypoints: np.ndarray = keypoints[0].copy() / np.array([model_width, model_height])
                person_scores: np.ndarray = scores[0].copy()

                # Create pose and add to result
                pose = PosePointData(norm_keypoints, person_scores, confidence_threshold)
                first_poses.append(pose)
                pose_found = True
                break  # Stop after finding first pose

            # If no pose found for this image, add None
            if not pose_found:
                first_poses.append(None)

        return first_poses
