# Standard library imports
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
from modules.pose.Pose import Pose, PosePointData

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
    def __init__(self, path: str, model_type:PoseModelType, fps: float = 30.0, confidence_threshold: float = 0.3, verbose: bool = False) -> None:
        super().__init__()

        if model_type is PoseModelType.NONE:
            print('Pose Detection WARNING: ModelType is NONE')
        self.model_config_file: str = path + '/' + POSE_MODEL_FILE_NAMES[model_type.value][0]
        self.model_checkpoint_file: str = path + '/' + POSE_MODEL_FILE_NAMES[model_type.value][1]
        self.model_width: int = 192
        self.model_height: int = 256
        self.confidence_threshold: float = confidence_threshold

        self.interval: float = 1.0 / fps

        self.verbose: bool = verbose
        self._running: bool = False

        self._poses_dict: dict[int, Pose] = {}
        self._poses_timestamp: dict[int, Timestamp] = {}
        self._poses_lock: Lock = Lock()  # Add lock for thread safety
        self._callbacks: set = set()

        self._notify_update_event: Event = Event()


        hot_reloader = HotReloadMethods(self.__class__)

    def stop(self) -> None:
        self._running = False
        self._notify_update_event.set()  # Wake up the thread if waiting

    def run(self) -> None:
        model:torch.nn.Module = init_model(self.model_config_file, self.model_checkpoint_file, device='cuda:0')
        model.half()
        pipeline: Compose = Compose(model.cfg.test_dataloader.dataset.pipeline) # pyright: ignore

        self._running = True

        while self._running:
            self._notify_update_event.wait(timeout=1.0)
            self._notify_update_event.clear()
            start_time: float = time.perf_counter()

            try:
                # Step 1: Pre-process - get poses and prepare images
                poses, images = self._preprocess_poses()

                # Step 2: Run inference if we have images
                if images:
                    # Run inference
                    data_samples: list[list[PoseDataSample]] = PoseDetection._run_inference(model, pipeline, images)

                    # Step 3: Post-process results and send callbacks
                    self._postprocess_results(poses, data_samples)

            except Exception as e:
                if self.verbose:
                    print(f"Pose Detection Error: {str(e)}")
                    traceback.print_exc()

            if self.verbose:
                detection_time = time.perf_counter() - start_time
                if detection_time > self.interval:
                    print(f"Pose Detection Time: {detection_time:.3f} seconds")

    def _preprocess_poses(self) -> tuple[list[Pose], list[np.ndarray]]:
        """Extract poses from queue and prepare images for inference."""
        poses_with_images: list[Pose] = []
        poses_without_images: list[Pose] = []

        for pose in self.get_poses(consume=True).values():
            if pose.crop_image is not None:
                poses_with_images.append(pose)
            else:
                poses_without_images.append(pose)

        # Handle poses without images immediately (keep in pipeline)
        for pose in poses_without_images:
            self.callback(pose)

        # Prepare images for inference
        images: list[np.ndarray] = [pose.crop_image for pose in poses_with_images if pose.crop_image is not None]

        return poses_with_images, images

    def _postprocess_results(self, poses: list[Pose], data_samples: list[list[PoseDataSample]]) -> None:
        """Process inference results and call callbacks with updated poses."""
        point_data_list: list[PosePointData | None] = PoseDetection._extract_point_data_from_samples(
            data_samples, self.model_width, self.model_height, self.confidence_threshold
        )

        # Match results with original poses
        for i, pose in enumerate(poses):
            if i < len(point_data_list) and point_data_list[i] is not None:
                updated_pose: Pose = replace(
                    pose,
                    point_data=point_data_list[i]
                )
                self.callback(updated_pose)
            else:
                # No pose detected, callback with original pose
                self.callback(pose)

    # GETTERS AND SETTERS
    def add_pose(self, pose: Pose) -> None:
        if self._running and pose.tracklet.id is not None:
            with self._poses_lock:
                if self._poses_dict.get(pose.tracklet.id) is not None:
                    existing_pose: Pose = self._poses_dict[pose.tracklet.id]
                    self.callback(existing_pose)

                    if self.verbose:
                        diff1: float = (pose.tracklet.time_stamp - existing_pose.tracklet.time_stamp).total_seconds()
                        diff2: float = (Timestamp.now() - self._poses_timestamp.get(pose.tracklet.id, Timestamp.now())).total_seconds()
                        print(f"Pose Detection Warning: Pose ID {pose.tracklet.id} already in queue, overwriting. {diff1:.3f}, {diff2:.3f}")

                self._poses_dict[pose.tracklet.id] = pose
                self._poses_timestamp[pose.tracklet.id] = Timestamp.now()

    def get_poses(self, consume: bool) -> dict[int, Pose]:
        with self._poses_lock:
            poses: dict[int, Pose] = self._poses_dict.copy()
            if consume:
                self._poses_dict = {}
            return poses

    def notify_update(self) -> None:
        if self._running:
            self._notify_update_event.set()

    # CALLBACKS
    def callback(self, pose: Pose) -> None:
        for c in self._callbacks:
            c(pose)

    def addMessageCallback(self, callback) -> None:
        self._callbacks.add(callback)

    def clearMessageCallbacks(self) -> None:
        self._callbacks = set()

    # STATIC METHODS
    @staticmethod
    def _run_inference(model: torch.nn.Module, pipeline: Compose, imgs: list[np.ndarray], verbose: bool= False) -> list[list[PoseDataSample]]:
        with torch.cuda.amp.autocast(): # pyright: ignore

            start_time = time.perf_counter()

            scope = model.cfg.get('default_scope', 'mmpose') # pyright: ignore
            if scope is not None:
                init_default_scope(scope)
            # pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

            bboxes = [None] * len(imgs)

            # Process each image and its bboxes to create data samples
            data_list = []
            img_lengths = []  # Track number of data samples per image

            for img_idx, (img, img_bboxes) in enumerate(zip(imgs, bboxes)):
                h, w = img.shape[:2]

                # Handle bboxes for this image
                if img_bboxes is None or len(img_bboxes) == 0:
                    img_bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
                else:
                    if isinstance(img_bboxes, list):
                        img_bboxes = np.array(img_bboxes)

                    img_bboxes = bbox_xywh2xyxy(img_bboxes)

                # Create data samples for each bbox in this image
                for bbox in img_bboxes:
                    data_info = dict(img=img)
                    data_info['bbox'] = bbox[None]  # shape (1, 4)
                    data_info['bbox_score'] = np.ones(1, dtype=np.float32)
                    data_info.update(model.dataset_meta) # pyright: ignore
                    data_list.append(pipeline(data_info))

                img_lengths.append(len(img_bboxes))

            results_by_image = []

            if data_list:
                # Process all images in a single batch
                batch = pseudo_collate(data_list)


                start_time = time.perf_counter()
                with torch.no_grad():
                    all_results = model.test_step(batch) # pyright: ignore
                    end_time = time.perf_counter()
                    # print(f"Pose Detection: Inference time: {end_time - start_time:.4f} seconds")

                # Split results back by image
                start_idx = 0
                for length in img_lengths:
                    results_by_image.append(all_results[start_idx:start_idx + length])
                    start_idx += length


            # if verbose:
            #     print(f"Pose Detection Processing Time: {time.perf_counter() - start_time  :.3f} seconds")

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

    @staticmethod
    def resize_with_pad(image, target_width, target_height, padding_color=(0, 0, 0)) -> np.ndarray:
        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio: float = original_width / original_height

        # Determine the new dimensions while maintaining the aspect ratio
        if target_width / target_height > aspect_ratio:
            new_height: int = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width: int = target_width
            new_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image: np.ndarray = cv2.resize(image, (new_width, new_height))

        # Create a new image with the target dimensions and the padding color
        padded_image: np.ndarray = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

        # Calculate the position to place the resized image
        x_offset: int = (target_width - new_width) // 2
        y_offset: int = (target_height - new_height) // 2

        # Place the resized image on the padded image
        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

        return padded_image