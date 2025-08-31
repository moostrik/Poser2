# Standard library imports
from dataclasses import replace
from queue import Queue
from threading import Thread, Lock
import os
import time

# Third-party imports
import cv2
import numpy as np

import torch
import torch.nn as nn
from mmpose.apis import inference_topdown, init_model
from typing import List, Optional, Union
from mmpose.structures import PoseDataSample
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from PIL import Image

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.builder import build_pose_estimator
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy

# Local application imports
from modules.pose.PoseDefinitions import Pose, PosePoints, ModelType, ModelFileNames

class PoseDetectionMulti(Thread):
    # Class no longer has class variables
    
    def __init__(self, path: str, model_type:ModelType, fps: float = 30.0, verbose: bool = False) -> None:
        super().__init__()
        
        # Convert class variables to instance variables
        self.model_load_lock: Lock = Lock()
        self.model_loaded: bool = False
        self.model_type: ModelType = model_type
        self.model_config_file: str = path + '/' + ModelFileNames[model_type.value][0]
        self.model_checkpoint_file: str = path + '/' + ModelFileNames[model_type.value][1]
        self.model_width: int = 192
        self.model_height: int = 256
        self.model_session: torch.nn.Module = None
        self.pipeline: Compose = None
        
        print(self.model_checkpoint_file, self.model_config_file)

        if self.model_type is ModelType.NONE:
            print('Pose Detection WARNING: ModelType is NONE')

        self.interval: float = 1.0 / fps

        self.verbose: bool = verbose
        self._running: bool = False
        # Replace Queue with dictionary
        self._poses_dict: dict = {}
        self._poses_lock: Lock = Lock()  # Add lock for thread safety
        self._callbacks: set = set()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self.load_model_once()

        self._running = True
        
        next_time: float = time.time()
        while self._running:
            try:
                # Process multiple poses at a time from the dictionary
                current_poses = []
                pose_ids = []
                images = []
                
                with self._poses_lock:
                    # Collect up to 8 images to process in batch
                    batch_size = min(8, len(self._poses_dict))
                    if batch_size > 0:
                        for i, pose_id in enumerate(list(self._poses_dict.keys())[:batch_size]):
                            pose = self._poses_dict.pop(pose_id)
                            if pose.image is not None:
                                current_poses.append(pose)
                                pose_ids.append(pose_id)
                                images.append(pose.image)
                
                # If we got poses to process, handle them in batch
                if images:
                    # Process all images in a single batch call
                    start_time = time.perf_counter()
                    all_poses = self.run_session(
                        self.model_session, 
                        self.pipeline,
                        images)
                    end_time = time.perf_counter()
                    
                    processing_time = end_time - start_time
                    print(f"Pose Detection Processing Time: {processing_time:.4f} seconds")

                    # Match results with original poses
                    for i, pose in enumerate(current_poses):
                        if i < len(all_poses) and all_poses[i]:
                            # Use the first detected person in each image
                            # (or could implement person matching logic here)
                            updated_pose = replace(
                                pose,
                                points=all_poses[i][0] if all_poses[i] else None
                            )
                            self.callback(updated_pose)
                        else:
                            # No pose detected, but still send back the original
                            self.callback(pose)
                while next_time < time.time():
                    next_time += self.interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # else:
                #     # No poses to process, wait a bit
                #     import time
                #     time.sleep(0.01)

            except Exception as e:
                if self.verbose:
                    print(f"Pose Detection Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

    # GETTERS AND SETTERS
    def add_pose(self, pose: Pose) -> None:
        if self._running and pose.id is not None:
            with self._poses_lock:
                # if self._poses_dict.get(pose.id) is not None:
                #     print(f"Pose Detection Warning: Pose ID {pose.id} already in queue, overwriting.")
                self._poses_dict[pose.id] = pose
    
    # CALLBACKS
    def callback(self, pose: Pose) -> None:
        for c in self._callbacks:
            c(pose)

    def addMessageCallback(self, callback) -> None:
        self._callbacks.add(callback)

    def clearMessageCallbacks(self) -> None:
        self._callbacks = set()

    # Changed from staticmethod to instance method
    def load_model_once(self) -> None:     
        with self.model_load_lock:
            if not self.model_loaded:
                model = init_model(self.model_config_file, self.model_checkpoint_file, device='cuda:0')
                try:
                    model.half()
                except Exception:
                    print("Pose Detection: Could not convert model to half precision.")
                self.model_session = model
                self.model_loaded = True
                
                self.pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # Changed from staticmethod to instance method
    def run_session(self, session: torch.nn, pipeline, images: list[np.ndarray]) -> list[list[PosePoints]]:
        with torch.cuda.amp.autocast():
            all_results = self.inference_topdown_multi(session, pipeline, images)
        
        poses = []
        
        # Process all results for all images
        for image_results in all_results:
            image_poses = []
            for result in image_results:
                pred_instances = result.pred_instances
                keypoints = pred_instances.keypoints
                scores = pred_instances.keypoint_scores

                for i in range(len(keypoints)):
                    person_keypoints = keypoints[i]  # [num_keypoints, 2]
                    person_scores = scores[i]        # [num_keypoints]

                    # Normalize keypoints to [0, 1] range
                    norm_keypoints = person_keypoints.copy()
                    norm_keypoints[:, 0] /= self.model_width   # x / width
                    norm_keypoints[:, 1] /= self.model_height  # y / height

                    pose = PosePoints(norm_keypoints, person_scores)
                    image_poses.append(pose)

            poses.append(image_poses)

        return poses
    
    # Changed from staticmethod to instance method
    def inference_topdown_multi(self, model: nn.Module, pipeline,
                  imgs: List[np.ndarray],
                  bboxes: Optional[List[Union[List, np.ndarray]]] = None,
                  bbox_format: str = 'xyxy') -> List[List[PoseDataSample]]:
        """Inference multiple images with a top-down pose estimator.

        Args:
            model (nn.Module): The top-down pose estimator
            imgs (List[np.ndarray]): List of images to process in batch
            bboxes (List[np.ndarray], optional): List of bboxes for each image.
                If None, entire image will be used. Defaults to None.
            bbox_format (str): The bbox format indicator. Options are ``'xywh'``
                and ``'xyxy'``. Defaults to ``'xyxy'``

        Returns:
            List[List[PoseDataSample]]: The inference results for each image.
        """
        scope = model.cfg.get('default_scope', 'mmpose')
        if scope is not None:
            init_default_scope(scope)
        # pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
        
        if bboxes is None:
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
                
                if bbox_format == 'xywh':
                    img_bboxes = bbox_xywh2xyxy(img_bboxes)
            
            # Create data samples for each bbox in this image
            for bbox in img_bboxes:
                data_info = dict(img=img)
                data_info['bbox'] = bbox[None]  # shape (1, 4)
                data_info['bbox_score'] = np.ones(1, dtype=np.float32)
                data_info.update(model.dataset_meta)
                data_list.append(pipeline(data_info))
            
            img_lengths.append(len(img_bboxes))
        
        results_by_image = []
        
        if data_list:
            # Process all images in a single batch
            batch = pseudo_collate(data_list)
            
            
            start_time = time.perf_counter()
            with torch.no_grad():
                all_results = model.test_step(batch)
                end_time = time.perf_counter()
                # print(f"Pose Detection: Inference time: {end_time - start_time:.4f} seconds")

            # Split results back by image
            start_idx = 0
            for length in img_lengths:
                results_by_image.append(all_results[start_idx:start_idx + length])
                start_idx += length
        
        return results_by_image

   

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