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
    _model_load_lock: Lock = Lock()
    _model_loaded: bool =  False
    _model_type: ModelType = ModelType.NONE
    _model_config_file: str = ""
    _model_checkpoint_file: str = ""
    _model_width: int = 192
    _model_height: int = 256
    _model_session: torch.nn.Module
    _pipeline: Compose

    def __init__(self, path: str, model_type:ModelType, fps: float = 30.0, verbose: bool = False) -> None:
        super().__init__()
        
        if PoseDetectionMulti._model_type is ModelType.NONE:
            PoseDetectionMulti._model_type = model_type
            PoseDetectionMulti._model_config_file = path + '/' + ModelFileNames[model_type.value][0]
            PoseDetectionMulti._model_checkpoint_file = path + '/' + ModelFileNames[model_type.value][1]
            print(PoseDetectionMulti._model_checkpoint_file, PoseDetectionMulti._model_config_file)
        else:
            if PoseDetectionMulti._model_type is not model_type:
                print('Pose Detection WARNING: ModelType is different from the first instance')

        if PoseDetectionMulti._model_type is ModelType.NONE:
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
                current_poses = []
                pose_ids = []
                images = []
                
                with self._poses_lock:
                    batch_size = min(8, len(self._poses_dict))
                    if batch_size > 0:
                        for i, pose_id in enumerate(list(self._poses_dict.keys())[:batch_size]):
                            pose = self._poses_dict.pop(pose_id)
                            if pose.image is not None:
                                current_poses.append(pose)
                                pose_ids.append(pose_id)
                                images.append(pose.image)
                
                # Split into two batches if more than 4 images
                batches = []
                if len(images) > 4:
                    batches = [
                        (current_poses[:4], images[:4]),
                        (current_poses[4:], images[4:])
                    ]
                elif images:
                    batches = [(current_poses, images)]

                start_time = time.perf_counter()
                cntr = 0
                for poses_batch, images_batch in batches:
                    all_poses = self.run_session(
                        PoseDetectionMulti._model_session, 
                        PoseDetectionMulti._pipeline,
                        images_batch)
                    end_time = time.perf_counter()
                    
                    
                    processing_time = end_time - start_time
                    print(f"Pose Detection Processing Time {cntr}: {processing_time:.4f} seconds")
                    cntr += 1

                    for i, pose in enumerate(poses_batch):
                        if i < len(all_poses) and all_poses[i]:
                            updated_pose = replace(
                                pose,
                                points=all_poses[i][0] if all_poses[i] else None
                            )
                            self.callback(updated_pose)
                        else:
                            self.callback(pose)

                while next_time < time.time():
                    next_time += self.interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

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

    @staticmethod
    def load_model_once() -> None:     
        with PoseDetectionMulti._model_load_lock:
            if not PoseDetectionMulti._model_loaded:
                model = init_model(PoseDetectionMulti._model_config_file, PoseDetectionMulti._model_checkpoint_file, device='cuda:0')
                try:
                    model.half()
                except Exception:
                    print("Pose Detection: Could not convert model to half precision.")
                PoseDetectionMulti._model_session = model
                PoseDetectionMulti._model_loaded = True
                
                PoseDetectionMulti._pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    @staticmethod
    def run_session(session: torch.nn, pipeline,  images: list[np.ndarray]) -> list[list[PosePoints]]:
        
        with torch.cuda.amp.autocast():
            all_results = PoseDetectionMulti.inference_topdown_multi(session, pipeline, images)
        
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
                    norm_keypoints[:, 0] /= 192   # x / width
                    norm_keypoints[:, 1] /= 256  # y / height

                    pose = PosePoints(norm_keypoints, person_scores)
                    image_poses.append(pose)

            poses.append(image_poses)

        return poses
    
    @staticmethod
    def inference_topdown_multi(model: nn.Module, pipeline,
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
    def inference_topdown(model: nn.Module,
                      img: Union[np.ndarray, str],
                      bboxes: Optional[Union[List, np.ndarray]] = None,
                      bbox_format: str = 'xyxy') -> List[PoseDataSample]:

        scope = model.cfg.get('default_scope', 'mmpose')
        if scope is not None:
            init_default_scope(scope)
        pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

        if bboxes is None or len(bboxes) == 0:
            # get bbox from the image size
            if isinstance(img, str):
                w, h = Image.open(img).size
            else:
                h, w = img.shape[:2]

            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        else:
            if isinstance(bboxes, list):
                bboxes = np.array(bboxes)

            assert bbox_format in {'xyxy', 'xywh'}, \
                f'Invalid bbox_format "{bbox_format}".'

            if bbox_format == 'xywh':
                bboxes = bbox_xywh2xyxy(bboxes)

        # construct batch data samples
        data_list = []
        for bbox in bboxes:
            if isinstance(img, str):
                data_info = dict(img_path=img)
            else:
                data_info = dict(img=img)
            data_info['bbox'] = bbox[None]  # shape (1, 4)
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_info.update(model.dataset_meta)
            data_list.append(pipeline(data_info))

        if data_list:
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            batch = pseudo_collate(data_list)
            with torch.no_grad():
                results = model.test_step(batch)
        else:
            results = []

        return results


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