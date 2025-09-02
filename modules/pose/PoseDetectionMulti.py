# Standard library imports
import time
from dataclasses import replace
from threading import Thread, Lock, Event

# Third-party imports
import cv2
import numpy as np
import torch
from pandas import Timestamp

from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy

# Local application imports
from modules.pose.PoseDefinitions import Pose, PosePoints, ModelType, ModelFileNames

class PoseDetectionMulti(Thread):    
    def __init__(self, path: str, model_type:ModelType, fps: float = 30.0, verbose: bool = False) -> None:
        super().__init__()
        
        if model_type is ModelType.NONE:
            print('Pose Detection WARNING: ModelType is NONE')
        self.model_config_file: str = path + '/' + ModelFileNames[model_type.value][0]
        self.model_checkpoint_file: str = path + '/' + ModelFileNames[model_type.value][1]
        self.model_width: int = 192
        self.model_height: int = 256
        
        self.interval: float = 1.0 / fps

        self.verbose: bool = verbose
        self._running: bool = False
        
        self._poses_dict: dict = {}
        self._poses_timestamp: dict = {}
        self._poses_lock: Lock = Lock()  # Add lock for thread safety
        self._callbacks: set = set()
        
        self._notify_update_event: Event = Event()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        model:torch.nn.Module = init_model(self.model_config_file, self.model_checkpoint_file, device='cuda:0')
        model.half()        
        pipeline: Compose = Compose(model.cfg.test_dataloader.dataset.pipeline)

        self._running = True
        
        next_time: float = time.time()
        while self._running:
            self._notify_update_event.wait(timeout=0.1)
            self._notify_update_event.clear()
            start_time = time.perf_counter()
            try:
                # Process multiple poses at a time from the dictionary
                current_poses: list[Pose] = []
                empty_poses: list[Pose] = []
                images: list[np.ndarray] = []

                with self._poses_lock:
                    for pose in self._poses_dict.values():
                        if pose.image is not None:
                            current_poses.append(pose)
                        else:
                            empty_poses.append(pose)
                    self._poses_dict = {}
                
                for pose in empty_poses:
                    self.callback(pose)
                     
                images = [pose.image for pose in current_poses]

                if images:
                    data_samples = PoseDetectionMulti.run_interference(model,pipeline,images, False)
                    all_poses = PoseDetectionMulti.process_pose_data_samples(data_samples, self.model_width, self.model_height)

                    # Match results with original poses
                    for i, pose in enumerate(current_poses):
                        if i < len(all_poses) and all_poses[i]:
                            # Use the first detected person in each image
                            updated_pose = replace(
                                pose,
                                points=all_poses[i][0] if all_poses[i] else None
                            )
                            self.callback(updated_pose)
                        else:
                            # No pose detected, but still send back the original
                            self.callback(pose)

            except Exception as e:
                if self.verbose:
                    print(f"Pose Detection Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
            
            if self.verbose:
                detection_time = time.perf_counter() - start_time
                if detection_time > self.interval:
                    print(f"Pose Detection Time: {detection_time:.3f} seconds")

    # GETTERS AND SETTERS
    def add_pose(self, pose: Pose) -> None:
        if self._running and pose.id is not None:
            with self._poses_lock:
                if self._poses_dict.get(pose.id) is not None:
                    existing_pose: Pose = self._poses_dict[pose.id]
                    e_time = existing_pose.time_stamp
                    n_time = pose.time_stamp
                    difference = (n_time - e_time).total_seconds()
                    
                    t2 = self._poses_timestamp.get(pose.id, Timestamp.now())
                    diff2 = (Timestamp.now() - t2).total_seconds()

                    print(f"Pose Detection Warning: Pose ID {pose.id} already in queue, overwriting. {existing_pose.cam_id} {pose.cam_id} {difference:.3f}, {diff2:.3f}")
                    
                    # print(f"Existing Pose: {self._poses_dict[pose.id].time_stamp}, New Pose: {pose.time_stamp}")

                self._poses_dict[pose.id] = pose
                self._poses_timestamp[pose.id] = Timestamp.now()
                
        
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
    def run_interference(model: torch.nn.Module, pipeline: Compose, imgs: list[np.ndarray], verbose: bool= False) -> list[list[PoseDataSample]]:
        with torch.cuda.amp.autocast():
            
            start_time = time.perf_counter()
            
            scope = model.cfg.get('default_scope', 'mmpose')
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
                    
            
            if verbose:
                print(f"Pose Detection Processing Time: {time.perf_counter() - start_time  :.3f} seconds")

            return results_by_image
    
    @staticmethod
    def process_pose_data_samples(data_samples: list, model_width: int, model_height: int) -> list[list[PosePoints]]:
        poses: list[list[PosePoints]] = []
        for image_results in data_samples:
            image_poses: list[PosePoints] = []
            for result in image_results:
                pred_instances = result.pred_instances
                keypoints = pred_instances.keypoints
                scores = pred_instances.keypoint_scores

                for i in range(len(keypoints)):
                    person_keypoints = keypoints[i]  # [num_keypoints, 2]
                    person_scores = scores[i]        # [num_keypoints]

                    # Normalize keypoints to [0, 1] range
                    norm_keypoints = person_keypoints.copy()
                    norm_keypoints[:, 0] /= model_width   # x / width
                    norm_keypoints[:, 1] /= model_height  # y / height

                    pose = PosePoints(norm_keypoints, person_scores)
                    image_poses.append(pose)

            poses.append(image_poses)

        return poses
    
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