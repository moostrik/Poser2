# Standard library imports
from dataclasses import replace
from typing import Callable
import time

import numpy as np

from modules.pose.features import BBox
from modules.pose.nodes._utils.ImageProcessor import ImageProcessor
from modules.pose.Frame import FrameDict
from modules.cam.depthcam.Definitions import FrameType
from modules.utils.PerformanceTimer import PerformanceTimer


CropCallback = Callable[[FrameDict, dict[int, np.ndarray]], None]
PairCropCallback = Callable[[FrameDict, dict[int, tuple[np.ndarray, np.ndarray]]], None]

class ImageCropProcessorConfig:
    """Configuration for image cropping."""

    def __init__(self, expansion: float = 0.1, output_width: int = 192, output_height: int = 256) -> None:
        self.crop_scale: float = 1.0 + expansion
        self.output_width: int = output_width
        self.output_height: int = output_height


class ImageCropProcessor:
    """Batch processor for cropping images based on pose bounding boxes.

    Crops current frames for regular processing and provides frame pairs
    (previous, current) cropped at the same location for optical flow.
    """

    def __init__(self, config: ImageCropProcessorConfig) -> None:
        self._config: ImageCropProcessorConfig = config
        self._image_processor: ImageProcessor = ImageProcessor(
            crop_scale=self._config.crop_scale,
            output_width=self._config.output_width,
            output_height=self._config.output_height
        )
        self._images: dict[int, np.ndarray] = {}
        self._previous_images: dict[int, np.ndarray] = {}
        self._callbacks: set[CropCallback] = set()
        self._flow_callbacks: set[PairCropCallback] = set()

        # Performance timer
        self._process_timer: PerformanceTimer = PerformanceTimer(
            name="CPUCrop Process", sample_count=10000, report_interval=100, color="red", omit_init=10
        )

    def set_image(self, cam_id: int, frame_type: FrameType, image: np.ndarray) -> None:
        """Store image from a specific camera. Only VIDEO frames are stored.

        Shifts current image to previous before storing new current.
        """
        if frame_type == FrameType.VIDEO:
            if cam_id in self._images:
                self._previous_images[cam_id] = self._images[cam_id]
            self._images[cam_id] = image

    def process(self, poses: FrameDict) -> None:
        """Process all poses at once, crop images, and notify callbacks.

        Crops current frame for regular callbacks and both previous/current frames
        at the same location (using current bbox) for flow callbacks.
        """
        start = time.perf_counter()

        cropped_poses: FrameDict = {}
        cropped_images: dict[int, np.ndarray] = {}
        cropped_frame_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Clean up previous images for lost tracks
        lost_ids = set(self._previous_images.keys()) - set(poses.keys())
        for lost_id in lost_ids:
            del self._previous_images[lost_id]

        for pose_id, pose in poses.items():
            if pose_id in self._images:
                try:
                    image = self._images[pose_id]
                    bbox_rect = pose.bbox.to_rect()

                    # Crop current frame
                    result_image, result_roi = self._image_processor.process_pose_image(bbox_rect, image)
                    cropped_poses[pose_id] = replace(pose, bbox=BBox.from_rect(result_roi))
                    cropped_images[pose_id] = result_image

                    # Only add to pairs if previous frame exists
                    if pose_id in self._previous_images:
                        prev_crop, _ = self._image_processor.process_pose_image(bbox_rect, self._previous_images[pose_id])
                        cropped_frame_pairs[pose_id] = (prev_crop, result_image)
                except Exception as e:
                    print(f"ImageCropProcessor: Error processing pose {pose_id}: {e}")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._process_timer.add_time(elapsed_ms)

        # Notify regular callbacks with cropped current frames
        for callback in self._callbacks:
            try:
                callback(cropped_poses, cropped_images)
            except Exception as e:
                print(f"ImageCropProcessor: Error in callback: {e}")

        # Only notify pair callbacks if there are actual pairs
        if cropped_frame_pairs:
            for callback in self._flow_callbacks:
                try:
                    callback(cropped_poses, cropped_frame_pairs)
                except Exception as e:
                    print(f"ImageCropProcessor: Error in flow callback: {e}")

    def add_callback(self, callback: Callable[[FrameDict, dict[int, np.ndarray]], None]) -> None:
        """Register callback to receive cropped poses and images."""
        self._callbacks.add(callback)

    def remove_callback(self, callback: Callable[[FrameDict, dict[int, np.ndarray]], None]) -> None:
        """Unregister callback."""
        self._callbacks.discard(callback)

    def add_pair_callback(self, callback: PairCropCallback) -> None:
        """Register callback to receive cropped frame pairs for optical flow."""
        self._flow_callbacks.add(callback)

    def remove_pair_callback(self, callback: PairCropCallback) -> None:
        """Unregister pair callback."""
        self._flow_callbacks.discard(callback)

    def reset(self) -> None:
        """Clear all stored images."""
        self._images.clear()
        self._previous_images.clear()