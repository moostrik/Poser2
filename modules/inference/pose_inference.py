"""PoseInference — the async step of a pose pipeline's frame domain: the crops and the pose model.

Reads the images the app's ``Uploader`` put on the GPU, and on ``submit`` cuts every pose's box out of the
newest images (``CropExtractor``) and hands the crops to the pose model
(``pose.Predictor``), which emits the poses with their keypoints on its own thread, dropping a batch under
load. The crops go to the other consumers of the same crops too (segmentation, optical flow, the board),
and the images of the batch to the board.
"""

import logging
from typing import Callable

from modules.pose import FrameDict, FrameDictCallbackMixin
from modules.utils import Broadcast

from . import crop, pose
from .crop_extractor import CropExtractor
from .source import ImageDict
from .uploader import Uploader

logger = logging.getLogger(__name__)


class PoseInference(FrameDictCallbackMixin):
    """The crop and pose predictor as one async step; see the module docstring."""

    def __init__(self, crop_settings: crop.Settings, pose_settings: pose.Settings, uploader: Uploader) -> None:
        super().__init__()
        self._uploader: Uploader = uploader
        self._crop_extractor = CropExtractor(crop_settings)
        self._predictor = pose.Predictor(pose_settings)
        self._camera_images_callbacks: Broadcast[ImageDict] = Broadcast()
        self._crop_extractor.add_image_callback(self._predictor.submit)
        self._predictor.add_frames_callback(self._notify_frames_callbacks)

    def start(self) -> None:
        self._predictor.start()

    def stop(self) -> None:
        self._predictor.stop()

    def submit(self, frames: FrameDict) -> None:
        """Crop the poses' boxes from the newest images and queue them for the pose model."""
        images, prev_images, timestamps = self._uploader.snapshot()
        self._camera_images_callbacks(images)
        self._crop_extractor.process(frames, images, prev_images, timestamps)

    def add_camera_images_callback(self, callback: Callable[[ImageDict], None]) -> None:
        """The GPU images of each batch, before the crops are cut."""
        self._camera_images_callbacks.add_callback(callback)

    def add_crop_callback(self, callback: crop.ImageCallback) -> None:
        """The poses with their crops, after the pose model has them."""
        self._crop_extractor.add_image_callback(callback)

    def reset(self) -> None:
        self._predictor.reset()
        self._uploader.reset()
