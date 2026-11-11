"""PoseInference — the async step of a pose pipeline's frame domain: the crops and the pose model.

Reads the images the app's ``Uploader`` put on the GPU, and on ``submit`` cuts every pose's box out of the
newest images (``CropExtractor``) and hands the crops to the pose model
(``pose.Predictor``), which emits the poses with their keypoints on its own thread, dropping a batch under
load. The crops go to the other consumers of the same crops too (segmentation, optical flow, the board),
each with the event that says they are written.
"""

import logging

from modules.pose import FrameDict, FrameDictCallbackMixin

from . import crop, pose
from .crop_extractor import CropExtractor
from .uploader import Uploader

logger = logging.getLogger(__name__)


class PoseInference(FrameDictCallbackMixin):
    """The crop and pose predictor as one async step; see the module docstring."""

    def __init__(self, crop_settings: crop.Settings, pose_settings: pose.Settings, uploader: Uploader) -> None:
        super().__init__()
        self._uploader: Uploader = uploader
        self._crop_extractor = CropExtractor(crop_settings)
        self._predictor = pose.Predictor(pose_settings)
        self._crop_extractor.add_image_callback(self._predictor.submit)
        self._predictor.add_frames_callback(self._notify_frames_callbacks)

    def start(self) -> None:
        self._predictor.start()

    def stop(self) -> None:
        self._predictor.stop()

    def submit(self, frames: FrameDict) -> None:
        """Crop the poses' boxes from the newest images and queue them for the pose model, on the caller's thread
        and without waiting for the GPU: the crops wait for the images, and the pose model for the crops, on
        their own streams."""
        images, prev_images, timestamps, ready = self._uploader.snapshot()
        self._crop_extractor.process(frames, images, prev_images, timestamps, ready)

    def add_crop_callback(self, callback: crop.ImageCallback) -> None:
        """The poses with their crops, after the pose model has them."""
        self._crop_extractor.add_image_callback(callback)

    def reset(self) -> None:
        self._predictor.reset()
        self._uploader.reset()
