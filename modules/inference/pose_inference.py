"""PoseInference — the async step of a pose pipeline's frame domain: the crops and the pose model.

On ``submit`` it is handed the poses with the images they are for (the frame source's ``Uploader.snapshot`` of
that moment, carried by the pipeline), cuts every pose's box out of exactly those images (``CropExtractor``),
and hands the crops to the pose model (``pose.Predictor``), which emits the poses with their keypoints on its own
thread, dropping a batch under load. The crops go to the other consumers of the same crops too (segmentation,
optical flow, the board), each with the event that says they are written. ``cut_crops`` cuts the crops for them
without the pose model. Poses without images are an error: every frame source hands its images on.
"""

import logging
from typing import Any

from modules.pose import FrameDict, FrameDictCallbackMixin

from . import crop, pose
from .crop_extractor import CropExtractor
from .uploader import ImageSnapshot, Uploader

logger = logging.getLogger(__name__)


class PoseInference(FrameDictCallbackMixin):
    """The crop and pose predictor as one async step; see the module docstring."""

    def __init__(self, crop_settings: crop.Settings, pose_settings: pose.Settings, uploader: Uploader) -> None:
        super().__init__()
        self._uploader: Uploader = uploader
        self._crop_extractor = CropExtractor(crop_settings)
        self._predictor = pose.Predictor(pose_settings)
        self._predictor.add_frames_callback(self._notify_frames_callbacks)

    def start(self) -> None:
        self._predictor.start()

    def stop(self) -> None:
        self._predictor.stop()

    def submit(self, frames: FrameDict, images: Any) -> None:
        """Crop the poses' boxes from `images` and queue them for the pose model, on the caller's thread and without
        waiting for the GPU: the crops wait for the images, and the pose model for the crops, on their own
        streams."""
        cropped, crops = self.cut_crops(frames, images)
        self._predictor.submit(cropped, crops)

    def cut_crops(self, frames: FrameDict, images: Any) -> tuple[FrameDict, crop.CropImageDict]:
        """Crop the poses' boxes from `images` (an `ImageSnapshot`), for the crop callbacks, without the pose model:
        the poses with their crops as their boxes and the images' capture times as their stamps, and the crops."""
        if not isinstance(images, tuple) or len(images) != 4:
            raise ValueError("PoseInference: poses came without the images they are for (an Uploader.snapshot)")
        snapshot: ImageSnapshot = images
        current, previous, timestamps, ready = snapshot
        return self._crop_extractor.process(frames, current, previous, timestamps, ready)

    def add_crop_callback(self, callback: crop.ImageCallback) -> None:
        """The poses with their crops, as soon as they are cut."""
        self._crop_extractor.add_image_callback(callback)

    def reset(self) -> None:
        self._predictor.reset()
        self._uploader.reset()
