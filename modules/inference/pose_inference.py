"""PoseInference — the async step of a pose pipeline's frame domain: the crops and the pose model.

On ``submit`` it is handed the poses with the images they are for (the `ImageSnapshot` an uploader handed on,
carried by the frame source and the pipeline), cuts every pose's box out of exactly those images (``CropExtractor``),
and hands the crops to the pose model (``pose.Predictor``), which emits the poses with their keypoints on its own
thread, dropping a batch under load. The crops go to the other consumers of the same crops too (segmentation,
optical flow, the board), each with the event that says they are written. ``cut_crops`` cuts the crops for them
without the pose model. Poses without images are an error: every frame source hands its images on.

With the host pose model off (``pose.enabled``) the crops are still cut, and the poses that already carry the
camera's own keypoints (an OAK4 pose chain, put on by the frame source) are handed on as they are; the others
are not posed. With it on, the model's keypoints replace the camera's.
"""

import logging
from typing import Any

import numpy as np

from modules.pose import Frame, FrameDict, FrameDictCallbackMixin, Points2D

from . import crop, pose
from .crop_extractor import CropExtractor
from .source import ImageSnapshot

logger = logging.getLogger(__name__)


def has_points(frame: Frame) -> bool:
    """Whether a pose carries keypoints: any point with a score."""
    return bool(np.any(frame[Points2D].scores > 0.0))


class PoseInference(FrameDictCallbackMixin):
    """The crop and pose predictor as one async step; see the module docstring."""

    def __init__(self, crop_settings: crop.Settings, pose_settings: pose.Settings) -> None:
        super().__init__()
        self._pose_settings: pose.Settings = pose_settings
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
        streams. With the host model off, the poses carrying the camera's keypoints go on as they are."""
        cropped, crops = self.cut_crops(frames, images)
        if self._pose_settings.enabled:
            self._predictor.submit(cropped, crops)
            return
        posed: FrameDict = {track_id: frame for track_id, frame in cropped.items() if has_points(frame)}
        if posed:
            self._notify_frames_callbacks(posed)

    def cut_crops(self, frames: FrameDict, images: Any) -> tuple[FrameDict, crop.CropImageDict]:
        """Crop the poses' boxes from `images` (an `ImageSnapshot`), for the crop callbacks, without the pose model:
        the poses with their crops as their boxes and the images' capture times as their stamps, and the crops."""
        if not isinstance(images, tuple) or len(images) != 4:
            raise ValueError("PoseInference: poses came without the images they are for (an ImageSnapshot)")
        snapshot: ImageSnapshot = images
        current, previous, timestamps, ready = snapshot
        return self._crop_extractor.process(frames, current, previous, timestamps, ready)

    def add_crop_callback(self, callback: crop.ImageCallback) -> None:
        """The poses with their crops, as soon as they are cut."""
        self._crop_extractor.add_image_callback(callback)

    def reset(self) -> None:
        self._predictor.reset()
