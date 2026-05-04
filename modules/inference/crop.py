import logging
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn.functional as F

from modules.pose.features import BBox
from modules.pose.frame import FrameDict, replace
from .source import ImageDict as SourceImageDict
from modules.utils import Rect, Point2f
from modules.settings import BaseSettings, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Image:
    """GPU-resident cropped region for a single tracked pose.

    Format: float16 RGB CHW [0, 1].

    Attributes:
        crop:      Resized crop on GPU (3, H, W) float16 RGB CHW [0, 1]
        prev_crop: Previous frame cropped at current bbox for optical flow,
                   or None if no previous frame is available.
    """
    crop: torch.Tensor
    prev_crop: torch.Tensor | None = None


ImageDict: TypeAlias = dict[int, Image]
ImageCallback: TypeAlias = Callable[[FrameDict, ImageDict], None]


class Settings(BaseSettings):
    """Configuration for GPU-based image cropping."""
    expansion_width:  Field[float] = Field(0.0, min=0.0, max=1.0, access=Field.INIT)
    expansion_height: Field[float] = Field(0.0, min=0.0, max=1.0, access=Field.INIT)
    output_width:     Field[int]   = Field(384, access=Field.INIT)
    output_height:    Field[int]   = Field(512, access=Field.INIT)
    max_poses:        Field[int]   = Field(4, min=1, max=16, access=Field.INIT)
    enable_prev_crop: Field[bool]  = Field(True, access=Field.INIT)


class Extractor:
    """Crops GPU images at pose bounding-box locations and emits ImageDict.

    Receives SourceImageDict and prev-images from source.Uploader, plus a FrameDict
    with BBox features. For each pose, crops the current image and optionally
    the previous image at the same bbox for optical-flow use.

    All tensors are float16 RGB CHW [0, 1] at a fixed output resolution.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._callbacks: set[ImageCallback] = set()
        self._stream: torch.cuda.Stream = torch.cuda.Stream()

    def process(
        self,
        poses: FrameDict,
        images: SourceImageDict,
        prev_images: SourceImageDict,
    ) -> None:
        """Crop images for each pose and notify callbacks.

        Args:
            poses:       FrameDict with BBox features, keyed by track_id
            images:      Current GPU images (from source.Uploader.snapshot)
            prev_images: Previous GPU images (from source.Uploader.snapshot)
        """
        cropped_poses: FrameDict = {}
        crop_images: ImageDict = {}
        pose_count = 0

        with torch.cuda.stream(self._stream):
            for pose_id, pose in poses.items():
                cam_id = pose.cam_id
                if cam_id not in images:
                    continue

                if pose_count >= self._config.max_poses:
                    logger.warning(
                        f"Extractor: Exceeded max poses ({self._config.max_poses}), skipping {pose_id}"
                    )
                    continue

                try:
                    gpu_image = images[cam_id]
                    img_height, img_width = gpu_image.shape[1:3]
                    bbox_rect = pose[BBox].to_rect().zoom(
                        Point2f(1.0 + self._config.expansion_width, 1.0 + self._config.expansion_height)
                    )
                    crop_roi = self._calculate_crop_roi(bbox_rect, img_width, img_height)
                    crop_tensor = self._gpu_crop_resize(gpu_image, crop_roi, img_width, img_height)

                    prev_crop: torch.Tensor | None = None
                    if self._config.enable_prev_crop and cam_id in prev_images:
                        prev_img = prev_images[cam_id]
                        prev_h, prev_w = prev_img.shape[1:3]
                        prev_crop = self._gpu_crop_resize(prev_img, crop_roi, prev_w, prev_h)

                    normalized_roi = crop_roi.scale(Point2f(1.0 / img_width, 1.0 / img_height))
                    cropped_poses[pose_id] = replace(pose, {BBox: BBox.from_rect(normalized_roi)})
                    crop_images[pose_id] = Image(
                        crop=crop_tensor,
                        prev_crop=prev_crop,
                    )
                    pose_count += 1

                except Exception:
                    logger.exception(f"Extractor: Error processing pose {pose_id}")

        self._stream.synchronize()

        for callback in self._callbacks:
            try:
                callback(cropped_poses, crop_images)
            except Exception:
                logger.exception("Error in crop callback")

    def _calculate_crop_roi(self, bbox_rect: Rect, img_width: int, img_height: int) -> Rect:
        image_rect = Rect(0.0, 0.0, float(img_width), float(img_height))
        roi = bbox_rect.affine_transform(image_rect)
        crop_roi = Rect(0, 0, self._config.output_width, self._config.output_height)
        return crop_roi.aspect_fill(roi)

    def _gpu_crop_resize(self, src: torch.Tensor, roi: Rect, src_w: int, src_h: int) -> torch.Tensor:
        x1 = max(0, int(roi.x))
        y1 = max(0, int(roi.y))
        x2 = min(src_w, int(roi.x + roi.width))
        y2 = min(src_h, int(roi.y + roi.height))

        if x1 >= x2 or y1 >= y2:
            return torch.zeros(
                (3, self._config.output_height, self._config.output_width),
                device='cuda', dtype=torch.float16,
            )

        crop_chw = src[:, y1:y2, x1:x2]

        pad_left   = max(0, -int(roi.x))
        pad_right  = max(0, int(roi.x + roi.width) - src_w)
        pad_top    = max(0, -int(roi.y))
        pad_bottom = max(0, int(roi.y + roi.height) - src_h)

        if pad_left or pad_right or pad_top or pad_bottom:
            crop_chw = F.pad(crop_chw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        crop_h, crop_w = crop_chw.shape[1], crop_chw.shape[2]
        if crop_h != self._config.output_height or crop_w != self._config.output_width:
            crop_chw = F.interpolate(
                crop_chw.unsqueeze(0),
                size=(self._config.output_height, self._config.output_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return crop_chw

    def add_image_callback(self, callback: ImageCallback) -> None:
        self._callbacks.add(callback)
