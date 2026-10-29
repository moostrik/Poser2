import logging

import torch
import torch.nn.functional as F

from modules.pose import BBox, FrameDict, replace
from .crop import Image, CropImageDict, ImageCallback, Settings
from .source import ImageDict
from modules.utils import Rect, Point2f

logger = logging.getLogger(__name__)


class CropExtractor:
    """Crops GPU images at pose bounding-box locations and emits CropImageDict.

    Receives ImageDict and prev-images from Uploader, plus a FrameDict
    with BBox features. For each pose, crops the current image and optionally
    the previous image at the same bbox for optical-flow use.

    All tensors are float16 RGB CHW [0, 1] at a fixed output resolution.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._callbacks: list[ImageCallback] = []       # run in registration order
        self._stream: torch.cuda.Stream = torch.cuda.Stream()

    def process(
        self,
        poses: FrameDict,
        images: ImageDict,
        prev_images: ImageDict,
    ) -> None:
        """Crop images for each pose and notify callbacks.

        Args:
            poses:       FrameDict with BBox features, keyed by track_id
            images:      Current GPU images (from Uploader.snapshot)
            prev_images: Previous GPU images (from Uploader.snapshot)
        """
        cropped_poses: FrameDict = {}
        crop_images: CropImageDict = {}
        pose_count = 0
        # Live settings, read once so every pose in this batch is cropped with the same padding.
        zoom: float = 1.0 + self.settings.padding

        with torch.cuda.stream(self._stream):
            for track_id, pose in poses.items():
                cam_id = pose.cam_id
                if cam_id not in images:
                    continue

                if pose_count >= self.settings.max_players:
                    logger.warning(
                        f"Exceeded max poses ({self.settings.max_players}), skipping {track_id}"
                    )
                    continue

                try:
                    gpu_image = images[cam_id]
                    img_height, img_width = gpu_image.shape[1:3]
                    bbox_rect = pose[BBox].to_rect().zoom(zoom)
                    crop_roi = self._calculate_crop_roi(bbox_rect, img_width, img_height)
                    crop_tensor = self._gpu_crop_resize(gpu_image, crop_roi, img_width, img_height)

                    prev_crop: torch.Tensor | None = None
                    if self.settings.enable_prev_crop and cam_id in prev_images:
                        prev_img = prev_images[cam_id]
                        prev_h, prev_w = prev_img.shape[1:3]
                        prev_crop = self._gpu_crop_resize(prev_img, crop_roi, prev_w, prev_h)

                    normalized_roi = crop_roi.scale(Point2f(1.0 / img_width, 1.0 / img_height))
                    cropped_poses[track_id] = replace(pose, {BBox: BBox.from_rect(normalized_roi)})
                    crop_images[track_id] = Image(
                        crop=crop_tensor,
                        prev_crop=prev_crop,
                    )
                    pose_count += 1

                except Exception:
                    logger.exception(f"Error processing pose {track_id}")

        self._stream.synchronize()

        for callback in self._callbacks:
            try:
                callback(cropped_poses, crop_images)
            except Exception:
                logger.exception("Error in crop callback")

    def _calculate_crop_roi(self, bbox_rect: Rect, img_width: int, img_height: int) -> Rect:
        image_rect = Rect(0.0, 0.0, float(img_width), float(img_height))
        roi = bbox_rect.affine_transform(image_rect)
        crop_roi = Rect(0, 0, self.settings.output_width, self.settings.output_height)
        return crop_roi.aspect_fill(roi)

    def _gpu_crop_resize(self, src: torch.Tensor, roi: Rect, src_w: int, src_h: int) -> torch.Tensor:
        x1 = max(0, int(roi.x))
        y1 = max(0, int(roi.y))
        x2 = min(src_w, int(roi.x + roi.width))
        y2 = min(src_h, int(roi.y + roi.height))

        if x1 >= x2 or y1 >= y2:
            return torch.zeros(
                (3, self.settings.output_height, self.settings.output_width),
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
        if crop_h != self.settings.output_height or crop_w != self.settings.output_width:
            # Antialiased: a crop is usually larger than the output, and plain bilinear would sample
            # only a 2×2 neighbourhood of each output pixel and alias. No effect when enlarging.
            crop_chw = F.interpolate(
                crop_chw.unsqueeze(0),
                size=(self.settings.output_height, self.settings.output_width),
                mode='bilinear',
                align_corners=False,
                antialias=True,
            ).squeeze(0)

        return crop_chw

    def add_image_callback(self, callback: ImageCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)
