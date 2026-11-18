import logging
from threading import Lock
from typing import Any

import torch
import torch.nn.functional as F

from modules.pose import BBox, FrameDict, replace
from .crop import Image, CropImageDict, ImageCallback, CropExtractorSettings
from .source import ImageDict, ImageSnapshot
from modules.utils import Rect, Point2f

logger = logging.getLogger(__name__)


class CropExtractor:
    """The crop step: every pose's box cut out of the images it is for, at each size it is given, in one pass.

    `submit` takes the poses with the `ImageSnapshot` they are for (an uploader's, carried by the frame source and the
    pipeline) and cuts each pose's box from exactly those images, at every size (`CropExtractorSettings`: the pose
    model's, and, where a segmentation runs, its own), from the one image and box; with `enable_prev_crop` the
    previous image is cut at the same box too, for the optical flow. Each size's consumers get the poses with their
    crops (`add_image_callback`): the pose leaves with its crop as its box and the image's capture time as its stamp,
    so its keypoints, box and time all belong to the one image. The crops are queued on the crop stream, not waited
    for: each carries the event recorded after them all, and their readers wait on that.

    All tensors are float16 RGB CHW [0, 1].
    """

    def __init__(self, settings: CropExtractorSettings, *more: CropExtractorSettings) -> None:
        self.settings: CropExtractorSettings = settings
        self._sizes: tuple[CropExtractorSettings, ...] = (settings, *more)
        self._callback_lock: Lock = Lock()
        self._callbacks: dict[int, list[ImageCallback]] = {id(size): [] for size in self._sizes}   # per size, in registration order
        self._stream: torch.cuda.Stream = torch.cuda.Stream()

    def add_image_callback(self, callback: ImageCallback, size: CropExtractorSettings | None = None) -> None:
        """The poses with their crops at `size` (one of the settings this was made with; the first by default), as
        soon as they are cut."""
        key: int = id(size if size is not None else self.settings)
        if key not in self._callbacks:
            raise ValueError("CropExtractor: no crops at that size; it cuts the sizes it was made with")
        with self._callback_lock:
            if callback not in self._callbacks[key]:
                self._callbacks[key].append(callback)

    def submit(self, frames: FrameDict, images: Any) -> None:
        """Cut the poses' boxes from `images` (an `ImageSnapshot`) at every size and hand each size's crops on, on the
        caller's thread and without waiting for the GPU. Poses without their images are an error: every frame source
        hands its images on."""
        if not isinstance(images, tuple) or len(images) != 4:
            raise ValueError("CropExtractor: poses came without the images they are for (an ImageSnapshot)")
        snapshot: ImageSnapshot = images
        current, previous, timestamps, ready = snapshot
        crops_ready: torch.cuda.Event = torch.cuda.Event()      # recorded after the last crop, before anyone reads one
        cut: list[tuple[FrameDict, CropImageDict]] = []
        with torch.cuda.stream(self._stream):
            if ready is not None:
                self._stream.wait_event(ready)
            for image in current.values():
                image.record_stream(self._stream)       # read here: not reused before this stream has run
            for image in previous.values():
                image.record_stream(self._stream)
            for size in self._sizes:
                cut.append(self._cut(size, frames, current, previous, timestamps, crops_ready))
            crops_ready.record(self._stream)

        for size, (cropped, crops) in zip(self._sizes, cut):
            with self._callback_lock:
                callbacks: list[ImageCallback] = list(self._callbacks[id(size)])
            for callback in callbacks:
                try:
                    callback(cropped, crops)
                except Exception:
                    logger.exception("Error in crop callback")

    def _cut(self, size: CropExtractorSettings, frames: FrameDict, current: ImageDict, previous: ImageDict,
             timestamps: dict[int, float], crops_ready: torch.cuda.Event) -> tuple[FrameDict, CropImageDict]:
        """One size's crops of every pose, queued on the crop stream."""
        cropped: FrameDict = {}
        crops: CropImageDict = {}
        zoom: float = 1.0 + size.padding     # read once, so every pose of this set is cut with the same padding
        for track_id, pose in frames.items():
            cam_id = pose.cam_id
            if cam_id not in current:
                continue
            if len(crops) >= size.max_players:
                logger.warning(f"Exceeded max poses ({size.max_players}), skipping {track_id}")
                continue
            try:
                image = current[cam_id]
                height, width = image.shape[1:3]
                roi = self._roi(size, pose[BBox].to_rect().zoom(zoom), width, height)
                crop = self._crop(size, image, roi)
                prev_crop: torch.Tensor | None = None
                if size.enable_prev_crop and cam_id in previous:
                    prev_crop = self._crop(size, previous[cam_id], roi)
                box = roi.scale(Point2f(1.0 / width, 1.0 / height))
                cropped[track_id] = replace(pose, {BBox: BBox.from_rect(box)}, timestamps.get(cam_id, pose.timestamp))
                crops[track_id] = Image(crop=crop, prev_crop=prev_crop, ready=crops_ready)
            except Exception:
                logger.exception(f"Error processing pose {track_id}")
        return cropped, crops

    @staticmethod
    def _roi(size: CropExtractorSettings, box: Rect, width: int, height: int) -> Rect:
        roi = box.affine_transform(Rect(0.0, 0.0, float(width), float(height)))
        return Rect(0, 0, size.output_width, size.output_height).aspect_fill(roi)

    @staticmethod
    def _crop(size: CropExtractorSettings, src: torch.Tensor, roi: Rect) -> torch.Tensor:
        src_h, src_w = src.shape[1:3]
        x1 = max(0, int(roi.x))
        y1 = max(0, int(roi.y))
        x2 = min(src_w, int(roi.x + roi.width))
        y2 = min(src_h, int(roi.y + roi.height))
        if x1 >= x2 or y1 >= y2:
            return torch.zeros((3, size.output_height, size.output_width), device='cuda', dtype=torch.float16)

        crop = src[:, y1:y2, x1:x2]
        pad_left = max(0, -int(roi.x))
        pad_right = max(0, int(roi.x + roi.width) - src_w)
        pad_top = max(0, -int(roi.y))
        pad_bottom = max(0, int(roi.y + roi.height) - src_h)
        if pad_left or pad_right or pad_top or pad_bottom:
            crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        if crop.shape[1] != size.output_height or crop.shape[2] != size.output_width:
            # Antialiased: a crop is usually larger than the output, and plain bilinear would sample
            # only a 2×2 neighbourhood of each output pixel and alias. No effect when enlarging.
            crop = F.interpolate(crop.unsqueeze(0), size=(size.output_height, size.output_width), mode='bilinear',
                                 align_corners=False, antialias=True).squeeze(0)
        return crop
