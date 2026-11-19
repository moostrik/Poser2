import logging
import time
from collections import deque
from threading import Lock

import torch
import torch.nn.functional as F

from modules.pose import BBox, FrameDict, replace
from .crop import AskedCallback, Image, CropImageDict, ImageCallback, CropExtractorSettings
from .source import ImageDict, ImageSnapshot
from modules.utils import PerformanceTimer, Rect, Point2f

logger = logging.getLogger(__name__)


KEPT: int = 3      # snapshots kept: the frame source's waiting bangs (`TrackerFrameSource`'s WAITING) and the one it runs


class CropExtractor:
    """The crop step: every pose's box cut out of the images it is for, at each size it is given, in one pass.

    `submit_snapshot` takes each `ImageSnapshot` from the uploader and keeps the last `KEPT`, by their capture times.
    `submit` takes the poses with the capture times they are for (carried by the frame source and the pipeline),
    finds the snapshot of those times and cuts each pose's box from exactly its images, at every size
    (`CropExtractorSettings`: the pose model's, and, where a segmentation runs, its own), from the one image and box; with `enable_prev_crop` the
    previous image is cut at the same box too, for the optical flow. Each size's consumers get the poses with their
    crops (`add_image_callback`): the pose leaves with its crop as its box and the image's capture time as its stamp,
    so its keypoints, box and time all belong to the one image. The crops are queued on the crop stream, not waited
    for: each carries the event recorded after them all, and the models wait on that on their own streams. Readers
    outside the package get them finished (`add_finished_image_callback`): after the other callbacks the crop stream
    is waited for, only when a size that is cut has such a callback.

    A size is cut every `interval`-th submit and handed on only then. After the crops, each size's asked callbacks
    hear, per camera in the capture times, its capture time and the poses cut for its image: none on a submit the
    interval passes or whose snapshot is no longer kept, so a consumer that waits for the results knows none come.

    All tensors are float16 RGB CHW [0, 1].
    """

    def __init__(self, settings: CropExtractorSettings, *more: CropExtractorSettings) -> None:
        self.settings: CropExtractorSettings = settings
        self._sizes: tuple[CropExtractorSettings, ...] = (settings, *more)
        self._submits: int = 0                          # the submits taken, for each size's `interval`
        self._callback_lock: Lock = Lock()
        self._callbacks: dict[int, list[ImageCallback]] = {id(size): [] for size in self._sizes}   # per size, in registration order
        self._asked_callbacks: dict[int, list[AskedCallback]] = {id(size): [] for size in self._sizes}
        self._finished_callbacks: dict[int, list[ImageCallback]] = {id(size): [] for size in self._sizes}
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._snapshot_lock: Lock = Lock()              # the uploader's thread keeps them, the frame source's takes them
        self._snapshots: deque[ImageSnapshot] = deque(maxlen=KEPT)
        self._finish_timer: PerformanceTimer = PerformanceTimer(name="Wait crops finished")

    def add_image_callback(self, callback: ImageCallback, size: CropExtractorSettings | None = None) -> None:
        """The poses with their crops at `size` (one of the settings this was made with; the first by default), as
        soon as they are cut."""
        key: int = self._key(size)
        with self._callback_lock:
            if callback not in self._callbacks[key]:
                self._callbacks[key].append(callback)

    def add_finished_image_callback(self, callback: ImageCallback, size: CropExtractorSettings | None = None) -> None:
        """The poses with their crops at `size` (the first by default), written: after the other callbacks, once the
        crop stream has run."""
        key: int = self._key(size)
        with self._callback_lock:
            if callback not in self._finished_callbacks[key]:
                self._finished_callbacks[key].append(callback)

    def add_asked_callback(self, callback: AskedCallback, size: CropExtractorSettings | None = None) -> None:
        """Per camera in each submit, its capture time and the poses cut for its image at `size` (the first by
        default), none when the interval passed it; after the image callbacks."""
        key: int = self._key(size)
        with self._callback_lock:
            if callback not in self._asked_callbacks[key]:
                self._asked_callbacks[key].append(callback)

    def _key(self, size: CropExtractorSettings | None) -> int:
        key: int = id(size if size is not None else self.settings)
        if key not in self._callbacks:
            raise ValueError("CropExtractor: no crops at that size; it cuts the sizes it was made with")
        return key

    def submit_snapshot(self, snapshot: ImageSnapshot) -> None:
        """An uploader's snapshot (its snapshot callback), kept for the poses of its capture times; the oldest of
        `KEPT` goes."""
        with self._snapshot_lock:
            self._snapshots.append(snapshot)

    def _snapshot(self, cam_timestamps: dict[int, float]) -> ImageSnapshot | None:
        with self._snapshot_lock:
            for snapshot in reversed(self._snapshots):
                if snapshot[2] == cam_timestamps:
                    return snapshot
        return None

    def submit(self, frames: FrameDict, cam_timestamps: dict[int, float] | None) -> None:
        """Cut the poses' boxes from the snapshot of `cam_timestamps` at every size whose turn it is and hand each
        size's crops on, then tell each size's asked callbacks, on the caller's thread and without waiting for the GPU
        unless a finished callback is due. A snapshot no longer kept is logged, and its poses are cut from nothing."""
        if cam_timestamps is None:
            raise ValueError("CropExtractor: poses came without the capture times they are for")
        snapshot: ImageSnapshot | None = self._snapshot(cam_timestamps)
        due: list[CropExtractorSettings] = [size for size in self._sizes if self._submits % size.interval == 0]
        self._submits += 1
        timestamps: dict[int, float] = cam_timestamps
        if snapshot is None:
            logger.warning("CropExtractor: the images of %s are no longer kept; their poses are not cut", cam_timestamps)
            due = []
        else:
            current, previous, timestamps, ready = snapshot
        crops_ready: torch.cuda.Event = torch.cuda.Event()      # recorded after the last crop, before anyone reads one
        cut: dict[int, tuple[FrameDict, CropImageDict]] = {}
        if due and snapshot is not None:
            with torch.cuda.stream(self._stream):
                if ready is not None:
                    self._stream.wait_event(ready)
                for image in current.values():
                    image.record_stream(self._stream)       # read here: not reused before this stream has run
                for image in previous.values():
                    image.record_stream(self._stream)
                for size in due:
                    cut[id(size)] = self._cut(size, frames, current, previous, timestamps, crops_ready)
                crops_ready.record(self._stream)

        for size in due:
            cropped, crops = cut[id(size)]
            with self._callback_lock:
                callbacks: list[ImageCallback] = list(self._callbacks[id(size)])
            for callback in callbacks:
                try:
                    callback(cropped, crops)
                except Exception:
                    logger.exception("Error in crop callback")
        for size in self._sizes:
            cropped = cut[id(size)][0] if id(size) in cut else {}
            with self._callback_lock:
                asked_callbacks: list[AskedCallback] = list(self._asked_callbacks[id(size)])
            for cam_id, timestamp in timestamps.items():
                cam_asked: FrameDict = {track_id: pose for track_id, pose in cropped.items() if pose.cam_id == cam_id}
                for asked_callback in asked_callbacks:
                    try:
                        asked_callback(timestamp, cam_id, cam_asked)
                    except Exception:
                        logger.exception("Error in asked callback")

        with self._callback_lock:
            finished: list[tuple[int, list[ImageCallback]]] = [(id(size), list(self._finished_callbacks[id(size)]))
                                                               for size in due if self._finished_callbacks[id(size)]]
        if finished:
            start: float = time.perf_counter()
            self._stream.synchronize()
            self._finish_timer.add_time((time.perf_counter() - start) * 1000.0, report=False)
        for key, finished_callbacks in finished:
            cropped, crops = cut[key]
            for finished_callback in finished_callbacks:
                try:
                    finished_callback(cropped, crops)
                except Exception:
                    logger.exception("Error in finished crop callback")

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
                crops[track_id] = Image(crop=crop, prev_crop=prev_crop, ready=crops_ready, box=box)
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
