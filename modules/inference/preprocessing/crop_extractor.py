import logging
import time
from collections import deque
from threading import Lock

import torch
import torch.nn.functional as F

from modules.pose import BBox, FrameDict, replace
from modules.settings import BaseSettings, Field
from .crop import AskedCallback, Image, CropImageDict, ImageCallback
from .source import ImageDict, ImageSnapshot
from modules.utils import Broadcast, PerformanceTimer, Rect, Point2f

logger = logging.getLogger(__name__)


KEPT: int = 3      # snapshots kept


class CropExtractorSettings(BaseSettings):
    """One size the crop step cuts: the padding around each box, the output's size, the players and the interval."""
    padding:          Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05,
                                           description="Margin added around the box before cropping (fraction; RTMPose uses 0.25)")
    output_width:     Field[int]   = Field(384, access=Field.INIT)
    output_height:    Field[int]   = Field(512, access=Field.INIT)
    max_players:        Field[int]   = Field(4, min=1, max=16, access=Field.INIT)
    interval:         Field[int]   = Field(1, min=1, max=8, description="Frames from one cut to the next: 1 every frame, 2 every other")


class CropExtractor:
    """The crop step: every pose's box cut out of the images it is for, at each size it is given, in one pass.

    `set_snapshot` takes each `ImageSnapshot` from the uploader and keeps the last `KEPT`, by their capture times.
    `process` takes the poses with the capture times they are for (carried by the frame source and the pipeline),
    finds the snapshot of those times and cuts each pose's box from exactly its images, at every size
    (`CropExtractorSettings`: the pose model's, and, where a segmentation runs, its own), from the one image and box.
    Each size's consumers get the poses with their crops (`add_image_callback`): the pose leaves with its crop as its box and the image's capture time as its stamp,
    so its keypoints, box and time all belong to the one image. The crops are queued on the crop stream, not waited
    for: each carries the event recorded after them all, and the models wait on that on their own streams. Readers
    outside the package get them finished (`add_finished_image_callback`): after the other callbacks the crop stream
    is waited for, only when a size that is cut has such a callback.

    A size is cut every `interval`-th process and handed on only then. After the crops, each size's asked callbacks
    hear, per camera in the capture times, its capture time and the poses cut for its image: none on a process the
    interval passes or whose snapshot is no longer kept, so a consumer that waits for the results knows none come.

    All tensors are float16 RGB CHW [0, 1].
    """

    def __init__(self, settings: CropExtractorSettings, *more: CropExtractorSettings) -> None:
        self.settings: CropExtractorSettings = settings
        self._sizes: tuple[CropExtractorSettings, ...] = (settings, *more)
        self._processed: int = 0                        # the processes taken, for each size's `interval`
        # each size's callbacks, by the size's id
        self._callbacks: dict[int, Broadcast[[FrameDict, CropImageDict]]] = {
            id(size): Broadcast(name="crop") for size in self._sizes}
        self._asked_callbacks: dict[int, Broadcast[[float, int, FrameDict]]] = {
            id(size): Broadcast(name="crop asked") for size in self._sizes}
        self._finished_callbacks: dict[int, Broadcast[[FrameDict, CropImageDict]]] = {
            id(size): Broadcast(name="finished crop") for size in self._sizes}
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._snapshot_lock: Lock = Lock()              # the uploader's thread keeps them, the frame source's takes them
        self._snapshots: deque[ImageSnapshot] = deque(maxlen=KEPT)
        self._finish_timer: PerformanceTimer = PerformanceTimer(name="Wait crops finished")

    def add_image_callback(self, callback: ImageCallback, size: CropExtractorSettings | None = None) -> None:
        """The poses with their crops at `size` (one of the settings this was made with; the first by default), as
        soon as they are cut."""
        key: int = self._key(size)
        self._callbacks[key].add_callback(callback)

    def add_finished_image_callback(self, callback: ImageCallback, size: CropExtractorSettings | None = None) -> None:
        """The poses with their crops at `size` (the first by default), written: after the other callbacks, once the
        crop stream has run."""
        key: int = self._key(size)
        self._finished_callbacks[key].add_callback(callback)

    def add_asked_callback(self, callback: AskedCallback, size: CropExtractorSettings | None = None) -> None:
        """Per camera in each process, its capture time and the poses cut for its image at `size` (the first by
        default), none when the interval passed it; after the image callbacks."""
        key: int = self._key(size)
        self._asked_callbacks[key].add_callback(callback)

    def _key(self, size: CropExtractorSettings | None) -> int:
        key: int = id(size if size is not None else self.settings)
        if key not in self._callbacks:
            raise ValueError("CropExtractor: no crops at that size; it cuts the sizes it was made with")
        return key

    def set_snapshot(self, snapshot: ImageSnapshot) -> None:
        """An uploader's snapshot (its snapshot callback), kept for the poses of its capture times; the oldest of
        `KEPT` goes."""
        with self._snapshot_lock:
            self._snapshots.append(snapshot)

    def _snapshot(self, cam_timestamps: dict[int, float]) -> ImageSnapshot | None:
        with self._snapshot_lock:
            for snapshot in reversed(self._snapshots):
                if snapshot[1] == cam_timestamps:
                    return snapshot
        return None

    def process(self, frames: FrameDict, cam_timestamps: dict[int, float] | None) -> None:
        """Cut the poses' boxes from the snapshot of `cam_timestamps` at every size whose turn it is and hand each
        size's crops on, then tell each size's asked callbacks, on the caller's thread and without waiting for the GPU
        unless a finished callback is due. A snapshot no longer kept is logged, and its poses are cut from nothing."""
        if cam_timestamps is None:
            raise ValueError("CropExtractor: poses came without the capture times they are for")
        snapshot: ImageSnapshot | None = self._snapshot(cam_timestamps)
        due: list[CropExtractorSettings] = [size for size in self._sizes if self._processed % size.interval == 0]
        self._processed += 1
        timestamps: dict[int, float] = cam_timestamps
        if snapshot is None:
            logger.warning("CropExtractor: the images of %s are no longer kept; their poses are not cut", cam_timestamps)
            due = []
        else:
            current, timestamps, ready = snapshot
        crops_ready: torch.cuda.Event = torch.cuda.Event()      # recorded after the last crop, before anyone reads one
        cut: dict[int, tuple[FrameDict, CropImageDict]] = {}
        if due and snapshot is not None:
            with torch.cuda.stream(self._stream):
                if ready is not None:
                    self._stream.wait_event(ready)
                for image in current.values():
                    image.record_stream(self._stream)       # read here: not reused before this stream has run
                for size in due:
                    cut[id(size)] = self._cut(size, frames, current, timestamps, crops_ready)
                crops_ready.record(self._stream)

        for size in due:
            cropped, crops = cut[id(size)]
            self._callbacks[id(size)](cropped, crops)
        for size in self._sizes:
            cropped = cut[id(size)][0] if id(size) in cut else {}
            for cam_id, timestamp in timestamps.items():
                cam_asked: FrameDict = {track_id: pose for track_id, pose in cropped.items() if pose.cam_id == cam_id}
                self._asked_callbacks[id(size)](timestamp, cam_id, cam_asked)

        finished: list[int] = [id(size) for size in due if self._finished_callbacks[id(size)]]
        if finished:
            start: float = time.perf_counter()
            self._stream.synchronize()
            self._finish_timer.add_time((time.perf_counter() - start) * 1000.0, report=False)
        for key in finished:
            cropped, crops = cut[key]
            self._finished_callbacks[key](cropped, crops)

    def _cut(self, size: CropExtractorSettings, frames: FrameDict, current: ImageDict,
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
                box = roi.scale(Point2f(1.0 / width, 1.0 / height))
                cropped[track_id] = replace(pose, {BBox: BBox.from_rect(box)}, timestamps.get(cam_id, pose.timestamp))
                crops[track_id] = Image(crop=crop, ready=crops_ready, box=box)
            except Exception:
                logger.exception(f"Error processing pose {track_id}")
        return cropped, crops

    @staticmethod
    def _roi(size: CropExtractorSettings, box: Rect, width: int, height: int) -> Rect:
        roi = box.denormalized(Rect(0.0, 0.0, float(width), float(height)))
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
            crop = F.interpolate(crop.unsqueeze(0), size=(size.output_height, size.output_width), mode='bilinear',
                                 align_corners=False, antialias=True).squeeze(0)
        return crop
