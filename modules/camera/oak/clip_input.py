"""An OAK-D's clip input: a recorded clip frame sent into the device where the sensor's would enter, so its warp,
detector and tracker run on it as on the rig (CAMERA.md, the clip camera)."""

import logging
from datetime import timedelta
from time import perf_counter
from typing import Protocol

import depthai as dai
import numpy as np
from cv2 import COLOR_BGR2GRAY, cvtColor

from .. import ClipFrame

logger = logging.getLogger(__name__)


class ClipQueue(Protocol):
    """The host's end of the device's clip input."""
    def send(self, msg: dai.ImgFrame) -> None: ...


class ClipInput:
    """One open device's clip queue: each clip frame goes in as the sensor would send it, planar BGR on a color
    camera, grey on a mono one, stamped with its recorded capture. The send is the hand-over to the device."""

    def __init__(self, queue: ClipQueue, color: bool, expected_size: tuple[int, int], cam_id: int) -> None:
        self.queue: ClipQueue = queue
        self.color: bool = color
        self.expected_size: tuple[int, int] = expected_size     # the sensor's with `clip_warp`, else the delivered one
        self.cam_id: int = cam_id
        self._warned_size: tuple[int, int] | None = None        # the last size warned of, so it warns once per size

    def send(self, clip: ClipFrame) -> None:
        """The clip frame into the device, stamped with its recorded capture on the play's clock, or with its send
        for a recording without one."""
        frame: np.ndarray = clip.image
        height, width = frame.shape[:2]
        self._check_size(width, height)
        img = dai.ImgFrame()
        if self.color:
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setData(to_planar(frame))  # pyright: ignore[reportCallIssue, reportArgumentType]
        else:
            img.setType(dai.ImgFrame.Type.RAW8)
            if frame.ndim == 3:
                frame = cvtColor(frame, COLOR_BGR2GRAY)  # clip frames arrive BGR (OpenCV convention); a gray-decoded clip skips this
            img.setData(frame.flatten())  # pyright: ignore[reportCallIssue, reportArgumentType]
        timestamp: float = clip.timestamp if clip.timestamp is not None else perf_counter()
        img.setTimestamp(timedelta(seconds=timestamp))
        img.setWidth(width)
        img.setHeight(height)
        self.queue.send(img)

    def _check_size(self, width: int, height: int) -> None:
        """Warn once per size when the clip's frames are not the size the pipeline was built for."""
        if (width, height) == self.expected_size or (width, height) == self._warned_size:
            return
        self._warned_size = (width, height)
        logger.warning(
            "cam %d: clip frames are %dx%d, pipeline is built for %dx%d — frame-relative "
            "settings (vfov, tracker height gates, pose distance) are tuned for the latter",
            self.cam_id, width, height, self.expected_size[0], self.expected_size[1],
        )


def to_planar(image: np.ndarray) -> np.ndarray:
    """An interleaved BGR image (H, W, 3) as planar BGR888p bytes: the B rows, then G, then R."""
    return image.transpose(2, 0, 1).flatten()
