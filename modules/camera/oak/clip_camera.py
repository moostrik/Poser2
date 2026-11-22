"""An OAK camera fed recorded clips: they go through the device's pipeline, so its detector and tracker run as on
the rig."""

import logging
from datetime import timedelta
from time import perf_counter

import depthai as dai
import numpy as np
from cv2 import COLOR_BGR2GRAY, cvtColor

from .. import ClipFrame, ClipSource, FrameType
from ..geometry import frame_size, mode_size

from .camera import Camera
from .pipeline import build_pipeline, CLIPS_STREAM
from .settings import OakCameraSettings

logger = logging.getLogger(__name__)


class ClipCamera(Camera):
    """An OAK-D fed recorded clips instead of its sensor, through the device's own pipeline; with `clip_passthrough`
    the clips go straight to the callbacks, no device."""

    def __init__(self, player: ClipSource, settings: OakCameraSettings, cam_id: int = 0) -> None:
        super().__init__(settings, cam_id)

        self.player: ClipSource = player
        self.clips: dai.DataInputQueue | None = None

        self.passthrough: bool = settings.clip_passthrough
        self.warp_clips: bool = settings.clip_warp

        # Last clip size reported by _check_frame_size, so the warning fires once per size.
        self._warned_frame_size: tuple[int, int] | None = None

    def start(self) -> None:  # override
        if self.passthrough:
            self.running = True         # no device to open: the camera runs from its start
            self.player.add_frame_callback(self._passthrough_callback)
        else:
            self.player.add_frame_callback(self._clip_callback)
        super().start()

    def run(self) -> None:  # override
        if self.passthrough:
            self.stop_event.wait()
        else:
            super().run()

    @property
    def connected(self) -> bool:  # override
        return self.running if self.passthrough else self.device is not None

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:  # override
        build_pipeline(pipeline, self.model_path, self.fps, self.color, self.square, self.yolo, self.resolution,
                       self.mount, frame_height=self.frame_height, clips=True, warp_clips=self.warp_clips)

    def _setup_queues(self) -> None:  # override
        """The clip input and the frame queues; no IMU (it would report a box on a bench) and no
        sensor controls (there is no sensor)."""
        assert self.device is not None
        self.clips = self.device.getInputQueue(name=CLIPS_STREAM, maxSize=1, blocking=False)  # pyright: ignore[reportCallIssue]
        assert self.clips is not None
        self.queues.append(self.clips)
        self._setup_frame_queues()

    def _close(self) -> None:  # override
        self.clips = None
        super()._close()

    def _clip_callback(self, clip: ClipFrame) -> None:
        """A decoded clip frame, into the device as the sensor would have sent it."""
        if not self.running or clip.cam_id != self.cam_id or clip.kind != FrameType.VIDEO or self.clips is None:
            return

        frame: np.ndarray = clip.image
        height, width = frame.shape[:2]
        self._check_frame_size(width, height)
        img = dai.ImgFrame()
        if self.color:
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setData(self.to_planar(frame))  # pyright: ignore[reportCallIssue, reportArgumentType]
        else:
            img.setType(dai.ImgFrame.Type.RAW8)
            if frame.ndim == 3:
                frame = cvtColor(frame, COLOR_BGR2GRAY)  # clip frames arrive BGR (OpenCV convention); a gray-decoded clip skips this
            img.setData(frame.flatten())  # pyright: ignore[reportCallIssue, reportArgumentType]
        img.setTimestamp(timedelta(seconds=perf_counter()))
        img.setWidth(width)
        img.setHeight(height)
        self.clips.send(img)

    def _check_frame_size(self, width: int, height: int) -> None:
        """Warn once per size when a clip's frames are not the size the pipeline was built for: the sensor's with
        `clip_warp`, else the delivered one."""
        expected: tuple[int, int] = (mode_size(self.color, self.resolution) if self.warp_clips
                                     else frame_size(self.color, self.resolution, self.square, self.frame_height))
        if (width, height) == expected or (width, height) == self._warned_frame_size:
            return
        self._warned_frame_size = (width, height)
        logger.warning(
            "cam %d: clip frames are %dx%d, pipeline is built for %dx%d — frame-relative "
            "settings (vfov, tracker height gates, pose distance) are tuned for the latter",
            self.cam_id, width, height, expected[0], expected[1],
        )

    def _passthrough_callback(self, clip: ClipFrame) -> None:
        """Without a device: the clip frames go straight to the callbacks, no tracklets, stamped with their recorded
        capture on the play's clock, or with their arrival for a recording without one."""
        if clip.cam_id != self.cam_id:
            return
        arrival: float = perf_counter()
        self._emit_frame(clip.kind, clip.image, clip.timestamp if clip.timestamp is not None else arrival, arrival)

    @staticmethod
    def to_planar(image: np.ndarray) -> np.ndarray:
        """An interleaved BGR image (H, W, 3) as planar BGR888p bytes: the B rows, then G, then R."""
        return image.transpose(2, 0, 1).flatten()
