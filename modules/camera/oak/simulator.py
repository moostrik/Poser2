"""An OAK camera fed from recordings: the `Player`'s clips go through the real device's pipeline, so
the detector and the tracker run as on the rig, and come out as the contract's frames and tracklets."""

import logging
from datetime import timedelta
from time import process_time

import depthai as dai
import numpy as np
from cv2 import resize, COLOR_BGR2GRAY, cvtColor

from .. import FrameType, Player, PlayerSettings, frame_size, mode_size

from .camera import Camera
from .pipeline import build_pipeline, CLIPS_STREAM
from .settings import OakCameraSettings

logger = logging.getLogger(__name__)


class Simulator(Camera):

    def __init__(self, player: Player, settings: OakCameraSettings, player_settings: PlayerSettings) -> None:
        super().__init__(settings)

        self.player: Player = player
        self.clips: dai.DataInputQueue | None = None

        self.passthrough: bool = player_settings.sim_passthrough
        self.warp_clips: bool = player_settings.apply_warp

        # Last clip size reported by _check_frame_size, so the warning fires once per size.
        self._warned_frame_size: tuple[int, int] | None = None

    def start(self) -> None:  # override
        if self.passthrough:
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
        self.clips = self.device.getInputQueue(name=CLIPS_STREAM, maxSize=1, blocking=False)  # type: ignore
        assert self.clips is not None
        self.queues.append(self.clips)
        self._setup_frame_queues()

    def _close(self) -> None:  # override
        self.clips = None
        super()._close()

    def _clip_callback(self, cam_id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        """A decoded clip frame, into the device as the sensor would have sent it."""
        if not self.running or cam_id != self.id or frame_type != FrameType.VIDEO or self.clips is None:
            return

        height, width = frame.shape[:2]
        self._check_frame_size(width, height)
        img = dai.ImgFrame()
        if self.color:
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setData(self.to_planar(frame, (width, height)))  # type: ignore
        else:
            img.setType(dai.ImgFrame.Type.RAW8)
            if frame.ndim == 3:
                frame = cvtColor(frame, COLOR_BGR2GRAY)  # clip frames arrive BGR (OpenCV convention); a gray-decoded clip skips this
            img.setData(frame.flatten())  # type: ignore
        img.setTimestamp(timedelta(seconds=process_time()))
        img.setWidth(width)
        img.setHeight(height)
        self.clips.send(img)

    def _check_frame_size(self, width: int, height: int) -> None:
        """Warn once per size when a clip's frames differ from what the pipeline was built for.

        Nothing breaks mechanically: XLinkIn's max data size is a ceiling, the ImgFrame carries
        the clip's own dimensions, and the detector resizes whatever arrives. But every
        frame-relative setting is tuned for the configured size and is silently off by the
        ratio — the tracker's vfov and height gates, and the pose distance thresholds.

        The fix is in the preset, not here: set `camera.resolution` to the label the clip was
        shot at and the whole chain derives correctly again.

        A clip that is being warped (`apply_warp`) feeds the warp's INPUT, so it must be the
        sensor's size; one that passes through must be the delivered size, its derived height and all.
        """
        expected: tuple[int, int] = (mode_size(self.color, self.resolution) if self.warp_clips
                                     else frame_size(self.color, self.resolution, self.square, self.frame_height))
        if (width, height) == expected or (width, height) == self._warned_frame_size:
            return
        self._warned_frame_size = (width, height)
        logger.warning(
            "cam %d: clip frames are %dx%d, pipeline is built for %dx%d — frame-relative "
            "settings (vfov, tracker height gates, pose distance) are tuned for the latter",
            self.id, width, height, expected[0], expected[1],
        )

    def _passthrough_callback(self, cam_id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        """Without a device: the clip frames go straight to the callbacks, no tracklets."""
        if cam_id != self.id:
            return
        self._emit_frame(frame_type, frame)

    @staticmethod
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return resize(arr, shape).transpose(2, 0, 1).flatten()
