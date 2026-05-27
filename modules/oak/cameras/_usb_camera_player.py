import depthai as dai
import numpy as np
from threading import Barrier
from ._usb_camera import UsbCamera
from ._definitions import FrameType, Input, Output, Tracklet, DeviceInfo
from ._pipeline import build_pipeline, get_model_path, PipelineConfig
from .settings import CameraSettings
from modules.utils import FPS
from ..player.settings import SimulatorSettings
from ..player import Player
from cv2 import resize, COLOR_RGB2GRAY, cvtColor
from time import process_time
from datetime import timedelta


class UsbCameraPlayer(UsbCamera):

    def __init__(self, syncplayer: Player, core_settings: CameraSettings, player_settings: SimulatorSettings, barrier: Barrier | None = None, device_info: DeviceInfo | None = None) -> None:
        super().__init__(core_settings, barrier, device_info)

        self.sync_player: Player = syncplayer
        self.passthrough: bool = player_settings.sim_passthrough

    def start(self) -> None:  # override
        if self.passthrough:
            self.sync_player.addFrameCallback(self._passthrough_frame_callback)
        else:
            self.sync_player.addFrameCallback(self._video_frame_callback)
        super().start()

    def run(self) -> None:  # override
        if self.passthrough:
            self.stop_event.wait()
        else:
            super().run()

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:  # override
        config = PipelineConfig(
            fps=self.fps,
            square=self.square,
            do_color=self.do_color,
            do_yolo=self.do_yolo,
            do_720p=self.do_720p,
            perspective=self.perspective,
            simulate=True,
            nn_path=get_model_path(self.model_path, self.square, True) if self.do_yolo else None,
        )
        self._pipeline_handles = build_pipeline(pipeline, config)

    def _setup_queues(self) -> None:  # override
        assert self._pipeline_handles is not None
        handles = self._pipeline_handles
        if handles.video_frame_in is not None:
            self.inputs[Input.VIDEO_FRAME_IN] = handles.video_frame_in.createInputQueue()
        video_q = handles.video_out.createOutputQueue(maxSize=1, blocking=False)
        self.outputs[Output.VIDEO_FRAME_OUT] = video_q
        video_q.addCallback(self._video_callback)
        self.fps_counters[FrameType.VIDEO] = FPS(120)
        if self.do_yolo and handles.tracklets_out is not None:
            tracklets_q = handles.tracklets_out.createOutputQueue(maxSize=1, blocking=False)
            self.outputs[Output.TRACKLETS_OUT] = tracklets_q
            tracklets_q.addCallback(self._tracker_callback)

    def _video_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        if not self.running:
            return

        frame_time = timedelta(seconds=process_time())
        height, width = frame.shape[:2]
        if id == self.id:
            if frame_type == FrameType.VIDEO and Input.VIDEO_FRAME_IN in self.inputs:
                img = dai.ImgFrame()
                # img.setInstanceNum(int(dai.CameraBoardSocket.CAM_A))
                if self.do_color:
                    img.setType(dai.ImgFrame.Type.BGR888p)
                    img.setData(self.to_planar(frame, (width, height)).astype(np.uint8))  # type: ignore[arg-type]
                else:
                    img.setType(dai.ImgFrame.Type.RAW8)
                    if frame.shape[2] == 3:
                        frame = cvtColor(frame, COLOR_RGB2GRAY)
                    img.setData(frame.flatten().astype(np.uint8))  # type: ignore[arg-type]
                img.setTimestamp(frame_time)
                img.setWidth(width)
                img.setHeight(height)
                self.inputs[Input.VIDEO_FRAME_IN].send(img)

    def _passthrough_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        if id != self.id:
            return
        self._update_frame_callbacks(frame_type, frame)

    @staticmethod
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return resize(arr, shape).transpose(2, 0, 1).flatten()
