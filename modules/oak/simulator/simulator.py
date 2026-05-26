
from depthai import Pipeline
from ..camera.camera import *
from ..camera.pipeline import build_pipeline, get_model_path, PipelineConfig
from ..camera.settings import CameraSettings
from .settings import SimulatorSettings
from .player import Player
from cv2 import resize, COLOR_RGB2GRAY, cvtColor
from time import process_time
from datetime import timedelta

class Simulator(Camera):

    def __init__(self, syncplayer: Player, core_settings: CameraSettings, player_settings: SimulatorSettings) -> None:
        super().__init__(core_settings)

        self.sync_player: Player = syncplayer
        self.ex_video:  dai.DataInputQueue

        self.passthrough: bool = player_settings.sim_passthrough

    def start(self) -> None: # override
        if self.passthrough:
            self.sync_player.addFrameCallback(self._passthrough_frame_callback)
        else:
            self.sync_player.addFrameCallback(self._video_frame_callback)
        super().start()

    def run(self) -> None: # override
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
        self.inputs[Input.VIDEO_FRAME_IN] =    self.device.getInputQueue('ex_video')
        video_q = self.device.getOutputQueue('video')
        video_q.setMaxSize(1)
        video_q.setBlocking(False)
        self.outputs[Output.VIDEO_FRAME_OUT] = video_q
        self.outputs[Output.VIDEO_FRAME_OUT].addCallback(self._video_callback)
        self.fps_counters[FrameType.VIDEO] = FPS(120)
        if self.do_yolo:
            tracklets_q = self.device.getOutputQueue('tracklets')
            tracklets_q.setMaxSize(1)
            tracklets_q.setBlocking(False)
            self.outputs[Output.TRACKLETS_OUT] = tracklets_q
            self.outputs[Output.TRACKLETS_OUT].addCallback(self._tracker_callback)

    def _video_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        if not self.running:
            return

        frame_time = timedelta(seconds = process_time())
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