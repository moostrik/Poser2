from modules.cam.depthcam.Core import *
from modules.cam.depthcam.Definitions import FrameType
from modules.cam.depthcam.Pipeline import get_stereo_config, get_frame_types
from modules.cam.depthplayer.SyncPlayer import SyncPlayer
from modules.cam.depthplayer.Player import DecoderType

class CorePlayer(Core):

    def __init__(self, gui, syncplayer: SyncPlayer, device_id: str, model_path:str, fps: int = 30,
                 do_color: bool = True, do_stereo: bool = True, do_person: bool = True,
                 lowres: bool = False, show_stereo: bool = False) -> None:
        super().__init__(gui, device_id, model_path, fps, do_color, do_stereo, do_person, lowres, show_stereo)

        self.sync_player: SyncPlayer = syncplayer

    def start(self) -> None: # override
        self.sync_player.addFrameCallback(self._video_frame_callback)
        super().start()
        # Thread.start(self)

    def stop(self) -> None: # override
        self.sync_player.discardFrameCallback(self._video_frame_callback)
        super().stop()
        # pass

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None: # override
        setup_pipeline(pipeline, self.model_path, self.fps, self.do_color, self.do_stereo, self.do_person, self.lowres, self.show_stereo)

    def _frame_callback(self, message_group: dai.MessageGroup) -> None: # override
        return
        super()._frame_callback(message_group)

    def _tracker_callback(self, msg: dai.RawTracklets) -> None: # override
        return
        super()._tracker_callback(msg)

    def _video_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        if id == self.id:
            # send frames to camera

            self._update_callbacks(frame_type, frame)