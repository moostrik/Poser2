from modules.cam.depthcam.Core import Core
from modules.cam.depthcam.Definitions import FrameType
from modules.cam.depthcam.Pipeline import get_stereo_config, get_frame_types

class CorePlayer(Core):
    def __init__(self, model_path:str, fps: int = 30, do_color: bool = True, do_stereo: bool = True, do_person: bool = True, lowres: bool = False, show_stereo: bool = False) -> None:
        super().__init__(model_path, fps, do_color, do_stereo, do_person, lowres, show_stereo)