from modules.cam.depthplayer.SyncPlayer import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame

class SyncPlayerGui(SyncPlayer):

    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__(input_path, num_cams, types, decoder)

        elem: list = []
        elem.append([E(eT.TEXT, 'Depth Min ')])

        self.frame = Frame('CAMERA COLOR', elem, 240)

    def get_gui_frame(self):
        return self.frame
