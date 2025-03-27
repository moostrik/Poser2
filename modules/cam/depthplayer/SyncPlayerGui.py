from modules.cam.depthplayer.SyncPlayer import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame

class SyncPlayerGui(SyncPlayer):

    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__(input_path, num_cams, types, decoder)

        folders: list[str] = self.get_folders()
        f: str = folders[0] if len(folders) > 0 else ''

        elem: list = []
        elem.append([E(eT.TEXT, 'Folders '),
                     E(eT.CMBO, 'Folders', self.set_folder, f, folders)])

        self.frame = Frame('CAMERA COLOR', elem, 240)

    def get_gui_frame(self):
        return self.frame


    def set_folder(self, folder: str) -> None:
        self.play(False)
        self.play(True, folder)