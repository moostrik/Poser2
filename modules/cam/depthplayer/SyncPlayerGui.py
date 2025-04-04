from modules.cam.depthplayer.SyncPlayer import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame

class SyncPlayerGui(SyncPlayer):

    def __init__(self, gui: Gui | None, settings: Settings) -> None:
        self.gui: Gui | None = gui
        super().__init__(settings)

        folders: list[str] = self.get_folder_names()
        f: str = folders[0] if len(folders) > 0 else ''

        elem: list = []
        elem.append([E(eT.TEXT, 'Folders '),
                     E(eT.CMBO, 'Folders', self.set_folder, f, folders),
                     E(eT.BTTN, 'Stop', self.set_stop)])

        self.frame = Frame('CAMERA COLOR', elem, 240)

    def get_gui_frame(self):
        return self.frame

    def gui_check(self) -> None:
        if self.gui is not None:
            self.gui.updateElement('Folders', '')

    def set_folder(self, folder: str) -> None:
        self.play(True, folder)

    def set_stop(self) -> None:
        self.play(False, '')