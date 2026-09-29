from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.audio.DB import DB

class DBGui(DB):
    def __init__(self, gui: Gui ,file_path: str) -> None:
        super().__init__(file_path)
        self.gui: Gui = gui

        elem: list = []

        elem.append([E(eT.TEXT, 'ID'),
                     E(eT.ITXT, 'A_ID',         self.emptyFunction,   'none', expand=False)])
        self.frame = Frame('AUDIO', elem, 50)

    def get_gui_frame(self):
        return self.frame

    def update_gui(self) -> None:
        self.gui.updateElement('A_ID', self.get_audio_id())

    def update(self, movement_id: str) -> None: #override
        super().update(movement_id)
        self.update_gui()

    def emptyFunction(self, value) -> None:
        pass
