from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.save.ImgFolder import ImgFolder

class ImgFolderGui(ImgFolder):
    def __init__(self, gui: Gui | None, max_queue_size=100, output_path: str = 'images/', temp_path: str = 'temp/', min_files: int = 10) -> None:
        super().__init__(max_queue_size, output_path, temp_path, min_files)
        self.gui: Gui | None = gui

        elem: list = []
        elem.append([E(eT.CHCK, 'Active',   self.set_active,    True),
                     E(eT.TEXT, 'Min Files'),
                     E(eT.SLDR, 'MinFiles',     self.set_min_files, 12, (1, 20), 1)])

        self.frame = Frame('SAVE SEQUENCE', elem, 50)

    def get_gui_frame(self):
          return self.frame

    def set_min_files(self, value: int) -> None:
        self.min_files = value