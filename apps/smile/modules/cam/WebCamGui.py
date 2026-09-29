from modules.cam.WebCam import WebCam
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame


class WebCamGui(WebCam):
    def __init__(self, gui: Gui | None, forceSize: tuple[int, int] | None = None, doDepth: bool = True, doPerson: bool = True) -> None:
        self.gui: Gui | None = gui
        super().__init__(forceSize, doDepth, doPerson)

        elem: list = []
        elem.append([E(eT.TEXT, 'TRANSFORM'),
                     E(eT.CHCK, 'Flip H',           super().setFlipH,               False),
                     E(eT.CHCK, 'Flip V',           super().setFlipV,               False)])

        self.frame = Frame('CAMERA', elem, 80)

    def get_gui_frame(self):
          return self.frame