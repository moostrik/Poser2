from modules.cam.VimbaCam import VimbaCam
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame


class VimbaCamGui(VimbaCam):
    def __init__(self, gui: Gui | None, forceSize: tuple[int, int] | None = None) -> None:
        self.gui: Gui | None = gui
        super().__init__(forceSize)

        elem: list = []
        elem.append([E(eT.TEXT, 'TRANSFORM'),
                     E(eT.CHCK, 'Flip H',           super().setFlipH,               False),
                     E(eT.CHCK, 'Flip V',           super().setFlipV,               False)])
        elem.append([E(eT.TEXT, 'GAIN     '),
                     E(eT.CHCK, 'Auto',             super().setGainAuto,            False),
                     E(eT.SLDR, 'Target',           super().setGainTarget,          40,         [0, 100],   1)])

        self.frame = Frame('CAMERA', elem, 80)

    def get_gui_frame(self):
          return self.frame