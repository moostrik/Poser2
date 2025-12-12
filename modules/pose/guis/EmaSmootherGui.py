from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.filters.EmaSmoothers import EmaSmootherConfig



class EmaSmootherGui:
    def __init__(self, config: EmaSmootherConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: EmaSmootherConfig = config

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Smooth   '),
            E(eT.TEXT, 'attack'),
            E(eT.SLDR, name + 'attack',     self.set_attack,    0.05,    [0.0, 1.0],    0.01),
            E(eT.TEXT, 'release'),
            E(eT.SLDR, name + 'release',    self.set_release,   0.025,   [0.0, 1.0],    0.01)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_attack(self, value: float) -> None:
        self.config.attack = value

    def set_release(self, value: float) -> None:
        self.config.release = value