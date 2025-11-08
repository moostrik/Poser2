from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.filters.general.Smoothers import SmootherConfig



class SmootherGui:
    def __init__(self, config: SmootherConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: SmootherConfig = config

        elm: list = []
        elm.append([E(eT.TEXT, 'SMOOTH   '),
            E(eT.TEXT, 'minc'),
            E(eT.SLDR, name + 'min_cutoff', self.set_min_cutoff,    1.0,    [0.01,2.0], 0.01),
            E(eT.TEXT, 'beta'),
            E(eT.SLDR, name + 'beta',       self.set_beta,          0.05,   [0.0,0.5],  0.01),
            E(eT.TEXT, 'dc'),
            E(eT.SLDR, name + 'd_cutoff',   self.set_d_cutoff,      1.0,    [0.0,2.0],  0.1)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_min_cutoff(self, value: float) -> None:
        self.config.min_cutoff = value

    def set_beta(self, value: float) -> None:
        self.config.beta = value

    def set_d_cutoff(self, value: float) -> None:
        self.config.d_cutoff = value