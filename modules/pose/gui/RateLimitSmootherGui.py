from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.filters.RateLimitSmoothers import RateLimitSmootherConfig

from numpy import pi

class RateLimitSmootherGui:
    def __init__(self, config: RateLimitSmootherConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: RateLimitSmootherConfig = config

        elm: list = []
        elm.append([E(eT.TEXT, 'CHASE    '),
            E(eT.TEXT, 'increase'),
            E(eT.SLDR, name + 'max_increase',   self.set_max_increase,  pi, [0, pi],    0.1),
            E(eT.TEXT, 'decrease'),
            E(eT.SLDR, name + 'max_decrease',   self.set_max_decrease,  pi, [0, pi],    0.1)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_max_increase(self, value: float) -> None:
        self.config.max_increase = value

    def set_max_decrease(self, value: float) -> None:
        self.config.max_decrease = value