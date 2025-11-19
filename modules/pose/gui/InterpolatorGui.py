from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.interpolators.ChaseInterpolators import ChaseInterpolatorConfig



class InterpolatorGui:
    def __init__(self, config: ChaseInterpolatorConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: ChaseInterpolatorConfig = config


        elm: list = []
        elm.append([E(eT.TEXT, 'CHASE    '),
            E(eT.TEXT, 'response'),
            E(eT.SLDR, name + 'responsiveness',     self.set_responsiveness,    0.2,   [0.01, 1.0],  0.01),
            E(eT.TEXT, 'friction'),
            E(eT.SLDR, name + 'friction',           self.set_friction,          0.03,  [0.01, .99],  0.01)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_input_frequency(self, value: float) -> None:
        self.input_frequency = value

    def set_responsiveness(self, value: float) -> None:
        self.config.responsiveness = value

    def set_friction(self, value: float) -> None:
        self.config.friction = value