from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.filters.Predictors import PredictorConfig, PredictionMethod



class PredictionGui:
    def __init__(self, config: PredictorConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: PredictorConfig = config

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Predict  '),
            E(eT.TEXT, 'method'),
            E(eT.SLDR, name + 'method', self.set_method,    2,    [0,2], 1)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_method(self, value: float) -> None:
        self.config.method = PredictionMethod(int(value))