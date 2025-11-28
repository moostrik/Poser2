from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from ..nodes import SimilarityExtractorConfig, AggregationMethod



class SimilarityExtractorGui:
    def __init__(self, config: SimilarityExtractorConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: SimilarityExtractorConfig = config

        aggregation_method_options: list[str] = [method.name for method in AggregationMethod]

        elm: list = []
        elm.append([E(eT.TEXT, 'Aggregate'),
            E(eT.TEXT, 'method'),
            E(eT.CMBO, name + 'method', self.set_method,    1.0,    aggregation_method_options, 0.1),
            E(eT.TEXT, 'exponent'),
            E(eT.SLDR, name + 'exponent',       self.set_exponent,          2.0,   [0.5,3.5],  0.1)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_method(self, value: str) -> None:
        self.config.method = AggregationMethod[value]

    def set_exponent(self, value: float) -> None:
        self.config.exponent = value