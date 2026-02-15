from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT, SLIDERHEIGHT
from modules.pose.batch.WindowSimilarity import WindowSimilarityConfig
from modules.pose.features.base.NormalizedScalarFeature import AggregationMethod


class WindowSimilarityGui:
    def __init__(self, config: WindowSimilarityConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: WindowSimilarityConfig = config

        # Get enum values for method dropdown
        method_names = [m.name for m in AggregationMethod]

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Window'),
            E(eT.SLDR, name + 'window_length', self.set_window_length, config.window_length, [1, 300], 1),
            E(eT.TEXT, 'Exponent'),
            E(eT.SLDR, name + 'exponent', self.set_exponent, config.exponent, [0.5, 4.0], 0.1),
            E(eT.TEXT, 'Method'),
            E(eT.CMBO, name + 'method', self.set_method, config.method.name, method_names)])
        elm.append([
            E(eT.TEXT, 'Time Decay'),
            E(eT.SLDR, name + 'time_decay_exp', self.set_time_decay_exp, config.time_decay_exp, [0.1, 4.0], 0.1),
            E(eT.CHCK, name + 'use_time_penalty', self.set_use_time_penalty, config.use_time_penalty),
            E(eT.CHCK, name + 'use_motion_weighting', self.set_use_motion_weighting, config.use_motion_weighting),
        ])

        # Calculate height: 2 slider rows + checkboxes
        gui_height: int = SLIDERHEIGHT * 2 + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_window_length(self, value: float) -> None:
        self.config.window_length = int(value)

    def set_exponent(self, value: float) -> None:
        self.config.exponent = value

    def set_method(self, value: str) -> None:
        self.config.method = AggregationMethod[value]

    def set_time_decay_exp(self, value: float) -> None:
        self.config.time_decay_exp = value

    def set_use_time_penalty(self, value: bool) -> None:
        self.config.use_time_penalty = value

    def set_use_motion_weighting(self, value: bool) -> None:
        self.config.use_motion_weighting = value
