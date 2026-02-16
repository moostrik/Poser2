from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, SLIDERHEIGHT
from modules.pose.batch.WindowCorrelation import WindowCorrelationConfig
from modules.pose.features.base.NormalizedScalarFeature import AggregationMethod


class WindowCorrelationGui:
    def __init__(self, config: WindowCorrelationConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: WindowCorrelationConfig = config

        method_names = [m.name for m in AggregationMethod]

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Window'),
            E(eT.SLDR, name + 'window_length', self.set_window_length, config.window_length, [4, 120], 1),
            E(eT.TEXT, 'Max Lag'),
            E(eT.SLDR, name + 'max_lag', self.set_max_lag, config.max_lag, [1, 60], 1),
            E(eT.TEXT, 'Aggregation'),
            E(eT.CMBO, name + 'method', self.set_method, config.method.name, method_names),
        ])
        elm.append([
            E(eT.TEXT, 'Min Var'),
            E(eT.SLDR, name + 'min_variance', self.set_min_variance, config.min_variance, [0.001, 0.1], 0.001),
            E(eT.TEXT, 'Mot'),
            E(eT.CHCK, name + 'use_motion_weighting', self.set_use_motion_weighting, config.use_motion_weighting),
            E(eT.TEXT, 'Print'),
            E(eT.CHCK, name + 'verbose', self.set_verbose, config.verbose),
        ])
        elm.append([
            E(eT.TEXT, 'Enable'),
            E(eT.CHCK, name + 'enabled', self.set_enabled, config.enabled),
            E(eT.TEXT, 'Remap'),
            E(eT.SLDR, name + 'remap_low', self.set_remap_low, config.remap_low, [-1.0, 1.0], 0.01),
            E(eT.SLDR, name + 'remap_high', self.set_remap_high, config.remap_high, [-1.0, 1.0], 0.01),
        ])

        gui_height: int = SLIDERHEIGHT * 3 + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_enabled(self, value: bool) -> None:
        self.config.enabled = value

    def set_window_length(self, value: float) -> None:
        self.config.window_length = int(value)

    def set_max_lag(self, value: float) -> None:
        self.config.max_lag = int(value)

    def set_min_variance(self, value: float) -> None:
        self.config.min_variance = value

    def set_use_motion_weighting(self, value: bool) -> None:
        self.config.use_motion_weighting = value

    def set_method(self, value: str) -> None:
        self.config.method = AggregationMethod[value]

    def set_remap_low(self, value: float) -> None:
        self.config.remap_low = value

    def set_remap_high(self, value: float) -> None:
        self.config.remap_high = value

    def set_verbose(self, value: bool) -> None:
        self.config.verbose = value
