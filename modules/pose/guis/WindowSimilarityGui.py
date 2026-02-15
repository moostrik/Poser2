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
            E(eT.SLDR, name + 'window_length', self.set_window_length, config.window_length, [1, 60], 1),
            E(eT.TEXT, 'Time Decay'),
            E(eT.CHCK, name + 'use_time_penalty', self.set_use_time_penalty, config.use_time_penalty),
            E(eT.SLDR, name + 'time_decay_exp', self.set_time_decay_exp, config.time_decay_exp, [0.1, 4.0], 0.1),
            E(eT.TEXT, 'Aggregation'),
            E(eT.CMBO, name + 'method', self.set_method, config.method.name, method_names)])
        elm.append([
            E(eT.TEXT, 'Ang'),
            E(eT.CHCK, name + 'use_angle_similarity', self.set_use_angle_similarity, config.use_angle_similarity),
            E(eT.SLDR, name + 'angle_scale', self.set_angle_scale, config.angle_scale, [0.1, 2.0], 0.05),
            E(eT.TEXT, 'Vel'),
            E(eT.CHCK, name + 'use_velocity_similarity', self.set_use_velocity_similarity, config.use_velocity_similarity),
            E(eT.SLDR, name + 'vel_scale', self.set_vel_scale, config.vel_scale, [0.1, 2.0], 0.05),
            E(eT.TEXT, 'Mot'),
            E(eT.CHCK, name + 'use_motion_weighting', self.set_use_motion_weighting, config.use_motion_weighting),
            E(eT.TEXT, 'Print'),
            E(eT.CHCK, name + 'verbose', self.set_verbose, config.verbose),
        ])
        elm.append([
            E(eT.TEXT, 'Remap'),
            E(eT.TEXT, 'Low'),
            E(eT.SLDR, name + 'remap_low', self.set_remap_low, config.remap_low, [0.0, 1.0], 0.01),
            E(eT.TEXT, 'High'),
            E(eT.SLDR, name + 'remap_high', self.set_remap_high, config.remap_high, [0.0, 1.0], 0.01),
        ])

        # Calculate height: 3 rows
        gui_height: int = SLIDERHEIGHT * 3 + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_window_length(self, value: float) -> None:
        self.config.window_length = int(value)

    def set_angle_scale(self, value: float) -> None:
        self.config.angle_scale = value

    def set_use_angle_similarity(self, value: bool) -> None:
        self.config.use_angle_similarity = value

    def set_vel_scale(self, value: float) -> None:
        self.config.vel_scale = value

    def set_use_velocity_similarity(self, value: bool) -> None:
        self.config.use_velocity_similarity = value

    def set_method(self, value: str) -> None:
        self.config.method = AggregationMethod[value]

    def set_time_decay_exp(self, value: float) -> None:
        self.config.time_decay_exp = value

    def set_use_time_penalty(self, value: bool) -> None:
        self.config.use_time_penalty = value

    def set_use_motion_weighting(self, value: bool) -> None:
        self.config.use_motion_weighting = value

    def set_verbose(self, value: bool) -> None:
        self.config.verbose = value

    def set_remap_low(self, value: float) -> None:
        self.config.remap_low = value

    def set_remap_high(self, value: float) -> None:
        self.config.remap_high = value
