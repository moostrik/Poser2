"""GUI for configuring motion extraction thresholds."""

from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.extractors.AngleMotionExtractor import AngleMotionExtractorConfig


class AngleMotionExtractorGui:
    """GUI for tuning motion extraction noise floor and max threshold."""

    def __init__(self, config: AngleMotionExtractorConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: AngleMotionExtractorConfig = config

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Motion   '),
            E(eT.TEXT, 'n'),
            E(eT.SLDR, name + '_n', self.set_n_top_motions, config.n_top_motions, [1, 10], 1),
            E(eT.TEXT, 'noise'),
            E(eT.SLDR, name + '_noise', self.set_noise_threshold, config.noise_threshold, [0.0, 0.2], 0.01),
            E(eT.TEXT, 'max'),
            E(eT.SLDR, name + '_max', self.set_max_threshold, config.max_threshold, [0.1, 1.0], 0.05),
        ])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_n_top_motions(self, value: float) -> None:
        self.config.n_top_motions = int(value)

    def set_noise_threshold(self, value: float) -> None:
        self.config.noise_threshold = value

    def set_max_threshold(self, value: float) -> None:
        self.config.max_threshold = value
