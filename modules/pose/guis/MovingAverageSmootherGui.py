"""GUI for configuring Moving Average smoothing."""

from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.filters.MovingAverageSmoothers import MovingAverageConfig, WindowType


# Window type names for dropdown
WINDOW_TYPE_NAMES: list[str] = [wt.name for wt in WindowType]


class MovingAverageSmootherGui:
    """GUI for configuring moving average smoothing parameters."""

    def __init__(self, config: MovingAverageConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: MovingAverageConfig = config

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Window   '),
            E(eT.TEXT, 'size'),
            E(eT.SLDR, name + 'window_size', self.set_window_size, config.window_size, [5, 120], 1),
            E(eT.TEXT, 'type'),
            E(eT.CMBO, name + 'window_type', self.set_window_type, config.window_type.name, WINDOW_TYPE_NAMES),
        ])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_window_size(self, value: float) -> None:
        """Update window size in config."""
        self.config.window_size = int(value)

    def set_window_type(self, value: str) -> None:
        """Update window type in config."""
        try:
            self.config.window_type = WindowType[value]
        except KeyError:
            pass  # Invalid type name, ignore
