"""GUI for configuring easing function selection."""

from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.filters.EasingNode import EasingConfig, EASING_FUNCTIONS


class EasingGui:
    """GUI for selecting easing function."""

    def __init__(self, config: EasingConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: EasingConfig = config

        # Get list of available easing function names
        easing_names: list[str] = list(EASING_FUNCTIONS.keys())

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Easing   '),
            E(eT.CMBO, name + 'easing', self.set_easing, config.easing_name, easing_names),
        ])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_easing(self, value: str) -> None:
        """Update the easing function name in config."""
        if value in EASING_FUNCTIONS:
            self.config.easing_name = value
