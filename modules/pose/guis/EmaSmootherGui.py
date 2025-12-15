import math

from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT
from modules.pose.nodes.filters.EmaSmoothers import EmaSmootherConfig



class EmaSmootherGui:
    def __init__(self, config: EmaSmootherConfig, gui: Gui, name: str) -> None:
        self.gui: Gui = gui
        self.config: EmaSmootherConfig = config

        elm: list = []
        elm.append([
            E(eT.TEXT, 'Smooth   '),
            E(eT.TEXT, 'attack'),
            E(eT.SLDR, name + 'attack',     self.set_attack,    500,    [0, 1000],    100),
            E(eT.TEXT, 'release'),
            E(eT.SLDR, name + 'release',    self.set_release,   800,    [0, 1000],    100)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
          return self.frame

    def set_attack(self, value: float) -> None:
        self.config.attack = self._time_to_alpha(value)

    def set_release(self, value: float) -> None:
        self.config.release = self._time_to_alpha(value)


    def _time_to_alpha(self, time_ms: float) -> float:
        """Convert settling time to per-second alpha.

        Args:
            time_ms: Time (milliseconds) to reach 95% of target value.

        Frame-rate independent.
        """
        if time_ms <= 0:
            return 1.0  # Instant
        # 95% settling time ≈ 3 * tau
        tau_seconds = time_ms / 3000.0
        alpha = 1.0 - math.exp(-1.0 / tau_seconds)
        return alpha

    def _alpha_to_time(self, alpha: float) -> float:
        """Convert per-second alpha to settling time (95% of target)."""
        if alpha >= 1.0:
            return 0.0  # Instant
        if alpha <= 0.0:
            return float('inf')
        tau_seconds = -1.0 / math.log(1.0 - alpha)
        # 95% settling time ≈ 3 * tau
        return tau_seconds * 3000.0