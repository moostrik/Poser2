# Standard library imports
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase
from modules.pose.Pose import Pose

# Local application imports
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT


@dataclass
class SmootherSettings:
    """Configuration for OneEuroFilter-based smoothing."""
    frequency: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.025
    d_cutoff: float = 1.0
    reset_on_reappear: bool = False


class GuiSettings:
    def __init__(self, settings: SmootherSettings, gui: Gui, name: str, on_change: Callable) -> None:
        self.gui: Gui = gui
        self.settings: SmootherSettings = settings
        self.on_change: Callable = on_change

        elm: list = []
        elm.append([E(eT.TEXT, 'TIME     '),
            E(eT.TEXT, 'minc'),
            E(eT.SLDR, name + 'min_cutoff', self.set_min_cutoff,    1.0,    [0.0,2.0],  0.1),
            E(eT.TEXT, 'beta'),
            E(eT.SLDR, name + 'beta',       self.set_beta,          0.05,   [0.0,0.5],  0.01),
            E(eT.TEXT, 'dc'),
            E(eT.SLDR, name + 'd_cutoff',   self.set_d_cutoff,      1.0,    [0.0,2.0],  0.1)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame(name, elm, gui_height)

    def get_gui_frame(self):
        return self.frame

    def set_min_cutoff(self, value: float) -> None:
        self.settings.min_cutoff = value
        self.on_change()

    def set_beta(self, value: float) -> None:
        self.settings.beta = value
        self.on_change()

    def set_d_cutoff(self, value: float) -> None:
        self.settings.d_cutoff = value
        self.on_change()


class PoseSmootherBase(PoseFilterBase):
    """
    Base class for single-pose smoothing filters using OneEuroFilter.

    Handles:
    - Filter state management for a single pose
    - Settings and GUI integration

    Subclasses implement:
    - Filter initialization for a new pose
    - Smoothing logic for specific data types (points, angles, bbox)
    - Settings update propagation to filters
    """

    def __init__(self, frequency: float, name: str, gui: Gui | None = None) -> None:
        super().__init__()
        self.settings = SmootherSettings(
            frequency=frequency,
            min_cutoff=2.0,
            beta=0.05,
            d_cutoff=1.0,
            reset_on_reappear=False
        )

        self.name: str = name
        self.gui_settings: GuiSettings | None = None
        if gui is not None:
            self.gui_settings = GuiSettings(self.settings, gui, name, self._update_settings)

        # State for the current pose (managed by subclasses)
        self._state: Any = None

    def process(self, pose: Pose) -> Pose:
        """Smooth data for a single pose."""
        # Initialize filter state if needed
        if self._state is None:
            self._state = self._create_state()

        # Smooth the pose data
        smoothed_pose: Pose = self._smooth(pose, self._state)

        # Cleanup state if pose is lost
        if pose.lost:
            self._state = None

        return smoothed_pose

    @abstractmethod
    def _create_state(self) -> Any:
        """Create initial filter state for a new pose."""
        pass

    @abstractmethod
    def _smooth(self, pose: Pose, state: Any) -> Pose:
        """Apply smoothing to a single pose."""
        pass

    @abstractmethod
    def _update_filters(self, state: Any) -> None:
        """Update filter parameters for the pose's filters."""
        pass

    def _update_settings(self) -> None:
        """Update filter parameters for the current pose."""
        if self._state is not None:
            self._update_filters(self._state)

    def get_gui_frame(self):
        if self.gui_settings:
            return self.gui_settings.get_gui_frame()
        return Frame(self.name, [], 50)  # Empty frame if no GUI settings

    def reset(self) -> None:
        """Reset the filter's internal state."""
        self._state = None