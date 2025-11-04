# Standard library imports
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

# Pose imports
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.Pose import Pose, PoseDict

# Local application imports
from modules.Settings import Settings
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
    """Base class for pose data smoothing using OneEuroFilter.

    Handles:
    - Per-tracklet filter state management
    - Tracklet lifecycle (initialization and cleanup)
    - Common pose processing loop

    Subclasses implement:
    - Filter initialization for new tracklets
    - Smoothing logic for specific data types (points, angles, bbox)
    - Settings update propagation to filters
    """

    def __init__(self, settings: Settings, name: str, gui: Gui | None = None) -> None:
        super().__init__()
        self.settings = SmootherSettings(
            frequency=settings.camera_fps,
            min_cutoff=2.0,
            beta=0.05,
            d_cutoff=1.0,
            reset_on_reappear=False
        )

        self.name: str = name
        self.gui_settings: GuiSettings | None = None
        if gui is not None:
            self.gui_settings = GuiSettings(self.settings, gui, name, self._update_settings)

        # Per-tracklet state: tracklet_id -> filter state (managed by subclasses)
        self._tracklets: dict[int, Any] = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth data for all poses."""
        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Initialize tracklet filters if needed
            if tracklet_id not in self._tracklets:
                self._tracklets[tracklet_id] = self._create_tracklet_state()

            # Smooth the pose data
            smoothed_pose: Pose = self._smooth(pose, tracklet_id)
            smoothed_poses[pose_id] = smoothed_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]

        self._notify_callbacks(smoothed_poses)

    @abstractmethod
    def _create_tracklet_state(self) -> Any:
        """Create initial filter state for a new tracklet.

        Returns:
            Tracklet-specific state (filters, validity masks, etc.)
        """
        pass

    @abstractmethod
    def _smooth(self, pose: Pose, tracklet_id: int) -> Pose:
        """Apply smoothing to a single pose.

        Args:
            pose: Input pose to smooth
            tracklet_id: ID of the tracklet (for accessing filter state)

        Returns:
            Smoothed pose with updated data
        """
        pass

    @abstractmethod
    def _update_tracklet_filters(self, tracklet_state: Any) -> None:
        """Update filter parameters for a tracklet's filters.

        Args:
            tracklet_state: The tracklet's filter state
        """
        pass

    def _update_settings(self) -> None:
        """Update filter parameters for all tracklets."""
        for tracklet_state in self._tracklets.values():
            self._update_tracklet_filters(tracklet_state)

    def get_gui_frame(self):
        if self.gui_settings:
            return self.gui_settings.get_gui_frame()
        return Frame(self.name, [], 50)  # Empty frame if no GUI settings