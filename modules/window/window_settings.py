from modules.settings import BaseSettings, Field, Widget
from .monitor import MonitorId


class WindowSettings(BaseSettings):
    """One window: shown or hidden, the monitor it is on, fullscreen, and its windowed place on that monitor."""
    enabled:    Field[bool]      = Field(True, description="Show the window")
    monitor:    Field[MonitorId] = Field(MonitorId.M0, description="The monitor the window is on", access=Field.READWRITE)
    fullscreen: Field[bool]      = Field(False, description="Borderless, the size of its monitor", access=Field.READWRITE)
    x:          Field[int]       = Field(20, newline=True, unit="px", description="Windowed: the left edge on its monitor", access=Field.READWRITE)
    y:          Field[int]       = Field(40, unit="px", description="Windowed: the top edge on its monitor", access=Field.READWRITE)
    width:      Field[int]       = Field(1280, unit="px", description="Windowed: the width", access=Field.READWRITE)
    height:     Field[int]       = Field(720, unit="px", description="Windowed: the height", access=Field.READWRITE)


class WindowManagerSettings(BaseSettings):
    """The window loop and its windows: an app's `windows` group subclasses it and declares its windows as
    `window_0`, `window_1`, … (`Group(WindowSettings)`), their number fixed at start."""
    avg_fps:           Field[int]  = Field(60, access=Field.READ, pinned=True, label="Render", unit="fps", description="Average render frame rate")
    min_fps:           Field[int]  = Field(60, access=Field.READ, pinned=True, label="Render min", unit="fps", description="Minimum render frame rate")
    monitors:          Field[str]  = Field("", access=Field.READ, description="The connected monitors: id, name, resolution")
    title:             Field[str]  = Field("Poser", access=Field.INIT, visible=False)
    shown_windows:     Field[list[str]] = Field(["window_0"], access=Field.READ, visible=False, description="The shown windows, by name")
    v_sync:            Field[bool] = Field(True, newline=True, description="Pace the loop by the display of the v-sync window")
    v_sync_window:     Field[str]  = Field("window_0", widget=Widget.text_select, options=shown_windows, description="The shown window swapped last, with v-sync", access=Field.READWRITE)
    fps_limit:         Field[int]  = Field(0, min=0, max=480, widget=Widget.number_field, unit="fps", description="The loop's rate when v-sync does not pace it; 0 = off")
    fullscreen_mouse:  Field[bool] = Field(False, description="Show the mouse pointer in fullscreen (a window always shows it)")
    gl_debug:          Field[bool] = Field(False, access=Field.INIT, description="Log each GL error at its call; slows every call")
    default_placement: Field[bool] = Field(False, widget=Widget.button, description="Place every window at its default place on its monitor, windowed")

    @property
    def windows(self) -> list[WindowSettings]:
        """The declared windows, window 0 first."""
        return [child for child in self.children.values() if isinstance(child, WindowSettings)]

    @property
    def window_names(self) -> list[str]:
        """The declared windows' names (`window_0`, …), in the order of `windows`."""
        return [name for name, child in self.children.items() if isinstance(child, WindowSettings)]
