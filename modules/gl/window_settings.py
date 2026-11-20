from modules.settings import BaseSettings, Field, Widget
from .monitor import MonitorId


class WindowSettings(BaseSettings):
    """One window: shown or hidden, the monitor it is on, fullscreen, and its windowed place on that monitor."""
    enabled:    Field[bool]      = Field(True, description="Show the window")
    monitor:    Field[MonitorId] = Field(MonitorId.M0, description="The monitor the window is on")
    fullscreen: Field[bool]      = Field(False, description="Borderless, the size of its monitor")
    x:          Field[int]       = Field(20, newline=True, unit="px", description="Windowed: the left edge on its monitor")
    y:          Field[int]       = Field(40, unit="px", description="Windowed: the top edge on its monitor")
    width:      Field[int]       = Field(1280, unit="px", description="Windowed: the width")
    height:     Field[int]       = Field(720, unit="px", description="Windowed: the height")


class WindowManagerSettings(BaseSettings):
    """The window loop: its pacing, the mouse in fullscreen, and the connected monitors."""
    title:             Field[str]  = Field("Poser", access=Field.INIT, visible=False)
    v_sync:            Field[bool] = Field(True, description="Pace the loop by the display of the v-sync window")
    v_sync_window:     Field[int]  = Field(0, min=0, max=7, description="The window swapped last, with v-sync")
    fullscreen_mouse:  Field[bool] = Field(False, description="Show the mouse pointer in fullscreen (a window always shows it)")
    default_placement: Field[bool] = Field(False, widget=Widget.button, description="Place window 0 on monitor 0 at 20, 40, 1280 × 720, windowed")
    avg_fps:           Field[int]  = Field(60, access=Field.READ, pinned=True, label="Render", unit="fps", description="Average render frame rate")
    min_fps:           Field[int]  = Field(60, access=Field.READ, pinned=True, label="Render min", unit="fps", description="Minimum render frame rate")
    connected:         Field[str]  = Field("", access=Field.READ, description="The connected monitors: id, name, resolution")

