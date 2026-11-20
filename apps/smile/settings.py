"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.gl import WindowManagerSettings, WindowSettings
from modules.settings import BaseSettings, Group
from modules.panel import PanelSettings
from .render import RenderSettings, SequencerSettings
from .comfy.settings import GeneratorSettings
from .cam import CameraSettings
from .audio import AudioSettings
from modules.inout import ArtNetSenderSettings
from .online.settings import OnlineSettings
from .player import WallsSettings


class WindowsSettings(BaseSettings):
    """The render's windows: window 0 the operator's, windows 1-2 the walls (`sphere.WALLS`)"""
    loop    : Group[WindowManagerSettings] = Group(WindowManagerSettings)
    window_0: Group[WindowSettings]        = Group(WindowSettings)
    window_1: Group[WindowSettings]        = Group(WindowSettings)
    window_2: Group[WindowSettings]        = Group(WindowSettings)

    @property
    def windows(self) -> list[WindowSettings]:
        return [self.window_0, self.window_1, self.window_2]


class Settings(BaseSettings):
    camera:    Group[CameraSettings]    = Group(CameraSettings)
    generator: Group[GeneratorSettings] = Group(GeneratorSettings)
    sequence:  Group[SequencerSettings] = Group(SequencerSettings)
    render:    Group[RenderSettings]    = Group(RenderSettings)
    windows:   Group[WindowsSettings]   = Group(WindowsSettings)
    audio:     Group[AudioSettings]     = Group(AudioSettings)
    walls:     Group[WallsSettings]     = Group(WallsSettings)
    artnet:    Group[ArtNetSenderSettings]    = Group(ArtNetSenderSettings)
    online:    Group[OnlineSettings]    = Group(OnlineSettings)
    panel:     Group[PanelSettings]     = Group(PanelSettings)
