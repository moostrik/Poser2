"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.settings import BaseSettings, Group
from modules.panel import PanelSettings
from .render import RenderSettings, WindowsSettings
from .sequence import SequencerSettings
from .comfy import GeneratorSettings
from .cam import CameraSettings
from .audio import AudioSettings
from modules.inout import ArtNetSenderSettings
from .online import OnlineSettings
from .player import WallsSettings


class SetupSettings(BaseSettings):
    """The machine and the site: the camera, the generator, the sequence, the render, the walls and the links."""
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


class Settings(BaseSettings):
    setup: Group[SetupSettings] = Group(SetupSettings)
