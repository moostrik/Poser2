"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.settings import BaseSettings, Group
from modules.panel import PanelSettings
from .render import RenderSettings, SequencerSettings
from .comfy.settings import GeneratorSettings
from .cam import CameraSettings
from .audio import AudioSettings
from modules.inout import ArtNetSenderSettings
from .online.settings import OnlineSettings
from .player import WallsSettings


class Settings(BaseSettings):
    camera:    Group[CameraSettings]    = Group(CameraSettings)
    generator: Group[GeneratorSettings] = Group(GeneratorSettings)
    sequence:  Group[SequencerSettings] = Group(SequencerSettings)
    render:    Group[RenderSettings]    = Group(RenderSettings)
    audio:     Group[AudioSettings]     = Group(AudioSettings)
    walls:     Group[WallsSettings]     = Group(WallsSettings)
    artnet:    Group[ArtNetSenderSettings]    = Group(ArtNetSenderSettings)
    online:    Group[OnlineSettings]    = Group(OnlineSettings)
    panel:     Group[PanelSettings]     = Group(PanelSettings)
