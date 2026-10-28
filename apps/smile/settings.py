"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.settings import BaseSettings, Group, NiceSettings
from apps.smile.render.settings import RenderSettings, SequencerSettings
from apps.smile.comfy.settings import PipelineSettings
from apps.smile.cam.settings import CameraSettings
from apps.smile.audio.settings import AudioSettings
from apps.smile.artnet.settings import ArtnetSettings
from apps.smile.online.settings import OnlineSettings
from apps.smile.player.settings import PlayersSettings


class Settings(BaseSettings):
    camera:   Group[CameraSettings]    = Group(CameraSettings)
    pipeline: Group[PipelineSettings]  = Group(PipelineSettings)
    sequence: Group[SequencerSettings] = Group(SequencerSettings)
    render:   Group[RenderSettings]    = Group(RenderSettings)
    audio:    Group[AudioSettings]     = Group(AudioSettings)
    players:  Group[PlayersSettings]   = Group(PlayersSettings)
    artnet:   Group[ArtnetSettings]    = Group(ArtnetSettings)
    online:   Group[OnlineSettings]    = Group(OnlineSettings)
    server:   Group[NiceSettings]      = Group(NiceSettings)
