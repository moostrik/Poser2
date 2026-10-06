"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.settings import BaseSettings, Group, NiceSettings
from modules.render.settings import RenderSettings, SequencerSettings
from modules.comfy.settings import PipelineSettings
from modules.cam.settings import CameraSettings
from modules.audio.settings import AudioSettings
from modules.artnet.settings import ArtnetSettings


class Settings(BaseSettings):
    camera:   Group[CameraSettings]    = Group(CameraSettings)
    pipeline: Group[PipelineSettings]  = Group(PipelineSettings)
    sequence: Group[SequencerSettings] = Group(SequencerSettings)
    render:   Group[RenderSettings]    = Group(RenderSettings)
    audio:    Group[AudioSettings]     = Group(AudioSettings)
    artnet:   Group[ArtnetSettings]    = Group(ArtnetSettings)
    server:   Group[NiceSettings]      = Group(NiceSettings)
