"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.settings import BaseSettings, Group, NiceSettings
from modules.render.settings import RenderSettings, SequencerSettings
from modules.comfy.settings import ComfySettings
from modules.face.settings import FaceSettings
from modules.cam.settings import CameraSettings
from modules.audio.settings import AudioSettings
from modules.osc.settings import OscSettings
from modules.artnet.settings import ArtnetSettings
from modules.save.settings import ImgFolderSettings


class Settings(BaseSettings):
    camera:   Group[CameraSettings]    = Group(CameraSettings)
    face:     Group[FaceSettings]      = Group(FaceSettings)
    comfy:    Group[ComfySettings]     = Group(ComfySettings)
    save:     Group[ImgFolderSettings] = Group(ImgFolderSettings)
    sequence: Group[SequencerSettings] = Group(SequencerSettings)
    render:   Group[RenderSettings]    = Group(RenderSettings)
    audio:    Group[AudioSettings]     = Group(AudioSettings)
    osc:      Group[OscSettings]       = Group(OscSettings)
    artnet:   Group[ArtnetSettings]    = Group(ArtnetSettings)
    server:   Group[NiceSettings]      = Group(NiceSettings)
