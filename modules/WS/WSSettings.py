from enum import IntEnum

from modules.settings import BaseSettings, Field


class CompMode(IntEnum):
    NONE      = 0
    TEST      = 1
    CALIBRATE = 2
    VISUALS   = 3


class WSSettings(BaseSettings):
    """All settings for the WSDraw thread."""
    # Construction / wiring
    max_poses:        Field[int]      = Field(3,     min=1,  max=16,    access=Field.INIT, description="Max tracked poses")
    light_rate:       Field[float]    = Field(30.0,  min=1,  max=120,   access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]      = Field(300,   min=10, max=1000,  access=Field.INIT, description="Light strip resolution (pixels)")
    udp_port:         Field[int]      = Field(8000,  min=1,  max=65535, access=Field.INIT, description="UDP sender port")
    udp_ip_light:     Field[list]     = Field(["127.0.0.1"], access=Field.INIT, description="IP addresses for light UDP output")
    udp_ip_sound:     Field[list]     = Field(["127.0.0.2"], access=Field.INIT, description="IP addresses for sound UDP output")
    # Runtime
    mode:             Field[CompMode] = Field(CompMode.VISUALS,          description="Composition mode")
    # Visual
    void_width:       Field[float]    = Field(0.05,  min=0.0, max=1.0,   step=0.01,  description="Void width (normalized)")
    void_edge:        Field[float]    = Field(0.01,  min=0.0, max=1.0,   step=0.005, description="Void edge softness")
    use_void:         Field[bool]     = Field(True,                                   description="Enable void zones")
    pattern_width:    Field[float]    = Field(0.2,   min=0.0, max=1.0,   step=0.01,  description="Pattern width (normalized)")
    pattern_edge:     Field[float]    = Field(0.2,   min=0.0, max=1.0,   step=0.01,  description="Pattern edge softness")
    line_sharpness:   Field[float]    = Field(1.5,   min=0.0, max=10.0,  step=0.1,   description="Line sharpness (higher = sharper)")
    line_speed:       Field[float]    = Field(1.5,   min=0.0, max=10.0,  step=0.1,   description="Line speed (higher = faster)")
    line_width:       Field[float]    = Field(0.1,   min=0.0, max=1.0,   step=0.01,  description="Line width (normalized)")
    line_amount:      Field[float]    = Field(20.0,  min=0.0, max=100.0, step=1.0,   description="Number of lines")
