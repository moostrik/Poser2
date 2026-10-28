from modules.settings import BaseSettings, Field, Widget


class ArtnetSettings(BaseSettings):
    active:   Field[bool]  = Field(True, widget=Widget.switch, description="Send Art-Net to the lights")
    ip:       Field[str]   = Field("10.0.0.11", widget=Widget.ip_field, description="Art-Net node IP address")
    universe: Field[int]   = Field(1, min=0, max=254, widget=Widget.number_field, description="Art-Net universe")
    fps:      Field[float] = Field(40.0, min=1.0, max=60.0, step=1.0, description="Packets sent per second")
