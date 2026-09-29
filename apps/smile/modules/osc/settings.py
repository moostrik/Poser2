from modules.settings import BaseSettings, Field, Widget


class OscSettings(BaseSettings):
    active: Field[bool]  = Field(True, widget=Widget.switch, description="Send OSC messages")
    ip:     Field[str]   = Field("127.0.0.1", widget=Widget.ip_field, description="Target OSC IP address")
    port:   Field[int]   = Field(4114, min=1024, max=65535, widget=Widget.number_field, description="Target OSC port")
    fps:    Field[float] = Field(30.0, min=1.0, max=120.0, step=1.0, description="Bundles sent per second")
