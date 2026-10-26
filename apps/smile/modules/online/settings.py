from modules.settings import BaseSettings, Field, Widget


class OnlineSettings(BaseSettings):
    active:         Field[bool]  = Field(False, widget=Widget.switch, pinned=True, description="SMILE Online: take turns from people outside, through the relay")
    relay:          Field[str]   = Field("ws://127.0.0.1:8800/ws/smile", description="The relay's websocket address (wss:// on the VPS)")
    token_file:     Field[str]   = Field("data/online_token.txt", access=Field.INIT, description="File with the relay's token (one line, not in git)")
    turn_window:    Field[float] = Field(10.0, min=3.0, max=60.0, step=1.0, description="Seconds a turn has to show a qualifying face, else it passes")
    eye_free_after: Field[float] = Field(3.0, min=0.0, max=30.0, step=0.5, description="Seconds the room must have had no face before a turn starts")
    connected:      Field[bool]  = Field(False, access=Field.READ, newline=True, description="Connected to the relay")
    status:         Field[str]   = Field("off", access=Field.READ, description="What the online side does now")
    waiting:        Field[int]   = Field(0, access=Field.READ, description="People in line at the relay")
    turns:          Field[int]   = Field(0, access=Field.READ, description="Online turns since the start")
