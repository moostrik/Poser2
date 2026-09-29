from modules.settings import BaseSettings, Field, Widget


class ComfySettings(BaseSettings):
    host:          Field[str]   = Field("127.0.0.1", access=Field.INIT, widget=Widget.ip_field, description="ComfyUI server address")
    port:          Field[int]   = Field(8188, access=Field.INIT, widget=Widget.number_field, description="ComfyUI server port")
    workflow:      Field[str]   = Field("data/comfy/AIIAv3 portrait grin v2.json", access=Field.INIT, description="Workflow in ComfyUI API format")
    styles:        Field[str]   = Field("prompts/smile.json", access=Field.INIT, description="Style texts per movement id (also the music key)")

    frames:        Field[int]   = Field(150, min=10, max=300, step=10, description="Grin animation frames")
    fps:           Field[float] = Field(15.0, min=5.0, max=30.0, step=1.0, description="Grin playback speed (frames per second)")
    prebuffer:     Field[int]   = Field(8, min=1, max=60, step=1, description="Frames buffered before the grin starts playing")
    style_repeat:  Field[int]   = Field(26, min=1, max=100, step=1, description="Visitors per style before switching to the next")
    job_timeout:   Field[float] = Field(90.0, min=10.0, max=300.0, step=5.0, description="Give up on a generation after this many seconds")

    connected:     Field[bool]  = Field(False, access=Field.READ, description="Connected to ComfyUI")
    status:        Field[str]   = Field("", access=Field.READ, description="Generator status")
    style:         Field[str]   = Field("", access=Field.READ, description="Current style id")
    description:   Field[str]   = Field("", access=Field.READ, description="Last QwenVL description")
    last_duration: Field[float] = Field(0.0, access=Field.READ, description="Duration of the last generation (s)")
