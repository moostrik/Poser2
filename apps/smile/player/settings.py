from modules.settings import BaseSettings, Field, Widget


class PlayersSettings(BaseSettings):
    """The clip players on the sphere's walls (the two secondary windows): what the one pool keeps and how
    the cells play. The pool's values are read at the start; the play values at every play or frame."""
    pool:           Field[int]   = Field(30, min=1, max=200, access=Field.INIT, description="Clips kept decoded in RAM (about 200 MB each)")
    recent:         Field[int]   = Field(4, min=0, max=24, access=Field.INIT, description="The last recordings, always playing on every third of the sphere; 0 = none")
    recycle_every:  Field[float] = Field(20.0, min=0.0, max=600.0, step=1.0, access=Field.INIT, description="A clip from disk joins when no clip has joined for this many seconds; 0 = never")
    hwaccel:        Field[int]   = Field(-1, min=-1, max=7, access=Field.INIT, widget=Widget.number_field, description="Decode on the GPU with this DXGI adapter index; -1 = software")
    fps:            Field[float] = Field(20.0, min=0.0, max=60.0, step=1.0, description="Playback speed of the clips, frames per second; 0 plays each clip at its own rate")
    spread:         Field[float] = Field(0.1, min=0.0, max=0.9, step=0.05, description="Random speed difference per play, as a fraction of fps, so the cells drift apart")
    trail:          Field[float] = Field(0.0, min=0.0, max=0.99, step=0.01, description="Blend of each frame with the previous output; 0 = none")
