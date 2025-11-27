from dataclasses import dataclass, field

@dataclass
class Settings:
    title: str =         field(default="Poser")
    monitor: int =       field(default=0)
    width: int =         field(default=1920)
    height: int =        field(default=1000)
    x: int =             field(default=0)
    y: int =             field(default=80)
    fullscreen: bool =   field(default=False)
    fps: int =           field(default=60)
    v_sync: bool =       field(default=True)

    secondary_list: list[int] = field(default_factory=list)

    num_cams: int =      field(default=3)
    num_players: int =   field(default=3)

    cams_a_row: int =    field(default=3)
    num_R: int =         field(default=3)
    stream_capacity: int = field(default=600)  # 10 seconds at 60 FPS