from dataclasses import dataclass, field

@dataclass
class Settings:
    max_poses: int =                field(default=1)
    stream_capacity: int =          field(default=300)  # seconds * fps
    stream_sample_rate: int =       field(default=100)  # in milliseconds

    corr_rate: float =              field(default=100)  # in Hz (fps)
    corr_num_workers: int =         field(default=8)
    corr_buffer_duration: int =     field(default=90)   # seconds * fps
    corr_stream_timeout: float =    field(default=2.0)
    corr_max_nan_ratio: float =     field(default=0.15)
    corr_dtw_band: int =            field(default=10)
    corr_similarity_exp: float =    field(default=2.0)