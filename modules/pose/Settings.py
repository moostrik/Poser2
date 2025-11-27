from dataclasses import dataclass, field

from modules.pose.detection.MMDetection import ModelType

@dataclass
class Settings:
    active: bool =                  field(default=True)
    max_poses: int =                field(default=1)
    model_type: ModelType =         field(default=ModelType.SMALL)
    model_path: str =               field(default="models")
    confidence_threshold: float =   field(default=0.3)
    verbose: bool =                 field(default=False)
    crop_expansion: float =         field(default=0.0)

    stream_capacity: int =          field(default=100)
    stream_sample_interval: int =   field(default=100)  # in milliseconds