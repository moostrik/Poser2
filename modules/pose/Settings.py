from dataclasses import dataclass, field

from modules.pose.detection.MMDetection import ModelType


@dataclass
class Settings:
    active: bool =                  field(default=True)
    model_type: ModelType =         field(default=ModelType.SMALL)
    model_path: str =               field(default="models")
    num_warmups: int =              field(default=1)
    confidence_threshold: float =   field(default=0.3)
    verbose: bool =                 field(default=False)
    crop_expansion: float =         field(default=0.0)
    stream_capacity: int =          field(default=100)