from dataclasses import dataclass, field

from modules.pose.detection.MMDetection import ModelType

@dataclass
class Settings:
    max_poses: int =                field(default=1)
    model_type: ModelType =         field(default=ModelType.SMALL)
    model_path: str =               field(default="models")
    confidence_threshold: float =   field(default=0.3)
    verbose: bool =                 field(default=False)
    crop_expansion: float =         field(default=0.0)