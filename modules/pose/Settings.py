from dataclasses import dataclass, field
from enum import IntEnum, auto


class ModelType(IntEnum):
    NONE =      0
    MM =        auto()
    ONNX =      auto()
    TENSORRT =  auto()
POSE_MODEL_TYPE_NAMES: list[str] = [e.name for e in ModelType]


class ModelSize(IntEnum):
    LARGE =     0
    MEDIUM =    auto()
    SMALL =     auto()
    TINY =      auto()
POSE_MODEL_SIZE_NAMES: list[str] = [e.name for e in ModelSize]

@dataclass
class Settings:

    max_poses: int =                field(default=1)
    model_type: ModelType =         field(default=ModelType.ONNX)
    model_size: ModelSize =         field(default=ModelSize.LARGE)
    model_path: str =               field(default="models")
    model_width: int =              field(default=192)
    model_height: int =             field(default=256)


    confidence_low: float =         field(default=0.5)
    confidence_high: float =        field(default=0.7)
    verbose: bool =                 field(default=False)
    crop_expansion: float =         field(default=0.0)

    # Segmentation settings
    segmentation_enabled: bool =    field(default=True)
    segmentation_model_name: str =  field(default="rvm_mobilenetv3_fp16.onnx")

    # Flow settings
    flow_enabled: bool =            field(default=True)