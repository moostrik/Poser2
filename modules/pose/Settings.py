from dataclasses import dataclass, field
from enum import IntEnum, auto


class ModelType(IntEnum):
    NONE =  0
    ONNX =  auto()
    TRT =   auto()
POSE_MODEL_TYPE_NAMES: list[str] = [e.name for e in ModelType]


class Resolution(IntEnum):
    """Model resolution presets."""
    STANDARD = 0
    HIGH = auto()
    ULTRA = auto()
RESOLUTION_NAMES: list[str] = [e.name for e in Resolution]


# Model path definitions - edit these to change model files
MODEL_PATHS = {
    # Pose models (no 512 version - falls back to 384x288)
    'pose': {
        'onnx': {
            Resolution.STANDARD: "rtmpose-l_256x192.onnx",
            Resolution.HIGH: "rtmpose-l_384x288.onnx",
            Resolution.ULTRA: "rtmpose-l_384x288.onnx",  # Fallback
        },
        'tensorrt': {
            Resolution.STANDARD: "rtmpose-l_256x192_b3.trt",
            Resolution.HIGH: "rtmpose-l_384x288_b3.trt",
            Resolution.ULTRA: "rtmpose-l_384x288_b3.trt",  # Fallback
        }
    },
    # Segmentation models
    'segmentation': {
        'onnx': {
            Resolution.STANDARD: "rvm_mobilenetv3_256x192.onnx",
            Resolution.HIGH: "rvm_mobilenetv3_384x288.onnx",
            Resolution.ULTRA: "rvm_mobilenetv3_512x384.onnx",
        },
        'tensorrt': {
            Resolution.STANDARD: "rvm_mobilenetv3_256x192_b3.trt",
            Resolution.HIGH: "rvm_mobilenetv3_384x288_b3.trt",
            Resolution.ULTRA: "rvm_mobilenetv3_512x384_b3.trt",
        }
    },
    # Flow models
    'flow': {
        'onnx': {
            Resolution.STANDARD: "raft-sintel_256x192_i12.onnx",
            Resolution.HIGH: "raft-sintel_384x288_i12.onnx",
            Resolution.ULTRA: "raft-sintel_512x384_i12.onnx",
        },
        'tensorrt': {
            Resolution.STANDARD: "raft-sintel_256x192_i12_b3.trt",
            Resolution.HIGH: "raft-sintel_384x288_i12_b3.trt",
            Resolution.ULTRA: "raft-sintel_512x384_i12_b3.trt",
        }
    }
}

# Resolution dimensions
RESOLUTION_DIMS = {
    Resolution.STANDARD: (192, 256),  # (width, height)
    Resolution.HIGH: (288, 384),
    Resolution.ULTRA: (384, 512),
}


@dataclass
class Settings:

    max_poses: int =                    field(default=1)
    model_type: ModelType =             field(default=ModelType.TRT)

    # Per-model resolution settings
    pose_resolution: Resolution =       field(default=Resolution.STANDARD)
    segmentation_resolution: Resolution = field(default=Resolution.STANDARD)
    flow_resolution: Resolution =       field(default=Resolution.STANDARD)

    model_path: str =                   field(default="models")

    confidence_low: float =             field(default=0.5)
    confidence_high: float =            field(default=0.7)
    verbose: bool =                     field(default=False)
    crop_expansion: float =             field(default=0.0)

    # Feature toggles
    segmentation_enabled: bool =        field(default=True)
    flow_enabled: bool =                field(default=True)

    # Backward compatibility - deprecated single resolution setting
    @property
    def resolution(self) -> Resolution:
        """Deprecated: Use pose_resolution instead."""
        return self.pose_resolution

    @resolution.setter
    def resolution(self, value: Resolution) -> None:
        """Deprecated: Sets all resolutions to the same value."""
        self.pose_resolution = value
        self.segmentation_resolution = value
        self.flow_resolution = value

    # Pose dimensions
    @property
    def width(self) -> int:
        """Pose model width (backward compatibility)."""
        return RESOLUTION_DIMS[self.pose_resolution][0]

    @property
    def height(self) -> int:
        """Pose model height (backward compatibility)."""
        return RESOLUTION_DIMS[self.pose_resolution][1]

    # Specific dimension properties for each model
    @property
    def pose_width(self) -> int:
        return RESOLUTION_DIMS[self.pose_resolution][0]

    @property
    def pose_height(self) -> int:
        return RESOLUTION_DIMS[self.pose_resolution][1]

    @property
    def segmentation_width(self) -> int:
        return RESOLUTION_DIMS[self.segmentation_resolution][0]

    @property
    def segmentation_height(self) -> int:
        return RESOLUTION_DIMS[self.segmentation_resolution][1]

    @property
    def flow_width(self) -> int:
        return RESOLUTION_DIMS[self.flow_resolution][0]

    @property
    def flow_height(self) -> int:
        return RESOLUTION_DIMS[self.flow_resolution][1]

    def _get_model_path(self, model_category: str, resolution: Resolution) -> str:
        """Get model path from MODEL_PATHS definition."""
        if self.model_type == ModelType.ONNX:
            return MODEL_PATHS[model_category]['onnx'][resolution]
        elif self.model_type == ModelType.TRT:
            return MODEL_PATHS[model_category]['tensorrt'][resolution]
        return ""

    @property
    def pose_model(self) -> str:
        return self._get_model_path('pose', self.pose_resolution)

    @property
    def segmentation_model(self) -> str:
        return self._get_model_path('segmentation', self.segmentation_resolution)

    @property
    def flow_model(self) -> str:
        return self._get_model_path('flow', self.flow_resolution)