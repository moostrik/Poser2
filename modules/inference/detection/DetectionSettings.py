from ..model_types import ModelType, Resolution, RESOLUTION_DIMS, ASPECT_RATIOS
from modules.settings import BaseSettings, Field

_MODELS = {
    'onnx': {
        Resolution.STANDARD: "rtmpose-l_256x192.onnx",
        Resolution.HIGH: "rtmpose-l_384x288.onnx",
        Resolution.ULTRA: "rtmpose-l_384x288.onnx",        # Fallback
        Resolution.EXTREME: "rtmpose-l_384x288.onnx",      # Fallback
    },
    'tensorrt': {
        Resolution.STANDARD: "rtmpose-l_256x192",
        Resolution.HIGH:     "rtmpose-l_384x288",
        Resolution.ULTRA:    "rtmpose-l_384x288",           # Fallback
        Resolution.EXTREME:  "rtmpose-l_384x288",           # Fallback
    }
}


class DetectionSettings(BaseSettings):
    """Settings for pose detection (RTMPose)."""

    enabled:    Field[bool]       = Field(True)
    max_poses:  Field[int]        = Field(1, min=1, max=16, access=Field.INIT)
    model_type: Field[ModelType]  = Field(ModelType.TRT, access=Field.INIT)
    resolution: Field[Resolution] = Field(Resolution.STANDARD, access=Field.INIT)
    model_path: Field[str]        = Field("models", access=Field.INIT, visible=False)
    verbose:    Field[bool]       = Field(False, access=Field.INIT)

    @property
    def width(self) -> int:
        return RESOLUTION_DIMS[self.resolution][0]

    @property
    def height(self) -> int:
        return RESOLUTION_DIMS[self.resolution][1]

    @property
    def aspect_ratio(self) -> float:
        return ASPECT_RATIOS[self.resolution]

    @property
    def model(self) -> str:
        if self.model_type == ModelType.ONNX:
            return _MODELS['onnx'][self.resolution]
        elif self.model_type == ModelType.TRT:
            return f"{_MODELS['tensorrt'][self.resolution]}_b{self.max_poses}.trt"
        return ""
