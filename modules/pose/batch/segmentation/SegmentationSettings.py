from ..model_types import ModelType, Resolution, RESOLUTION_DIMS
from modules.settings import Settings, Field

_MODELS = {
    'onnx': {
        Resolution.STANDARD: "rvm_mobilenetv3_256x192.onnx",
        Resolution.HIGH: "rvm_mobilenetv3_384x288.onnx",
        Resolution.ULTRA: "rvm_mobilenetv3_512x384.onnx",
        Resolution.EXTREME: "rvm_mobilenetv3_1024x768.onnx",
    },
    'tensorrt': {
        Resolution.STANDARD: "rvm_mobilenetv3_256x192_b3.trt",
        Resolution.HIGH: "rvm_mobilenetv3_384x288_b3.trt",
        Resolution.ULTRA: "rvm_mobilenetv3_512x384_b3.trt",
        Resolution.EXTREME: "rvm_mobilenetv3_1024x768_b3.trt",
    }
}


class SegmentationSettings(Settings):
    """Settings for person segmentation (RVM)."""

    max_poses:      Field[int]        = Field(1, min=1, max=16, access=Field.INIT)
    model_type:     Field[ModelType]  = Field(ModelType.TRT, access=Field.INIT)
    resolution:     Field[Resolution] = Field(Resolution.STANDARD, access=Field.INIT)
    model_path:     Field[str]        = Field("models", access=Field.INIT, visible=False)
    verbose:        Field[bool]       = Field(False, access=Field.INIT)
    enabled:        Field[bool]       = Field(True, access=Field.INIT)
    reset_interval: Field[int]        = Field(60, min=1, access=Field.INIT)

    @property
    def width(self) -> int:
        return RESOLUTION_DIMS[self.resolution][0]

    @property
    def height(self) -> int:
        return RESOLUTION_DIMS[self.resolution][1]

    @property
    def model(self) -> str:
        if self.model_type == ModelType.ONNX:
            return _MODELS['onnx'][self.resolution]
        elif self.model_type == ModelType.TRT:
            return _MODELS['tensorrt'][self.resolution]
        return ""
