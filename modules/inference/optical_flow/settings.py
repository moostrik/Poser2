from ..model_types import ModelType, Resolution, RESOLUTION_DIMS
from modules.settings import BaseSettings, Field

_MODELS = {
    'onnx': {
        Resolution.STANDARD: "raft-sintel_256x192_i12.onnx",
        Resolution.HIGH: "raft-sintel_384x288_i12.onnx",
        Resolution.ULTRA: "raft-sintel_512x384_i12.onnx",
        Resolution.EXTREME: "raft-sintel_1024x768_i12.onnx",
    },
    'tensorrt': {
        Resolution.STANDARD: "raft-sintel_256x192_i12.trt",
        Resolution.HIGH: "raft-sintel_384x288_i12.trt",
        Resolution.ULTRA: "raft-sintel_512x384_i12.trt",
        Resolution.EXTREME: "raft-sintel_1024x768_i12.trt",
    }
}


class Settings(BaseSettings):
    """Settings for optical flow (RAFT)."""

    max_poses:  Field[int]        = Field(1, min=1, max=16, access=Field.INIT)
    model_type: Field[ModelType]  = Field(ModelType.TRT, access=Field.INIT)
    resolution: Field[Resolution] = Field(Resolution.STANDARD, access=Field.INIT)
    model_path: Field[str]        = Field("data/models", access=Field.INIT, visible=False)
    verbose:    Field[bool]       = Field(False, access=Field.INIT)
    enabled:    Field[bool]       = Field(True)

    @property
    def width(self) -> int:
        return RESOLUTION_DIMS[self.resolution][0]

    @property
    def height(self) -> int:
        return RESOLUTION_DIMS[self.resolution][1]

    @property
    def model(self) -> str:
        if self.model_type == ModelType.ONNX:
            return f"data/models/{_MODELS['onnx'][self.resolution]}"
        elif self.model_type == ModelType.TRT:
            return _MODELS['tensorrt'][self.resolution]
        return ""
