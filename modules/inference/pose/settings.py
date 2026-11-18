from ..engine import ModelType, Resolution, RESOLUTION_DIMS
from modules.settings import BaseSettings, Field

# The model of each resolution RTMPose comes in; a resolution without one has no pose model
_MODELS: dict[Resolution, str] = {
    Resolution.STANDARD: "rtmpose-l_256x192",
    Resolution.HIGH:     "rtmpose-l_384x288",
}


class PoseEstimatorSettings(BaseSettings):
    """The pose model's settings (RTMPose)."""

    enabled:    Field[bool]       = Field(True)
    max_players:  Field[int]        = Field(1, min=1, max=16, access=Field.INIT)
    model_type: Field[ModelType]  = Field(ModelType.TRT, access=Field.INIT)
    resolution: Field[Resolution] = Field(Resolution.STANDARD, access=Field.INIT)
    model_path: Field[str]        = Field("data/models", access=Field.INIT, visible=False)
    verbose:    Field[bool]       = Field(False, access=Field.INIT)
    dropped:    Field[int]        = Field(0, access=Field.READ, description="Batches replaced by a newer one before the model ran them")

    @property
    def width(self) -> int:
        return RESOLUTION_DIMS[self.resolution][0]

    @property
    def height(self) -> int:
        return RESOLUTION_DIMS[self.resolution][1]

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def model(self) -> str:
        """The model file of the resolution; a resolution RTMPose does not come in is an error."""
        name: str | None = _MODELS.get(self.resolution)
        if name is None:
            raise ValueError(f"Pose: no model at {self.resolution.name}; RTMPose comes in {', '.join(r.name for r in _MODELS)}")
        if self.model_type == ModelType.ONNX:
            return f"data/models/{name}.onnx"
        elif self.model_type == ModelType.TRT:
            return f"{name}.trt"
        return ""
