from modules.settings import BaseSettings, Field

from .model_type import ModelType


class ModelWorkerSettings(BaseSettings):
    """The settings every model worker has: its runtime, the folders its files sit in, and its log switch."""

    model_type:  Field[ModelType] = Field(ModelType.TRT, access=Field.INIT, description="Runtime the model runs on")
    onnx_path:   Field[str]       = Field("data/models", access=Field.INIT, visible=False, description="ONNX model files directory")
    engine_path: Field[str]       = Field("data/models", access=Field.INIT, visible=False, description="TensorRT engines directory")
    verbose:     Field[bool]      = Field(False, access=Field.INIT)

    def model_file(self, name: str) -> str:
        """The file of the model `name` for the runtime in use."""
        if self.model_type is ModelType.ONNX:
            return f"{self.onnx_path}/{name}.onnx"
        return f"{self.engine_path}/{name}.trt"
