"""What a model's backend does, the same for ONNX Runtime and TensorRT: a model file loaded on the thread that runs it,
bound to that thread's stream, and run on GPU tensors by name, its outputs on the GPU, queued on the stream."""

from abc import ABC, abstractmethod

import torch

from .model_type import ModelType


class Engine(ABC):
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self.input_names: list[str] = []
        self.output_names: list[str] = []

    @abstractmethod
    def setup(self, stream: torch.cuda.Stream) -> None:
        """Load the model, on the thread that will run it, to run on `stream`."""

    @abstractmethod
    def dtype(self, name: str) -> torch.dtype:
        """The torch type of an input or output."""

    @abstractmethod
    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The outputs by name, on the GPU, for the inputs by name (on the GPU, in their types, written on the stream
        or waited for on it), queued on the stream: a reader on the CPU synchronizes it first."""


def make_engine(model_type: ModelType, model_file: str, max_batch: int | None = None) -> Engine:
    """The engine for a model type; TensorRT is imported only when it is used. `max_batch`: ONNX Runtime's batch
    padded to it, the size it is warmed up at (TensorRT needs none)."""
    if model_type is ModelType.ONNX:
        from .onnx_engine import OnnxEngine
        return OnnxEngine(model_file, max_batch)
    from .trt_engine import TrtEngine
    return TrtEngine(model_file)
