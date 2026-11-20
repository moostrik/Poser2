from enum import IntEnum, auto


class ModelType(IntEnum):
    NONE =  0
    ONNX =  auto()
    TRT =   auto()
