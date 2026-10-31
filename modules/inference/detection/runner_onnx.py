"""The person detector's model through ONNX Runtime."""

import logging
from typing import cast

import numpy as np
import onnxruntime as ort

from ..ort_runtime import preload_cuda

logger = logging.getLogger(__name__)


class RunnerONNX:
    def __init__(self, model_file: str) -> None:
        self.model_file: str = model_file
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ''

    def setup(self) -> None:
        """Load the model, on the thread that will run it."""
        preload_cuda()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(self.model_file, options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        provider: str = self._session.get_providers()[0]
        if provider != 'CUDAExecutionProvider':
            logger.warning("Person detector runs on %s: CUDA is not available to ONNX Runtime, expect dropped frames", provider)
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """The output (N, 4 + classes, candidates) of a float32 batch (N, 3, height, width)."""
        assert self._session is not None
        return cast(list[np.ndarray], self._session.run(None, {self._input_name: batch}))[0]
