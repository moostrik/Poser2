"""Shared ONNX Runtime setup: its CUDA provider loads the CUDA and cuDNN libraries torch brought."""
from threading import Lock
import logging

import torch   # noqa: F401  loaded first, so onnxruntime finds its CUDA libraries next to it
import onnxruntime as ort

logger = logging.getLogger(__name__)

_preload_lock = Lock()
_preloaded = False


def preload_cuda() -> None:
    """Load CUDA and cuDNN for the CUDA provider once, from torch's wheel (no CUDA toolkit on the PATH)."""
    global _preloaded
    with _preload_lock:
        if _preloaded:
            return
        _preloaded = True
        try:
            ort.preload_dlls()
        except Exception as e:
            logger.warning("onnxruntime could not preload CUDA libraries: %s", e)
