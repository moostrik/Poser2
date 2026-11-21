from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class HasGeneratorImages(Protocol):
    """The generator's newest image (a step preview or the portrait, BGR) and the depth of the last grin frame (the
    bloom's mask)."""
    def get_diffusion_image(self) -> np.ndarray | None: ...
    def set_diffusion_image(self, image: np.ndarray) -> None: ...
    def get_diffusion_depth(self) -> np.ndarray | None: ...
    def set_diffusion_depth(self, image: np.ndarray) -> None: ...


class GeneratorImageStoreMixin:
    """Thread-safe storage of the generator's newest image and depth."""

    def __init__(self) -> None:
        self._diffusion_lock = Lock()
        self._diffusion_image: np.ndarray | None = None
        self._diffusion_depth: np.ndarray | None = None

    def get_diffusion_image(self) -> np.ndarray | None:
        with self._diffusion_lock:
            return self._diffusion_image

    def set_diffusion_image(self, image: np.ndarray) -> None:
        with self._diffusion_lock:
            self._diffusion_image = image

    def get_diffusion_depth(self) -> np.ndarray | None:
        with self._diffusion_lock:
            return self._diffusion_depth

    def set_diffusion_depth(self, image: np.ndarray) -> None:
        with self._diffusion_lock:
            self._diffusion_depth = image
