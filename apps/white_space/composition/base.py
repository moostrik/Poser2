"""Composition base class."""

from abc import ABC, abstractmethod

import numpy as np

from modules.settings import BaseSettings
from modules.tracker import Tracklet
from modules.pose.frame import Frame

from .transport import Transport


class Composition(ABC):
    """Base class for all LED compositions.

    The host calls ``render`` once per tick with:
      - ``transport``: shared clock snapshot (time, dt, bpm, phase, beat)
      - ``white``:     float32 array of shape (resolution,), pre-zeroed each tick
      - ``blue``:      float32 array of shape (resolution,), pre-zeroed each tick

    Implementations write *additively* into ``white`` and ``blue``.
    The host clips both arrays to [0, 1] after all compositions have rendered.
    """

    def __init__(self, resolution: int, settings: BaseSettings) -> None:
        self.resolution: int          = resolution
        self._settings:  BaseSettings = settings
        self.target_rpm: float | None = None  # motor speed target — None means no opinion; 0.0 explicitly stops the motor

    @abstractmethod
    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        """Additively write into the white and blue output arrays."""
        ...

    def set_pose_inputs(self, frames: list[Frame], tracklets: dict[int, Tracklet]) -> None:
        """Receive the latest pose snapshot each tick. Default: no-op."""

    def reset(self) -> None:
        """Reset internal time/phase state. Default: no-op."""
