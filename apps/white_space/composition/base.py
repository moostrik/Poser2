"""Composition base class and shared settings primitives."""

from abc import ABC, abstractmethod

import numpy as np

from modules.settings import BaseSettings, Field
from modules.tracker import Tracklet
from modules.pose.frame import Frame

from .transport import Transport


class ChannelSettings(BaseSettings):
    """Shared per-channel knobs for waveform-style compositions (white or blue)."""
    level:  Field[float] = Field(0.5,  min=0.0,   max=1.0,  step=0.01, description="Brightness level")
    speed:  Field[float] = Field(0.5,  min=-10.0, max=10.0, step=0.01, description="Animation speed")
    phase:  Field[float] = Field(0.0,  min=0.0,   max=1.0,  step=0.01, description="Phase offset (0–1)")
    width:  Field[float] = Field(0.5,  min=0.0,   max=1.0,  step=0.01, description="Pattern width")
    amount: Field[int]   = Field(36,   min=1,     max=200,  step=1,    description="Pattern count")


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
        self.resolution: int      = resolution
        self._settings:  BaseSettings = settings

    @abstractmethod
    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        """Additively write into the white and blue output arrays."""
        ...

    def set_pose_inputs(self, frames: list[Frame], tracklets: dict[int, Tracklet]) -> None:
        """Receive the latest pose snapshot each tick. Default: no-op."""

    def reset(self) -> None:
        """Reset internal time/phase state. Default: no-op."""
