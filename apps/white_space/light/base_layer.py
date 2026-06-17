"""Layer base class and shared settings primitives."""

from abc import ABC, abstractmethod

import numpy as np

from modules.settings import BaseSettings, Field
from modules.tracker import Tracklet
from modules.pose import frame as pose_frame

from .draw import BlendType, blend_values
from .frame import Frame, BUFFER_DTYPE


class LayerSettings(BaseSettings):
    """Base settings for every layer — carries the blend mode used to composite
    the layer into the master frame."""
    blend: Field[BlendType] = Field(BlendType.ADD, description="How this layer blends into the master")


class ChannelSettings(BaseSettings):
    """Shared per-channel knobs for waveform-style layers (white or blue)."""
    level:  Field[float] = Field(0.5,  min=0.0,   max=1.0,  step=0.01, description="Brightness level")
    speed:  Field[float] = Field(0.5,  min=-10.0, max=10.0, step=0.01, description="Animation speed")
    phase:  Field[float] = Field(0.0,  min=0.0,   max=1.0,  step=0.01, description="Phase offset (0–1)")
    width:  Field[float] = Field(0.5,  min=0.0,   max=1.0,  step=0.01, description="Pattern width")
    amount: Field[int]   = Field(36,   min=1,     max=200,  step=1,    description="Pattern count")


class BaseLayer(ABC):
    """Base class for all LED layers.

    ``render`` is a template method called once per tick by the renderer:
      1. zero this layer's own scratch buffers
      2. call ``_draw`` (subclass writes additively into the scratch buffers)
      3. blend the scratch buffers into ``frame.white`` / ``frame.blue`` using the
         layer's own ``blend`` setting

    Subclasses implement ``_draw`` and write *additively* into the supplied
    ``white`` / ``blue`` scratch arrays (shape ``(resolution,)``, pre-zeroed).
    """

    def __init__(self, resolution: int, settings: LayerSettings) -> None:
        self.resolution: int           = resolution
        self._settings:  LayerSettings = settings
        self._scratch_white: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._scratch_blue:  np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self.target_rpm: float | None = None  # motor speed target — None means no opinion; 0.0 explicitly stops the motor

    def render(self, frame: Frame) -> None:
        self._scratch_white.fill(0.0)
        self._scratch_blue.fill(0.0)
        self._draw(frame, self._scratch_white, self._scratch_blue)
        blend_values(frame.white, self._scratch_white, 0, self._settings.blend)
        blend_values(frame.blue,  self._scratch_blue,  0, self._settings.blend)

    @abstractmethod
    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        """Additively write into the white and blue scratch arrays."""
        ...

    def set_pose_inputs(self, frames: list[pose_frame.Frame], tracklets: dict[int, Tracklet]) -> None:
        """Receive the latest pose snapshot each tick. Default: no-op."""

    def reset(self) -> None:
        """Reset internal time/phase state. Default: no-op."""
