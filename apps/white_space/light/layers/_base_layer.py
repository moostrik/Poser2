"""Layer base class and shared settings primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from modules.settings import BaseSettings, Field

from ._utilities import BlendType, blend_values
from ..frame import Frame, BUFFER_DTYPE

if TYPE_CHECKING:
    from ...board import Board


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

    Concrete layers subclass one of the two regime bases (`LowLayer` / `HighLayer`),
    never BaseLayer directly — the regime drives the light_phase ring shift and the
    debug auto-follow's motor derivation.
    """

    SHIFTED: bool = False   # rides the fast ring → gets the light_phase roll (HighLayer)

    def __init__(self, resolution: int, settings: LayerSettings, board: Board) -> None:
        self.resolution: int           = resolution
        self._settings:  LayerSettings = settings
        self._board                    = board  # full blackboard — layers pull the slices they need
        self._scratch_white: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._scratch_blue:  np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)

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

    def reset(self) -> None:
        """Reset internal state. Called explicitly (a show state's ``enter()`` via the
        Compositor) when the layer should start fresh. Default: no-op."""


class LowLayer(BaseLayer):
    """The lamp regime (< ~200 rpm): each output pixel drives a discrete physical lamp
    rather than a POV ring (see ``low/__init__.py`` for the hardware mapping). Encodes
    that mapping once — subclasses write lamps by name instead of hand-computing pixels."""

    def _add_lamps(self, white: np.ndarray, blue: np.ndarray, *,
                   front_white: float = 0.0, back_white: float = 0.0,
                   left_blue: float = 0.0, right_blue: float = 0.0) -> None:
        """Additively write the four physical lamps into the scratch arrays:
        front/back white = ``white[0]`` / ``white[R//2]``, left/right blue =
        ``blue[0]`` / ``blue[R//2]``."""
        half = self.resolution // 2
        white[0]    += front_white
        white[half] += back_white
        blue[0]     += left_blue
        blue[half]  += right_blue


class HighLayer(BaseLayer):
    """The ring regime (fast rotation): pixels draw the persistence-of-vision ring, so
    the content rides the fast spin — ``SHIFTED`` grants the ``light_phase`` ring roll,
    and the debug auto-follow derives HIGH from any selected HighLayer."""

    SHIFTED = True
