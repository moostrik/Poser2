"""Layer base classes and shared settings primitives.

Two regimes, two bases, one axis: a ``HighLayer`` draws pixels — the persistence-of-vision
ring — and a ``LowLayer`` writes the four bar lights by name. The split is structural
because it is the fixture's truth: in slot mode (commanded below ``FIXTURE_SLOW_RPM``) the
firmware drives the lamps from four values and ignores the ring, in ring mode it steps the
ring and ignores the lamps (see ``inout/osc_light_sender.py`` for the wire contract). A low
layer therefore has no pixel buffers to write to at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from modules.settings import BaseSettings, Field

from ._utilities import BlendType, blend_values
from ..frame import Frame, BarLightId, BUFFER_DTYPE

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
    """Base class for all light layers.

    ``render`` is a template method called once per tick by the Compositor: zero the
    layer's own scratch, call ``_draw`` (the subclass writes additively into it), blend the
    scratch into the frame with the layer's own ``blend`` setting. Each regime base owns
    its scratch and its ``_draw`` signature.

    Concrete layers subclass one of the two regime bases (`LowLayer` / `HighLayer`),
    never BaseLayer directly — the regime drives the light_phase ring shift and the
    debug auto-follow's motor derivation.
    """

    SHIFTED: bool = False   # rides the fast ring → gets the light_phase roll (HighLayer)

    def __init__(self, resolution: int, settings: LayerSettings, board: Board) -> None:
        self.resolution: int           = resolution
        self._settings:  LayerSettings = settings
        self._board                    = board  # full blackboard — layers pull the slices they need

    @abstractmethod
    def render(self, frame: Frame) -> None:
        """Draw this tick into ``frame`` (the regime base's template method)."""
        ...

    def reset(self) -> None:
        """Reset internal state. Called explicitly (a show state's ``enter()`` via the
        Compositor) when the layer should start fresh. Default: no-op."""


class LowLayer(BaseLayer):
    """The lamp regime (commanded below ``FIXTURE_SLOW_RPM``): the fixture drives the four
    bar lights directly and reads no ring pixels, so a low layer writes ``bar_lights`` —
    a ``(4,)`` array indexed by ``BarLightId`` — and nothing else."""

    def __init__(self, resolution: int, settings: LayerSettings, board: Board) -> None:
        super().__init__(resolution, settings, board)
        self._scratch_bar_lights: np.ndarray = np.zeros(len(BarLightId), dtype=BUFFER_DTYPE)

    def render(self, frame: Frame) -> None:
        self._scratch_bar_lights.fill(0.0)
        self._draw(frame, self._scratch_bar_lights)
        blend_values(frame.bar_lights, self._scratch_bar_lights, 0, self._settings.blend)

    @abstractmethod
    def _draw(self, frame: Frame, bar_lights: np.ndarray) -> None:
        """Additively write the four bar lights (pre-zeroed, indexed by ``BarLightId``)."""
        ...

    @staticmethod
    def _add_bar_lights(bar_lights: np.ndarray, *,
                        front_white: float = 0.0, back_white: float = 0.0,
                        left_blue: float = 0.0, right_blue: float = 0.0) -> None:
        """Additively write the bar lights by name."""
        bar_lights[BarLightId.FRONT_WHITE] += front_white
        bar_lights[BarLightId.BACK_WHITE]  += back_white
        bar_lights[BarLightId.LEFT_BLUE]   += left_blue
        bar_lights[BarLightId.RIGHT_BLUE]  += right_blue


class HighLayer(BaseLayer):
    """The ring regime (commanded at or above ``FIXTURE_SLOW_RPM``): pixels draw the
    persistence-of-vision ring, so the content rides the fast spin — ``SHIFTED`` grants the
    ``light_phase`` ring roll, and the debug auto-follow derives HIGH from any selected
    HighLayer. Subclasses write additively into the ``white`` / ``blue`` scratch arrays
    (shape ``(resolution,)``, pre-zeroed)."""

    SHIFTED = True

    def __init__(self, resolution: int, settings: LayerSettings, board: Board) -> None:
        super().__init__(resolution, settings, board)
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
