"""Layer base classes and shared settings primitives.

Two modes, two bases, one axis: a ``ProjectionLayer`` draws pixels — the persistence-of-vision
projection — and a ``BeamLayer`` writes the four beam lights by name. The split is structural
because it is the fixture's truth: in beam mode (commanded below ``FIXTURE_PROJECTION_RPM``) the
firmware drives the lamps from four values and ignores the pixels, in projection mode it steps the
pixels and ignores the lamps (see ``inout/osc_light_sender.py`` for the wire contract). A beam
layer therefore has no pixel buffers to write to at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from modules.settings import BaseSettings, Field

from ._utilities import BlendType, blend_values
from ..frame import Frame, BeamLightId, BUFFER_DTYPE
from ..motor import MotorMode

if TYPE_CHECKING:
    from ...board import Board


class LayerSettings(BaseSettings):
    """Base settings for every layer — carries the blend mode used to composite
    the layer into the master frame."""
    blend: Field[BlendType] = Field(BlendType.ADD, description="How this layer blends into the master")


class ChannelSettings(BaseSettings):
    """Shared per-channel knobs for waveform-style layers (white or blue)."""
    level:  Field[float] = Field(0.5,  min=0.0,   max=1.0,  step=0.01, description="Brightness level")
    speed:  Field[float] = Field(0.5,  min=-32.0, max=32.0, step=0.01, description="Animation speed")
    phase:  Field[float] = Field(0.0,  min=0.0,   max=1.0,  step=0.01, description="Phase offset (0–1)")
    width:  Field[float] = Field(0.5,  min=0.0,   max=1.0,  step=0.01, description="Pattern width")
    amount: Field[int]   = Field(36,   min=1,     max=200,  step=1,    description="Pattern count")


class BaseLayer(ABC):
    """Base class for all light layers.

    ``render`` is a template method called once per tick by the Compositor: zero the
    layer's own scratch, call ``_draw`` (the subclass writes additively into it), blend the
    scratch into the frame with the layer's own ``blend`` setting. Each mode base owns
    its scratch and its ``_draw`` signature.

    Concrete layers subclass one of the two mode bases (`BeamLayer` / `ProjectionLayer`),
    never BaseLayer directly — ``MODE`` is what the debug auto-follow reads to command the
    motor.
    """

    MODE: MotorMode = MotorMode.STOPPED   # the fixture mode this layer draws for

    def __init__(self, resolution: int, settings: LayerSettings, board: Board) -> None:
        self.resolution: int           = resolution
        self._settings:  LayerSettings = settings
        self._board                    = board  # full blackboard — layers pull the slices they need

    @abstractmethod
    def render(self, frame: Frame) -> None:
        """Draw this tick into ``frame`` (the mode base's template method)."""
        ...

    def reset(self) -> None:
        """Reset internal state. Called explicitly (a show state's ``enter()`` via the
        Compositor) when the layer should start fresh. Default: no-op."""


class BeamLayer(BaseLayer):
    """Beam mode (commanded below ``FIXTURE_PROJECTION_RPM``): the fixture drives the four
    beam lights directly and reads no projection, so a beam layer writes ``beam_lights`` —
    a ``(4,)`` array indexed by ``BeamLightId`` — and nothing else."""

    MODE = MotorMode.BEAM

    def __init__(self, resolution: int, settings: LayerSettings, board: Board) -> None:
        super().__init__(resolution, settings, board)
        self._scratch_beam_lights: np.ndarray = np.zeros(len(BeamLightId), dtype=BUFFER_DTYPE)

    def render(self, frame: Frame) -> None:
        self._scratch_beam_lights.fill(0.0)
        self._draw(frame, self._scratch_beam_lights)
        blend_values(frame.beam_lights, self._scratch_beam_lights, 0, self._settings.blend)

    @abstractmethod
    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        """Additively write the four beam lights (pre-zeroed, indexed by ``BeamLightId``)."""
        ...

    @staticmethod
    def _add_beam_lights(beam_lights: np.ndarray, *,
                         front_white: float = 0.0, back_white: float = 0.0,
                         left_blue: float = 0.0, right_blue: float = 0.0) -> None:
        """Additively write the beam lights by name."""
        beam_lights[BeamLightId.FRONT_WHITE] += front_white
        beam_lights[BeamLightId.BACK_WHITE]  += back_white
        beam_lights[BeamLightId.LEFT_BLUE]   += left_blue
        beam_lights[BeamLightId.RIGHT_BLUE]  += right_blue


class ProjectionLayer(BaseLayer):
    """Projection mode (commanded at or above ``FIXTURE_PROJECTION_RPM``): pixels draw the
    persistence-of-vision projection, so the content rides the fast spin. A projection layer authors
    in azimuth and nothing rotates it here — the light sender applies the projection offset on
    the way out, so the frame on the board stays azimuth-true. Subclasses write additively into
    the ``white`` / ``blue`` scratch arrays (shape ``(resolution,)``, pre-zeroed)."""

    MODE = MotorMode.PROJECTION

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
