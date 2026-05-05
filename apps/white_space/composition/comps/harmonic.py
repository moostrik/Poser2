"""Harmonic composition — BPM-locked spatiotemporal LFO/wave-interference sources."""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Group

from ..base import Composition
from ..transport import Transport


class HarmonicSourceSettings(BaseSettings):
    """One LFO source contributing to white and/or blue channels."""
    enabled:         Field[bool]  = Field(True,  description="Enable this source")
    bpm_multiplier:  Field[float] = Field(1.0,  min=0.0, max=32.0, step=0.25,
                                           description="Oscillations per beat (1.0 = 1 cycle/beat)")
    spatial_cycles:  Field[float] = Field(3.0,  min=0.0, max=64.0, step=0.25,
                                           description="Spatial wavelength cycles across the full strip")
    phase_offset:    Field[float] = Field(0.0,  min=0.0, max=1.0,  step=0.01,
                                           description="Temporal phase offset (0–1)")
    spatial_phase:   Field[float] = Field(0.0,  min=0.0, max=1.0,  step=0.01,
                                           description="Spatial phase offset (0–1)")
    amplitude_white: Field[float] = Field(0.5,  min=0.0, max=1.0,  step=0.01,
                                           description="Contribution to white channel")
    amplitude_blue:  Field[float] = Field(0.0,  min=0.0, max=1.0,  step=0.01,
                                           description="Contribution to blue channel")


class HarmonicSettings(BaseSettings):
    """Settings for the LFO/harmonic-interference composition."""
    source_0: Group[HarmonicSourceSettings] = Group(HarmonicSourceSettings)
    source_1: Group[HarmonicSourceSettings] = Group(HarmonicSourceSettings)
    source_2: Group[HarmonicSourceSettings] = Group(HarmonicSourceSettings)
    source_3: Group[HarmonicSourceSettings] = Group(HarmonicSourceSettings)


class Harmonic(Composition):
    """BPM-locked spatiotemporal LFO and harmonic interference composition.

    Each source generates a 1-D spatial sine wave::

        value(x, t) = sin(2π * (spatial_cycles * x + hz * t + phase_offset + spatial_phase))

    where ``hz = bpm_multiplier * bpm / 60``.  Sources accumulate additively
    into white and blue according to their individual amplitudes.
    """

    def __init__(self, resolution: int, config: HarmonicSettings) -> None:
        super().__init__(resolution, config)
        self._config = config
        # Normalised position per pixel: 0.0 → 1.0 (exclusive)
        self._x: np.ndarray = np.linspace(0.0, 1.0, resolution, endpoint=False, dtype=np.float32)

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        sources = (
            self._config.source_0,
            self._config.source_1,
            self._config.source_2,
            self._config.source_3,
        )
        for src in sources:
            if not src.enabled:
                continue
            hz = src.bpm_multiplier * transport.bpm / 60.0
            arg = (
                self._x * src.spatial_cycles
                + hz * transport.time
                + src.phase_offset
                + src.spatial_phase
            ) * math.tau
            wave = (np.sin(arg) * 0.5 + 0.5).astype(np.float32)
            if src.amplitude_white > 0.0:
                white += wave * src.amplitude_white
            if src.amplitude_blue > 0.0:
                blue  += wave * src.amplitude_blue
