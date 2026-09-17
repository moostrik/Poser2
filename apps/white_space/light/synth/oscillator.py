"""Oscillator — the light synth's first building block (``docs/LIGHT_SYNTH.md``, *The oscillator*).

An ordinary LFO with one addition: it is given a set of positions, each a phase offset, and gives
a value for every one of them. The **core** knows where each position is in the cycle; a
**waveform** reads that and makes the output: the pulse draws lines, the sine is an LFO's. It knows
nothing of people, colours, mirroring or the mask, and takes every input as an argument, so what
plays an input (a slot, a setting, another oscillator) is the caller's.

The waveforms are static methods so ``HotReloadMethods`` can patch them while the app runs.
"""

import math

import numpy as np

from modules.settings import BaseSettings, Field

Value = float | np.ndarray      # an input's value: one per tick, or one per position


class OscillatorSettings(BaseSettings):
    """One drawn oscillator's patch: each input's base and the amount its source moves it by."""
    interval:           Field[float] = Field(14.0, min=1.0,    max=180.0, step=0.5,  description="Distance from one line to the next (deg)")
    interval_amount:    Field[float] = Field(0.0,  min=-3.0,   max=3.0,   step=0.05, description="Its source moves the interval by this (octaves)")
    pulse_width:        Field[float] = Field(0.5,  min=0.0,    max=1.0,   step=0.01, description="Line thickness: 0 none, 1 solid (fraction of interval)", newline=True)
    pulse_width_amount: Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, description="Its source moves the pulse width by this")
    phase:              Field[float] = Field(0.0,  min=-0.5,   max=0.5,   step=0.01, description="Where the lines sit: 0 a line at the person, 0.5 a gap (intervals)", newline=True)
    phase_amount:       Field[float] = Field(0.0,  min=-2.0,   max=2.0,   step=0.01, description="Its source moves the phase by this (intervals)")
    speed:              Field[float] = Field(0.0,  min=-90.0,  max=90.0,  step=0.1,  description="Lines travelling: positive outward, 0 still (deg/s)", newline=True)
    speed_amount:       Field[float] = Field(0.0,  min=-90.0,  max=90.0,  step=0.1,  description="Its source moves the speed by this (deg/s)")
    hardness:           Field[float] = Field(1.0,  min=0.0,    max=1.0,   step=0.01, description="Line flanks: 1 hard, 0 softest", newline=True)
    hardness_amount:    Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, description="Its source moves the hardness by this")
    push:               Field[float] = Field(0.0,  min=-180.0, max=180.0, step=0.5,  description="Speed a hit adds for a moment: positive outward (deg/s)", newline=True)


class LfoSettings(BaseSettings):
    """An LFO in time: an oscillator with one position, used as a source. It has no interval and
    no speed, only how fast it cycles, and its level is what is played."""
    rate:         Field[float] = Field(0.5, min=0.0,  max=10.0, step=0.05, description="Cycles per second (Hz)")
    phase:        Field[float] = Field(0.0, min=-0.5, max=0.5,  step=0.01, description="Where in its cycle it starts (cycles)")
    level:        Field[float] = Field(0.0, min=0.0,  max=1.0,  step=0.01, description="How far it swings: 0 silent, 1 full", newline=True)
    level_amount: Field[float] = Field(0.0, min=-1.0, max=1.0,  step=0.01, description="Its source moves the level by this")


class Oscillator:
    """The core and the waveforms; see the module docstring."""

    def __init__(self) -> None:
        self._travelled = 0.0           # how far the wave has moved, in cycles; only its fraction matters

    def reset(self) -> None:
        self._travelled = 0.0

    def update(self, dt: float, interval: float, speed: float) -> None:
        """Advance the wave by ``speed`` (position units per second) over ``dt``. The travel is
        counted in cycles, so a later change of interval opens the wave from position 0."""
        self._travelled = (self._travelled + speed * dt / interval) % 1.0

    def cycle(self, positions: np.ndarray, interval: float, phase: Value) -> np.ndarray:
        """Where each position is in the cycle: a whole number at every crest."""
        return positions / interval - phase - self._travelled

    @staticmethod
    def pulse(cycle: np.ndarray, pulse_width: Value, hardness: Value) -> np.ndarray:
        """The pulse wave, 0..1: on for ``pulse_width`` of every cycle, centred on the whole
        cycle. At ``hardness`` 1 every value is 0 or 1; below it the flank is a cosine fall
        centred on the edge, never wider than the line or the gap has room for, so the centre of
        a line stays 1, the centre of a gap 0, width 0 dark and width 1 solid."""
        d = np.abs((cycle + 0.5) % 1.0 - 0.5)                      # distance from the nearest crest, 0..½
        width = np.clip(pulse_width, 0.0, 1.0)
        edge = width / 2.0
        ramp = (1.0 - np.clip(hardness, 0.0, 1.0)) * 2.0 * np.minimum(edge, 0.5 - edge)
        t = (d - (edge - ramp / 2.0)) / np.maximum(ramp, 1e-9)    # 0 where the fall starts, 1 where it ends
        soft = 0.5 + 0.5 * np.cos(math.pi * np.clip(t, 0.0, 1.0))
        level = np.where(ramp <= 1e-9, d <= edge, soft)             # no flank: exactly off or full
        level = np.where(width <= 0.0, 0.0, np.where(width >= 1.0, 1.0, level))
        return level.astype(np.float32)

    @staticmethod
    def sine(cycle: np.ndarray, level: Value) -> np.ndarray:
        """The sine wave, −level..level, highest at every whole cycle: an LFO's output."""
        return (level * np.cos(math.tau * cycle)).astype(np.float32)
