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

from modules.settings import BaseSettings, Field, Widget

from .slot import Curve

Value = float | np.ndarray      # an input's value: one per tick, or one per position

KNOB = Widget.knob
CURVE = 36                      # the Curve select's width: narrower than a select's own


class OscillatorSettings(BaseSettings):
    """One drawn oscillator's patch, a modulation matrix row per input, titled with the input's
    name: **Base** (the input's own knob), the **Amount** its source moves it by, the source's
    **Curve**, the **Source** knob reading the live source, and **Bypass**, which switches the
    modulation off and leaves the base."""
    interval:           Field[float] = Field(14.0, min=1.0,    max=180.0, step=0.5,  widget=KNOB, label="Base",   description="Distance from one line to the next (deg)", row_label="Interval", newline=True)
    interval_amount:    Field[float] = Field(0.0,  min=-3.0,   max=3.0,   step=0.05, widget=KNOB, label="Amount", description="How far the source moves the interval (octaves)")
    interval_curve:     Field[Curve] = Field(Curve.LINEAR,                           width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    interval_source:    Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Source", description="What plays the interval, live", access=Field.READ)
    interval_bypass:    Field[bool]  = Field(False,                                               label="Bypass", description="Switch the modulation off: the interval is its base")
    pulse_width:        Field[float] = Field(0.5,  min=0.0,    max=1.0,   step=0.01, widget=KNOB, label="Base",   description="Line thickness: 0 none, 1 solid (fraction of interval)", row_label="Pulse Width", newline=True)
    pulse_width_amount: Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Amount", description="How far the source moves the pulse width")
    pulse_width_curve:  Field[Curve] = Field(Curve.LINEAR,                           width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    pulse_width_source: Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Source", description="What plays the pulse width, live", access=Field.READ)
    pulse_width_bypass: Field[bool]  = Field(False,                                               label="Bypass", description="Switch the modulation off: the pulse width is its base")
    phase:              Field[float] = Field(0.0,  min=-0.5,   max=0.5,   step=0.01, widget=KNOB, label="Base",   description="Where the lines sit: 0 a line at the person, 0.5 a gap (intervals)", row_label="Phase", newline=True)
    phase_amount:       Field[float] = Field(0.0,  min=-2.0,   max=2.0,   step=0.01, widget=KNOB, label="Amount", description="How far the source moves the phase (intervals)")
    phase_curve:        Field[Curve] = Field(Curve.LINEAR,                           width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    phase_source:       Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Source", description="What plays the phase, live", access=Field.READ)
    phase_bypass:       Field[bool]  = Field(False,                                               label="Bypass", description="Switch the modulation off: the phase is its base")
    speed:              Field[float] = Field(0.0,  min=-90.0,  max=90.0,  step=0.1,  widget=KNOB, label="Base",   description="Lines travelling: positive outward, 0 still (deg/s)", row_label="Speed", newline=True)
    speed_amount:       Field[float] = Field(0.0,  min=-90.0,  max=90.0,  step=0.1,  widget=KNOB, label="Amount", description="How far the source moves the speed (deg/s)")
    speed_curve:        Field[Curve] = Field(Curve.LINEAR,                           width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    speed_source:       Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Source", description="What plays the speed, live", access=Field.READ)
    speed_bypass:       Field[bool]  = Field(False,                                               label="Bypass", description="Switch the modulation off: the speed is its base")
    hardness:           Field[float] = Field(1.0,  min=0.0,    max=1.0,   step=0.01, widget=KNOB, label="Base",   description="Line flanks: 1 hard, 0 softest", row_label="Hardness", newline=True)
    hardness_amount:    Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Amount", description="How far the source moves the hardness")
    hardness_curve:     Field[Curve] = Field(Curve.LINEAR,                           width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    hardness_source:    Field[float] = Field(0.0,  min=-1.0,   max=1.0,   step=0.01, widget=KNOB, label="Source", description="What plays the hardness, live", access=Field.READ)
    hardness_bypass:    Field[bool]  = Field(False,                                               label="Bypass", description="Switch the modulation off: the hardness is its base")
    push:               Field[float] = Field(0.0,  min=-180.0, max=180.0, step=0.5,  widget=KNOB, label="Amount", description="Speed a hit adds for a moment: positive outward (deg/s)", row_label="Push", newline=True)


class LfoSettings(BaseSettings):
    """An LFO in time: an oscillator with one position, used as a source. It has no interval and
    no speed, only how fast it cycles; its level is what is played, a matrix row like an input's."""
    rate:         Field[float] = Field(0.5, min=0.0,  max=10.0, step=0.05, widget=KNOB, label="Rate",   description="Cycles per second (Hz)", row_label="LFO", newline=True)
    phase:        Field[float] = Field(0.0, min=-0.5, max=0.5,  step=0.01, widget=KNOB, label="Phase",  description="Where in its cycle it starts (cycles)")
    level:        Field[float] = Field(0.0, min=0.0,  max=1.0,  step=0.01, widget=KNOB, label="Base",   description="How far it swings: 0 silent, 1 full", row_label="Level", newline=True)
    level_amount: Field[float] = Field(0.0, min=-1.0, max=1.0,  step=0.01, widget=KNOB, label="Amount", description="How far the source moves the level")
    level_curve:  Field[Curve] = Field(Curve.LINEAR,                       width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    level_source: Field[float] = Field(0.0, min=-1.0, max=1.0,  step=0.01, widget=KNOB, label="Source", description="What plays the level, live", access=Field.READ)
    level_bypass: Field[bool]  = Field(False,                                           label="Bypass", description="Switch the modulation off: the level is its base")


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
