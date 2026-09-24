"""Strobe — the light synth's gate in time (``docs/LIGHT_SYNTH.md``, *The strobe*).

Each tick, each line of an oscillator's output is on or off: no level between, the synth's own
rule, and the synth's one deliberate step in time. One strobe per oscillator; its four parameters
each have a slot. Time is the clock's tick index, so every voice's strobe sits on one grid: two
people at the same rate go dark on the same ticks. The rates are powers of two, so the grids nest
and a change of rate is seamless.

The gate, in ticks, with ``T = round(ticks per second / rate)`` ticks per cycle and the delay in
seconds per line:

    offset(k)        = round(phase · T) + round(delay · ticks per second · k)
    on(tick, line k) = ((tick − offset(k)) mod T) < round(width · T)

The lit ticks are the first ``round(width · T)`` of a cycle and the dark ticks the rest, so the
dark ticks of nested rates coincide. ``k`` is a line's count from the person, in intervals: whole
for standing lines, drifting smoothly as a line travels. The delay is in seconds, not cycles, so a
run's speed (one line per delay) is the same at every rate; the offsets are modulo ``T``, so the
run repeats every ``T / (delay · ticks per second)`` lines.

The methods are static so ``HotReloadMethods`` can patch them while the app runs.
"""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Widget

from .slot import Curve

KNOB = Widget.knob
CURVE = 36                      # the Curve select's width: narrower than a select's own


class StrobeSettings(BaseSettings):
    """One strobe's patch: a slot per parameter, a modulation matrix row titled with the
    parameter's name: **Base**, **Amount**, **Curve** and **Bypass**, as an oscillator's. The
    button sets every Bypass of the strobe, or clears them when all are set; what it does is the
    caller's."""
    bypass_all:    Field[bool]  = Field(False, widget=Widget.button,                          label="Bypass All", description="Bypass every slot, so the panel plays this strobe; again to lift them all", newline=True)
    rate:          Field[float] = Field(0.0,  min=0.0,  max=16.0, step=1.0,  widget=KNOB, label="Base",   description="Strobes per second: 0 off, else 1, 2, 4, 8 or 16", row_label="Rate", newline=True)
    rate_amount:   Field[float] = Field(0.0,  min=-16.0, max=16.0, step=1.0, widget=KNOB, label="Amount", description="How far the source moves the rate (strobes per second)")
    rate_curve:    Field[Curve] = Field(Curve.LINEAR,                        width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    rate_bypass:   Field[bool]  = Field(False,                                            label="Bypass", description="Switch the modulation off: the rate is its base")
    width:         Field[float] = Field(0.5,  min=0.0,  max=1.0,  step=0.01, widget=KNOB, label="Base",   description="Lit part of every cycle: 0 always dark, 1 always lit", row_label="Width", newline=True)
    width_amount:  Field[float] = Field(0.0,  min=-1.0, max=1.0,  step=0.01, widget=KNOB, label="Amount", description="How far the source moves the width")
    width_curve:   Field[Curve] = Field(Curve.LINEAR,                        width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    width_bypass:  Field[bool]  = Field(False,                                            label="Bypass", description="Switch the modulation off: the width is its base")
    phase:         Field[float] = Field(0.0,  min=-0.5, max=0.5,  step=0.01, widget=KNOB, label="Base",   description="Where the cycle starts (cycles)", row_label="Phase", newline=True)
    phase_amount:  Field[float] = Field(0.0,  min=-1.0, max=1.0,  step=0.01, widget=KNOB, label="Amount", description="How far the source moves the phase (cycles)")
    phase_curve:   Field[Curve] = Field(Curve.LINEAR,                        width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    phase_bypass:  Field[bool]  = Field(False,                                            label="Bypass", description="Switch the modulation off: the phase is its base")
    delay:         Field[float] = Field(0.0,  min=-0.5, max=0.5,  step=0.005, widget=KNOB, label="Base",   description="Seconds between one line's strobe and the next: 0 all together, positive the dark runs outward", row_label="Delay", newline=True)
    delay_amount:  Field[float] = Field(0.0,  min=-0.5, max=0.5,  step=0.005, widget=KNOB, label="Amount", description="How far the source moves the delay (s per line)")
    delay_curve:   Field[Curve] = Field(Curve.LINEAR,                         width=CURVE, label="Curve",  description="How the source's magnitude is eased")
    delay_bypass:  Field[bool]  = Field(False,                                             label="Bypass", description="Switch the modulation off: the delay is its base")


class Strobe:
    """The quantized rate and the gate; see the module docstring."""

    @staticmethod
    def rate(value: float, ticks_per_second: int) -> int:
        """The strobes per second a slot's value gives: 0 below ½, else the nearest power of two
        on a log scale, at most the largest power of two within half the ticks per second (every
        other tick at a power-of-two tick rate)."""
        if value < 0.5:
            return 0
        top = 2 ** int(math.log2(max(1, int(ticks_per_second) // 2)))
        return min(2 ** max(0, round(math.log2(value))), top)

    @staticmethod
    def period(rate: int, ticks_per_second: int) -> int:
        """Ticks per cycle: whole, and nested for the powers of two at a power-of-two tick rate."""
        return max(1, round(ticks_per_second / rate))

    @staticmethod
    def gate(tick: int, line: np.ndarray, rate: int, width: float, phase: float, delay: float,
             ticks_per_second: int) -> np.ndarray:
        """On (1) or off (0) this tick for each line, ``line`` being each pixel's line count from
        the person in intervals and ``delay`` the seconds between one line and the next. Rate 0 is
        always on."""
        if rate <= 0:
            return np.ones(line.shape, dtype=np.float32)
        T = Strobe.period(rate, ticks_per_second)
        lit = int(round(min(max(width, 0.0), 1.0) * T))
        offset = int(round(phase * T)) + np.rint(delay * ticks_per_second * line).astype(np.int64)
        position = (int(tick) - offset) % T
        return np.where(position < lit, 1.0, 0.0).astype(np.float32)
