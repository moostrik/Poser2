"""Envelope — the light synth's second building block (``docs/LIGHT_SYNTH.md``, *The envelope*).

An oscillator repeats; an envelope goes up once, holds and comes down once, 0..1, eased so its ends
are smooth. Over positions it is a shape with a fixed length (the window). Over time it follows a
gate that is opened and closed from outside (presence, the push). It knows nothing of people or
colours.
"""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Widget

from .slot import Curve

Value = float | np.ndarray

CURVE = 36                      # the Curve select's width: narrower than a select's own


class WindowSettings(BaseSettings):
    """What the voice reads of the window: the taper of the envelope over distance on the pulse
    width, and presence, the envelope over time on the reaches. The reaches are the caller's.
    While the window opens or closes the taper widens to the whole window, so lines enter and
    leave thin at any resting taper, each direction eased by its own curve; the cap keeps every
    line inside the reach."""
    taper:           Field[float] = Field(0.2, min=0.0, max=1.0,  step=0.01, widget=Widget.knob, label="Taper",   description="Last part of a reach over which the lines thin out", row_label="Shape", newline=True)
    attack_seconds:  Field[float] = Field(1.0, min=0.0, max=5.0,  step=0.1,  widget=Widget.knob, label="Attack",  description="Window opens after arrival (s)")
    release_seconds: Field[float] = Field(1.5, min=0.0, max=5.0,  step=0.1,  widget=Widget.knob, label="Release", description="Window closes after leaving (s)")
    attack_curve:    Field[Curve] = Field(Curve.LINEAR, width=CURVE,                             label="Attack Curve",  description="How the taper eases from the whole window to its own value as the window opens")
    release_curve:   Field[Curve] = Field(Curve.LINEAR, width=CURVE,                             label="Release Curve", description="How the taper eases back to the whole window as the window closes")
    cap:             Field[bool]  = Field(False,                                                 label="Cap",     description="Lines never pass the reach: no overhang")


class Envelope:
    """An envelope over time (an instance, following its gate) or over positions (``over_positions``)."""

    def __init__(self) -> None:
        self._level = 0.0               # 0..1, linear; the value is this eased

    def reset(self) -> None:
        self._level = 0.0

    def update(self, gate: bool, dt: float, rise: float, fall: float) -> float:
        """Move toward 1 over ``rise`` seconds while the gate is open and toward 0 over ``fall``
        seconds while it is closed; 0 seconds is at once. A gate that closes early turns the rise
        into a fall from where it is."""
        if gate:
            self._level = 1.0 if rise <= 0.0 else min(1.0, self._level + dt / rise)
        else:
            self._level = 0.0 if fall <= 0.0 else max(0.0, self._level - dt / fall)
        return self.value

    @property
    def value(self) -> float:
        return float(Envelope.ease(self._level))

    @property
    def level(self) -> float:
        """The linear level: 0 exactly when the envelope is over."""
        return self._level

    @staticmethod
    def ease(level: Value) -> Value:
        """A linear 0..1 level eased, so both ends are smooth."""
        return 0.5 - 0.5 * np.cos(math.pi * np.clip(level, 0.0, 1.0))

    @staticmethod
    def over_positions(positions: np.ndarray, rise: Value, fall: Value, length: Value) -> np.ndarray:
        """The envelope as a shape along ``positions`` (≥ 0): up over ``rise`` from position 0, 1
        until ``length − fall``, down to 0 at ``length``, 0 beyond. ``rise``, ``fall`` and
        ``length`` are in the positions' units, each one value or one per position."""
        up = np.where(np.asarray(rise) > 0.0, positions / np.maximum(rise, 1e-12), 1.0)
        down = (length - positions) / np.maximum(fall, 1e-12)
        level = np.minimum(np.clip(up, 0.0, 1.0), np.clip(down, 0.0, 1.0))
        return np.asarray(Envelope.ease(level), dtype=np.float32)
