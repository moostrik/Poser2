"""Slot — the light synth's modulation slot (``docs/LIGHT_SYNTH.md``, *Modulation*).

Every input has one, a modulation matrix row: the input's own value (its base), what is connected
to it (the source), how far and which way the source moves it (the amount), the source's curve,
and a bypass that switches the modulation off. A source is one value per tick or one per position:
0..1, or −1..1 for an LFO. The slot adds no smoothing and no steps; a curve is smooth and keeps the
source's sign.
"""

from enum import IntEnum, auto

import numpy as np

Value = float | np.ndarray


class Curve(IntEnum):
    """How a source's magnitude is eased before it moves the input; the sign is kept."""
    LINEAR      = 0
    EASE_IN     = auto()    # little at first, much at the end
    EASE_OUT    = auto()    # much at first, little at the end
    EASE_IN_OUT = auto()    # both


class Slot:
    """``input = base + amount × source``; static methods so ``HotReloadMethods`` can patch them."""

    @staticmethod
    def bypassed(bypass: bool, source: Value) -> Value:
        """The source, or nothing while the modulation is bypassed: bypassed, an input is exactly
        its base, and its amount is left as it is for when the bypass is lifted."""
        return 0.0 if bypass else source

    @staticmethod
    def curve(source: Value, curve: int) -> Value:
        """The source eased by its curve, its sign kept, so a bipolar source stays symmetric; 0
        and ±1 are left where they are."""
        if curve == int(Curve.LINEAR):
            return source
        magnitude = np.clip(np.abs(source), 0.0, 1.0)
        if curve == int(Curve.EASE_IN):
            eased = magnitude * magnitude
        elif curve == int(Curve.EASE_OUT):
            eased = 1.0 - (1.0 - magnitude) ** 2
        else:
            eased = 0.5 - 0.5 * np.cos(np.pi * magnitude)
        return np.sign(source) * eased

    @staticmethod
    def modulate(base: float, amount: float, source: Value) -> Value:
        """The amount in the input's own unit."""
        return base + amount * source

    @staticmethod
    def modulate_octaves(base: float, amount: float, source: Value) -> Value:
        """The amount in octaves, for the interval: a doubling looks the same size anywhere."""
        return base * 2.0 ** (amount * source)

    @staticmethod
    def unit(value: Value) -> Value:
        """The end of an input that lives in 0..1 (pulse width, hardness): a stop, not a step."""
        return np.clip(value, 0.0, 1.0)
