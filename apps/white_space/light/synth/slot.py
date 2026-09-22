"""Slot — the light synth's modulation slot (``docs/LIGHT_SYNTH.md``, *Modulation*).

Every parameter has one, a modulation matrix row: the parameter's own value (its base), what is connected
to it (the source), how far and which way the source moves it (the amount), the source's curve,
and a bypass that switches the modulation off. A source is one value per tick or one per position:
0..1, or −1..1 for an LFO. The slot adds no smoothing and no steps; a curve is smooth and keeps the
source's sign.
"""

from enum import IntEnum, auto

import numpy as np
import pytweening

Value = float | np.ndarray


class Curve(IntEnum):
    """How a source's magnitude is eased before it moves the parameter; the sign is kept. The curves
    are pytweening's, every family in its three forms: in (little at first), out (much at first)
    and in-out. Back and Elastic overshoot on the way; Bounce turns back on itself."""
    LINEAR              = 0
    EASE_IN_QUAD        = auto()
    EASE_OUT_QUAD       = auto()
    EASE_IN_OUT_QUAD    = auto()
    EASE_IN_CUBIC       = auto()
    EASE_OUT_CUBIC      = auto()
    EASE_IN_OUT_CUBIC   = auto()
    EASE_IN_QUART       = auto()
    EASE_OUT_QUART      = auto()
    EASE_IN_OUT_QUART   = auto()
    EASE_IN_QUINT       = auto()
    EASE_OUT_QUINT      = auto()
    EASE_IN_OUT_QUINT   = auto()
    EASE_IN_SINE        = auto()
    EASE_OUT_SINE       = auto()
    EASE_IN_OUT_SINE    = auto()
    EASE_IN_EXPO        = auto()
    EASE_OUT_EXPO       = auto()
    EASE_IN_OUT_EXPO    = auto()
    EASE_IN_CIRC        = auto()
    EASE_OUT_CIRC       = auto()
    EASE_IN_OUT_CIRC    = auto()
    EASE_IN_BACK        = auto()
    EASE_OUT_BACK       = auto()
    EASE_IN_OUT_BACK    = auto()
    EASE_IN_ELASTIC     = auto()
    EASE_OUT_ELASTIC    = auto()
    EASE_IN_OUT_ELASTIC = auto()
    EASE_IN_BOUNCE      = auto()
    EASE_OUT_BOUNCE     = auto()
    EASE_IN_OUT_BOUNCE  = auto()


_GRID = np.linspace(0.0, 1.0, 1025)     # where a curve is sampled; read between with np.interp


class Slot:
    """``parameter = base + amount × source``; static methods so ``HotReloadMethods`` can patch them."""

    @staticmethod
    def bypassed(bypass: bool, source: Value) -> Value:
        """The source, or nothing while the modulation is bypassed: bypassed, a parameter is exactly
        its base, and its amount is left as it is for when the bypass is lifted."""
        return 0.0 if bypass else source

    _tables: dict[int, np.ndarray] = {}     # a curve sampled over _GRID, built on first use

    @staticmethod
    def curve(source: Value, curve: int) -> Value:
        """The source eased by its curve, its sign kept, so a bipolar source stays symmetric; 0
        and ±1 are left where they are. pytweening's functions take one number and a source may
        be one value per position, so a curve is sampled once into a table and read between."""
        if curve == int(Curve.LINEAR):
            return source
        table = Slot._tables.get(int(curve))
        if table is None:
            table = Slot._tables[int(curve)] = Slot._sample(int(curve))
        magnitude = np.clip(np.abs(source), 0.0, 1.0)
        return np.sign(source) * np.interp(magnitude, _GRID, table)

    @staticmethod
    def _sample(curve: int) -> np.ndarray:
        """pytweening's function of a curve over the grid: ``EASE_IN_OUT_QUAD`` → ``easeInOutQuad``."""
        words = Curve(curve).name.lower().split("_")
        function = getattr(pytweening, words[0] + "".join(word.capitalize() for word in words[1:]))
        table = np.array([function(float(t)) for t in _GRID])
        table[0], table[-1] = 0.0, 1.0      # pinned: pytweening's elastic ends a hair past 1
        return table

    @staticmethod
    def modulate(base: float, amount: float, source: Value) -> Value:
        """The amount in the parameter's own unit."""
        return base + amount * source

    @staticmethod
    def modulate_octaves(base: float, amount: float, source: Value) -> Value:
        """The amount in octaves, for the interval: a doubling looks the same size anywhere."""
        return base * 2.0 ** (amount * source)

    @staticmethod
    def unit(value: Value) -> Value:
        """The end of a parameter that lives in 0..1 (pulse width, hardness): a stop, not a step."""
        return np.clip(value, 0.0, 1.0)
