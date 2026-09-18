"""Slot — the light synth's modulation slot (``docs/LIGHT_SYNTH.md``, *Modulation*).

Every input has one: its base, what is connected to it, and how far that moves it. The slot adds
no smoothing, no curve and no steps; a source that needs shaping is shaped in the source. A source
is one value per tick or one per position: 0..1, or −1..1 for an LFO.
"""

import numpy as np

Value = float | np.ndarray


class Slot:
    """``input = base + amount × source``; static methods so ``HotReloadMethods`` can patch them."""

    @staticmethod
    def held(hold: bool, source: Value) -> Value:
        """The source, or nothing while the input is held: held, an input is exactly its base,
        and its amount is left as it is for when the hold is let go."""
        return 0.0 if hold else source

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
