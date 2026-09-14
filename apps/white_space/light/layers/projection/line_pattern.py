"""LinePattern — the pose instrument's line math: a thresholded spatial LFO and the 1-D
morphology that keeps a pattern legible on the projection (see ``docs/POSE_INSTRUMENT.md``).

Everything is on/off: a pattern is a boolean mask, one value per pixel. Lengths are in pixels.
All operations are static methods on the class, so ``HotReloadMethods`` can patch them while the
app runs (it re-executes class bodies, not module-level functions).
"""

import math
from enum import IntEnum, auto

import numpy as np


class Waveform(IntEnum):
    """The wave thresholded into lines; compared through ``int``, since a reload redefines the class."""
    SINE     = 0
    TRIANGLE = auto()
    SAW      = auto()


class LinePattern:
    """Pure numpy line-pattern primitives; see the module docstring."""

    @staticmethod
    def lines(x: np.ndarray, interval: float, duty: float, harmonic: float, order: int,
              harmonic_phase: float, phase: float) -> np.ndarray:
        """The lines at distances ``x`` (px) from the person: on where the LFO is at or above its
        threshold.

        ``u = x / interval + phase``; the LFO is ``cos 2πu`` mixed by ``harmonic`` with
        ``cos 2π(order·u + harmonic_phase)``, and a pixel is lit where it reaches ``cos(π·duty)``. With
        no harmonic, a line of ``duty × interval`` px is centred on every whole ``u``; duty 0 is
        dark and 1 is solid.
        """
        if duty <= 0.0:
            return np.zeros(x.shape, dtype=bool)
        if duty >= 1.0:
            return np.ones(x.shape, dtype=bool)
        u = x / interval + phase
        lfo = np.cos(math.tau * u)
        if harmonic > 0.0:
            lfo = (1.0 - harmonic) * lfo + harmonic * np.cos(math.tau * (order * u + harmonic_phase))
        return lfo >= math.cos(math.pi * duty)

    @staticmethod
    def window_count(mask: np.ndarray, before: int, after: int) -> np.ndarray:
        """Lit pixels in the circular window ``[i − before, i + after]`` for every pixel ``i``."""
        n = mask.size
        padded = np.concatenate((mask[n - before:], mask, mask[:after]))
        cumulative = np.concatenate(([0], np.cumsum(padded, dtype=np.int32)))
        width = before + after + 1
        return cumulative[width:width + n] - cumulative[:n]

    @staticmethod
    def dilate(mask: np.ndarray, before: int, after: int) -> np.ndarray:
        """Lit where any pixel of the circular window ``[i − before, i + after]`` is lit."""
        if before <= 0 and after <= 0:
            return mask.copy()
        return LinePattern.window_count(mask, before, after) > 0

    @staticmethod
    def erode(mask: np.ndarray, before: int, after: int) -> np.ndarray:
        """Lit where every pixel of the circular window ``[i − before, i + after]`` is lit."""
        if before <= 0 and after <= 0:
            return mask.copy()
        return LinePattern.window_count(mask, before, after) == before + after + 1

    @staticmethod
    def fill_gaps(mask: np.ndarray, min_px: int) -> np.ndarray:
        """Close every gap narrower than ``min_px``; wider gaps and all lines keep their exact pixels."""
        if min_px <= 1 or mask.all() or not mask.any():
            return mask.copy()
        before = (min_px - 1) // 2
        after = min_px - 1 - before
        return LinePattern.erode(LinePattern.dilate(mask, before, after), after, before)

    @staticmethod
    def remove_slivers(mask: np.ndarray, min_px: int) -> np.ndarray:
        """Drop every line narrower than ``min_px``; wider lines and all gaps keep their exact pixels."""
        if min_px <= 1 or mask.all() or not mask.any():
            return mask.copy()
        before = (min_px - 1) // 2
        after = min_px - 1 - before
        return LinePattern.dilate(LinePattern.erode(mask, before, after), after, before)

    @staticmethod
    def legible(mask: np.ndarray, min_px: int) -> np.ndarray:
        """No line and no gap narrower than ``min_px``: gaps are filled first, then slivers dropped
        (dropping a line only merges gaps, so no narrow gap reappears). A mask that already
        satisfies the limit comes back unchanged."""
        return LinePattern.remove_slivers(LinePattern.fill_gaps(mask, min_px), min_px)
