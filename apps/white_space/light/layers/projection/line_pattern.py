"""LinePattern — the pose instrument's line math: a spatial oscillator thresholded into lines,
and the 1-D morphology that keeps a pattern visible on the projection (see
``docs/POSE_INSTRUMENT.md``, *The pattern*).

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
    def lines(x: np.ndarray, interval: float, waveform: int, fundamental: float, harmonic: float,
              cutoff: int, overtone_phase: float, phase: float) -> np.ndarray:
        """The lines at distances ``x`` (px) from the person: on where the wave is at or above
        the level.

        ``u = x / interval − phase`` (a positive phase moves the lines outward). The wave is the
        filter, the two drawbars its weights:
        ``(fundamental · w(u) + harmonic · w(cutoff · u + overtone_phase)) / (fundamental + harmonic)``
        with ``w`` the waveform, crest 1 at every whole ``u`` and trough −1; a pixel is lit where
        the wave reaches ``1 − (fundamental + harmonic)``. Both drawbars in is dark, both out is
        solid. One drawbar alone lights at most half the interval: the fundamental a line at every
        whole ``u``, ``acos(1 − fundamental) / π`` of the interval wide for the sine; the harmonic
        ``cutoff`` sub-lines per interval.
        """
        registration = fundamental + harmonic
        if registration <= 0.0:
            return np.zeros(x.shape, dtype=bool)
        if registration >= 2.0:
            return np.ones(x.shape, dtype=bool)
        u = x / interval - phase
        wave = fundamental * LinePattern.wave(u, waveform)
        if harmonic > 0.0:
            wave = wave + harmonic * LinePattern.wave(cutoff * u + overtone_phase, waveform)
        return wave / registration >= 1.0 - registration

    @staticmethod
    def wave(u: np.ndarray, waveform: int) -> np.ndarray:
        """The waveform over ``u`` in cycles: −1..1, crest at every whole ``u``. The sine and the
        triangle fall both ways from the crest; the saw falls outward only."""
        if waveform == int(Waveform.TRIANGLE):
            return 1.0 - 4.0 * np.abs((u + 0.5) % 1.0 - 0.5)
        if waveform == int(Waveform.SAW):
            return 1.0 - 2.0 * (u % 1.0)
        return np.cos(math.tau * u)

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
    def core(mask: np.ndarray, fraction: float, min_px: int) -> np.ndarray:
        """The central ``fraction`` of every run of ``mask`` (linear, no wrap), for the tint. A
        core is never narrower than ``min_px``, and a run whose rims would be narrower than
        ``min_px`` is taken whole, so nothing under the limit is left either side."""
        if fraction <= 0.0 or not mask.any():
            return np.zeros(mask.shape, dtype=bool)
        if fraction >= 1.0:
            return mask.copy()
        edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
        starts, ends = edges[::2], edges[1::2]
        lengths = ends - starts
        cores = np.minimum(lengths, np.maximum(np.rint(lengths * fraction).astype(np.int64), min_px))
        cores = np.where((lengths - cores) // 2 < min_px, lengths, cores)
        core_starts = starts + (lengths - cores) // 2
        marks = np.zeros(mask.size + 1, dtype=np.int32)
        np.add.at(marks, core_starts, 1)
        np.add.at(marks, core_starts + cores, -1)
        return np.cumsum(marks[:-1]) > 0

    @staticmethod
    def visible(mask: np.ndarray, min_px: int) -> np.ndarray:
        """No line and no gap narrower than ``min_px``, the visual limit: gaps are filled first,
        then slivers dropped (dropping a line only merges gaps, so no narrow gap reappears). A
        mask that already satisfies the limit comes back unchanged."""
        return LinePattern.remove_slivers(LinePattern.fill_gaps(mask, min_px), min_px)
