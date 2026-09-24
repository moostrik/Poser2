"""Tests for the waveform test layers (test_lines, test_chase, test_pulse): they draw the light
synth's pulse, so ``width`` is a fraction of the interval and ``hardness`` shapes the flanks."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from apps.white_space.light import Tick
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers import (TestLines, TestLinesSettings, TestChase, TestChaseSettings,
                                           TestPulse, TestPulseSettings)

RES = 400


def frame(time: float = 0.0) -> Frame:
    return Frame(RES, Tick(time, 1 / 30))


def runs(levels: np.ndarray) -> list[tuple[int, int]]:
    """(start, width) in pixels of every lit run of a hard output."""
    lit = levels > 0.5
    edges = np.flatnonzero(np.diff(np.concatenate(([False], lit, [False])).astype(int)))
    return [(int(s), int(e - s)) for s, e in zip(edges[::2], edges[1::2])]


class LinesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TestLinesSettings()
        self.layer = TestLines(RES, self.cfg, SimpleNamespace())
        for ch in (self.cfg.white, self.cfg.blue):
            ch.speed, ch.phase, ch.amount, ch.width, ch.hardness = 0.0, 0.0, 4, 0.25, 1.0
        self.cfg.white.level = 1.0
        self.cfg.blue.level = 0.0

    def test_hard_lines_are_off_or_full_and_the_width_of_the_interval(self) -> None:
        f = frame()
        self.layer.render(f)
        self.assertEqual(set(np.unique(f.white)), {0.0, 1.0})
        found = runs(f.white)
        self.assertEqual(len(found), 5)                                  # 4 lines, the one at 0 wraps
        for _, width in found[1:-1]:
            self.assertAlmostEqual(width, RES / 4 * 0.25, delta=1)
        self.assertEqual(float(f.blue.sum()), 0.0)                       # blue independent of white

    def test_hardness_softens_the_flanks_only(self) -> None:
        self.cfg.white.hardness = 0.3
        f = frame()
        self.layer.render(f)
        interval = RES // 4
        self.assertEqual(float(f.white[0]), 1.0)                         # a line centre stays full
        self.assertEqual(float(f.white[interval // 2]), 0.0)             # a gap centre stays off
        self.assertTrue(((f.white > 0.0) & (f.white < 1.0)).any())

    def test_lines_travel(self) -> None:
        self.cfg.white.speed = 1.0                                       # adj = speed*amount/10 = 0.4 cycles/s
        first, later = frame(0.0), frame(0.25)
        self.layer.render(first)
        self.layer.render(later)
        shift = int(round(0.4 * 0.25 * RES / 4))                         # 0.1 cycle of a 100-pixel interval
        np.testing.assert_array_equal(np.roll(first.white, shift), later.white)


class ChaseTest(unittest.TestCase):
    def test_width_half_and_hardness_zero_is_a_sine(self) -> None:
        cfg = TestChaseSettings()
        cfg.white.speed, cfg.white.phase, cfg.white.amount = 0.0, 0.0, 3
        cfg.white.width, cfg.white.hardness, cfg.white.level = 0.5, 0.0, 1.0
        cfg.blue.level = 0.0
        f = frame()
        TestChase(RES, cfg, SimpleNamespace()).render(f)
        cycle = np.arange(RES) * 3 / RES
        np.testing.assert_allclose(f.white, 0.5 + 0.5 * np.cos(math.tau * cycle), atol=1e-5)

    def test_hardness_one_is_a_square(self) -> None:
        cfg = TestChaseSettings()
        cfg.white.speed, cfg.white.phase, cfg.white.amount = 0.0, 0.0, 3
        cfg.white.width, cfg.white.hardness, cfg.white.level = 0.5, 1.0, 1.0
        cfg.blue.level = 0.0
        f = frame()
        TestChase(RES, cfg, SimpleNamespace()).render(f)
        self.assertEqual(set(np.unique(f.white)), {0.0, 1.0})
        self.assertAlmostEqual(float(f.white.mean()), 0.5, delta=0.02)


class PulseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TestPulseSettings()
        self.cfg.white.speed, self.cfg.white.phase, self.cfg.white.width = 1.0, 0.0, 0.5
        self.cfg.white.level = 0.8
        self.cfg.blue.level = 0.0
        self.layer = TestPulse(RES, self.cfg, SimpleNamespace())

    def test_hard_pulse_strobes_the_whole_ring(self) -> None:
        self.cfg.white.hardness = 1.0
        on, off = frame(0.0), frame(0.5)
        self.layer.render(on)
        self.layer.render(off)
        np.testing.assert_allclose(on.white, 0.8, atol=1e-6)
        self.assertEqual(float(off.white.sum()), 0.0)
        self.assertEqual(float(on.blue.sum()), 0.0)

    def test_soft_pulse_is_a_sine(self) -> None:
        self.cfg.white.hardness = 0.0
        f = frame(0.25)                                                  # a quarter cycle: halfway down
        self.layer.render(f)
        np.testing.assert_allclose(f.white, 0.4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
