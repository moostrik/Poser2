"""Tests for the Part-4 show layers: sound_light (levels → blue lamps, jitter window,
stale fallback) and the pose_instrument placeholder's sync-fill arc geometry."""

import math
import unittest
from time import monotonic
from types import SimpleNamespace

import numpy as np

from modules.board import SoundLevels
from modules.pose import features

from apps.white_space.light import Tick, BarLightId
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers.low.sound_light import SoundLight, SoundLightSettings, SoundFallback
from apps.white_space.light.layers.high.pose_instrument import (
    PoseInstrument, PoseInstrumentSettings, _shorter_arc)

RES = 200
HALF = RES // 2


def frame(time: float = 0.0) -> Frame:
    return Frame(RES, Tick(time, 1 / 30))


# -- sound_light -----------------------------------------------------------------

class SoundBoard(SimpleNamespace):
    def get_sound_levels(self) -> SoundLevels:
        return self.levels


class SoundLightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SoundLightSettings()
        self.board = SoundBoard(levels=SoundLevels())
        self.layer = SoundLight(RES, self.cfg, self.board)

    def _fresh(self, left: float, right: float) -> None:
        self.board.levels = SoundLevels(left=left, right=right, timestamp=monotonic())

    def test_levels_land_on_the_blue_lamps(self) -> None:
        self.cfg.smoothing_frames = 0
        self._fresh(0.8, 0.3)
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.bar_lights[BarLightId.LEFT_BLUE],  0.8, places=5)
        self.assertAlmostEqual(f.bar_lights[BarLightId.RIGHT_BLUE], 0.3, places=5)
        self.assertEqual(float(f.bar_lights[BarLightId.FRONT_WHITE]), 0.0)   # whites untouched
        self.assertEqual(float(f.bar_lights[BarLightId.BACK_WHITE]),  0.0)
        self.assertEqual(float(f.light_img.sum()), 0.0)                       # no pixels written

    def test_gain_scales(self) -> None:
        self.cfg.smoothing_frames = 0
        self.cfg.gain = 0.5
        self._fresh(0.8, 0.4)
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.bar_lights[BarLightId.LEFT_BLUE], 0.4, places=5)

    def test_smoothing_window_bridges_jitter(self) -> None:
        self.cfg.smoothing_frames = 2
        self._fresh(1.0, 1.0)
        self.layer.render(frame())
        self._fresh(0.0, 0.0)
        f = frame()
        self.layer.render(f)                                  # average of the last 2 frames
        self.assertAlmostEqual(f.bar_lights[BarLightId.LEFT_BLUE], 0.5, places=5)

    def test_stale_input_off_fallback(self) -> None:
        self.board.levels = SoundLevels(left=1.0, right=1.0, timestamp=monotonic() - 60.0)
        f = frame()
        self.layer.render(f)
        self.assertEqual(float(f.bar_lights.sum()), 0.0)      # never freezes at a stuck level

    def test_stale_input_pulse_fallback(self) -> None:
        self.cfg.fallback = SoundFallback.PULSE
        self.board.levels = SoundLevels()                      # never received
        f = frame(time=1.25)                                   # quarter period of the 0.2 Hz pulse
        self.layer.render(f)
        left, right = f.bar_lights[BarLightId.LEFT_BLUE], f.bar_lights[BarLightId.RIGHT_BLUE]
        self.assertGreater(left, 0.0)
        self.assertAlmostEqual(left, right, places=5)
        self.assertLessEqual(left, self.cfg.fallback_level + 1e-6)


# -- wind_down ---------------------------------------------------------------------

class WindDownTest(unittest.TestCase):
    """The dying wall: both white lamps fading over spin_down_seconds, monotonic, no
    pixels; progress is the readout."""

    def setUp(self) -> None:
        from apps.white_space.light.layers.low.wind_down import WindDown, WindDownSettings
        self.cfg = WindDownSettings()
        self.layer = WindDown(RES, self.cfg, board=None)

    def _wall(self) -> float:
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.bar_lights[BarLightId.FRONT_WHITE],
                               f.bar_lights[BarLightId.BACK_WHITE], places=6)   # both whites alike
        self.assertEqual(float(f.bar_lights[BarLightId.LEFT_BLUE]), 0.0)
        self.assertEqual(float(f.light_img.sum()), 0.0)                        # no pixels written
        return float(f.bar_lights[BarLightId.FRONT_WHITE])

    def test_timed_fade_reaches_zero(self) -> None:
        self.cfg.spin_down_seconds = 1.0
        levels = [self._wall() for _ in range(45)]         # 1.5 s of 1/30 ticks
        self.assertGreater(levels[0], 0.9)                 # starts at the full wall
        self.assertEqual(levels[-1], 0.0)                  # gone by time alone
        self.assertEqual(levels, sorted(levels, reverse=True))   # monotonic, no snap
        self.assertEqual(self.cfg.progress, 1.0)

    def test_level_scales_the_wall(self) -> None:
        self.cfg.level = 0.5
        self.assertAlmostEqual(self._wall(), 0.5, places=3)

    def test_reset_restarts_at_the_full_wall(self) -> None:
        self.cfg.spin_down_seconds = 1.0
        for _ in range(60):
            self._wall()
        self.assertEqual(self.cfg.progress, 1.0)
        self.layer.reset()
        self.assertEqual(self.cfg.progress, 0.0)
        self.assertGreater(self._wall(), 0.9)              # the wall is back at full


# -- pose_instrument (placeholder) --------------------------------------------------

class FakePose:
    """A fake pose frame: ``pose[FeatureType]`` over a prepared feature dict."""

    def __init__(self, by_type: dict) -> None:
        self._by_type = by_type

    def __getitem__(self, feature_type):
        return self._by_type[feature_type]


def _pose(azimuth_pos: float, sims: dict[int, float] | None = None) -> FakePose:
    """A fake pose at strip position ``azimuth_pos`` (0..1) with pairwise sims."""
    angles = np.full(64, np.nan)
    sim_values = np.full(16, np.nan)
    for j, v in (sims or {}).items():
        sim_values[j] = v
    return FakePose({
        features.Azimuth: SimpleNamespace(value=azimuth_pos * math.tau),
        features.BBox: {features.BBoxElement.height: 1.0},
        features.Angles: SimpleNamespace(values=angles),
        features.Similarity: SimpleNamespace(values=sim_values),
    })


class InstrumentBoard(SimpleNamespace):
    def get_tracklets(self):
        return self.tracklets

    def get_frames(self, stage: int):
        return self.frames


class PoseInstrumentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PoseInstrumentSettings()
        self.board = InstrumentBoard(tracklets={}, frames={})
        self.layer = PoseInstrument(RES, self.cfg, self.board, pose_stage=4)

    def _people(self, poses: dict[int, SimpleNamespace]) -> None:
        self.board.frames = poses
        self.board.tracklets = {id: SimpleNamespace(is_active=True) for id in poses}

    def test_shorter_arc_geometry(self) -> None:
        centre, width = _shorter_arc(0.1, 0.3)
        self.assertAlmostEqual(centre, 0.2)
        self.assertAlmostEqual(width, 0.2)
        centre, width = _shorter_arc(0.95, 0.05)              # wraps through 0
        self.assertAlmostEqual(centre, 0.0)
        self.assertAlmostEqual(width, 0.1)

    def test_band_and_marker_per_participant(self) -> None:
        self._people({0: _pose(0.5)})
        f = frame()
        self.layer.render(f)
        self.assertGreater(f.white[HALF], 0.0)                # band at the azimuth
        self.assertGreater(f.blue[HALF], 0.0)                 # the blue-light spot

    def test_sync_fill_lights_the_arc_between_matched_pair(self) -> None:
        self._people({0: _pose(0.1, sims={1: 0.9}), 1: _pose(0.3, sims={0: 0.9})})
        f = frame()
        self.layer.render(f)
        mid = int(0.2 * RES)                                  # arc midpoint, clear of both bands
        self.assertAlmostEqual(f.white[mid], self.cfg.fill_level * 0.9, places=4)

    def test_sync_fill_respects_the_threshold(self) -> None:
        self._people({0: _pose(0.1, sims={1: 0.5}), 1: _pose(0.3, sims={0: 0.5})})
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.white[int(0.2 * RES)], 0.0, places=5)

    def test_sync_fill_takes_the_shorter_arc_across_the_wrap(self) -> None:
        self._people({0: _pose(0.95, sims={1: 0.9}), 1: _pose(0.05, sims={0: 0.9})})
        f = frame()
        self.layer.render(f)
        self.assertGreater(f.white[0], 0.0)                   # through the wrap
        self.assertAlmostEqual(f.white[HALF], 0.0, places=5)  # never the long way round

    def test_inactive_participants_are_ignored(self) -> None:
        self.board.frames = {0: _pose(0.5)}
        self.board.tracklets = {0: SimpleNamespace(is_active=False)}
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(float(f.white.sum()) + float(f.blue.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
