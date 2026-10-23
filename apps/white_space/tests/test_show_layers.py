"""Tests for the beam show layers: beam_blue_sound (levels → blue lamps, jitter window, stale
fallback) and wind_down (the dying wall, plain and under the sync strobe). The pose instrument's
are in test_pose_instrument.py."""

import unittest
from time import monotonic
from types import SimpleNamespace

from modules.board import SoundLevels

from apps.white_space.light import Tick, BeamLightId
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers import SyncStrobeMode
from apps.white_space.light.layers.beam.blue_sound import BeamBlueSound, BeamBlueSoundSettings, SoundFallback

RES = 200


def frame(time: float = 0.0) -> Frame:
    return Frame(RES, Tick(time, 1 / 30))


# -- beam_blue_sound -----------------------------------------------------------------

class SoundBoard(SimpleNamespace):
    def get_sound_levels(self) -> SoundLevels:
        return self.levels


class BeamBlueSoundTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = BeamBlueSoundSettings()
        self.board = SoundBoard(levels=SoundLevels())
        self.layer = BeamBlueSound(RES, self.cfg, self.board)

    def _fresh(self, left: float, right: float) -> None:
        self.board.levels = SoundLevels(left=left, right=right, timestamp=monotonic())

    def test_levels_land_on_the_blue_lamps(self) -> None:
        self.cfg.smoothing_frames = 0
        self._fresh(0.8, 0.3)
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE],  0.8, places=5)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.RIGHT_BLUE], 0.3, places=5)
        self.assertEqual(float(f.beam_lights[BeamLightId.FRONT_WHITE]), 0.0)   # whites untouched
        self.assertEqual(float(f.beam_lights[BeamLightId.BACK_WHITE]),  0.0)
        self.assertEqual(float(f.light_img.sum()), 0.0)                       # no pixels written

    def test_gain_scales(self) -> None:
        self.cfg.smoothing_frames = 0
        self.cfg.gain = 0.5
        self._fresh(0.8, 0.4)
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE], 0.4, places=5)

    def test_smoothing_window_bridges_jitter(self) -> None:
        self.cfg.smoothing_frames = 2
        self._fresh(1.0, 1.0)
        self.layer.render(frame())
        self._fresh(0.0, 0.0)
        f = frame()
        self.layer.render(f)                                  # average of the last 2 frames
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE], 0.5, places=5)

    def test_stale_input_off_fallback(self) -> None:
        self.board.levels = SoundLevels(left=1.0, right=1.0, timestamp=monotonic() - 60.0)
        f = frame()
        self.layer.render(f)
        self.assertEqual(float(f.beam_lights.sum()), 0.0)      # never freezes at a stuck level

    def test_stale_input_pulse_fallback(self) -> None:
        self.cfg.fallback = SoundFallback.PULSE
        self.board.levels = SoundLevels()                      # never received
        f = frame(time=1.25)                                   # quarter period of the 0.2 Hz pulse
        self.layer.render(f)
        left, right = f.beam_lights[BeamLightId.LEFT_BLUE], f.beam_lights[BeamLightId.RIGHT_BLUE]
        self.assertGreater(left, 0.0)
        self.assertAlmostEqual(left, right, places=5)
        self.assertLessEqual(left, self.cfg.fallback_level + 1e-6)


# -- wind_down ---------------------------------------------------------------------

class BeamWindDownTest(unittest.TestCase):
    """The dying wall: both white lamps fading over spin_down_seconds, monotonic, no
    pixels; progress is the readout."""

    def setUp(self) -> None:
        from apps.white_space.light.layers.beam.wind_down import BeamWindDown, BeamWindDownSettings
        self.cfg = BeamWindDownSettings()
        self.layer = BeamWindDown(RES, self.cfg, board=None)

    def _wall(self) -> float:
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.FRONT_WHITE],
                               f.beam_lights[BeamLightId.BACK_WHITE], places=6)   # both whites alike
        self.assertEqual(float(f.beam_lights[BeamLightId.LEFT_BLUE]), 0.0)
        self.assertEqual(float(f.light_img.sum()), 0.0)                        # no pixels written
        return float(f.beam_lights[BeamLightId.FRONT_WHITE])

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

    def test_the_plain_wall_carries_no_sync(self) -> None:
        f = frame()
        self.layer.render(f)
        self.assertEqual(f.sync, 0.0)


class BeamSyncWindDownTest(unittest.TestCase):
    """The strobing wall (S11): the same fade under the sync strobe — on the strobes' shared tick
    grid at the top rung (every other tick at 32 fps), a strobe frame swaps the wall to the blue
    lamps (INVERT) or blanks it (BLACK); it carries no sync — /global/sync is the instrument's."""

    def setUp(self) -> None:
        from apps.white_space.light.layers.beam.wind_down import BeamWindDown, BeamWindDownSettings
        from apps.white_space.light.layers.projection.pose_instrument import SyncStrobeSettings
        self.cfg = BeamWindDownSettings()
        self.strobe = SyncStrobeSettings()
        self.layer = BeamWindDown(RES, self.cfg, board=None, strobe=self.strobe)
        self.index = 0

    def _next(self) -> Frame:
        """Render the next tick at 32 fps; tick 0 is a strobe frame."""
        f = Frame(RES, Tick(self.index / 32, 1 / 32, 1 / 32, self.index))
        self.index += 1
        self.layer.render(f)
        self.assertEqual(float(f.light_img.sum()), 0.0)                        # no pixels written
        return f

    def test_invert_swaps_the_wall_to_the_blue_lamps_on_strobe_ticks(self) -> None:
        f = self._next()                                                        # tick 0: a strobe frame
        self.assertEqual(float(f.beam_lights[BeamLightId.FRONT_WHITE]), 0.0)
        self.assertEqual(float(f.beam_lights[BeamLightId.BACK_WHITE]), 0.0)
        self.assertGreater(float(f.beam_lights[BeamLightId.LEFT_BLUE]), 0.9)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE], f.beam_lights[BeamLightId.RIGHT_BLUE], places=6)
        f = self._next()                                                        # tick 1: the wall
        self.assertGreater(float(f.beam_lights[BeamLightId.FRONT_WHITE]), 0.9)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.FRONT_WHITE], f.beam_lights[BeamLightId.BACK_WHITE], places=6)
        self.assertEqual(float(f.beam_lights[BeamLightId.LEFT_BLUE]), 0.0)
        self.assertEqual(float(f.beam_lights[BeamLightId.RIGHT_BLUE]), 0.0)

    def test_black_blanks_the_wall_on_strobe_ticks(self) -> None:
        self.strobe.mode = SyncStrobeMode.BLACK
        self.assertEqual(float(self._next().beam_lights.sum()), 0.0)            # tick 0: dark
        self.assertGreater(float(self._next().beam_lights[BeamLightId.FRONT_WHITE]), 0.9)

    def test_the_fade_advances_on_strobe_ticks_too(self) -> None:
        from apps.white_space.light.layers.beam.wind_down import BeamWindDown, BeamWindDownSettings
        plain_cfg = BeamWindDownSettings()
        plain = BeamWindDown(RES, plain_cfg, board=None)
        self.cfg.spin_down_seconds = plain_cfg.spin_down_seconds = 1.0
        for i in range(10):
            self._next()
            plain.render(Frame(RES, Tick(i / 32, 1 / 32, 1 / 32, i)))
        self.assertAlmostEqual(self.cfg.progress, plain_cfg.progress, places=6)

    def test_the_wall_fades_through_the_strobe_and_carries_no_sync(self) -> None:
        self.cfg.spin_down_seconds = 1.0
        frames = [self._next() for _ in range(48)]                              # 1.5 s
        self.assertEqual(self.cfg.progress, 1.0)
        self.assertEqual({f.sync for f in frames}, {0.0})                       # the sound's sync is the instrument's
        whites = [float(f.beam_lights[BeamLightId.FRONT_WHITE]) for f in frames[1::2]]   # the wall frames
        self.assertEqual(whites, sorted(whites, reverse=True))                  # fading through the strobe
        self.assertEqual(whites[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
