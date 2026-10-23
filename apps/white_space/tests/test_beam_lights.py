"""The beam lights on the light side and in the render's model: the named writes of a
BeamLayer, and the projection of the four lights as lines on the walls."""

import math
import unittest

import numpy as np

from apps.white_space.light import Frame, Tick, BeamLightId, BEAM_LIGHT_HEADINGS, BUFFER_DTYPE
from apps.white_space.light.layers import BeamTest, BeamTestSettings
from modules.board import Flash, FlashStoreMixin
from apps.white_space.render.layers.beam_light_projection import beam_profile, project_beam_lights, paint_flashes

R = 360   # one pixel per degree keeps the expected indices readable


def _frame() -> Frame:
    return Frame(R, Tick(0.0, 1.0 / 30.0))


class LowLayerWritesTest(unittest.TestCase):

    def test_named_writes_land_on_the_bar_lights_and_nowhere_else(self) -> None:
        cfg = BeamTestSettings()
        cfg.front_white, cfg.back_white = 0.1, 0.2
        cfg.left_blue,   cfg.right_blue = 0.3, 0.4
        f = _frame()
        BeamTest(R, cfg, board=None).render(f)
        np.testing.assert_allclose(f.beam_lights, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
        self.assertEqual(float(f.light_img.sum()), 0.0)

    def test_headings_follow_the_firmware_sampling_offsets(self) -> None:
        # white 2 at +1800, blue 1 (the blue[0] slot, LEFT) at +2700, blue 2 (RIGHT) at +900 of 3600.
        self.assertAlmostEqual(BEAM_LIGHT_HEADINGS[BeamLightId.FRONT_WHITE], 0.0)
        self.assertAlmostEqual(BEAM_LIGHT_HEADINGS[BeamLightId.BACK_WHITE],  math.pi)
        self.assertAlmostEqual(BEAM_LIGHT_HEADINGS[BeamLightId.LEFT_BLUE],  -math.pi / 2)
        self.assertAlmostEqual(BEAM_LIGHT_HEADINGS[BeamLightId.RIGHT_BLUE],  math.pi / 2)


class BeamProfileTest(unittest.TestCase):
    """The line's shape: a solid core of `beam`, a soft falloff of `blur` on each side."""

    def test_core_is_solid_and_the_edges_fade_to_zero(self) -> None:
        profile = beam_profile(math.radians(20.0), math.radians(10.0), R)   # 20 px core, 10 px each side
        self.assertEqual(len(profile), 2 * (10 + 10) + 1)
        centre = len(profile) // 2
        np.testing.assert_allclose(profile[centre - 10:centre + 11], 1.0, rtol=1e-9)   # the core
        self.assertAlmostEqual(profile[0], 0.0, places=9)                              # both ends dark
        self.assertAlmostEqual(profile[-1], 0.0, places=9)
        self.assertAlmostEqual(profile[centre - 15], 0.5, places=6)                    # halfway down the fade

    def test_profile_is_symmetric(self) -> None:
        profile = beam_profile(math.radians(9.0), math.radians(7.0), R)
        np.testing.assert_allclose(profile, profile[::-1], rtol=1e-9)

    def test_falloff_is_monotonic_and_smooth_at_both_joins(self) -> None:
        profile = beam_profile(math.radians(20.0), math.radians(20.0), R)
        centre = len(profile) // 2
        edge = profile[centre:]                                            # core out to darkness
        self.assertTrue(np.all(np.diff(edge) <= 1e-12))                    # never rises
        # A raised cosine flattens into both the core and the darkness — no kink at either join.
        self.assertLess(abs(edge[10] - edge[11]), abs(edge[19] - edge[20]))
        self.assertLess(abs(edge[-2] - edge[-1]), abs(edge[19] - edge[20]))

    def test_zero_blur_is_a_hard_edge(self) -> None:
        profile = beam_profile(math.radians(20.0), 0.0, R)
        self.assertEqual(len(profile), 21)
        np.testing.assert_allclose(profile, 1.0, rtol=1e-9)

    def test_zero_width_is_a_single_pixel(self) -> None:
        np.testing.assert_allclose(beam_profile(0.0, 0.0, R), [1.0])

    def test_profile_never_outgrows_the_full_turn(self) -> None:
        profile = beam_profile(math.tau, math.tau, R)
        self.assertLessEqual(len(profile), R)


class ProjectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.out = np.zeros((1, R, 3), dtype=BUFFER_DTYPE)
        self.beam = math.radians(20.0)          # 20 px core at 1 px/deg
        self.blur = math.radians(10.0)          # 10 px falloff each side

    @staticmethod
    def _all(level: float) -> np.ndarray:
        return np.full(len(BeamLightId), level, dtype=BUFFER_DTYPE)

    def _one(self, light: BeamLightId, level: float = 1.0) -> np.ndarray:
        v = np.zeros(len(BeamLightId), dtype=BUFFER_DTYPE)
        v[light] = level
        return v

    def _project(self, beam_lights: np.ndarray, heading: float = 0.0) -> None:
        project_beam_lights(beam_lights, heading, self.beam, self.blur, self.out)

    def test_lights_land_at_their_headings_from_heading_zero(self) -> None:
        self._project(self._all(1.0))
        white, blue = self.out[0, :, 0], self.out[0, :, 1]
        self.assertAlmostEqual(white[0],         1.0, places=5)   # front
        self.assertAlmostEqual(white[R // 2],    1.0, places=5)   # back
        self.assertAlmostEqual(blue[3 * R // 4], 1.0, places=5)   # left  = −90°
        self.assertAlmostEqual(blue[R // 4],     1.0, places=5)   # right = +90°
        self.assertEqual(blue[0], 0.0)                            # no blue on the whites' spots
        self.assertEqual(white[R // 4], 0.0)
        self.assertTrue(np.all(self.out[0, :, 2] == 0.0))         # reserved channel untouched

    def test_lines_follow_the_heading(self) -> None:
        self._project(self._one(BeamLightId.FRONT_WHITE), math.radians(90.0))
        lit = np.flatnonzero(self.out[0, :, 0] > 0.0)      # the core is a plateau — take its centre
        self.assertEqual(int(round(float(lit.mean()))), R // 4)

    def test_line_is_symmetric_around_its_centre(self) -> None:
        self._project(self._one(BeamLightId.BACK_WHITE))
        white = self.out[0, :, 0]
        centre = R // 2
        for d in range(1, 31):
            self.assertAlmostEqual(white[centre - d], white[centre + d], places=6, msg=f"offset {d}")

    def test_line_wraps_across_the_zero_azimuth_seam(self) -> None:
        self._project(self._one(BeamLightId.FRONT_WHITE), math.radians(-2.0))
        white = self.out[0, :, 0]
        self.assertAlmostEqual(white[R - 2], 1.0, places=5)   # centre, just before the seam
        self.assertAlmostEqual(white[R - 1], 1.0, places=5)   # core carries on …
        self.assertAlmostEqual(white[0], 1.0, places=5)       # … across the seam
        self.assertGreater(white[12], 0.0)                    # into the fade on the far side
        self.assertAlmostEqual(white[21], 0.0, places=6)      # and out (core 10 + blur 10 from centre)

    def test_core_and_fade_have_the_requested_widths(self) -> None:
        self._project(self._one(BeamLightId.FRONT_WHITE))
        white = self.out[0, :, 0]
        self.assertAlmostEqual(white[10], 1.0, places=5)      # solid to the core's edge
        self.assertGreater(white[10], white[15])              # then falls off
        self.assertGreater(white[15], 0.0)
        self.assertAlmostEqual(white[20], 0.0, places=6)      # dark past core + blur

    def test_level_scales_and_zero_or_nan_paint_nothing(self) -> None:
        levels = np.array([0.0, 0.25, float('nan'), 0.0], dtype=BUFFER_DTYPE)
        self._project(levels)
        self.assertAlmostEqual(self.out[0, R // 2, 0], 0.25, places=5)
        self.assertTrue(np.all(self.out[0, :, 1] == 0.0))

    def test_buffer_is_cleared_between_calls(self) -> None:
        self._project(self._all(1.0))
        self._project(self._all(0.0))
        self.assertTrue(np.all(self.out == 0.0))


class FlashPaintTest(unittest.TestCase):
    """Recent flashes drawn over the beam view at the heading they lit, fading with age."""

    NOW = 100.0
    SECONDS = 1.0

    def setUp(self) -> None:
        self.out = np.zeros((1, R, 3), dtype=BUFFER_DTYPE)
        self.beam = math.radians(4.0)
        self.blur = 0.0

    def _paint(self, *flashes: Flash, seconds: float = SECONDS) -> None:
        paint_flashes(list(flashes), self.NOW, seconds, self.beam, self.blur, self.out)

    def test_a_new_flash_lights_its_heading_at_full(self) -> None:
        self._paint(Flash(math.radians(90.0), 0.8, 0.0, self.NOW))
        self.assertAlmostEqual(float(self.out[0, 90, 0]), 0.8, places=5)
        self.assertEqual(float(self.out[0, 0, 0]), 0.0)             # nothing at the current bar

    def test_it_fades_linearly_and_is_gone_after_the_seconds(self) -> None:
        self._paint(Flash(math.radians(90.0), 1.0, 0.0, self.NOW - 0.25))
        self.assertAlmostEqual(float(self.out[0, 90, 0]), 0.75, places=5)
        self.out.fill(0.0)
        self._paint(Flash(math.radians(90.0), 1.0, 0.0, self.NOW - self.SECONDS))
        self.assertTrue(np.all(self.out == 0.0))

    def test_blue_lights_both_blue_lamps(self) -> None:
        self._paint(Flash(0.0, 0.0, 0.5, self.NOW))
        blue = self.out[0, :, 1]
        self.assertAlmostEqual(float(blue[R // 4]), 0.5, places=5)       # right, +90°
        self.assertAlmostEqual(float(blue[3 * R // 4]), 0.5, places=5)   # left, −90°
        self.assertTrue(np.all(self.out[0, :, 0] == 0.0))

    def test_zero_seconds_draws_nothing(self) -> None:
        self._paint(Flash(0.0, 1.0, 1.0, self.NOW), seconds=0.0)
        self.assertTrue(np.all(self.out == 0.0))

    def test_the_steady_image_underneath_is_kept(self) -> None:
        self.out[0, :, 0] = 0.4                                          # a DIM line everywhere
        self._paint(Flash(math.radians(90.0), 1.0, 0.0, self.NOW - 0.9))  # a faint old flash
        self.assertAlmostEqual(float(self.out[0, 90, 0]), 0.4, places=5)  # MAX, never darker
        self.assertAlmostEqual(float(self.out[0, 200, 0]), 0.4, places=5)


class FlashStoreTest(unittest.TestCase):

    def test_flashes_come_back_oldest_first_and_bounded(self) -> None:
        store = FlashStoreMixin()
        for i in range(100):
            store.add_flash(float(i), 1.0, 0.0)
        flashes = store.get_flashes()
        self.assertLess(len(flashes), 100)
        self.assertEqual(flashes[-1].azimuth, 99.0)
        self.assertEqual([f.azimuth for f in flashes], sorted(f.azimuth for f in flashes))


if __name__ == '__main__':
    unittest.main()
