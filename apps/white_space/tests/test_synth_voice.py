"""Tests for the light synth's voice: two oscillators on one time, the window per side, presence,
the push, and one patch drawing differently for different sources (docs/LIGHT_SYNTH.md)."""

import unittest

import numpy as np

from apps.white_space.light.synth import Voice, Input, OscillatorSettings, WindowSettings, PresenceSettings, PushSettings

STEP = 0.1                                              # degrees per pixel
OFFSETS = np.arange(-600, 601) * STEP                   # a strip 60° each side of the person
DISTANCE = np.abs(OFFSETS)
LEFT = OFFSETS < 0.0
RIGHT = OFFSETS > 0.0
NO_SOURCES = ({}, {})
MIN_INTERVAL = 4.0


def lines(output: np.ndarray) -> list[tuple[float, float]]:
    """(centre, width) in degrees of the whole lines on the person's right: the half line a
    phase of 0 leaves at the person is not one."""
    lit = (output > 0.5) & RIGHT
    edges = np.flatnonzero(np.diff(np.concatenate(([False], lit, [False])).astype(int)))
    found = [(float((OFFSETS[s] + OFFSETS[e - 1]) / 2.0), float((e - s) * STEP)) for s, e in zip(edges[::2], edges[1::2])]
    return [(centre, width) for centre, width in found if centre - width / 2.0 > 2 * STEP]


def centres(output: np.ndarray) -> list[float]:
    return [centre for centre, _ in lines(output)]


def has_line_at(output: np.ndarray, position: float) -> bool:
    return any(abs(centre - position) < 0.2 for centre in centres(output))


class VoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.one, self.two = OscillatorSettings(), OscillatorSettings()
        self.window, self.presence, self.push = WindowSettings(), PresenceSettings(), PushSettings()
        self.presence.attack_seconds = 0.0              # present at once, unless a test says otherwise
        self.one.interval = self.two.interval = 10.0
        self.one.pulse_width = self.two.pulse_width = 0.3

    def _voice(self) -> Voice:
        return Voice(self.one, self.two, self.window, self.presence, self.push)

    def _arrived(self) -> Voice:
        voice = self._voice()
        voice.update(0.01, True, False, NO_SOURCES, MIN_INTERVAL)
        return voice

    def _render(self, voice: Voice, reach_left: float = 40.0, reach_right: float = 40.0, sources=NO_SOURCES):
        return voice.render(DISTANCE, LEFT, reach_left, reach_right, sources)

    # -- the two sides and the window --

    def test_equal_reaches_draw_the_same_on_both_sides(self) -> None:
        for output in self._render(self._arrived()):
            np.testing.assert_array_equal(output, output[::-1])

    def test_the_reach_is_the_one_thing_that_differs_between_the_sides(self) -> None:
        output, _ = self._render(self._arrived(), reach_left=20.0, reach_right=50.0)
        self.assertFalse(output[OFFSETS < -20.0].any())
        self.assertTrue(output[(OFFSETS > 20.0) & (OFFSETS < 40.0)].any())
        near = np.abs(OFFSETS) < 15.0                   # inside both windows' full part the sides agree
        np.testing.assert_array_equal(output[near], output[::-1][near])

    def test_nothing_is_drawn_past_the_reach_and_no_line_is_cut(self) -> None:
        self.one.interval = 4.0                         # lines at 4, 8, … 36: two of them in the taper, 32 to 40
        output, _ = self._render(self._arrived())
        self.assertFalse(output[DISTANCE >= 40.0].any())
        found = lines(output)
        full = 0.3 * 4.0
        for centre, width in found:
            if centre < 30.0:
                self.assertAlmostEqual(width, full, delta=2 * STEP)        # whole before the taper
        last_centre, last_width = found[-1]
        self.assertAlmostEqual(last_centre, 36.0, delta=2 * STEP)          # still centred where it belongs,
        self.assertLess(last_width, 0.7 * full)                            # thinned, not cut

    def test_a_dark_output_stays_dark_and_a_solid_one_is_solid_up_to_the_taper(self) -> None:
        self.one.pulse_width, self.two.pulse_width = 0.0, 1.0
        dark, solid = self._render(self._arrived())
        self.assertFalse(dark.any())
        self.assertTrue((solid[DISTANCE <= 32.0] == 1.0).all())           # the taper is the last fifth of 40
        self.assertFalse(solid[DISTANCE >= 40.0].any())
        self.assertFalse((solid[(DISTANCE > 32.0) & (DISTANCE < 40.0)] == 1.0).all())   # opened into lines

    def test_the_interval_never_goes_below_the_visual_limit(self) -> None:
        self.one.interval = 2.0
        self.one.pulse_width = 0.2
        voice = self._arrived()
        found = centres(self._render(voice)[0])
        self.assertAlmostEqual(found[1] - found[0], MIN_INTERVAL, delta=2 * STEP)

    # -- presence --

    def test_presence_opens_the_window_from_the_person_and_closes_it(self) -> None:
        self.presence.attack_seconds, self.presence.release_seconds = 1.0, 1.0
        voice = self._voice()
        self.assertFalse(voice.alive)
        voice.update(0.5, True, False, NO_SOURCES, MIN_INTERVAL)
        half, _ = self._render(voice)
        self.assertTrue(half[DISTANCE < 2.0].any())
        self.assertFalse(half[DISTANCE >= 20.0].any())                    # half open: half the reach
        voice.update(0.5, True, False, NO_SOURCES, MIN_INTERVAL)
        self.assertTrue(self._render(voice)[0][DISTANCE > 25.0].any())
        voice.update(0.5, False, False, NO_SOURCES, MIN_INTERVAL)
        self.assertTrue(voice.alive)
        voice.update(0.5, False, False, NO_SOURCES, MIN_INTERVAL)
        self.assertFalse(voice.alive)
        self.assertFalse(self._render(voice)[0].any())

    # -- the push and the time --

    def test_a_push_moves_each_output_its_own_way_and_the_lines_keep_the_gain(self) -> None:
        self.one.push, self.two.push = 6.0, -6.0       # output 1 outward, output 2 inward; both standing
        self.push.settle_seconds = 1.0
        voice = self._arrived()
        self.assertTrue(has_line_at(self._render(voice)[0], 10.0))
        voice.update(0.01, True, True, NO_SOURCES, MIN_INTERVAL)           # the hit
        for _ in range(150):
            voice.update(0.01, True, False, NO_SOURCES, MIN_INTERVAL)
        one, two = self._render(voice)
        self.assertTrue(has_line_at(one, 13.0))                            # 6 deg/s eased over a second: 3° out
        self.assertTrue(has_line_at(two, 7.0))                             # and 3° in
        settled = centres(one)
        for _ in range(100):
            voice.update(0.01, True, False, NO_SOURCES, MIN_INTERVAL)
        self.assertEqual(centres(self._render(voice)[0]), settled)         # settled: nothing comes back

    def test_both_oscillators_travel_on_the_voices_time(self) -> None:
        self.one.speed, self.two.speed = 2.0, -2.0
        voice = self._voice()
        for _ in range(100):
            voice.update(0.01, True, False, NO_SOURCES, MIN_INTERVAL)      # one second
        one, two = self._render(voice)
        self.assertTrue(has_line_at(one, 12.0))                            # 2° outward
        self.assertTrue(has_line_at(two, 8.0))                             # 2° inward

    # -- the patch and the sources --

    def test_one_patch_draws_differently_for_different_sources(self) -> None:
        self.one.pulse_width, self.one.pulse_width_amount = 0.0, 1.0
        thin = self._render(self._arrived(), sources=({Input.PULSE_WIDTH: 0.2}, {}))[0]
        thick = self._render(self._arrived(), sources=({Input.PULSE_WIDTH: 0.8}, {}))[0]
        self.assertGreater(np.count_nonzero(thick), 3 * np.count_nonzero(thin))
        self.assertFalse(self._render(self._arrived())[0].any())           # no source: the base, dark

    def test_a_source_per_pixel_varies_the_width_along_the_strip(self) -> None:
        self.one.pulse_width, self.one.pulse_width_amount = 0.2, 0.6
        swell = np.where(DISTANCE < 15.0, 1.0, 0.0)
        found = lines(self._render(self._arrived(), sources=({Input.PULSE_WIDTH: swell}, {}))[0])
        self.assertAlmostEqual(found[0][1], 8.0, delta=2 * STEP)           # the line at 10°
        self.assertAlmostEqual(found[1][1], 2.0, delta=2 * STEP)           # the line at 20°

    def test_the_interval_source_moves_the_pitch_in_octaves(self) -> None:
        self.one.interval_amount = 1.0
        self.one.pulse_width = 0.2
        voice = self._voice()
        voice.update(0.01, True, False, ({Input.INTERVAL: 1.0}, {}), MIN_INTERVAL)
        found = centres(self._render(voice, 60.0, 60.0)[0])
        self.assertAlmostEqual(found[1] - found[0], 20.0, delta=2 * STEP)

    def test_the_settings_are_read_live(self) -> None:
        voice = self._arrived()
        narrow = np.count_nonzero(self._render(voice)[0])
        self.one.pulse_width = 0.6                      # the panel changes the patch under a running voice
        self.assertGreater(np.count_nonzero(self._render(voice)[0]), 1.5 * narrow)


if __name__ == "__main__":
    unittest.main()
