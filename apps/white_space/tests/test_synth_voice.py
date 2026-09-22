"""Tests for the light synth's voice: two oscillators on one time, the window per side, presence,
the push, and one patch drawing differently for different sources (docs/LIGHT_SYNTH.md)."""

import unittest

import numpy as np

from apps.white_space.light.synth import Voice, Parameter, Curve, OscillatorSettings, WindowSettings, LfoSettings

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
        self.window, self.lfo = WindowSettings(), LfoSettings()
        self.window.attack_seconds = 0.0                # present at once, unless a test says otherwise
        self.one.pitch = self.two.pitch = 36.0        # 36 lines per turn: a 10° interval
        self.one.pulse_width = self.two.pulse_width = 0.3

    def _voice(self) -> Voice:
        return Voice(self.one, self.two, self.window, self.lfo)

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
        self.one.pitch = 90.0                           # a 4° interval: lines at 4, 8, … 36, two of them in the taper, 32 to 40
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

    def test_the_pitch_never_goes_above_the_visual_limit(self) -> None:
        self.one.pitch = 180.0                          # a 2° interval, asked for; the limit is 4°
        self.one.pulse_width = 0.2
        voice = self._arrived()
        found = centres(self._render(voice)[0])
        self.assertAlmostEqual(found[1] - found[0], MIN_INTERVAL, delta=2 * STEP)

    # -- presence --

    def test_presence_opens_the_window_from_the_person_and_closes_it(self) -> None:
        self.window.attack_seconds, self.window.release_seconds = 1.0, 1.0
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
        self.one.push_release_seconds = self.two.push_release_seconds = 1.0
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

    def test_each_oscillator_releases_its_push_on_its_own_time(self) -> None:
        self.one.push = self.two.push = 6.0
        self.one.push_release_seconds, self.two.push_release_seconds = 0.1, 2.0
        voice = self._arrived()
        voice.update(0.01, True, True, NO_SOURCES, MIN_INTERVAL)           # the hit
        for _ in range(50):                                                # half a second on
            voice.update(0.01, True, False, NO_SOURCES, MIN_INTERVAL)
        one_before, two_before = (centres(o) for o in self._render(voice))
        voice.update(0.01, True, False, NO_SOURCES, MIN_INTERVAL)
        one_after, two_after = (centres(o) for o in self._render(voice))
        self.assertEqual(one_after, one_before)                            # released: standing again
        self.assertNotEqual(two_after, two_before)                         # still being pushed

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
        thin = self._render(self._arrived(), sources=({Parameter.PULSE_WIDTH: 0.2}, {}))[0]
        thick = self._render(self._arrived(), sources=({Parameter.PULSE_WIDTH: 0.8}, {}))[0]
        self.assertGreater(np.count_nonzero(thick), 3 * np.count_nonzero(thin))
        self.assertFalse(self._render(self._arrived())[0].any())           # no source: the base, dark

    def test_a_source_per_pixel_varies_the_width_along_the_strip(self) -> None:
        self.one.pulse_width, self.one.pulse_width_amount = 0.2, 0.6
        swell = np.where(DISTANCE < 15.0, 1.0, 0.0)
        found = lines(self._render(self._arrived(), sources=({Parameter.PULSE_WIDTH: swell}, {}))[0])
        self.assertAlmostEqual(found[0][1], 8.0, delta=2 * STEP)           # the line at 10°
        self.assertAlmostEqual(found[1][1], 2.0, delta=2 * STEP)           # the line at 20°

    def test_the_pitch_source_moves_the_pitch_in_lines(self) -> None:
        self.one.pitch_amount = -18.0                   # from 36 to 18 lines per turn: the interval doubles
        self.one.pulse_width = 0.2
        voice = self._voice()
        voice.update(0.01, True, False, ({Parameter.PITCH: 1.0}, {}), MIN_INTERVAL)
        found = centres(self._render(voice, 60.0, 60.0)[0])
        self.assertAlmostEqual(found[1] - found[0], 20.0, delta=2 * STEP)

    def test_one_source_into_two_pitches_keeps_their_difference(self) -> None:
        self.one.pitch, self.two.pitch = 36.0, 72.0     # 10° and 5°
        self.one.pitch_amount = self.two.pitch_amount = 18.0
        self.one.pulse_width = self.two.pulse_width = 0.2
        voice = self._voice()
        voice.update(0.01, True, False, ({Parameter.PITCH: 1.0}, {Parameter.PITCH: 1.0}), MIN_INTERVAL)
        one, two = (centres(output) for output in self._render(voice, 60.0, 60.0))
        self.assertAlmostEqual(one[1] - one[0], 360.0 / 54.0, delta=2 * STEP)      # 54 and 90 lines: the
        self.assertAlmostEqual(two[1] - two[0], 360.0 / 90.0, delta=2 * STEP)      # difference of 36 kept, not the ratio

    def test_a_pitch_never_goes_below_one_line_per_half_turn(self) -> None:
        self.one.pitch, self.one.pulse_width = 2.0, 0.1
        self.one.pitch_amount = -10.0                   # would ask for a negative pitch
        voice = self._voice()
        voice.update(0.01, True, False, ({Parameter.PITCH: 1.0}, {}), MIN_INTERVAL)
        found = lines(self._render(voice, 60.0, 60.0)[0])
        self.assertEqual(len(found), 0)                 # the first whole line sits at 180°, out of reach
        self.assertTrue(self._render(voice, 60.0, 60.0)[0][DISTANCE < 9.0].all())   # the half line at the person, 18° wide

    # -- the LFO --

    def test_the_lfo_is_silent_at_level_zero(self) -> None:
        voice = self._arrived()
        for _ in range(50):
            self.assertEqual(voice.update_lfo(0.01, 0.0), 0.0)
        self.assertEqual(voice.lfo, 0.0)

    def test_the_lfo_swings_both_ways_by_its_level_at_its_rate(self) -> None:
        self.lfo.rate, self.lfo.level = 0.5, 0.6                           # a cycle every two seconds
        voice = self._arrived()
        outputs = [voice.update_lfo(0.01, 0.0) for _ in range(200)]        # one cycle
        self.assertAlmostEqual(max(outputs), 0.6, places=3)
        self.assertAlmostEqual(min(outputs), -0.6, places=3)
        self.assertAlmostEqual(outputs[-1], 0.6, places=3)                 # back where it started
        self.assertLess(max(abs(b - a) for a, b in zip(outputs, outputs[1:])), 0.02)   # smooth

    def test_a_source_brings_the_lfo_in_smoothly(self) -> None:
        self.lfo.rate, self.lfo.level_amount = 0.5, 1.0                    # the level is played, from 0
        voice = self._arrived()
        outputs = [voice.update_lfo(0.01, i / 300.0) for i in range(300)]
        self.assertEqual(outputs[0], 0.0)
        self.assertGreater(max(abs(o) for o in outputs[200:]), 0.5)
        self.assertLess(max(abs(b - a) for a, b in zip(outputs, outputs[1:])), 0.03)   # no step as it comes in

    def test_the_lfo_into_a_phase_rocks_the_lines_about_their_place(self) -> None:
        self.lfo.rate, self.lfo.level = 0.5, 1.0
        self.one.phase_amount = 0.25                                       # a quarter interval each way
        voice = self._arrived()
        seen = []
        for _ in range(200):                                               # one cycle
            voice.update_lfo(0.01, 0.0)
            found = centres(self._render(voice, sources=({Parameter.PHASE: voice.lfo}, {}))[0])
            seen.append(min(found, key=lambda centre: abs(centre - 10.0)))   # the line that rests at 10°
        self.assertAlmostEqual(max(seen), 12.5, delta=0.2)                 # a quarter of 10° out,
        self.assertAlmostEqual(min(seen), 7.5, delta=0.2)                  # a quarter in,
        self.assertAlmostEqual(seen[-1], seen[0], delta=0.3)               # and back where it began
        self.assertLess(max(abs(b - a) for a, b in zip(seen, seen[1:])), 0.2)   # rocking, never stepping
        still = centres(self._render(voice, sources=({Parameter.PHASE: voice.lfo}, {}))[1])
        self.assertAlmostEqual(still[0], 10.0, delta=0.2)                  # the other output does not move

    # -- the holds --

    def test_a_held_input_is_its_base_while_the_others_follow(self) -> None:
        self.one.pulse_width, self.one.pulse_width_amount = 0.2, 0.6
        self.one.pitch_amount = -18.0                   # the interval doubles to 20°
        sources = ({Parameter.PULSE_WIDTH: 1.0, Parameter.PITCH: 1.0}, {})
        voice = self._voice()
        voice.update(0.01, True, False, sources, MIN_INTERVAL)
        followed = lines(self._render(voice, 60.0, 60.0, sources)[0])
        self.assertAlmostEqual(followed[0][1], 0.8 * 20.0, delta=2 * STEP)      # width and pitch both from the source
        self.one.pulse_width_bypass = True
        held = lines(self._render(voice, 60.0, 60.0, sources)[0])
        self.assertAlmostEqual(held[0][1], 0.2 * 20.0, delta=2 * STEP)          # the width at its knob,
        self.assertAlmostEqual(held[1][0] - held[0][0], 20.0, delta=2 * STEP)   # the pitch still following
        self.assertEqual(self.one.pulse_width_amount, 0.6)                      # and the amount left as it was
        self.one.pulse_width_bypass = False
        again = lines(self._render(voice, 60.0, 60.0, sources)[0])
        self.assertAlmostEqual(again[0][1], 0.8 * 20.0, delta=2 * STEP)         # let go: it follows at once

    def test_a_bypassed_pitch_and_speed_ignore_their_sources(self) -> None:
        self.one.pitch_amount, self.one.speed_amount = 36.0, 10.0
        self.one.pitch_bypass = self.one.speed_bypass = True
        self.one.pulse_width = 0.2
        voice = self._voice()
        for _ in range(100):
            voice.update(0.01, True, False, ({Parameter.PITCH: 1.0, Parameter.SPEED: 1.0}, {}), MIN_INTERVAL)
        found = centres(self._render(voice)[0])
        self.assertAlmostEqual(found[0], 10.0, delta=0.2)                       # not doubled, not travelled

    def test_a_held_lfo_level_is_silent_whatever_its_source_says(self) -> None:
        self.lfo.rate, self.lfo.level_amount, self.lfo.level_bypass = 0.5, 1.0, True
        voice = self._arrived()
        self.assertEqual({voice.update_lfo(0.01, 1.0) for _ in range(50)}, {0.0})

    def test_an_oscillator_switched_off_is_dark_and_the_other_draws(self) -> None:
        self.one.enabled = False
        one, two = self._render(self._arrived())
        self.assertFalse(one.any())
        self.assertTrue(two.any())
        self.one.enabled = True
        self.assertTrue(self._render(self._arrived())[0].any())

    def test_the_settings_are_read_live(self) -> None:
        voice = self._arrived()
        narrow = np.count_nonzero(self._render(voice)[0])
        self.one.pulse_width = 0.6                      # the panel changes the patch under a running voice
        self.assertGreater(np.count_nonzero(self._render(voice)[0]), 1.5 * narrow)


if __name__ == "__main__":
    unittest.main()
