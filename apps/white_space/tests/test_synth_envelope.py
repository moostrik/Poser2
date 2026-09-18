"""Tests for the light synth's envelope (over positions and over time) and its modulation slot
(docs/LIGHT_SYNTH.md)."""

import unittest

import numpy as np

from apps.white_space.light.synth import Envelope, Slot, Curve

P = np.arange(0.0, 50.0, 0.1)


def at(values: np.ndarray, position: float) -> float:
    return float(values[int(round(position / 0.1))])


class EnvelopeOverPositionsTest(unittest.TestCase):
    def test_full_until_the_fall_and_nothing_past_the_length(self) -> None:
        shape = Envelope.over_positions(P, 0.0, 10.0, 40.0)
        self.assertTrue((shape[P <= 30.0] == 1.0).all())
        self.assertTrue((shape[P >= 40.0] == 0.0).all())
        self.assertAlmostEqual(at(shape, 35.0), 0.5, places=5)             # halfway down the fall
        self.assertTrue((np.diff(shape) <= 1e-7).all())                    # only ever down

    def test_the_fall_is_smooth_at_both_ends(self) -> None:
        shape = Envelope.over_positions(P, 0.0, 10.0, 40.0)
        self.assertLess(abs(at(shape, 30.1) - 1.0), 1e-3)                  # eased, not a corner
        self.assertLess(at(shape, 39.9), 1e-3)

    def test_a_rise_from_position_zero(self) -> None:
        shape = Envelope.over_positions(P, 5.0, 10.0, 40.0)
        self.assertEqual(at(shape, 0.0), 0.0)
        self.assertAlmostEqual(at(shape, 2.5), 0.5, places=5)
        self.assertEqual(at(shape, 5.0), 1.0)

    def test_no_fall_is_full_to_the_length(self) -> None:
        shape = Envelope.over_positions(P, 0.0, 0.0, 40.0)
        self.assertTrue((shape[P < 39.95] == 1.0).all())
        self.assertTrue((shape[P > 40.05] == 0.0).all())

    def test_no_length_is_nothing(self) -> None:
        self.assertFalse(Envelope.over_positions(P, 0.0, 0.0, 0.0).any())

    def test_a_length_per_position(self) -> None:
        length = np.where(P < 25.0, 40.0, 20.0)                            # as two sides with two reaches
        shape = Envelope.over_positions(P, 0.0, length * 0.2, length)
        self.assertEqual(at(shape, 24.0), 1.0)
        self.assertEqual(at(shape, 26.0), 0.0)


class EnvelopeOverTimeTest(unittest.TestCase):
    def test_it_rises_while_the_gate_is_open_and_falls_when_it_closes(self) -> None:
        envelope = Envelope()
        self.assertAlmostEqual(envelope.update(True, 0.5, 1.0, 2.0), 0.5, places=6)
        self.assertEqual(envelope.update(True, 0.5, 1.0, 2.0), 1.0)
        self.assertEqual(envelope.update(True, 5.0, 1.0, 2.0), 1.0)        # holds
        self.assertAlmostEqual(envelope.update(False, 1.0, 1.0, 2.0), 0.5, places=6)
        self.assertEqual(envelope.update(False, 1.0, 1.0, 2.0), 0.0)
        self.assertEqual(envelope.level, 0.0)

    def test_zero_seconds_is_at_once(self) -> None:
        envelope = Envelope()
        self.assertEqual(envelope.update(True, 0.01, 0.0, 1.0), 1.0)
        self.assertEqual(envelope.update(False, 0.01, 1.0, 0.0), 0.0)

    def test_a_gate_closing_early_falls_from_where_it_is(self) -> None:
        envelope = Envelope()
        values = [envelope.update(True, 0.01, 1.0, 1.0) for _ in range(30)]
        values += [envelope.update(False, 0.01, 1.0, 1.0) for _ in range(30)]
        self.assertLess(max(abs(b - a) for a, b in zip(values, values[1:])), 0.02)     # no step at the turn
        self.assertAlmostEqual(values[-1], 0.0, places=6)


class SlotTest(unittest.TestCase):
    def test_the_source_moves_the_input_from_its_base_by_the_amount(self) -> None:
        self.assertAlmostEqual(Slot.modulate(0.2, 0.6, 0.0), 0.2)
        self.assertAlmostEqual(Slot.modulate(0.2, 0.6, 0.5), 0.5)
        self.assertAlmostEqual(Slot.modulate(0.2, 0.6, 1.0), 0.8)
        self.assertAlmostEqual(Slot.modulate(0.8, -0.6, 1.0), 0.2)         # the amount's sign is the direction

    def test_an_lfo_moves_the_input_around_its_base(self) -> None:
        swing = Slot.modulate(0.5, 0.2, np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(swing, [0.3, 0.5, 0.7])

    def test_the_interval_moves_in_octaves(self) -> None:
        self.assertAlmostEqual(Slot.modulate_octaves(10.0, 1.0, 1.0), 20.0)
        self.assertAlmostEqual(Slot.modulate_octaves(10.0, -1.0, 1.0), 5.0)
        self.assertAlmostEqual(Slot.modulate_octaves(10.0, 1.0, 0.0), 10.0)
        up, down = Slot.modulate_octaves(10.0, 1.0, 0.5), Slot.modulate_octaves(10.0, 1.0, -0.5)
        self.assertAlmostEqual(up / 10.0, 10.0 / down)                     # an LFO's swing is symmetric in pitch

    def test_one_source_into_two_intervals_keeps_their_ratio(self) -> None:
        for source in (0.0, 0.3, 1.0):
            self.assertAlmostEqual(Slot.modulate_octaves(20.0, 0.7, source) / Slot.modulate_octaves(10.0, 0.7, source), 2.0)

    def test_a_unit_input_stops_at_its_ends(self) -> None:
        np.testing.assert_allclose(Slot.unit(np.array([-0.2, 0.4, 1.3])), [0.0, 0.4, 1.0])

    def test_bypassed_the_source_is_nothing_and_the_input_its_base(self) -> None:
        self.assertEqual(Slot.bypassed(True, 0.7), 0.0)
        self.assertEqual(Slot.bypassed(False, 0.7), 0.7)
        self.assertAlmostEqual(Slot.modulate(0.2, 0.6, Slot.bypassed(True, 1.0)), 0.2)

    def test_every_curve_is_smooth_keeps_the_sign_and_leaves_the_ends(self) -> None:
        source = np.linspace(-1.0, 1.0, 2001)
        for curve in Curve:
            eased = Slot.curve(source, curve)
            self.assertEqual(float(eased[0]), -1.0)
            self.assertEqual(float(eased[1000]), 0.0)
            self.assertEqual(float(eased[-1]), 1.0)
            np.testing.assert_allclose(eased, -eased[::-1], atol=1e-12)       # odd: the sign kept, symmetric
            self.assertTrue((np.diff(eased) >= -1e-12).all())                 # never turning back
            self.assertLess(float(np.abs(np.diff(eased)).max()), 0.005)       # no step

    def test_the_curves_ease_the_way_they_say(self) -> None:
        self.assertAlmostEqual(Slot.curve(0.5, Curve.LINEAR), 0.5)
        self.assertAlmostEqual(Slot.curve(0.5, Curve.EASE_IN), 0.25)           # little at first
        self.assertAlmostEqual(Slot.curve(0.5, Curve.EASE_OUT), 0.75)          # much at first
        self.assertAlmostEqual(Slot.curve(0.5, Curve.EASE_IN_OUT), 0.5)
        self.assertLess(Slot.curve(0.25, Curve.EASE_IN_OUT), 0.25)
        self.assertGreater(Slot.curve(0.75, Curve.EASE_IN_OUT), 0.75)


if __name__ == "__main__":
    unittest.main()
